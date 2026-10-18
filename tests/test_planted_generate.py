import re

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from planted_generate import DATA_DIR, RECIPES, TARGET, file_stem, generate, load
from planted_names import NAMES as DOMAINS
from planted_truth import FAULT_TYPES, UNMEASURABLE_FAULTS, VARIANTS, ColumnTruth, DatasetTruth

NAMES = [recipe.name for recipe in RECIPES]
PHONE = re.compile(r"^(\+90 5|05)\d{2} \d{3} \d{2} \d{2}$|^\+905\d{9}$")
EMAIL = re.compile(r"^[a-z0-9.]+@[a-z0-9.]+\.[a-z]+$")


def faults_of(truth: DatasetTruth, fault: str) -> list[ColumnTruth]:
    return [c for c in truth.columns if c.fault == fault]


@pytest.fixture(scope="module", params=NAMES)
def dataset(request):
    return generate(next(r for r in RECIPES if r.name == request.param))


def test_the_committed_files_are_what_the_generator_writes(dataset):
    frame, truth = dataset
    committed_frame, committed_truth = load(truth.dataset)
    assert committed_truth == truth
    pd.testing.assert_frame_equal(committed_frame, frame, check_dtype=False, rtol=1e-9)


def test_generation_is_deterministic():
    first, _ = generate(RECIPES[0])
    second, _ = generate(RECIPES[0])
    pd.testing.assert_frame_equal(first, second)


def test_different_seeds_give_different_data():
    a, _ = generate(RECIPES[0])
    b, _ = generate(RECIPES[1])
    assert not a.equals(b)


def test_names_are_neutral_and_the_truth_covers_every_column(dataset):
    frame, truth = dataset
    features = [c for c in frame.columns if c != TARGET]
    assert all(re.fullmatch(r"c\d{2}", name) for name in features)
    assert [c.name for c in truth.columns] == features
    assert truth.target == TARGET
    assert len(frame) == truth.n_rows


def test_the_target_has_both_outcomes_and_no_gaps(dataset):
    frame, truth = dataset
    assert frame[TARGET].notna().all()
    if truth.task_type == "classification":
        assert set(frame[TARGET]) == {0, 1}
        assert 0.15 < frame[TARGET].mean() < 0.55
    else:
        assert frame[TARGET].nunique() > 100


def test_every_fault_type_is_planted_in_at_least_two_datasets():
    counts = dict.fromkeys(FAULT_TYPES, 0)
    for recipe in RECIPES:
        _, truth = generate(recipe)
        for fault in {c.fault for c in truth.columns if c.fault}:
            counts[fault] += 1
    assert all(count >= 2 for count in counts.values()), counts


def test_the_set_has_between_six_and_ten_datasets():
    assert 6 <= len(RECIPES) <= 10
    assert len({r.seed for r in RECIPES}) == len(RECIPES)


def test_unmeasurable_faults_are_marked_and_the_rest_are_not():
    for recipe in RECIPES:
        _, truth = generate(recipe)
        for column in truth.columns:
            assert column.measurable == (column.fault not in UNMEASURABLE_FAULTS)
    assert set(UNMEASURABLE_FAULTS) == {"sentinel"}


def test_linear_leaks_follow_the_target_closely(dataset):
    frame, truth = dataset
    for column in faults_of(truth, "leak_linear"):
        assert abs(frame[column.name].corr(frame[TARGET])) > 0.95


def test_nonlinear_leaks_follow_the_target_in_rank_and_are_not_affine_in_it(dataset):
    frame, truth = dataset
    for column in faults_of(truth, "leak_nonlinear"):
        series = frame[column.name]
        assert series.corr(frame[TARGET], method="spearman") > 0.8
        assert series.min() > 0
        if truth.task_type == "regression":
            assert series.corr(frame[TARGET]) < 0.99


def test_categorical_leaks_are_a_noisy_thresholding_of_the_target(dataset):
    frame, truth = dataset
    for column in faults_of(truth, "leak_categorical"):
        groups = frame.groupby(column.name)[TARGET]
        assert frame[column.name].nunique() >= 2
        if truth.task_type == "classification":
            accuracy = groups.agg(lambda s: max(s.mean(), 1 - s.mean()) * len(s)).sum() / len(frame)
            assert accuracy > 0.8
        else:
            within = groups.var().mul(groups.size()).sum()
            assert 1 - within / (frame[TARGET].var() * len(frame)) > 0.7


def test_leaks_are_not_the_target_itself(dataset):
    frame, truth = dataset
    for column in truth.columns:
        if column.fault and column.fault.startswith("leak_"):
            assert not frame[column.name].equals(frame[TARGET])


def test_ids_are_unique(dataset):
    frame, truth = dataset
    for column in faults_of(truth, "id_sequential"):
        series = frame[column.name]
        assert series.is_unique and series.is_monotonic_increasing
        assert pd.api.types.is_integer_dtype(series)
    for column in faults_of(truth, "id_uuid"):
        series = frame[column.name]
        assert series.is_unique and series.str.len().eq(36).all()


def test_personal_data_looks_like_it(dataset):
    frame, truth = dataset
    for column in faults_of(truth, "pii_email"):
        assert frame[column.name].map(lambda v: bool(EMAIL.match(v))).all()
    for column in faults_of(truth, "pii_phone"):
        assert frame[column.name].map(lambda v: bool(PHONE.match(v))).all()
    for column in faults_of(truth, "pii_name"):
        assert frame[column.name].str.contains(" ").all()


def test_turkish_names_are_in_the_set():
    letters = set("ğüşıöçĞÜŞİÖÇ")
    seen = set()
    for recipe in RECIPES:
        frame, truth = generate(recipe)
        for column in faults_of(truth, "pii_name"):
            seen |= letters & set("".join(frame[column.name]))
    assert len(seen) >= 6


def test_duplicates_copy_their_original(dataset):
    frame, truth = dataset
    for column in faults_of(truth, "duplicate_exact"):
        original = frame[column.duplicate_of]
        pd.testing.assert_series_equal(frame[column.name], original, check_names=False)
    for column in faults_of(truth, "duplicate_scaled"):
        original = frame[column.duplicate_of]
        assert not frame[column.name].equals(original)
        assert frame[column.name].isna().equals(original.isna())
        assert abs(frame[column.name].corr(original)) > 0.9999
        assert truth.column(column.duplicate_of).role == "either"


def test_numeric_categories_are_few_unordered_integer_codes(dataset):
    frame, truth = dataset
    for column in faults_of(truth, "numeric_category"):
        series = frame[column.name]
        assert pd.api.types.is_integer_dtype(series)
        assert 5 <= series.nunique() <= 10
        assert column.role == "must_keep" and column.expected_op == "cast_to_category"
        means = frame.groupby(column.name)[TARGET].mean()
        assert means.rank().corr(pd.Series(means.index, index=means.index)) < 0.99


def test_count_decoys_are_integers_that_must_not_be_cast(dataset):
    frame, truth = dataset
    decoys = [c for c in truth.columns if c.avoid_op]
    for column in decoys:
        assert column.fault is None and column.role == "must_keep"
        assert frame[column.name].between(1, 6).all()


def test_constants_have_one_value(dataset):
    frame, truth = dataset
    for column in faults_of(truth, "constant"):
        assert frame[column.name].nunique() == 1


def test_sentinels_hide_in_an_informative_column(dataset):
    frame, truth = dataset
    for column in faults_of(truth, "sentinel"):
        series = frame[column.name]
        special = series.isin([-999, 9999])
        assert 0.03 < special.mean() < 0.12
        assert special.map({True: 1, False: 0}).nunique() == 2
        assert not column.measurable and column.role == "either"
        assert series[~special].corr(frame[TARGET][~special]) != 0


def test_only_plain_columns_have_missing_values(dataset):
    frame, truth = dataset
    for column in truth.columns:
        if column.fault in {"leak_linear", "leak_nonlinear", "leak_categorical", "id_uuid"}:
            assert frame[column.name].notna().all()
    assert frame.drop(columns=TARGET).isna().any().any()


def test_every_dataset_keeps_informative_columns(dataset):
    _, truth = dataset
    kept = [c for c in truth.columns if c.role == "must_keep"]
    assert len(kept) >= 6


def test_all_files_are_in_the_data_dir():
    for name in NAMES:
        for variant in VARIANTS:
            stem = file_stem(name, variant)
            assert (DATA_DIR / f"{stem}.csv").is_file()
            assert (DATA_DIR / f"{stem}.truth.json").is_file()


@pytest.mark.parametrize(
    "fields",
    [
        {"fault": "leak_linear", "role": "must_keep"},
        {"fault": "id_uuid", "role": "either"},
        {"fault": "duplicate_exact", "role": "either"},
        {"fault": "duplicate_exact", "role": "must_keep", "duplicate_of": "c02"},
        {"fault": "numeric_category", "role": "must_keep"},
        {"fault": "sentinel", "role": "either"},
        {"fault": "constant", "role": "must_drop", "measurable": False},
        {"role": "must_drop", "duplicate_of": "c02"},
        {"role": "must_drop", "expected_op": "cast_to_category"},
    ],
)
def test_inconsistent_column_truth_is_rejected(fields):
    with pytest.raises(ValidationError):
        ColumnTruth(name="c01", **fields)


def test_dataset_truth_rejects_a_copy_of_an_unknown_column():
    copy = ColumnTruth(name="c01", role="either", fault="duplicate_exact", duplicate_of="c09")
    with pytest.raises(ValidationError):
        DatasetTruth(
            dataset="d", seed=1, target="y", task_type="regression", n_rows=3, columns=[copy]
        )


def test_dataset_truth_rejects_the_target_among_the_features():
    with pytest.raises(ValidationError):
        DatasetTruth(
            dataset="d",
            seed=1,
            target="y",
            task_type="regression",
            n_rows=3,
            columns=[ColumnTruth(name="y", role="must_keep")],
        )


def test_numpy_seed_streams_are_not_shared_between_recipes():
    firsts = {np.random.default_rng(r.seed).random() for r in RECIPES}
    assert len(firsts) == len(RECIPES)


# --- the named variant ---


@pytest.fixture(scope="module", params=NAMES)
def pair(request):
    recipe = next(r for r in RECIPES if r.name == request.param)
    return generate(recipe, "anonymous"), generate(recipe, "named")


def test_the_committed_named_files_are_what_the_generator_writes(pair):
    _, (frame, truth) = pair
    committed_frame, committed_truth = load(truth.dataset, "named")
    assert committed_truth == truth
    pd.testing.assert_frame_equal(committed_frame, frame, check_dtype=False, rtol=1e-9)


def test_the_named_variant_has_the_same_data_in_the_same_order(pair):
    (anonymous, _), (named, _) = pair
    pd.testing.assert_frame_equal(named.set_axis(anonymous.columns, axis=1), anonymous)


def test_the_named_truth_is_the_anonymous_truth_with_other_names(pair):
    (_, anonymous), (named_frame, named) = pair
    assert named.variant == "named" and anonymous.variant == "anonymous"
    mapping = {a.name: n.name for a, n in zip(anonymous.columns, named.columns, strict=True)}
    mapping[anonymous.target] = named.target
    renamed = anonymous.model_copy(
        update={
            "variant": "named",
            "target": named.target,
            "columns": [
                c.model_copy(
                    update={
                        "name": mapping[c.name],
                        "duplicate_of": mapping.get(c.duplicate_of or ""),
                    }
                )
                for c in anonymous.columns
            ],
        }
    )
    assert renamed == named
    assert list(named_frame.columns) == [*(c.name for c in named.columns), named.target]


def test_named_columns_are_distinct_plain_words(pair):
    _, (frame, truth) = pair
    names = [*frame.columns]
    assert len(set(names)) == len(names)
    assert all(re.fullmatch(r"[a-z][a-z0-9_]*", name) for name in names)
    assert not any(re.fullmatch(r"c\d{2}", name) for name in names)
    assert truth.target != TARGET


def test_no_name_announces_its_fault(pair):
    _, (_, truth) = pair
    for column in truth.columns:
        assert not re.search(r"leak|sizinti|target|hedef|fault|planted", column.name)


def test_every_domain_has_enough_names_for_its_recipe():
    for recipe in RECIPES:
        domain = DOMAINS[recipe.name]
        assert len(domain.numeric) >= recipe.n_numeric
        assert len(domain.categorical) >= recipe.n_categorical
        planted = {i for i in recipe.items if not i.startswith("duplicate_")}
        assert planted <= set(domain.faults)
        assert set(domain.faults) <= planted


def test_the_named_set_mixes_turkish_and_english():
    words = {n for d in DOMAINS.values() for n in (*d.numeric, *d.categorical)}
    assert {"yas", "applicant_age"} <= words


# --- the second set ---

from planted_generate import ALL_RECIPES, RECIPES_B  # noqa: E402

NAMES_B = [recipe.name for recipe in RECIPES_B]


def test_the_second_set_has_the_same_recipes_with_other_seeds():
    assert [r.name for r in RECIPES_B] == [n.replace("planted-", "planted-b") for n in NAMES]
    for a, b in zip(RECIPES, RECIPES_B, strict=True):
        assert (a.task, a.n_rows, a.n_numeric, a.n_categorical, a.items) == (
            b.task,
            b.n_rows,
            b.n_numeric,
            b.n_categorical,
            b.items,
        )
        assert a.seed != b.seed and b.domain == a.name
    seeds = [r.seed for r in ALL_RECIPES]
    assert len(set(seeds)) == len(seeds) == 16


@pytest.mark.parametrize("name", NAMES_B)
@pytest.mark.parametrize("variant", VARIANTS)
def test_the_committed_files_of_the_second_set_are_what_the_generator_writes(name, variant):
    recipe = next(r for r in RECIPES_B if r.name == name)
    frame, truth = generate(recipe, variant)
    committed_frame, committed_truth = load(name, variant)
    assert committed_truth == truth
    pd.testing.assert_frame_equal(committed_frame, frame, check_dtype=False, rtol=1e-9)


def test_the_second_set_has_the_same_faults_in_other_data():
    for a, b in zip(RECIPES, RECIPES_B, strict=True):
        frame_a, truth_a = generate(a)
        frame_b, truth_b = generate(b)
        assert sorted(c.fault or "" for c in truth_a.columns) == sorted(
            c.fault or "" for c in truth_b.columns
        )
        assert frame_a.shape == frame_b.shape and not frame_a.equals(frame_b)


def test_the_second_set_has_the_same_names_as_the_first_in_the_named_variant():
    for a, b in zip(RECIPES, RECIPES_B, strict=True):
        _, truth_a = generate(a, "named")
        _, truth_b = generate(b, "named")
        assert truth_a.target == truth_b.target

        # A copy is named after the column it copies, which the seed picks.
        def plain(truth):
            return sorted(c.name for c in truth.columns if not (c.fault or "").startswith("dup"))

        assert plain(truth_a) == plain(truth_b)


def test_the_first_set_is_not_changed_by_the_second():
    assert [r.seed for r in RECIPES] == list(range(101, 109))
