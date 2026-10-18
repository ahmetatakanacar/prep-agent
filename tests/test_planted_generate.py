import re

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from planted_generate import DATA_DIR, RECIPES, TARGET, generate, load
from planted_truth import FAULT_TYPES, UNMEASURABLE_FAULTS, ColumnTruth, DatasetTruth

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
        assert (DATA_DIR / f"{name}.csv").is_file()
        assert (DATA_DIR / f"{name}.truth.json").is_file()


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
