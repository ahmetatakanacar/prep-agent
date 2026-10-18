"""Set D: the level mechanism, the recipes, the ground truth, the scorer's new metrics and the
guard around the test recipes.

Only the dev data is read here. Nothing in this file computes a profile score of a dev column or
looks at the data of a test recipe: the mechanism is checked on targets made for the purpose, and
the test recipes are checked through their ground truth.
"""

import re

import numpy as np
import pandas as pd
import pytest

from planted_generate import RECIPES, RECIPES_B, RECIPES_C, load
from planted_generate_d import generate_d, level_column, write_all_d
from planted_recipes_d import ALL_D, DEV, TEST
from planted_truth import D_FAULT_TYPES, VARIANTS, ColumnTruth
from prep_agent.signals import target_association

DEV_NAMES = [r.name for r in DEV]
TEST_NAMES = [r.name for r in TEST]
LEVELS = (0.6, 0.75, 0.85, 0.95)


# --- the level mechanism, on targets made for the purpose ---


def association(column, y, task) -> float:
    df = pd.DataFrame({"x": column, "y": y})
    scores = target_association(
        df, "y", ["x"], task=task, max_rows=20000, min_rows=100, folds=5, tree_depth=4
    )
    return scores["x"]


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("form", ["numeric", "category", "gap"])
def test_a_column_made_for_a_class_has_about_the_association_it_was_made_for(form, level):
    rng = np.random.default_rng(1)
    y = (rng.random(4000) < 0.4).astype(int)
    column = level_column(rng, y, "classification", level, form, 50, 10)
    assert abs(association(column, y, "classification") - level) <= 0.1


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("form", ["numeric", "category"])
def test_a_column_made_for_a_number_has_about_the_association_it_was_made_for(form, level):
    rng = np.random.default_rng(2)
    y = rng.normal(20, 5, 4000)
    column = level_column(rng, y, "regression", level, form, 50, 10)
    assert abs(association(column, y, "regression") - level) <= 0.1


def test_a_gap_column_has_values_only_for_positive_rows():
    rng = np.random.default_rng(3)
    y = (rng.random(1000) < 0.4).astype(int)
    column = level_column(rng, y, "classification", 0.75, "gap", 50, 10)
    assert np.isnan(column[y == 0]).all()
    assert 0.6 < np.isfinite(column[y == 1]).mean() < 0.9


@pytest.mark.parametrize(
    "task, form, level",
    [("regression", "gap", 0.8), ("classification", "numeric", 1.0), ("classification", "x", 0.5)],
)
def test_a_level_that_cannot_be_made_is_refused(task, form, level):
    with pytest.raises(ValueError):
        level_column(np.random.default_rng(0), np.zeros(10), task, level, form, 0, 1)


# --- the recipes ---


def test_there_are_six_dev_and_six_test_recipes_with_nothing_in_common():
    assert len(DEV) == len(TEST) == 6
    names = [r.name for r in ALL_D]
    assert len(set(names)) == 12 and all(r.split == "dev" for r in DEV)
    assert all(r.split == "test" for r in TEST)
    assert len({r.seed for r in ALL_D}) == 12
    assert set(DEV_NAMES).isdisjoint(TEST_NAMES)


def test_set_d_is_independent_of_the_first_three_sets():
    old = [*RECIPES, *RECIPES_B, *RECIPES_C]
    assert {r.seed for r in ALL_D}.isdisjoint({r.seed for r in old})
    assert {r.name for r in ALL_D}.isdisjoint({r.name for r in old})
    from planted_names import NAMES

    old_names = {
        n for d in NAMES.values() for n in (*d.numeric, *d.categorical, *d.faults.values())
    }
    old_names |= {d.target for d in NAMES.values()}
    for recipe in ALL_D:
        used = {c.name for c in recipe.base} | {f.name for f in recipe.faults} | {recipe.target}
        assert used.isdisjoint(old_names), used & old_names
    assert {r.n_rows for r in ALL_D}.isdisjoint({r.n_rows for r in old})


@pytest.mark.parametrize("recipes", [DEV, TEST])
def test_each_half_has_both_tasks_and_every_level_of_leak_and_legit_feature(recipes):
    assert {r.task for r in recipes} == {"classification", "regression"}
    leaks = [f.level for r in recipes for f in r.faults if f.fault == "leak_partial"]
    legit = [f.level for r in recipes for f in r.faults if f.fault == "legit_strong"]
    assert all(leaks.count(lv) == 3 for lv in LEVELS)
    # At and above the profile's threshold, and below it, there are legitimate features too.
    assert sum(lv >= 0.9 for lv in legit) >= 4 and sum(lv < 0.9 for lv in legit) >= 4
    assert sorted(set(legit)) == [0.6, 0.75, 0.85, 0.93, 0.97]


@pytest.mark.parametrize("recipes", [DEV, TEST])
def test_the_old_fault_types_are_in_each_half_in_less_obvious_forms(recipes):
    faults = [f for r in recipes for f in r.faults]
    kinds = {f.fault for f in faults}
    assert {
        "id_sequential", "id_uuid", "pii_email", "pii_phone", "pii_name",
        "duplicate_exact", "duplicate_scaled", "numeric_category", "constant",
    } <= kinds  # fmt: skip
    assert sum(f.form != "default" for f in faults) >= 5


def test_recipes_differ_in_size_and_in_columns():
    assert len({r.n_rows for r in ALL_D}) == 12
    assert len({len(r.base) for r in ALL_D}) >= 4
    assert all(r.faults for r in ALL_D)


# --- the ground truth ---


@pytest.mark.parametrize("recipe", ALL_D, ids=lambda r: r.name)
@pytest.mark.parametrize("variant", VARIANTS)
def test_the_committed_truth_is_what_the_generator_writes(recipe, variant):
    _, truth = generate_d(recipe, variant)
    _, committed = load_truth(recipe.name, variant)
    assert committed == truth


def load_truth(name, variant):
    from planted_generate import DATA_DIR, file_stem
    from planted_truth import DatasetTruth

    path = DATA_DIR / f"{file_stem(name, variant)}.truth.json"
    return None, DatasetTruth.model_validate_json(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("recipe", ALL_D, ids=lambda r: r.name)
def test_every_leak_and_legit_column_has_a_level_and_a_meaning(recipe):
    _, truth = generate_d(recipe, "named")
    levelled = [c for c in truth.columns if c.fault in D_FAULT_TYPES]
    assert len(levelled) == sum(f.fault in D_FAULT_TYPES for f in recipe.faults)
    for column in levelled:
        assert column.level in (0.6, 0.75, 0.85, 0.93, 0.95, 0.97)
        assert len(column.reason) > 20
        assert column.role == ("must_drop" if column.fault == "leak_partial" else "must_keep")


@pytest.mark.parametrize("recipe", ALL_D, ids=lambda r: r.name)
def test_anonymous_and_named_truth_differ_only_in_names(recipe):
    _, a = generate_d(recipe, "anonymous")
    _, n = generate_d(recipe, "named")
    assert a.variant == "anonymous" and n.variant == "named"
    assert a.target == "y" and n.target == recipe.target
    for left, right in zip(a.columns, n.columns, strict=True):
        assert left.model_copy(update={"name": "", "duplicate_of": None}) == right.model_copy(
            update={"name": "", "duplicate_of": None}
        )
    assert re.fullmatch(r"c\d{2}", a.columns[0].name)


@pytest.mark.parametrize("recipe", ALL_D, ids=lambda r: r.name)
def test_no_name_announces_its_fault(recipe):
    _, truth = generate_d(recipe, "named")
    for column in truth.columns:
        assert not re.search(r"leak|sizinti|target|hedef|fault|legit|strong", column.name)


def test_a_level_needs_a_reason_and_a_legit_feature_must_be_kept():
    with pytest.raises(ValueError, match="level and a reason"):
        ColumnTruth(name="c", role="must_drop", fault="leak_partial", level=0.6)
    with pytest.raises(ValueError, match="must have role must_keep"):
        ColumnTruth(name="c", role="must_drop", fault="legit_strong", level=0.9, reason="x")
    with pytest.raises(ValueError, match="only for leak_partial"):
        ColumnTruth(name="c", role="must_keep", level=0.9)


# --- the dev data ---


@pytest.fixture(scope="module", params=DEV, ids=lambda r: r.name)
def dev(request):
    recipe = request.param
    return recipe, {v: generate_d(recipe, v) for v in VARIANTS}


def test_the_committed_dev_files_are_what_the_generator_writes(dev):
    recipe, made = dev
    for variant, (frame, truth) in made.items():
        committed_frame, committed_truth = load(recipe.name, variant)
        assert committed_truth == truth
        pd.testing.assert_frame_equal(committed_frame, frame, check_dtype=False, rtol=1e-9)


def test_both_variants_hold_the_same_values(dev):
    _, made = dev
    anonymous, named = made["anonymous"][0], made["named"][0]
    pd.testing.assert_frame_equal(named.set_axis(anonymous.columns, axis=1), anonymous)


def test_the_dev_data_has_the_size_the_recipe_asks_for(dev):
    recipe, made = dev
    frame, truth = made["anonymous"]
    assert len(frame) == recipe.n_rows and frame.shape[1] == len(truth.columns) + 1
    if recipe.task == "classification":
        assert set(frame["y"]) == {0, 1}
        assert abs(frame["y"].mean() - recipe.pos_rate) < 0.06
    else:
        assert frame["y"].nunique() > 100


def test_gap_columns_have_values_only_where_the_class_is_positive(dev):
    recipe, made = dev
    frame, _ = made["named"]
    for fault in recipe.faults:
        if fault.form == "gap":
            present = frame[fault.name].notna()
            assert present.any() and (frame.loc[present, recipe.target] == 1).all()


def test_the_old_faults_take_their_less_obvious_forms(dev):
    recipe, made = dev
    frame, truth = made["named"]
    for fault in recipe.faults:
        if fault.fault == "id_sequential" and fault.form == "sparse":
            ids = frame[fault.name]
            assert ids.is_monotonic_increasing and ids.is_unique
            assert ids.max() - ids.min() > 2 * len(ids)
        if fault.fault == "id_sequential" and fault.form == "prefixed":
            assert frame[fault.name].str.fullmatch(r"[A-Z]{3}-\d{6}").all()
        if fault.fault == "id_uuid" and fault.form == "hex32":
            assert frame[fault.name].str.fullmatch(r"[0-9a-f]{32}").all()
        if fault.fault == "pii_phone" and fault.form == "int10":
            assert pd.api.types.is_integer_dtype(frame[fault.name])
            assert frame[fault.name].between(5 * 10**9, 6 * 10**9).all()
        if fault.fault == "pii_name" and fault.form == "surname_first":
            assert frame[fault.name].str.contains(", ").mean() > 0.8
        if fault.fault == "pii_name" and fault.form == "initial":
            assert frame[fault.name].str.contains(r"^[A-ZÇĞİÖŞÜ]\. ", regex=True).mean() > 0.8
        if fault.fault == "pii_email" and fault.form == "initial":
            assert frame[fault.name].str.fullmatch(r"[a-z]+\d*@[a-z.]+").all()


def test_a_coarse_scaled_copy_has_a_correlation_between_the_two_duplicate_bands(dev):
    recipe, made = dev
    frame, _ = made["named"]
    for fault in recipe.faults:
        if fault.fault == "duplicate_scaled" and fault.form == "coarse":
            corr = frame[fault.name].corr(frame[fault.of])
            assert 0.99 < corr < 0.9999


def test_a_relabelled_copy_matches_its_original_one_to_one(dev):
    recipe, made = dev
    frame, _ = made["named"]
    for fault in recipe.faults:
        if fault.form == "relabel":
            pairs = frame[[fault.of, fault.name]].dropna().drop_duplicates()
            assert pairs[fault.of].is_unique and pairs[fault.name].is_unique
            assert not frame[fault.of].equals(frame[fault.name])


# --- writing the files ---


def test_the_test_data_is_written_only_when_asked_for(tmp_path):
    write_all_d(tmp_path)
    files = {p.name for p in tmp_path.iterdir()}
    assert all(f"planted-t01{s}.truth.json" in files for s in ("", ".named"))
    assert not any(n.startswith("planted-t") and n.endswith(".csv") for n in files)
    assert "planted-d01.csv" in files and "planted-d01.named.csv" in files
    write_all_d(tmp_path / "all", with_test_data=True)
    assert (tmp_path / "all" / "planted-t06.named.csv").is_file()


def test_the_test_data_is_ignored_by_git():
    from pathlib import Path

    ignore = (Path(__file__).parent.parent / ".gitignore").read_text(encoding="utf-8")
    assert "benchmarks/planted/data/planted-t*.csv" in ignore
