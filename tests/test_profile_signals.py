"""Profile signals between columns: target association, duplicate columns, hash identifiers.

Every fixture is generated here from a fixed seed. The data is built so that the right answer is
known from how it was made, and no threshold is tuned on it: the positives are columns that are
functions of the target, the negatives are columns that are strong or useless on their own.
"""

import dataclasses
import hashlib
import uuid

import numpy as np
import pandas as pd
import pytest

from prep_agent.profiler import DEFAULT_CONFIG, profile_dataframe

N = 600


def rng(seed=0):
    return np.random.default_rng(seed)


def regression_frame(n=N, seed=1) -> pd.DataFrame:
    """y = 10a + 5b + noise: `a` alone explains about 75% of the variance, `b` about 19%."""
    r = rng(seed)
    a, b = r.normal(size=n), r.normal(size=n)
    y = 10 * a + 5 * b + r.normal(0, 3, n)
    return pd.DataFrame({"a": a, "b": b, "noise": r.normal(size=n), "y": y})


def classification_frame(n=N, seed=2) -> pd.DataFrame:
    """y is 1 when a noisy copy of `a` is positive: `a` alone is a good but imperfect predictor."""
    r = rng(seed)
    a, b = r.normal(size=n), r.normal(size=n)
    y = ((a + r.normal(0, 0.8, n)) > 0).astype(int)
    return pd.DataFrame({"a": a, "b": b, "noise": r.normal(size=n), "y": y})


def profile(df, target="y", **config):
    cfg = dataclasses.replace(DEFAULT_CONFIG, **config)
    return {c.name: c for c in profile_dataframe(df, target=target, config=cfg).columns}


def flagged(columns) -> set[str]:
    return {name for name, c in columns.items() if "suspected_target_leakage" in c.warnings}


# --- target association: columns that are functions of the target ---


def test_a_linear_function_of_a_numeric_target_is_flagged():
    df = regression_frame()
    df["leak"] = 2 * df["y"] + 7 + rng(3).normal(0, 0.05 * df["y"].std(), len(df))
    columns = profile(df)
    assert columns["leak"].target_association >= 0.95
    assert flagged(columns) == {"leak"}


def test_a_nonlinear_function_of_a_numeric_target_is_flagged():
    df = regression_frame()
    z = (df["y"] - df["y"].mean()) / df["y"].std()
    df["leak"] = np.exp(0.8 * z + rng(3).normal(0, 0.05, len(df)))
    assert flagged(profile(df)) == {"leak"}


def test_a_category_that_cuts_a_numeric_target_into_many_levels_is_flagged():
    df = regression_frame()
    levels = pd.qcut(df["y"], 8, labels=[f"L{i}" for i in range(8)])
    df["leak"] = levels.astype(str)
    assert flagged(profile(df)) == {"leak"}


def test_a_few_levels_cannot_reach_the_threshold_for_a_numeric_target():
    """Four levels of a bell-shaped target explain at most about 88% of its variance, so even
    a perfect category of that kind scores below the threshold. This is a limit of an absolute
    threshold, kept visible here; the score itself still shows in the profile."""
    df = regression_frame()
    df["coarse"] = pd.qcut(df["y"], 4, labels=list("wxyz")).astype(str)
    column = profile(df)["coarse"]
    assert 0.8 <= column.target_association < 0.9
    assert "suspected_target_leakage" not in column.warnings


def test_a_numeric_column_derived_from_a_class_is_flagged():
    df = classification_frame()
    df["leak"] = 2 * df["y"] + 1 + rng(3).normal(0, 0.1, len(df))
    assert flagged(profile(df)) == {"leak"}


def test_a_category_that_repeats_the_class_with_rare_errors_is_flagged():
    df = classification_frame()
    wrong = rng(3).random(len(df)) < 0.03
    df["leak"] = np.where(df["y"].astype(bool) ^ wrong, "yes", "no")
    columns = profile(df)
    assert columns["leak"].target_association >= 0.9
    assert flagged(columns) == {"leak"}


def test_a_column_that_is_missing_exactly_when_the_class_is_positive_is_flagged():
    df = classification_frame()
    df["leak"] = np.where(df["y"] == 1, np.nan, rng(3).normal(size=len(df)))
    assert flagged(profile(df)) == {"leak"}


def test_a_copy_of_the_target_is_flagged_for_text_and_boolean_classes():
    df = classification_frame()
    df["y"] = np.where(df["y"] == 1, "churn", "stay")
    df["leak"] = (df["y"] == "churn").to_numpy()
    assert flagged(profile(df)) == {"leak"}


def test_a_multiclass_target_is_handled_and_its_derived_column_flagged():
    r = rng(4)
    y = r.integers(0, 3, N)
    df = pd.DataFrame({"noise": r.normal(size=N), "leak": y * 10 + r.normal(0, 1, N), "y": y})
    columns = profile(df)
    assert columns["leak"].target_association >= 0.9
    assert flagged(columns) == {"leak"}


# --- target association: columns that must not be flagged ---


def test_a_strong_but_legitimate_regression_feature_is_not_flagged():
    columns = profile(regression_frame())
    assert 0.5 < columns["a"].target_association < 0.9
    assert flagged(columns) == set()


def test_a_strong_but_legitimate_classification_feature_is_not_flagged():
    columns = profile(classification_frame())
    assert 0.3 < columns["a"].target_association < 0.9
    assert flagged(columns) == set()


def test_a_useless_column_scores_about_zero():
    columns = profile(regression_frame())
    assert columns["noise"].target_association <= 0.1
    columns = profile(classification_frame())
    assert columns["noise"].target_association <= 0.1


def test_columns_unique_per_row_do_not_look_like_predictors():
    """Fitted on the rows, a unique column would 'predict' everything; scored out of fold it
    cannot, because a row it is asked about was never in the lookup."""
    df = classification_frame()
    df["ident"] = [f"user-{i}" for i in range(len(df))]
    df["rank"] = np.arange(len(df))
    df["many"] = rng(5).integers(0, 300, len(df)).astype(str)
    columns = profile(df)
    for name in ("many",):
        assert columns[name].target_association <= 0.1
    assert flagged(columns) == set()


def test_a_sequential_column_is_not_flagged_when_the_rows_are_in_random_order():
    df = regression_frame()
    df["rank"] = np.arange(len(df))
    assert flagged(profile(df)) == set()


def test_rows_without_a_target_value_are_left_out():
    df = regression_frame()
    df["leak"] = 2 * df["y"] + 7
    df.loc[rng(6).random(len(df)) < 0.1, "y"] = np.nan
    columns = profile(df)
    assert flagged(columns) == {"leak"}
    assert columns["a"].target_association > 0.5


def test_the_target_itself_constant_text_and_date_columns_have_no_score():
    df = classification_frame()
    df["const"] = 1
    df["when"] = pd.date_range("2024-01-01", periods=len(df)).astype(str)
    df["story"] = [f"a long free text sentence number {i} about nothing" for i in range(len(df))]
    columns = profile(df)
    assert columns["y"].target_association is None
    for name in ("const", "when", "story"):
        assert columns[name].target_association is None


def test_there_is_no_score_without_a_target():
    columns = profile(regression_frame(), target=None)
    assert all(c.target_association is None for c in columns.values())
    assert all("suspected_target_leakage" not in c.warnings for c in columns.values())


def test_there_is_no_score_for_a_text_target_with_no_task_hint():
    df = classification_frame()
    df["y"] = [f"free text label number {i} that is different" for i in range(len(df))]
    columns = profile(df)
    assert all(c.target_association is None for c in columns.values())


def test_too_few_rows_give_no_score():
    df = regression_frame(n=DEFAULT_CONFIG.leakage_min_rows - 1)
    df["leak"] = df["y"] * 3
    columns = profile(df)
    assert all(c.target_association is None for c in columns.values())


def test_a_constant_target_after_dropping_missing_rows_gives_no_score():
    df = regression_frame()
    df["y"] = np.where(np.arange(len(df)) == 0, 1.0, np.nan)
    assert all(c.target_association is None for c in profile(df).values())


# --- target association: how the score is made ---


def test_scores_are_between_zero_and_one_with_two_decimals():
    df = regression_frame()
    df["cat"] = rng(7).choice(["u", "v", "w"], len(df))
    for column in profile(df).values():
        score = column.target_association
        if score is not None:
            assert 0 <= score <= 1 and score == round(score, 2)


def test_the_score_does_not_depend_on_the_order_of_the_rows():
    df = regression_frame()
    df["leak"] = 2 * df["y"] + rng(3).normal(0, 1, len(df))
    shuffled = df.sample(frac=1, random_state=9).reset_index(drop=True)
    a, b = profile(df), profile(shuffled)
    for name in ("a", "b", "leak"):
        assert abs(a[name].target_association - b[name].target_association) <= 0.05


def test_a_large_frame_is_scored_on_a_fixed_sample():
    df = regression_frame(n=3000)
    df["leak"] = 2 * df["y"] + rng(3).normal(0, 1, len(df))
    first = profile(df, leakage_max_rows=500)
    second = profile(df, leakage_max_rows=500)
    assert {k: c.target_association for k, c in first.items()} == {
        k: c.target_association for k, c in second.items()
    }
    assert flagged(first) == {"leak"}
    assert first["a"].target_association > 0.5


def test_the_threshold_comes_from_the_config():
    df = regression_frame()
    assert flagged(profile(df, leakage_min_score=0.5)) == {"a"}
    assert flagged(profile(df)) == set()


def test_integer_column_labels_are_handled():
    df = regression_frame().rename(columns={"a": 0, "b": 1})
    df["leak"] = df["y"] * 2
    assert flagged(profile(df)) == {"leak"}
    assert 0 in {int(name) for name in profile(df) if name.isdigit()}


def test_the_profile_json_carries_the_score_only_when_there_is_one():
    df = regression_frame()
    dumped = profile_dataframe(df, target="y").model_dump(exclude_none=True)
    by_name = {c["name"]: c for c in dumped["columns"]}
    assert "target_association" in by_name["a"] and "target_association" not in by_name["y"]
    assert "duplicate_of" not in by_name["a"]


# --- duplicate columns ---


def duplicates(df, target=None) -> dict[str, str]:
    columns = profile(df, target=target)
    for c in columns.values():
        assert (c.duplicate_of is not None) == ("duplicate_column" in c.warnings)
    return {name: c.duplicate_of for name, c in columns.items() if c.duplicate_of}


def base(n=300, seed=10) -> pd.DataFrame:
    r = rng(seed)
    return pd.DataFrame(
        {
            "x": np.round(r.normal(100, 15, n), 4),
            "w": np.round(r.lognormal(3, 0.6, n), 4),
            "city": r.choice(["ankara", "izmir", "bursa", "adana"], n),
            "tier": r.choice(["gold", "silver"], n),
        }
    )


def test_an_exact_numeric_copy_points_to_its_original():
    df = base()
    df["x2"] = df["x"]
    assert duplicates(df) == {"x2": "x"}


def test_an_exact_text_copy_is_found():
    df = base()
    df["town"] = df["city"].copy()
    assert duplicates(df) == {"town": "city"}


def test_a_scaled_and_shifted_copy_with_rounding_is_found():
    df = base()
    df["x_cm"] = np.round(df["x"] * 2.54 + 3, 4)
    df["x_neg"] = -df["x"]
    assert duplicates(df) == {"x_cm": "x", "x_neg": "x"}


def test_an_integer_column_and_its_float_version_are_one_column():
    df = base()
    df["n"] = np.arange(300) % 7
    df["n_float"] = df["n"].astype(float)
    assert duplicates(df) == {"n_float": "n"}


def test_a_relabelled_category_is_found_by_its_one_to_one_mapping():
    df = base()
    df["tier_code"] = df["tier"].map({"gold": "G", "silver": "S"})
    df["city_code"] = df["city"].map({"ankara": 6, "izmir": 35, "bursa": 16, "adana": 1})
    assert duplicates(df) == {"tier_code": "tier", "city_code": "city"}


def test_the_original_is_the_earlier_column_and_a_copy_of_a_copy_points_to_it():
    df = base()
    df["x_a"] = df["x"] * 2
    df["x_b"] = df["x_a"] + 1
    df["x_c"] = df["x_b"]
    found = duplicates(df)
    assert found == {"x_a": "x", "x_b": "x", "x_c": "x"}


def test_a_copy_that_comes_first_in_the_file_is_the_original():
    df = base()
    df.insert(0, "x_front", df["x"] * 3)
    assert duplicates(df) == {"x": "x_front"}


def test_missing_values_in_both_columns_are_fine_if_they_are_in_the_same_rows():
    df = base()
    df.loc[df.index % 9 == 0, "x"] = np.nan
    df["x2"] = df["x"] * 5
    df["city2"] = df["city"]
    df.loc[df.index % 5 == 0, ["city", "city2"]] = None
    assert duplicates(df) == {"x2": "x", "city2": "city"}


def test_a_column_with_different_missing_values_is_not_a_copy():
    df = base()
    df["x2"] = df["x"]
    df.loc[3, "x2"] = np.nan
    assert duplicates(df) == {}


def test_two_measures_that_move_together_are_not_copies():
    df = base()
    df["x_close"] = df["x"] + rng(11).normal(0, 1.5, len(df))
    assert abs(df["x"].corr(df["x_close"])) > 0.99
    assert duplicates(df) == {}


def test_two_independent_columns_unique_in_every_row_are_not_copies():
    """Each value of one goes with exactly one value of the other, by chance, so a one-to-one
    mapping means nothing when every value is unique."""
    df = base()
    df["u1"] = rng(12).normal(size=len(df))
    df["u2"] = rng(13).normal(size=len(df))
    assert duplicates(df) == {}


def test_two_independent_categories_with_the_same_number_of_levels_are_not_copies():
    df = base()
    df["tier2"] = rng(14).choice(["a", "b"], len(df))
    df["city2"] = rng(15).choice(["p", "q", "r", "s"], len(df))
    assert duplicates(df) == {}


def test_a_mapping_that_is_not_one_to_one_is_not_a_copy():
    df = base()
    df["region"] = df["city"].map(
        {"ankara": "inland", "bursa": "inland", "izmir": "coast", "adana": "coast"}
    )
    df["city_x"] = df["city"].where(df.index % 50 != 0, "other")
    assert duplicates(df) == {}


def test_constant_columns_and_the_target_are_never_duplicates():
    df = base()
    df["k1"] = 1
    df["k2"] = 1
    df["y"] = df["x"] * 2
    assert duplicates(df, target="y") == {}


def test_duplicates_are_found_without_a_target_and_with_one():
    df = base()
    df["x2"] = df["x"]
    df["y"] = rng(16).integers(0, 2, len(df))
    assert duplicates(df) == duplicates(df, target="y") == {"x2": "x"}


def test_duplicate_detection_does_not_depend_on_the_correlation_sample():
    df = base(n=900)
    df["x_cm"] = np.round(df["x"] * 2.54, 4)
    assert duplicates_with(df, duplicate_corr_rows=40) == {"x_cm": "x"}


def duplicates_with(df, **config) -> dict[str, str]:
    columns = profile(df, target=None, **config)
    return {name: c.duplicate_of for name, c in columns.items() if c.duplicate_of}


# --- hash-shaped identifiers ---


def type_of(values, name="col") -> str:
    df = pd.DataFrame({name: values, "other": np.arange(len(values)) % 3})
    return {c.name: c for c in profile_dataframe(df).columns}[name].inferred_type


def uuids(n=50, seed=20):
    r = rng(seed)
    return [str(uuid.UUID(bytes=r.bytes(16), version=4)) for _ in range(n)]


def test_unique_uuids_are_identifiers_whatever_their_name():
    df = pd.DataFrame({"c01": uuids(), "x": np.arange(50) % 2})
    column = {c.name: c for c in profile_dataframe(df).columns}["c01"]
    assert column.inferred_type == "id_like" and "possible_id" in column.warnings


@pytest.mark.parametrize("algorithm", ["md5", "sha1", "sha256"])
def test_unique_hex_digests_are_identifiers(algorithm):
    values = [hashlib.new(algorithm, str(i).encode()).hexdigest() for i in range(50)]
    assert type_of(values) == "id_like"
    assert type_of([v.upper() for v in values]) == "id_like"


def test_uuid_variants_are_identifiers():
    base_values = uuids()
    assert type_of([f"{{{v}}}" for v in base_values]) == "id_like"
    assert type_of([v.upper() for v in base_values]) == "id_like"


def test_uuids_that_repeat_are_a_category_not_an_identifier():
    values = uuids(10) * 5
    assert type_of(values) == "categorical"


def test_too_few_unique_hashes_are_not_called_identifiers():
    assert type_of(uuids(DEFAULT_CONFIG.id_min_rows - 1)) != "id_like"


def test_long_digit_strings_are_not_hashes():
    """Forty digits are as long as a sha1, but a number has no letter a-f; without the hash rule
    the column is long unique text."""
    numbers = [f"{10**39 + i * 7919:040d}" for i in range(50)]
    assert type_of(numbers) == "text"


def test_a_value_that_is_not_hex_keeps_the_column_out():
    values = uuids()
    values[7] = "not-a-uuid-at-all-just-some-words-here"
    assert type_of(values) != "id_like"


def test_unique_long_text_is_still_text():
    values = [f"a sentence of free text, number {i}, that goes on for a while" for i in range(50)]
    assert type_of(values) == "text"
