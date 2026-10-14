import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from prep_agent import result_validator
from prep_agent.executor import ExecutionResult
from prep_agent.result_validator import (
    CLASS_SHIFT_THRESHOLD,
    LEAKAGE_CORRELATION_THRESHOLD,
    validate_result,
)
from prep_agent.validators import SEVERITY, has_errors

# Features a and b have an absolute correlation with t of 0.0 and about 0.63 in train.
TRAIN = pd.DataFrame(
    {
        "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "b": [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0],
        "t": [0, 1, 0, 1, 1, 0, 1, 0],
    }
)
TEST = pd.DataFrame({"a": [9.0, 10.0, 11.0, 12.0], "b": [5.0, 3.0, 5.0, 8.0], "t": [1, 0, 0, 1]})


def split(stratify=True):
    return {"op": "split", "stratify": stratify}


def records_of(frame: pd.DataFrame) -> np.ndarray:
    return pd.util.hash_pandas_object(frame, index=False).to_numpy()


def make_result(
    train=TRAIN,
    test=TEST,
    split_train=None,
    split_test=None,
    train_records=None,
    test_records=None,
) -> ExecutionResult:
    """`split_train` and `split_test` are the frames right after the split; by default they equal
    the output, as when there are no stateful steps. The records are the identities of the input
    rows behind them; by default rows that are equal at the split count as the same record, as
    when no column was dropped."""
    identity = Pipeline([("identity", FunctionTransformer())])
    split_train = (train if split_train is None else split_train).copy()
    split_test = (test if split_test is None else split_test).copy()
    return ExecutionResult(
        train=train.copy(),
        test=test.copy(),
        pipeline=identity,
        split_train=split_train,
        split_test=split_test,
        split_train_records=records_of(split_train) if train_records is None else train_records,
        split_test_records=records_of(split_test) if test_records is None else test_records,
    )


def issues_for(result, plan):
    return validate_result(result, plan)


def codes(issues):
    return [issue.code for issue in issues]


def test_clean_result_has_no_issues(make_plan):
    assert issues_for(make_result(), make_plan(split())) == []


# --- empty_split ---


@pytest.mark.parametrize("empty", ["train", "test"])
def test_empty_split_is_an_error(make_plan, empty):
    train = TRAIN.iloc[0:0] if empty == "train" else TRAIN
    test = TEST.iloc[0:0] if empty == "test" else TEST

    issues = issues_for(make_result(train, test), make_plan(split()))

    assert codes(issues) == ["empty_split"]
    assert issues[0].severity == "error"
    assert empty in issues[0].message


# --- column_mismatch ---


def test_different_columns_are_an_error(make_plan):
    issues = issues_for(make_result(test=TEST.drop(columns="b")), make_plan(split()))

    assert codes(issues) == ["column_mismatch"]
    assert issues[0].severity == "error"


def test_different_column_order_is_an_error(make_plan):
    issues = issues_for(make_result(test=TEST[["b", "a", "t"]]), make_plan(split()))

    assert codes(issues) == ["column_mismatch"]


# --- target_altered ---


def test_altered_target_is_an_error(make_plan):
    altered = TRAIN.copy()
    altered.loc[0, "t"] = 1 - altered.loc[0, "t"]

    issues = issues_for(make_result(split_train=altered), make_plan(split()))

    assert codes(issues) == ["target_altered"]
    assert issues[0].severity == "error"
    assert issues[0].column == "t"


def test_altered_test_target_is_an_error(make_plan):
    issues = issues_for(make_result(split_test=TEST.assign(t=0)), make_plan(split()))

    assert codes(issues) == ["target_altered"]


def test_unchanged_target_is_fine(make_plan):
    assert "target_altered" not in codes(issues_for(make_result(), make_plan(split())))


# --- train_test_overlap ---


def overlapping_test() -> pd.DataFrame:
    return pd.concat([TEST.iloc[:3], TRAIN.iloc[[2]]], ignore_index=True)


def test_overlap_is_a_warning_without_dedupe(make_plan):
    issues = issues_for(make_result(test=overlapping_test()), make_plan(split()))

    assert codes(issues) == ["train_test_overlap"]
    assert issues[0].severity == "warning"


def test_overlap_is_a_warning_even_with_dedupe(make_plan):
    plan = make_plan({"op": "dedupe", "keep": "first"}, split())

    issues = issues_for(make_result(test=overlapping_test()), plan)

    assert codes(issues) == ["train_test_overlap"]
    assert issues[0].severity == "warning"
    assert not has_errors(issues)


def test_the_overlap_message_names_identical_input_records_and_a_dedupe_before_the_split(
    make_plan,
):
    [issue] = issues_for(make_result(test=overlapping_test()), make_plan(split()))

    assert "1 test row(s)" in issue.message
    assert "identical in every column of the input" in issue.message
    assert "dedupe step before the split" in issue.message


def test_the_overlap_message_is_the_same_when_the_plan_has_a_dedupe(make_plan):
    plain = issues_for(make_result(test=overlapping_test()), make_plan(split()))
    deduped = issues_for(
        make_result(test=overlapping_test()),
        make_plan({"op": "dedupe", "keep": "first"}, split()),
    )

    assert plain[0].message == deduped[0].message


def test_train_test_overlap_has_warning_severity_in_the_table():
    assert SEVERITY["train_test_overlap"] == "warning"


def test_rows_equal_at_the_split_but_from_different_input_records_are_not_an_overlap(make_plan):
    # As after drop_columns: the rows look alike now, but the input told them apart.
    different_records = np.arange(100, 100 + len(overlapping_test()))
    result = make_result(test=overlapping_test(), test_records=different_records)

    assert issues_for(result, make_plan(split())) == []


def test_the_same_input_record_on_both_sides_is_an_overlap_whatever_the_frames_look_like(
    make_plan,
):
    train_records = np.arange(len(TRAIN))
    test_records = np.array([100, 101, 102, train_records[5]])

    issues = issues_for(
        make_result(train_records=train_records, test_records=test_records), make_plan(split())
    )

    assert codes(issues) == ["train_test_overlap"]
    assert "1 test row(s)" in issues[0].message


def test_rows_that_only_become_equal_after_the_pipeline_are_not_an_overlap(make_plan):
    # Distinct at the split, equal in the output (as after one-hot grouping or imputing).
    train = TRAIN.assign(a=0.0)
    test = TEST.assign(a=0.0, b=TRAIN.loc[0, "b"], t=TRAIN.loc[0, "t"])
    assert len(train.merge(test)) > 0, "the output frames must share a row"
    plan = make_plan({"op": "dedupe", "keep": "first"}, split())

    result = make_result(train, test, split_train=TRAIN, split_test=test.assign(a=TEST["a"]))

    assert "train_test_overlap" not in codes(issues_for(result, plan))


def test_rows_equal_at_the_split_are_an_overlap_even_if_the_output_differs(make_plan):
    plan = make_plan({"op": "dedupe", "keep": "first"}, split())
    split_test = overlapping_test()
    test = TEST.assign(t=split_test["t"])

    issues = issues_for(make_result(test=test, split_test=split_test), plan)
    assert codes(issues) == ["train_test_overlap"]
    assert issues[0].severity == "warning"


def test_same_features_with_another_target_is_not_an_overlap(make_plan):
    test = TEST.copy()
    test.loc[0, ["a", "b"]] = TRAIN.loc[2, ["a", "b"]].to_numpy()
    test.loc[0, "t"] = 1 - TRAIN.loc[2, "t"]

    assert issues_for(make_result(test=test), make_plan(split())) == []


def test_missing_values_in_equal_rows_still_count_as_overlap(make_plan):
    train = TRAIN.copy()
    train.loc[2, "a"] = float("nan")
    test = TEST.copy()
    test.loc[1] = train.loc[2]

    issues = issues_for(make_result(train, test), make_plan(split()))

    assert "train_test_overlap" in codes(issues)


# --- nan_in_output ---


def test_missing_feature_values_are_a_warning(make_plan):
    train = TRAIN.copy()
    train.loc[1, "a"] = float("nan")
    test = TEST.copy()
    test.loc[0, "b"] = float("nan")

    issues = issues_for(make_result(train, test), make_plan(split()))

    assert codes(issues) == ["nan_in_output", "nan_in_output"]
    assert {issue.column for issue in issues} == {"a", "b"}
    assert {issue.severity for issue in issues} == {"warning"}


def test_missing_target_values_are_not_reported_as_feature_nan(make_plan):
    train = TRAIN.astype({"t": "float64"})
    train.loc[0, "t"] = float("nan")

    assert "nan_in_output" not in codes(issues_for(make_result(train), make_plan(split())))


# --- non_numeric_output ---


def test_text_feature_is_a_warning(make_plan):
    train = TRAIN.assign(c=list("abababab"))
    test = TEST.assign(c=list("abab"))
    train = train[["a", "b", "c", "t"]]
    test = test[["a", "b", "c", "t"]]

    issues = issues_for(make_result(train, test), make_plan(split()))

    assert codes(issues) == ["non_numeric_output"]
    assert issues[0].severity == "warning"
    assert issues[0].column == "c"


def test_text_target_is_not_a_non_numeric_feature(make_plan):
    train = TRAIN.assign(t=["yes", "no"] * 4)
    test = TEST.assign(t=["yes", "no"] * 2)

    assert issues_for(make_result(train, test), make_plan(split())) == []


# --- possible_target_leakage ---


def with_leak(frame: pd.DataFrame, strength: float) -> pd.DataFrame:
    out = frame.copy()
    out.insert(2, "leak", out["t"] * strength + (1 - strength) * out["a"] / 8)
    return out


def test_feature_equal_to_target_is_a_warning(make_plan):
    train = with_leak(TRAIN, 1.0)
    test = with_leak(TEST, 1.0)

    issues = issues_for(make_result(train, test), make_plan(split()))

    assert codes(issues) == ["possible_target_leakage"]
    assert issues[0].severity == "warning"
    assert issues[0].column == "leak"


def test_negatively_correlated_feature_is_also_flagged(make_plan):
    train = TRAIN.assign(leak=-TRAIN["t"] * 3.0)
    test = TEST.assign(leak=-TEST["t"] * 3.0)
    train = train[["a", "b", "leak", "t"]]
    test = test[["a", "b", "leak", "t"]]

    assert codes(issues_for(make_result(train, test), make_plan(split()))) == [
        "possible_target_leakage"
    ]


def test_moderately_correlated_feature_is_not_flagged(make_plan):
    assert abs(TRAIN["b"].corr(TRAIN["t"])) < LEAKAGE_CORRELATION_THRESHOLD
    assert issues_for(make_result(), make_plan(split())) == []


def test_text_target_is_not_checked_for_leakage(make_plan):
    train = TRAIN.assign(t=["yes", "no"] * 4, leak=[1.0, 0.0] * 4)[["a", "b", "leak", "t"]]
    test = TEST.assign(t=["yes", "no"] * 2, leak=[1.0, 0.0] * 2)[["a", "b", "leak", "t"]]

    assert "possible_target_leakage" not in codes(
        issues_for(make_result(train, test), make_plan(split()))
    )


def test_constant_feature_does_not_crash_the_leakage_check(make_plan):
    train = TRAIN.assign(k=1.0)[["a", "b", "k", "t"]]
    test = TEST.assign(k=1.0)[["a", "b", "k", "t"]]

    assert issues_for(make_result(train, test), make_plan(split())) == []


# --- class_distribution_shift ---


def skewed_test() -> pd.DataFrame:
    return TEST.assign(t=[1, 1, 1, 1])


def test_class_shift_without_stratify_is_a_warning(make_plan):
    test = skewed_test()
    shift = abs(TRAIN["t"].mean() - test["t"].mean())
    assert shift > CLASS_SHIFT_THRESHOLD

    issues = issues_for(make_result(test=test), make_plan(split(stratify=False)))

    assert codes(issues) == ["class_distribution_shift"]
    assert issues[0].severity == "warning"


def test_class_shift_is_ignored_when_stratified(make_plan):
    assert issues_for(make_result(test=skewed_test()), make_plan(split(stratify=True))) == []


def test_similar_class_proportions_do_not_warn(make_plan):
    assert issues_for(make_result(), make_plan(split(stratify=False))) == []


def test_class_shift_is_ignored_for_regression(make_plan):
    plan = make_plan(split(stratify=False), task_type="regression")

    assert issues_for(make_result(test=skewed_test()), plan) == []


# --- thresholds ---


def test_thresholds_are_defined_in_one_place():
    assert result_validator.LEAKAGE_CORRELATION_THRESHOLD == 0.98
    assert 0 < result_validator.CLASS_SHIFT_THRESHOLD < 1
