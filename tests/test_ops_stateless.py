import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from prep_agent.ops import OpContext, OpError, get_op
from prep_agent.plan import CastToCategoryStep, DedupeStep, DropColumnsStep, DropMissingTargetStep

REASON = "test"
CTX = OpContext(target="y")


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [1, 1, 2, 2, 3, 3],
            "b": ["p", "q", "r", "r", "s", "t"],
            "y": [0.0, 1.0, None, None, 1.0, 0.0],
        },
        index=[10, 11, 12, 13, 14, 15],
    )


def run(op: str, df: pd.DataFrame, step, ctx: OpContext = CTX) -> pd.DataFrame:
    return get_op(op).apply(df, step, ctx)


# --- dedupe ----------------------------------------------------------------------------------


def test_dedupe_over_all_columns_removes_exact_duplicates(df):
    step = DedupeStep(op="dedupe", subset=None, keep="first", reason=REASON)

    result = run("dedupe", df, step)

    assert len(result) == 5
    assert result["b"].tolist() == ["p", "q", "r", "s", "t"]


def test_dedupe_subset_keep_first_and_last_pick_different_rows(df):
    first = run("dedupe", df, DedupeStep(op="dedupe", subset=["a"], keep="first", reason=REASON))
    last = run("dedupe", df, DedupeStep(op="dedupe", subset=["a"], keep="last", reason=REASON))

    assert first["b"].tolist() == ["p", "r", "s"]
    assert last["b"].tolist() == ["q", "r", "t"]


def test_dedupe_does_not_modify_input_and_returns_new_frame(df):
    original = df.copy()

    result = run("dedupe", df, DedupeStep(op="dedupe", subset=["a"], keep="first", reason=REASON))

    assert_frame_equal(df, original)
    assert result is not df


def test_dedupe_resets_the_index(df):
    result = run("dedupe", df, DedupeStep(op="dedupe", subset=["a"], keep="first", reason=REASON))

    assert result.index.equals(pd.RangeIndex(3))


def test_dedupe_without_duplicates_returns_an_equal_copy():
    df = pd.DataFrame({"a": [1, 2, 3], "y": [0, 1, 0]})

    result = run("dedupe", df, DedupeStep(op="dedupe", subset=None, keep="first", reason=REASON))

    assert_frame_equal(result, df)
    assert result is not df


def test_dedupe_unknown_subset_column_raises_op_error(df):
    step = DedupeStep(op="dedupe", subset=["nope"], keep="first", reason=REASON)

    with pytest.raises(OpError, match="nope"):
        run("dedupe", df, step)


# --- drop_columns ----------------------------------------------------------------------------


def test_drop_columns_removes_only_the_given_columns(df):
    step = DropColumnsStep(op="drop_columns", columns=["a", "b"], reason=REASON)

    result = run("drop_columns", df, step)

    assert list(result.columns) == ["y"]
    assert len(result) == len(df)
    assert result.index.equals(df.index)


def test_drop_columns_does_not_modify_input(df):
    original = df.copy()

    result = run(
        "drop_columns", df, DropColumnsStep(op="drop_columns", columns=["a"], reason=REASON)
    )

    assert_frame_equal(df, original)
    assert result is not df


def test_drop_columns_unknown_column_raises_op_error(df):
    step = DropColumnsStep(op="drop_columns", columns=["a", "nope"], reason=REASON)

    with pytest.raises(OpError, match="nope"):
        run("drop_columns", df, step)


def test_drop_columns_refuses_to_drop_the_target(df):
    step = DropColumnsStep(op="drop_columns", columns=["a", "y"], reason=REASON)

    with pytest.raises(OpError, match="target"):
        run("drop_columns", df, step)


# --- drop_missing_target ---------------------------------------------------------------------


def test_drop_missing_target_removes_rows_with_missing_target(df):
    result = run(
        "drop_missing_target", df, DropMissingTargetStep(op="drop_missing_target", reason=REASON)
    )

    assert result["y"].notna().all()
    assert result["b"].tolist() == ["p", "q", "s", "t"]
    assert result.index.equals(pd.RangeIndex(4))


def test_drop_missing_target_keeps_rows_with_missing_features():
    df = pd.DataFrame({"a": [None, 2.0, 3.0], "y": [0, 1, None]})

    result = run(
        "drop_missing_target", df, DropMissingTargetStep(op="drop_missing_target", reason=REASON)
    )

    assert len(result) == 2
    assert result["a"].isna().sum() == 1


def test_drop_missing_target_does_not_modify_input_and_returns_new_frame(df):
    original = df.copy()

    result = run(
        "drop_missing_target", df, DropMissingTargetStep(op="drop_missing_target", reason=REASON)
    )

    assert_frame_equal(df, original)
    assert result is not df


def test_drop_missing_target_without_missing_values_returns_an_equal_copy():
    df = pd.DataFrame({"a": [1, 2], "y": [0, 1]})

    result = run(
        "drop_missing_target", df, DropMissingTargetStep(op="drop_missing_target", reason=REASON)
    )

    assert_frame_equal(result, df)
    assert result is not df


def test_drop_missing_target_requires_the_target_column(df):
    step = DropMissingTargetStep(op="drop_missing_target", reason=REASON)

    with pytest.raises(OpError, match="target"):
        run("drop_missing_target", df.drop(columns="y"), step)


# --- cast_to_category ------------------------------------------------------------------------


def cast(columns: list[str]) -> CastToCategoryStep:
    return CastToCategoryStep(op="cast_to_category", columns=columns, reason=REASON)


def test_cast_writes_integers_as_text():
    df = pd.DataFrame({"hour": [0, 7, 23], "y": [1, 0, 1]})

    result = run("cast_to_category", df, cast(["hour"]))

    assert result["hour"].tolist() == ["0", "7", "23"]
    assert pd.api.types.is_string_dtype(result["hour"])


def test_cast_writes_whole_floats_without_the_decimal_point():
    # An integer column that has missing values is a float column, but its values are still
    # whole numbers and must give the same categories as the integer column would.
    df = pd.DataFrame({"code": [1.0, np.nan, 20.0, 1.0], "y": [0, 1, 0, 1]})

    result = run("cast_to_category", df, cast(["code"]))

    assert result["code"].dropna().tolist() == ["1", "20", "1"]


def test_cast_keeps_missing_values_missing():
    df = pd.DataFrame({"code": [1.0, np.nan, 3.0], "label": ["a", None, "c"], "y": [0, 1, 0]})

    result = run("cast_to_category", df, cast(["code", "label"]))

    assert result["code"].isna().tolist() == [False, True, False]
    assert result["label"].isna().tolist() == [False, True, False]


def test_cast_keeps_the_fraction_of_a_float_that_has_one():
    df = pd.DataFrame({"size": [0.5, 1.0, 2.25], "y": [0, 1, 0]})

    result = run("cast_to_category", df, cast(["size"]))

    assert result["size"].tolist() == ["0.5", "1", "2.25"]


def test_cast_leaves_text_alone_and_writes_booleans_as_text():
    df = pd.DataFrame({"city": ["ankara", "izmir"], "flag": [True, False], "y": [0, 1]})

    result = run("cast_to_category", df, cast(["city", "flag"]))

    assert result["city"].tolist() == ["ankara", "izmir"]
    assert result["flag"].tolist() == ["True", "False"]


def test_cast_leaves_other_columns_and_the_index_alone(df):
    result = run("cast_to_category", df, cast(["a"]))

    assert list(result.columns) == list(df.columns)
    assert result.index.equals(df.index)
    assert_frame_equal(result.drop(columns="a"), df.drop(columns="a"))
    assert result["a"].tolist() == ["1", "1", "2", "2", "3", "3"]


def test_cast_does_not_modify_input_and_returns_new_frame(df):
    original = df.copy()

    result = run("cast_to_category", df, cast(["a"]))

    assert_frame_equal(df, original)
    assert result is not df


def test_cast_unknown_column_raises_op_error(df):
    with pytest.raises(OpError, match="nope"):
        run("cast_to_category", df, cast(["a", "nope"]))


def test_cast_refuses_the_target(df):
    with pytest.raises(OpError, match="target"):
        run("cast_to_category", df, cast(["a", "y"]))
