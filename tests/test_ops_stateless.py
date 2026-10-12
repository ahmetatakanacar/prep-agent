import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from prep_agent.ops import OpContext, OpError, get_op
from prep_agent.plan import DedupeStep, DropColumnsStep, DropMissingTargetStep

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
