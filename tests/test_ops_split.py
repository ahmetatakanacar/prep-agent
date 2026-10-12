import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from prep_agent.ops import OpContext, OpError, get_op
from prep_agent.plan import SplitStep

REASON = "test"
CTX = OpContext(target="y")


@pytest.fixture
def df() -> pd.DataFrame:
    n = 100
    return pd.DataFrame(
        {"row": range(n), "x": [i * 0.5 for i in range(n)], "y": [1] * 20 + [0] * 80},
        index=range(1000, 1000 + n),
    )


def split(df, ctx=CTX, **kwargs):
    step = SplitStep(op="split", reason=REASON, **{"stratify": False, **kwargs})
    return get_op("split").apply(df, step, ctx)


def test_split_returns_train_and_test_with_requested_sizes(df):
    train, test = split(df, test_size=0.2)

    assert len(train) == 80
    assert len(test) == 20
    assert list(train.columns) == list(df.columns)
    assert list(test.columns) == list(df.columns)


@pytest.mark.parametrize("test_size, expected_test", [(0.25, 25), (0.1, 10), (0.5, 50)])
def test_split_respects_test_size(df, test_size, expected_test):
    train, test = split(df, test_size=test_size)

    assert len(test) == expected_test
    assert len(train) == len(df) - expected_test


def test_split_parts_are_disjoint_and_cover_all_rows(df):
    train, test = split(df)

    assert set(train["row"]).isdisjoint(test["row"])
    assert sorted(set(train["row"]) | set(test["row"])) == list(range(100))


def test_split_resets_both_indexes(df):
    train, test = split(df)

    assert train.index.equals(pd.RangeIndex(len(train)))
    assert test.index.equals(pd.RangeIndex(len(test)))


def test_stratified_split_preserves_class_ratios(df):
    train, test = split(df, test_size=0.2, stratify=True)

    assert test["y"].mean() == pytest.approx(0.2)
    assert train["y"].mean() == pytest.approx(0.2)
    assert test["y"].sum() == 4


def test_unstratified_split_does_not_force_class_ratios():
    df = pd.DataFrame({"x": range(40), "y": [1] * 4 + [0] * 36})

    ratios = {split(df, test_size=0.25, random_state=seed)[1]["y"].sum() for seed in range(10)}

    assert len(ratios) > 1


def test_same_random_state_gives_identical_result(df):
    first_train, first_test = split(df, random_state=7)
    second_train, second_test = split(df.copy(), random_state=7)

    assert_frame_equal(first_train, second_train)
    assert_frame_equal(first_test, second_test)


def test_stratified_split_is_reproducible(df):
    first = split(df, stratify=True, random_state=3)
    second = split(df, stratify=True, random_state=3)

    assert_frame_equal(first[0], second[0])
    assert_frame_equal(first[1], second[1])


def test_different_random_states_give_different_splits(df):
    _, test_a = split(df, random_state=1)
    _, test_b = split(df, random_state=2)

    assert set(test_a["row"]) != set(test_b["row"])


def test_split_does_not_modify_input(df):
    original = df.copy()

    split(df)

    assert_frame_equal(df, original)


def test_stratify_requires_the_target_column(df):
    with pytest.raises(OpError, match="target"):
        split(df.drop(columns="y"), stratify=True)


def test_stratify_rejects_missing_target_values(df):
    df.loc[1000, "y"] = None

    with pytest.raises(OpError, match="missing"):
        split(df, stratify=True)


def test_stratify_with_a_single_member_class_raises_op_error():
    df = pd.DataFrame({"x": range(20), "y": [1] + [0] * 19})

    with pytest.raises(OpError, match="split"):
        split(df, stratify=True)


def test_split_of_a_single_row_raises_op_error():
    df = pd.DataFrame({"x": [1], "y": [0]})

    with pytest.raises(OpError, match="split"):
        split(df)
