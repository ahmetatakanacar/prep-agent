import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from prep_agent.ops import get_op
from prep_agent.plan import EncodeStep, ImputeStep, ScaleStep

REASON = "test"


def build(op: str, step):
    return get_op(op).build_transformer(step)


def impute_step(strategy, columns=("age",), **kwargs):
    return ImputeStep(
        op="impute", columns=list(columns), strategy=strategy, reason=REASON, **kwargs
    )


def encode_step(method, columns=("color",), **kwargs):
    return EncodeStep(op="encode", columns=list(columns), method=method, reason=REASON, **kwargs)


def scale_step(method, columns=("x",)):
    return ScaleStep(op="scale", columns=list(columns), method=method, reason=REASON)


@pytest.fixture
def train_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [1.0, 2.0, 3.0, 100.0, None],
            "city": ["a", "b", "a", "a", None],
            "y": [0, 1, 0, 1, 0],
        }
    )


@pytest.fixture
def test_df() -> pd.DataFrame:
    return pd.DataFrame({"age": [None, 10.0], "city": [None, "c"], "y": [1, 0]}, index=[100, 101])


# --- shared behavior -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "op, step",
    [
        ("impute", impute_step("median")),
        ("encode", encode_step("onehot")),
        ("encode", encode_step("ordinal")),
        ("scale", scale_step("standard")),
    ],
)
def test_builder_returns_an_unfitted_transformer(op, step):
    transformer = build(op, step)

    with pytest.raises(NotFittedError):
        check_is_fitted(transformer)


def test_building_twice_gives_independent_transformers():
    step = scale_step("standard")

    assert build("scale", step) is not build("scale", step)


@pytest.mark.parametrize(
    "op, step, frame",
    [
        ("impute", impute_step("mean"), pd.DataFrame({"age": [1.0, None], "z": ["u", "v"]})),
        ("encode", encode_step("onehot"), pd.DataFrame({"color": ["r", "g"], "z": [1, 2]})),
        ("encode", encode_step("ordinal"), pd.DataFrame({"color": ["r", "g"], "z": [1, 2]})),
        ("scale", scale_step("minmax", ("x",)), pd.DataFrame({"x": [1.0, 5.0], "z": [1, 2]})),
    ],
)
def test_output_is_a_dataframe_that_keeps_untouched_columns_index_and_dtypes(op, step, frame):
    frame.index = [7, 8]

    out = build(op, step).fit(frame).transform(frame)

    assert isinstance(out, pd.DataFrame)
    assert out.index.tolist() == [7, 8]
    assert out["z"].tolist() == frame["z"].tolist()
    assert out["z"].dtype == frame["z"].dtype


# --- impute ----------------------------------------------------------------------------------


def test_median_imputation_uses_the_train_median(train_df, test_df):
    transformer = build("impute", impute_step("median")).fit(train_df)

    out = transformer.transform(test_df)

    assert out["age"].tolist() == [2.5, 10.0]
    assert out["y"].tolist() == [1, 0]
    assert out.index.tolist() == [100, 101]


def test_statistics_come_from_train_and_not_from_the_data_being_transformed(train_df, test_df):
    transformer = build("impute", impute_step("median")).fit(train_df)
    other_test = pd.DataFrame({"age": [None, 5000.0, 6000.0, 7000.0], "city": "a", "y": 0})

    assert transformer.transform(test_df)["age"].iloc[0] == 2.5
    assert transformer.transform(other_test)["age"].iloc[0] == 2.5


def test_mean_imputation_uses_the_train_mean(train_df, test_df):
    out = build("impute", impute_step("mean")).fit(train_df).transform(test_df)

    assert out["age"].iloc[0] == pytest.approx(26.5)


def test_most_frequent_imputation_of_a_text_column(train_df, test_df):
    out = build("impute", impute_step("most_frequent", ("city",))).fit(train_df).transform(test_df)

    assert out["city"].tolist() == ["a", "c"]


def test_constant_imputation(train_df, test_df):
    step = impute_step("constant", fill_value=-1)

    out = build("impute", step).fit(train_df).transform(test_df)

    assert out["age"].tolist() == [-1.0, 10.0]


def test_constant_imputation_of_a_text_column(train_df, test_df):
    step = impute_step("constant", ("city",), fill_value="missing")

    out = build("impute", step).fit(train_df).transform(test_df)

    assert out["city"].tolist() == ["missing", "c"]


def test_missing_indicator_columns_are_added_on_request(train_df, test_df):
    step = impute_step("median", add_indicator=True)

    out = build("impute", step).fit(train_df).transform(test_df)

    assert out["missingindicator_age"].astype(int).tolist() == [1, 0]
    assert out["age"].tolist() == [2.5, 10.0]


def test_no_indicator_columns_by_default(train_df, test_df):
    out = build("impute", impute_step("median")).fit(train_df).transform(test_df)

    assert set(out.columns) == {"age", "city", "y"}


def test_imputation_only_touches_the_listed_columns(train_df, test_df):
    out = build("impute", impute_step("median")).fit(train_df).transform(test_df)

    assert out["city"].isna().tolist() == [True, False]


def test_all_missing_train_column_is_kept():
    train = pd.DataFrame({"age": [None, None, None], "y": [0, 1, 0]})
    new = pd.DataFrame({"age": [None, 4.0], "y": [1, 0]})

    out = build("impute", impute_step("mean")).fit(train).transform(new)

    assert list(out.columns) == ["age", "y"]
    assert out["age"].notna().all()


# --- scale -----------------------------------------------------------------------------------


@pytest.fixture
def scale_train() -> pd.DataFrame:
    return pd.DataFrame({"x": [0.0, 10.0, 20.0, 30.0, 40.0], "z": [1, 2, 3, 4, 5]})


def test_standard_scaling_uses_train_mean_and_std(scale_train):
    new = pd.DataFrame({"x": [20.0, 40.0], "z": [9, 9]})

    out = build("scale", scale_step("standard")).fit(scale_train).transform(new)

    expected_std = np.std([0.0, 10.0, 20.0, 30.0, 40.0])
    assert out["x"].tolist() == pytest.approx([0.0, 20.0 / expected_std])
    assert out["z"].tolist() == [9, 9]


def test_minmax_scaling_uses_train_range_so_new_values_can_leave_zero_one(scale_train):
    new = pd.DataFrame({"x": [-40.0, 20.0, 80.0], "z": [0, 0, 0]})

    out = build("scale", scale_step("minmax")).fit(scale_train).transform(new)

    assert out["x"].tolist() == pytest.approx([-1.0, 0.5, 2.0])


def test_robust_scaling_uses_train_median_and_iqr(scale_train):
    new = pd.DataFrame({"x": [30.0, 0.0], "z": [0, 0]})

    out = build("scale", scale_step("robust")).fit(scale_train).transform(new)

    assert out["x"].tolist() == pytest.approx([0.5, -1.0])


def test_scaling_does_not_depend_on_the_transformed_data(scale_train):
    transformer = build("scale", scale_step("standard")).fit(scale_train)
    small = pd.DataFrame({"x": [20.0], "z": [0]})
    large = pd.DataFrame({"x": [20.0, 1e6, -1e6], "z": [0, 0, 0]})

    assert transformer.transform(small)["x"].iloc[0] == transformer.transform(large)["x"].iloc[0]


def test_scaling_leaves_missing_values_in_place():
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, None]})

    out = build("scale", scale_step("standard")).fit(frame).transform(frame)

    assert out["x"].isna().tolist() == [False, False, False, True]


def test_scaling_several_columns():
    frame = pd.DataFrame({"a": [0.0, 10.0], "b": [100.0, 200.0], "c": ["u", "v"]})

    out = build("scale", scale_step("minmax", ("a", "b"))).fit(frame).transform(frame)

    assert out["a"].tolist() == [0.0, 1.0]
    assert out["b"].tolist() == [0.0, 1.0]
    assert out["c"].tolist() == ["u", "v"]


# --- encode ----------------------------------------------------------------------------------


@pytest.fixture
def color_train() -> pd.DataFrame:
    return pd.DataFrame({"color": ["red", "blue", "red", "green"], "x": [1, 2, 3, 4]})


def test_onehot_produces_dense_named_columns_and_drops_the_original(color_train):
    out = build("encode", encode_step("onehot")).fit(color_train).transform(color_train)

    assert set(out.columns) == {"color_blue", "color_green", "color_red", "x"}
    assert out["color_red"].tolist() == [1, 0, 1, 0]
    assert out["color_blue"].tolist() == [0, 1, 0, 0]
    assert all(pd.api.types.is_numeric_dtype(out[c]) for c in out.columns)


def test_onehot_does_not_fail_on_categories_unseen_in_train(color_train):
    new = pd.DataFrame({"color": ["red", "purple"], "x": [5, 6]})

    out = build("encode", encode_step("onehot")).fit(color_train).transform(new)

    assert out.loc[1, ["color_blue", "color_green", "color_red"]].tolist() == [0, 0, 0]
    assert out["color_red"].tolist() == [1, 0]


def test_onehot_columns_come_from_train_categories_only(color_train):
    new = pd.DataFrame({"color": ["purple", "orange"], "x": [5, 6]})

    out = build("encode", encode_step("onehot")).fit(color_train).transform(new)

    assert set(out.columns) == {"color_blue", "color_green", "color_red", "x"}


def test_onehot_max_categories_groups_rare_categories_into_one_column():
    train = pd.DataFrame({"color": ["a"] * 5 + ["b"] * 3 + ["c", "d"]})
    new = pd.DataFrame({"color": ["a", "b", "c", "zzz"]})

    out = build("encode", encode_step("onehot", max_categories=3)).fit(train).transform(new)

    assert out.shape[1] == 3
    infrequent = next(c for c in out.columns if "infrequent" in c)
    assert out[infrequent].tolist() == [0, 0, 1, 1]


def test_onehot_encodes_several_columns():
    frame = pd.DataFrame({"a": ["u", "v"], "b": ["p", "q"], "n": [1, 2]})

    out = build("encode", encode_step("onehot", ("a", "b"))).fit(frame).transform(frame)

    assert set(out.columns) == {"a_u", "a_v", "b_p", "b_q", "n"}


def test_ordinal_uses_train_categories_and_marks_unseen_values(color_train):
    new = pd.DataFrame({"color": ["red", "purple", "blue"], "x": [5, 6, 7]})

    out = build("encode", encode_step("ordinal")).fit(color_train).transform(new)

    assert out["color"].tolist() == [2.0, -1.0, 0.0]
    assert out["x"].tolist() == [5, 6, 7]


def test_ordinal_keeps_missing_values_missing_when_train_had_them():
    train = pd.DataFrame({"color": ["red", None, "blue"], "x": [1, 2, 3]})
    new = pd.DataFrame({"color": ["red", None], "x": [5, 6]})

    out = build("encode", encode_step("ordinal")).fit(train).transform(new)

    assert out["color"].isna().tolist() == [False, True]


def test_ordinal_treats_missing_values_unseen_in_train_like_unseen_categories(color_train):
    new = pd.DataFrame({"color": ["red", None], "x": [5, 6]})

    out = build("encode", encode_step("ordinal")).fit(color_train).transform(new)

    assert out["color"].tolist() == [2.0, -1.0]


def test_fitting_on_one_frame_does_not_change_the_result_for_another(color_train):
    other = pd.DataFrame({"color": ["zebra", "yak", "red"], "x": [1, 2, 3]})
    transformer = build("encode", encode_step("ordinal")).fit(color_train)

    before = transformer.transform(color_train)
    transformer.transform(other)
    after = transformer.transform(color_train)

    assert_frame_equal(before, after)


@pytest.mark.parametrize(
    ("strategy", "column"), [("median", "age"), ("most_frequent", "city")], ids=["number", "text"]
)
def test_missing_indicators_are_zero_one_integers_for_every_column_type(
    train_df, test_df, strategy, column
):
    step = impute_step(strategy, (column,), add_indicator=True)

    out = build("impute", step).fit(train_df).transform(test_df)

    assert out[f"missingindicator_{column}"].dtype == "int64"
    assert out[f"missingindicator_{column}"].tolist() == [1, 0]
