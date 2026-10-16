from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from prep_agent import executor
from prep_agent.executor import PlanValidationError, execute
from prep_agent.plan import Plan
from prep_agent.profiler import load_csv, profile_dataframe
from prep_agent.validators import has_errors

FIXTURES = Path(__file__).parent / "fixtures"
CUSTOMERS_CSV = FIXTURES / "customers.csv"
CUSTOMERS_PLAN = FIXTURES / "customers_plan.json"
INVALID_PLAN = FIXTURES / "customers_plan_invalid.json"


def run_customers(plan_path: Path = CUSTOMERS_PLAN):
    df = load_csv(CUSTOMERS_CSV).df
    plan = Plan.model_validate_json(plan_path.read_text(encoding="utf-8"))
    return df, plan, execute(df, plan, profile_dataframe(df, target=plan.target))


def split(**extra):
    return {"op": "split", "stratify": True, "test_size": 0.25, **extra}


# `cat` is never encoded in the tests that do not exercise encoding; the validator rejects a plan
# that leaves it as text.
DROP_CAT = {"op": "drop_columns", "columns": ["cat"]}


def make_frame(n: int = 40) -> pd.DataFrame:
    i = np.arange(n)
    return pd.DataFrame(
        {
            "row": i,
            # Squares make the median of any subset differ clearly from the overall median.
            "x": np.where(i % 7 == 0, np.nan, (i + 1.0) ** 2),
            "z": 1000.0 * (i + 1.0) ** 2,
            "cat": [f"c{k % 25}" for k in i],
            "t": i % 2,
        }
    )


def run_frame(df, plan):
    return execute(df, plan, profile_dataframe(df, target=plan.target))


def test_fixture_plan_runs_end_to_end():
    df, plan, result = run_customers()

    assert len(result.train) == 16
    assert len(result.test) == 4
    expected = {"age", "monthly_spend", "plan_basic", "plan_enterprise", "plan_pro", "churned"}
    assert set(result.train.columns) == expected
    assert list(result.train.columns) == list(result.test.columns)
    assert list(result.train.columns)[-1] == "churned"
    assert not has_errors(result.plan_issues)
    assert not has_errors(result.result_issues)
    features = [c for c in result.train.columns if c != "churned"]
    assert not result.train[features].isna().any().any()
    assert not result.test[features].isna().any().any()


def test_input_frame_is_not_modified():
    df, _, _ = run_customers()

    pd.testing.assert_frame_equal(df, load_csv(CUSTOMERS_CSV).df)


def test_imputer_and_scaler_learn_from_train_only(make_plan):
    df = make_frame()
    plan = make_plan(
        DROP_CAT,
        split(),
        {"op": "impute", "columns": ["x"], "strategy": "median"},
        {"op": "scale", "columns": ["z"], "method": "standard"},
    )

    result = run_frame(df, plan)

    train_ids = set(result.train["row"])
    raw_train = df[df["row"].isin(train_ids)]
    raw_test = df[~df["row"].isin(train_ids)]
    imputer = result.pipeline.named_steps["2_impute"].named_transformers_["impute"]
    scaler = result.pipeline.named_steps["3_scale"].named_transformers_["scale"]

    assert imputer.statistics_[0] == raw_train["x"].median()
    assert raw_train["x"].median() != df["x"].median()
    assert raw_train["x"].median() != raw_test["x"].median()
    assert scaler.mean_[0] == pytest.approx(raw_train["z"].mean())
    assert scaler.mean_[0] != pytest.approx(df["z"].mean())

    missing_in_test = raw_test.loc[raw_test["x"].isna(), "row"]
    assert len(missing_in_test) > 0
    filled = result.test.loc[result.test["row"].isin(missing_in_test), "x"]
    assert (filled == raw_train["x"].median()).all()
    # Only train is standardised to zero mean; the test set keeps its own offset.
    assert result.train["z"].mean() == pytest.approx(0, abs=1e-9)
    assert abs(result.test["z"].mean()) > 1e-3


@pytest.mark.parametrize("method", ["onehot", "ordinal"])
def test_unseen_category_in_test_does_not_fail(make_plan, method):
    df = make_frame().fillna({"x": 0.0})
    plan = make_plan(
        {"op": "drop_missing_target"},
        split(),
        {"op": "encode", "columns": ["cat"], "method": method},
    )

    result = run_frame(df, plan)

    train_cats = set(df.loc[df["row"].isin(result.train["row"]), "cat"])
    test_cats = set(df.loc[df["row"].isin(result.test["row"]), "cat"])
    assert test_cats - train_cats, "the test must contain a category unseen in train"
    assert len(result.test) == 10
    assert not result.test.drop(columns="t").isna().any().any()


def test_two_runs_give_identical_frames(make_plan):
    df = make_frame()
    plan = make_plan(
        split(),
        {"op": "impute", "columns": ["x"], "strategy": "mean"},
        {"op": "encode", "columns": ["cat"], "method": "onehot", "max_categories": 5},
        {"op": "scale", "columns": ["z"], "method": "robust"},
    )

    first = run_frame(df, plan)
    second = run_frame(df, plan)

    pd.testing.assert_frame_equal(first.train, second.train, check_exact=True)
    pd.testing.assert_frame_equal(first.test, second.test, check_exact=True)


def test_pipeline_is_identity_without_stateful_steps(make_plan):
    df = make_frame().dropna()
    plan = make_plan(DROP_CAT, split())

    result = run_frame(df, plan)

    features = result.train.drop(columns="t")
    pd.testing.assert_frame_equal(result.pipeline.transform(features), features)
    assert len(result.train) + len(result.test) == len(df)


def test_target_never_enters_the_pipeline(make_plan):
    df = make_frame()
    plan = make_plan(DROP_CAT, split(), {"op": "impute", "columns": ["x"], "strategy": "median"})

    result = run_frame(df, plan)

    assert "t" not in result.pipeline.feature_names_in_
    assert list(result.train.columns)[-1] == "t"
    assert result.train["t"].tolist() == result.split_train["t"].tolist()


def result_codes(result):
    return {i.code for i in result.result_issues if i.severity == "error"}


def test_rows_that_only_match_after_encoding_are_not_an_overlap(make_plan):
    # Every `u` is distinct, but one-hot grouping puts nearly all of them into the same
    # "infrequent" column, so train and test end up with equal output rows.
    df = pd.DataFrame({"u": [f"u{i}" for i in range(40)], "t": np.arange(40) % 2})
    plan = make_plan(
        {"op": "dedupe", "keep": "first"},
        split(),
        {"op": "encode", "columns": ["u"], "method": "onehot", "max_categories": 2},
    )

    result = run_frame(df, plan)

    assert len(result.train.merge(result.test)) > 0, "the output frames must share a row"
    assert result.train.drop(columns="t").duplicated().any()
    assert "train_test_overlap" not in {i.code for i in result.result_issues}
    assert not has_errors(result.result_issues)


def test_rows_that_only_become_equal_by_dropping_a_column_are_not_an_overlap(make_plan):
    # Was an error ("rows that become equal before the split are an overlap error") while an
    # overlap meant rows equal at the split. It now means the same input record on both sides:
    # these rows differ in `id` in the input, so dropping `id` after the dedupe leaks nothing.
    i = np.arange(40)
    df = pd.DataFrame({"id": i, "x": (i // 2) % 2, "t": i % 2})
    plan = make_plan(
        {"op": "dedupe", "keep": "first"}, {"op": "drop_columns", "columns": ["id"]}, split()
    )

    result = run_frame(df, plan)

    assert len(result.split_train.merge(result.split_test)) > 0, "look-alike rows on both sides"
    assert not has_errors(result.plan_issues)
    assert "train_test_overlap" not in {i.code for i in result.result_issues}


def test_identical_input_records_on_both_sides_are_an_overlap_warning(make_plan):
    base = pd.DataFrame({"x": np.arange(20), "t": np.arange(20) % 2})
    df = pd.concat([base, base], ignore_index=True)

    result = run_frame(df, make_plan(split(test_size=0.5)))

    [issue] = [i for i in result.result_issues if i.code == "train_test_overlap"]
    assert issue.severity == "warning"
    assert not has_errors(result.result_issues)


def test_a_dedupe_removes_the_identical_input_records_before_the_split(make_plan):
    base = pd.DataFrame({"x": np.arange(20), "t": np.arange(20) % 2})
    df = pd.concat([base, base], ignore_index=True)

    result = run_frame(df, make_plan({"op": "dedupe", "keep": "first"}, split(test_size=0.5)))

    assert len(result.train) + len(result.test) == 20
    assert "train_test_overlap" not in {i.code for i in result.result_issues}


def oracle_shared_rows(df, *, dedupe, subset, keep, drop_missing, test_size, stratify):
    """The number of test rows whose input record is also in train, worked out on the input
    frame with its index kept, so no step has to be followed."""
    kept = df.dropna(subset=["t"]) if drop_missing else df
    if dedupe:
        kept = kept.drop_duplicates(subset=subset, keep=keep)
    train_at, test_at = train_test_split(
        kept.index.to_numpy(),
        test_size=test_size,
        random_state=42,
        stratify=kept["t"] if stratify else None,
    )
    records = pd.util.hash_pandas_object(df, index=False)
    train_records = set(records.loc[train_at])
    return sum(record in train_records for record in records.loc[test_at])


@pytest.mark.parametrize("dedupe", [False, True])
@pytest.mark.parametrize("keep", ["first", "last"])
@pytest.mark.parametrize("subset", [None, ["a", "b"]])
@pytest.mark.parametrize("drop_missing", [False, True])
def test_the_overlap_follows_input_records_through_row_removing_steps(
    make_plan, dedupe, keep, subset, drop_missing
):
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "id": np.arange(120),
            "a": rng.integers(0, 6, 120),
            "b": rng.integers(0, 3, 120),
            "t": rng.integers(0, 2, 120).astype(float),
        }
    )
    # Copies of earlier rows, id included, scattered over the frame. Missing targets only where
    # the plan drops them: a split that sees one is a plan error.
    df.iloc[rng.choice(120, 40, replace=False)] = df.iloc[rng.choice(120, 40)].to_numpy()
    if drop_missing:
        df.loc[rng.choice(120, 15, replace=False), "t"] = np.nan
    steps = []
    if drop_missing:
        steps.append({"op": "drop_missing_target"})
    steps.append({"op": "drop_columns", "columns": ["id"]})
    if dedupe:
        steps.append({"op": "dedupe", "keep": keep, "subset": subset})
    stratify = drop_missing
    steps.append(split(stratify=stratify, test_size=0.4))
    plan = make_plan(*steps)

    result = run_frame(df, plan)

    expected = oracle_shared_rows(
        df,
        dedupe=dedupe,
        subset=subset,
        keep=keep,
        drop_missing=drop_missing,
        test_size=0.4,
        stratify=stratify,
    )
    overlaps = [i for i in result.result_issues if i.code == "train_test_overlap"]
    if expected:
        [issue] = overlaps
        assert f"{expected} test row(s)" in issue.message
    else:
        assert overlaps == []


def test_plan_error_raises_before_anything_runs(monkeypatch):
    def fail(name):
        raise AssertionError(f"op {name!r} must not run when the plan has errors")

    monkeypatch.setattr(executor, "get_op", fail)

    with pytest.raises(PlanValidationError) as excinfo:
        run_customers(INVALID_PLAN)

    error_codes = {i.code for i in excinfo.value.issues if i.severity == "error"}
    assert {"stateful_before_split", "unknown_column"} <= error_codes


def test_data_that_does_not_match_the_profile_raises_execution_error(make_plan):
    df = make_frame()
    plan = make_plan(DROP_CAT, split(), {"op": "impute", "columns": ["x"], "strategy": "median"})
    profile = profile_dataframe(df, target="t")

    with pytest.raises(executor.ExecutionError):
        execute(df.drop(columns="x"), plan, profile)


@pytest.mark.parametrize("stratify", [True, False])
def test_a_missing_target_is_not_carried_into_the_output(make_plan, stratify):
    df = make_frame()
    df["t"] = df["t"].astype(float).mask(df["row"].isin([3, 10]))
    plan = make_plan(DROP_CAT, split(stratify=stratify))

    with pytest.raises(PlanValidationError, match="missing_target_at_split"):
        run_frame(df, plan)

    fixed = make_plan(DROP_CAT, {"op": "drop_missing_target"}, split(stratify=stratify))
    result = run_frame(df, fixed)
    assert not result.train["t"].isna().any()
    assert not result.test["t"].isna().any()


def test_a_regression_plan_for_a_text_target_is_rejected_before_it_runs():
    df = load_csv(CUSTOMERS_CSV).df
    plan = Plan.model_validate_json(CUSTOMERS_PLAN.read_text(encoding="utf-8"))
    plan = plan.model_copy(update={"task_type": "regression"})

    with pytest.raises(PlanValidationError, match="task_type_mismatch"):
        execute(df, plan, profile_dataframe(df, target=plan.target))


def test_expected_model_family_is_checked_before_anything_runs():
    df = load_csv(CUSTOMERS_CSV).df
    plan = Plan.model_validate_json(CUSTOMERS_PLAN.read_text(encoding="utf-8"))
    profile = profile_dataframe(df, target=plan.target)

    with pytest.raises(PlanValidationError, match="model_family_mismatch"):
        execute(df, plan, profile, expected_model_family="tree")


def census_like_frame(n: int = 200) -> pd.DataFrame:
    i = np.arange(n)
    return pd.DataFrame(
        {
            "age": 20.0 + i % 50,
            "hours": 10.0 + i % 40,
            "workclass": np.where(i % 9 == 0, None, np.where(i % 2 == 0, "Private", "Gov")),
            "occupation": np.where(i % 11 == 0, None, np.where(i % 3 == 0, "Sales", "Tech")),
            "sex": np.where(i % 2 == 0, "Male", "Female"),
            "t": np.where(i % 5 == 0, ">50K", "<=50K"),
        }
    )


def test_missing_indicators_of_text_columns_are_integers_the_validator_accepts(make_plan):
    plan = make_plan(
        split(),
        {
            "op": "impute",
            "columns": ["workclass", "occupation"],
            "strategy": "most_frequent",
            "add_indicator": True,
        },
        {"op": "encode", "columns": ["workclass", "occupation", "sex"], "method": "onehot"},
        {"op": "scale", "columns": ["age", "hours"], "method": "robust"},
    )

    result = run_frame(census_like_frame(), plan)

    indicators = [c for c in result.train.columns if c.startswith("missingindicator_")]
    assert indicators == ["missingindicator_workclass", "missingindicator_occupation"]
    for frame in (result.train, result.test):
        assert all(frame[c].dtype == "int64" for c in indicators)
        assert set(frame[indicators].to_numpy().ravel()) <= {0, 1}
    assert [i.code for i in result.result_issues if i.code == "non_numeric_output"] == []
