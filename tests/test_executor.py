from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
    df = load_csv(CUSTOMERS_CSV)
    plan = Plan.model_validate_json(plan_path.read_text(encoding="utf-8"))
    return df, plan, execute(df, plan, profile_dataframe(df, target=plan.target))


def split(**extra):
    return {"op": "split", "stratify": True, "test_size": 0.25, **extra}


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

    pd.testing.assert_frame_equal(df, load_csv(CUSTOMERS_CSV))


def test_imputer_and_scaler_learn_from_train_only(make_plan):
    df = make_frame()
    plan = make_plan(
        split(),
        {"op": "impute", "columns": ["x"], "strategy": "median"},
        {"op": "scale", "columns": ["z"], "method": "standard"},
    )

    result = run_frame(df, plan)

    train_ids = set(result.train["row"])
    raw_train = df[df["row"].isin(train_ids)]
    raw_test = df[~df["row"].isin(train_ids)]
    imputer = result.pipeline.named_steps["1_impute"].named_transformers_["impute"]
    scaler = result.pipeline.named_steps["2_scale"].named_transformers_["scale"]

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
    plan = make_plan(split())

    result = run_frame(df, plan)

    features = result.train.drop(columns="t")
    pd.testing.assert_frame_equal(result.pipeline.transform(features), features)
    assert len(result.train) + len(result.test) == len(df)


def test_target_never_enters_the_pipeline(make_plan):
    df = make_frame()
    plan = make_plan(split(), {"op": "impute", "columns": ["x"], "strategy": "median"})

    result = run_frame(df, plan)

    assert "t" not in result.pipeline.feature_names_in_
    assert list(result.train.columns)[-1] == "t"
    assert result.train["t"].tolist() == result.target_train.tolist()


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
    plan = make_plan(split(), {"op": "impute", "columns": ["x"], "strategy": "median"})
    profile = profile_dataframe(df, target="t")

    with pytest.raises(executor.ExecutionError):
        execute(df.drop(columns="x"), plan, profile)
