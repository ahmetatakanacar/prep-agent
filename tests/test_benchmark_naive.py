import numpy as np
import pandas as pd
import pytest

from bench_naive import HIGH_CARDINALITY_CAP, naive_plan
from prep_agent.executor import execute
from prep_agent.plan import (
    DropColumnsStep,
    DropMissingTargetStep,
    EncodeStep,
    ImputeStep,
    SplitStep,
)
from prep_agent.profiler import profile_dataframe
from prep_agent.validators import has_errors, validate_plan

N = 200


def _frame(**extra) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    base = {
        "row_id": range(N),
        "const": ["same"] * N,
        "age": np.where(rng.random(N) < 0.1, np.nan, rng.normal(40, 10, N)),
        "income": rng.normal(50, 5, N),
        "city": np.where(rng.random(N) < 0.1, None, rng.choice(["a", "b", "c"], N)),
        "plan": rng.choice(["yes", "no"], N),
        "y": rng.choice(["stay", "go"], N),
    }
    return pd.DataFrame(base | extra)


def _plan(df: pd.DataFrame, task_type="classification"):
    profile = profile_dataframe(df, target="y")
    return naive_plan(profile, task_type), profile


def _column(profile, name):
    return next(c for c in profile.columns if c.name == name)


def _steps(plan, step_type) -> list:
    return [step for step in plan.steps if isinstance(step, step_type)]


def test_drops_identifier_and_constant_columns_but_never_the_target():
    plan, _ = _plan(_frame())
    [drop] = _steps(plan, DropColumnsStep)
    assert sorted(drop.columns) == ["const", "row_id"]


def test_keeps_the_target_even_when_it_looks_like_an_identifier():
    df = _frame(y=range(N))
    plan, _ = _plan(df, "regression")
    [drop] = _steps(plan, DropColumnsStep)
    assert "y" not in drop.columns


def test_has_no_drop_step_when_there_is_nothing_to_drop():
    plan, _ = _plan(_frame().drop(columns=["row_id", "const"]))
    assert _steps(plan, DropColumnsStep) == []


def test_numeric_columns_with_missing_values_get_median_imputation():
    plan, _ = _plan(_frame())
    numeric = [s for s in _steps(plan, ImputeStep) if s.strategy == "median"]
    assert [s.columns for s in numeric] == [["age"]]


def test_categorical_columns_with_missing_values_get_most_frequent_imputation():
    plan, _ = _plan(_frame())
    categorical = [s for s in _steps(plan, ImputeStep) if s.strategy == "most_frequent"]
    assert [s.columns for s in categorical] == [["city"]]


def test_columns_without_missing_values_are_not_imputed():
    plan, _ = _plan(_frame().drop(columns=["age", "city"]))
    assert _steps(plan, ImputeStep) == []


def test_every_non_numeric_column_is_one_hot_encoded():
    plan, _ = _plan(_frame())
    [encode] = _steps(plan, EncodeStep)
    assert encode.method == "onehot"
    assert sorted(encode.columns) == ["city", "plan"]
    assert encode.max_categories is None


def test_numeric_columns_are_not_encoded_or_scaled():
    plan, _ = _plan(_frame())
    assert {step.op for step in plan.steps} == {
        "drop_columns",
        "split",
        "impute",
        "encode",
    }
    assert all("income" not in getattr(step, "columns", []) for step in plan.steps)


def test_high_cardinality_columns_are_one_hot_encoded_with_a_cap():
    df = _frame(code=[f"c{i % (HIGH_CARDINALITY_CAP + 30)}" for i in range(N)])
    profile = profile_dataframe(df, target="y")
    assert _column(profile, "code").n_unique > HIGH_CARDINALITY_CAP
    plan = naive_plan(profile, "classification")
    capped = [s for s in _steps(plan, EncodeStep) if s.max_categories is not None]
    assert [(s.columns, s.max_categories) for s in capped] == [(["code"], HIGH_CARDINALITY_CAP)]
    plain = [s for s in _steps(plan, EncodeStep) if s.max_categories is None]
    assert sorted(plain[0].columns) == ["city", "plan"]


def test_date_and_text_columns_are_treated_like_categories():
    df = _frame(
        when=pd.date_range("2020-01-01", periods=N).strftime("%m/%d/%Y"),
        note=[f"a fairly long free text note number {i} about nothing" for i in range(N)],
    )
    plan, profile = _plan(df)
    kinds = {c.name: c.inferred_type for c in profile.columns}
    assert kinds["when"] == "datetime"
    encoded = {name for step in _steps(plan, EncodeStep) for name in step.columns}
    assert {"when"} <= encoded
    assert not has_errors(validate_plan(plan, profile))


def test_classification_splits_stratified_and_regression_does_not():
    [split] = _steps(_plan(_frame())[0], SplitStep)
    assert (split.stratify, split.test_size, split.random_state) == (True, 0.2, 42)
    [split] = _steps(_plan(_frame(y=np.arange(N, dtype=float)), "regression")[0], SplitStep)
    assert split.stratify is False


def test_drops_rows_with_a_missing_target_only_when_there_are_some():
    assert _steps(_plan(_frame())[0], DropMissingTargetStep) == []
    y = pd.Series(["stay", "go"] * (N // 2), dtype=object)
    y[:5] = None
    plan, _ = _plan(_frame(y=y))
    assert len(_steps(plan, DropMissingTargetStep)) == 1


def test_steps_come_in_the_order_the_validator_requires():
    y = pd.Series(["stay", "go"] * (N // 2), dtype=object)
    y[:5] = None
    plan, _ = _plan(_frame(y=y))
    ops = [step.op for step in plan.steps]
    assert ops[:3] == ["drop_columns", "drop_missing_target", "split"]
    assert ops[3:] == ["impute", "impute", "encode"]


def test_the_model_family_is_any_and_the_target_comes_from_the_profile():
    plan, _ = _plan(_frame())
    assert (plan.model_family, plan.target, plan.task_type) == ("any", "y", "classification")


def test_every_step_explains_itself():
    plan, _ = _plan(_frame())
    assert all(step.reason.strip() for step in plan.steps)


def test_rejects_a_profile_without_a_target():
    with pytest.raises(ValueError, match="target"):
        naive_plan(profile_dataframe(_frame()), "classification")


def test_the_plan_validates_without_any_issue_and_executes():
    df = _frame()
    plan, profile = _plan(df)
    assert validate_plan(plan, profile) == []
    result = execute(df, plan, profile)
    assert result.result_issues == []
    assert not result.train.isna().any().any()


def test_numeric_columns_stored_as_zero_one_are_imputed_with_the_median():
    flags = pd.Series([1.0, 0.0] * (N // 2))
    flags[:10] = np.nan
    df = _frame(flag=flags)
    plan, profile = _plan(df)
    assert (
        profile.columns[[c.name for c in profile.columns].index("flag")].inferred_type == "binary"
    )
    numeric = [s for s in _steps(plan, ImputeStep) if s.strategy == "median"]
    assert sorted(numeric[0].columns) == ["age", "flag"]
    assert not has_errors(validate_plan(plan, profile))
