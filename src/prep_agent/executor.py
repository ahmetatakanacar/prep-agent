"""Executor: runs a validated plan on a dataset.

Order of work: the plan is validated first (errors raise `PlanValidationError` and nothing runs),
then the stateless steps run on the whole frame, then the split, then the target is separated
and the stateful steps are chained into one sklearn `Pipeline` that is fit on the train features
only and merely applied to the test features. The target never enters the pipeline; it is
appended as the last column of both outputs.

The fitted pipeline covers the stateful steps only. The stateless steps and the split are
not part of it, so it cannot be applied to a raw file; a standalone `pipeline.py` is a later
stage. Its steps are named `<plan step index>_<op>`.

The whole procedure is deterministic: the split is seeded by the plan and no step uses
randomness of its own.
"""

from dataclasses import dataclass, field, replace

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from prep_agent.ops import OpContext, OpError, get_op
from prep_agent.plan import Plan, Step
from prep_agent.profiler import DatasetProfile
from prep_agent.result_validator import validate_result
from prep_agent.validators import ValidationIssue, has_errors, partition_issues, validate_plan

__all__ = [
    "ExecutionError",
    "ExecutionResult",
    "PlanValidationError",
    "execute",
    "validate_result",
]


class PlanValidationError(ValueError):
    """Raised when the plan has errors. `issues` holds all plan issues, warnings included."""

    def __init__(self, issues: list[ValidationIssue]) -> None:
        self.issues = issues
        errors, _ = partition_issues(issues)
        details = "; ".join(f"{issue.code}: {issue.message}" for issue in errors)
        super().__init__(f"the plan has {len(errors)} error(s): {details}")


class ExecutionError(RuntimeError):
    """Raised when a step fails on the data although the plan was valid."""


@dataclass(frozen=True)
class ExecutionResult:
    """What `execute` returns.

    `split_train` and `split_test` are the frames right after the split: stateless steps
    applied, stateful ones not. The result validator uses them to check that the target was
    not altered and to find rows shared by train and test, which must be judged before the
    pipeline can make distinct rows look alike.
    """

    train: pd.DataFrame
    test: pd.DataFrame
    pipeline: Pipeline
    split_train: pd.DataFrame
    split_test: pd.DataFrame
    plan_issues: list[ValidationIssue] = field(default_factory=list)
    result_issues: list[ValidationIssue] = field(default_factory=list)


def execute(df: pd.DataFrame, plan: Plan, profile: DatasetProfile) -> ExecutionResult:
    """Run `plan` on `df`. The input frame is not modified."""
    plan_issues = validate_plan(plan, profile)
    if has_errors(plan_issues):
        raise PlanValidationError(plan_issues)

    try:
        result = _run(df, plan, plan_issues)
    except (OpError, ValueError, KeyError) as exc:
        raise ExecutionError(f"{type(exc).__name__}: {exc}") from exc
    return replace(result, result_issues=validate_result(result, plan))


def _run(df: pd.DataFrame, plan: Plan, plan_issues: list[ValidationIssue]) -> ExecutionResult:
    ctx = OpContext(target=plan.target)
    # The plan validator guarantees the order: stateless steps, one split, stateful steps.
    stateless = [(i, s) for i, s in enumerate(plan.steps) if get_op(s.op).kind == "stateless"]
    stateful = [(i, s) for i, s in enumerate(plan.steps) if get_op(s.op).kind == "stateful"]
    [split_step] = [s for s in plan.steps if get_op(s.op).kind == "split"]

    frame = df
    for _, step in stateless:
        frame = get_op(step.op).apply(frame, step, ctx)
    train, test = get_op(split_step.op).apply(frame, split_step, ctx)

    x_train = train.drop(columns=plan.target)
    x_test = test.drop(columns=plan.target)
    y_train = train[plan.target].copy()
    y_test = test[plan.target].copy()

    pipeline = _build_pipeline(stateful)
    x_train_out = pipeline.fit_transform(x_train)
    x_test_out = pipeline.transform(x_test)

    return ExecutionResult(
        train=pd.concat([x_train_out, y_train], axis=1),
        test=pd.concat([x_test_out, y_test], axis=1),
        pipeline=pipeline,
        split_train=train,
        split_test=test,
        plan_issues=plan_issues,
    )


def _build_pipeline(stateful: list[tuple[int, Step]]) -> Pipeline:
    if not stateful:
        return Pipeline([("identity", FunctionTransformer())]).set_output(transform="pandas")
    steps = [
        (f"{index}_{step.op}", get_op(step.op).build_transformer(step)) for index, step in stateful
    ]
    return Pipeline(steps).set_output(transform="pandas")
