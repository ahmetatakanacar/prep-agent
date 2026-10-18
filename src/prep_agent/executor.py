"""Executor: runs a validated plan on a dataset.

Order of work: the plan is validated first (errors raise `PlanValidationError` and nothing runs),
then the stateless steps run on the whole frame, then the split, then the target is separated
and the stateful steps are chained into one sklearn `Pipeline` that is fit on the train features
only and merely applied to the test features. The target never enters the pipeline; it is
appended as the last column of both outputs.

The fitted pipeline covers the stateful steps only. The stateless steps and the split are
not part of it, so it cannot be applied to a raw file; the standalone `pipeline.py` that repeats
every step is written by `codegen`. Its steps are named `<plan step index>_<op>`.

Every row keeps the identity of the input record it came from (a hash of all its input
columns) through the stateless steps and the split, so the result validator can tell whether the
same record reached both train and test.

The whole procedure is deterministic: the split is seeded by the plan and no step uses
randomness of its own.
"""

from collections.abc import Collection
from dataclasses import dataclass, field, replace

import numpy as np
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
    not altered. `split_train_records` and `split_test_records` give, row by row, the identity
    of the input record behind each of those rows: equal values mean the rows come from input
    rows that are equal in every column, which is what the overlap check needs.
    """

    train: pd.DataFrame
    test: pd.DataFrame
    pipeline: Pipeline
    split_train: pd.DataFrame
    split_test: pd.DataFrame
    split_train_records: np.ndarray
    split_test_records: np.ndarray
    plan_issues: list[ValidationIssue] = field(default_factory=list)
    result_issues: list[ValidationIssue] = field(default_factory=list)


def execute(
    df: pd.DataFrame,
    plan: Plan,
    profile: DatasetProfile,
    expected_model_family: str | None = None,
    decided_by_user: Collection[str] = (),
) -> ExecutionResult:
    """Run `plan` on `df`. The input frame is not modified.

    `expected_model_family` and `decided_by_user` are passed on to the plan validator (see
    `validate_plan`).
    """
    plan_issues = validate_plan(plan, profile, expected_model_family, decided_by_user)
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

    frame, records = df, _record_ids(df)
    for _, step in stateless:
        after = get_op(step.op).apply(frame, step, ctx)
        records = records[_surviving_positions(frame, after)]
        frame = after

    # The records travel through the split as a column: a split moves whole rows and ignores
    # the values in them, so the rows end up where they would without it.
    records_column = _unused_name(frame, "record")
    train, test = get_op(split_step.op).apply(
        frame.assign(**{records_column: records}), split_step, ctx
    )
    train_records = train.pop(records_column).to_numpy()
    test_records = test.pop(records_column).to_numpy()

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
        split_train_records=train_records,
        split_test_records=test_records,
        plan_issues=plan_issues,
    )


def _record_ids(df: pd.DataFrame) -> np.ndarray:
    return pd.util.hash_pandas_object(df, index=False).to_numpy()


def _unused_name(df: pd.DataFrame, name: str) -> str:
    while name in df.columns:
        name = f"_{name}_"
    return name


def _surviving_positions(before: pd.DataFrame, after: pd.DataFrame) -> np.ndarray:
    """Where the rows of `after` stood in `before`, for a stateless step that keeps or removes
    rows without reordering them.

    Rows are matched by content. When rows of `before` are equal, the one picked can differ
    from the one the step kept (dedupe with keep="last"), but it always has the same values in
    the columns the step looked at. That is enough: two rows kept by a dedupe never share an
    input record, and without a dedupe a step keeps or removes equal rows together.
    """
    if len(after) == len(before):
        return np.arange(len(before))
    before_hashes = _record_ids(before[list(after.columns)])
    positions = np.empty(len(after), dtype=np.intp)
    at = 0
    for k, row_hash in enumerate(_record_ids(after)):
        while at < len(before_hashes) and before_hashes[at] != row_hash:
            at += 1
        if at == len(before_hashes):
            raise ExecutionError("could not follow the rows through a step that removes rows")
        positions[k] = at
        at += 1
    return positions


def _build_pipeline(stateful: list[tuple[int, Step]]) -> Pipeline:
    if not stateful:
        return Pipeline([("identity", FunctionTransformer())]).set_output(transform="pandas")
    steps = [
        (f"{index}_{step.op}", get_op(step.op).build_transformer(step)) for index, step in stateful
    ]
    return Pipeline(steps).set_output(transform="pandas")
