"""Ops that learn nothing from the data, so they are safe to run before the split.

Every op returns a new frame with a fresh RangeIndex when rows were removed, and never
modifies its input.
"""

from collections.abc import Sequence

import pandas as pd

from prep_agent.ops.base import OpContext, OpError, StatelessOpSpec
from prep_agent.plan import DedupeStep, DropColumnsStep, DropMissingTargetStep


def _require_columns(df: pd.DataFrame, columns: Sequence[str], op: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise OpError(f"{op}: columns not found in the data: {missing}")


def _dedupe(df: pd.DataFrame, step: DedupeStep, ctx: OpContext) -> pd.DataFrame:
    if step.subset is not None:
        _require_columns(df, step.subset, "dedupe")
    return df.drop_duplicates(subset=step.subset, keep=step.keep).reset_index(drop=True)


def _drop_columns(df: pd.DataFrame, step: DropColumnsStep, ctx: OpContext) -> pd.DataFrame:
    _require_columns(df, step.columns, "drop_columns")
    if ctx.target in step.columns:
        raise OpError(f"drop_columns: cannot drop the target column {ctx.target!r}")
    return df.drop(columns=step.columns)


def _drop_missing_target(
    df: pd.DataFrame, step: DropMissingTargetStep, ctx: OpContext
) -> pd.DataFrame:
    if ctx.target not in df.columns:
        raise OpError(f"drop_missing_target: target column {ctx.target!r} not found in the data")
    return df.dropna(subset=[ctx.target]).reset_index(drop=True)


DEDUPE = StatelessOpSpec(
    name="dedupe",
    step_model=DedupeStep,
    description="Remove duplicate rows over all columns or a subset, keeping the first or last.",
    apply=_dedupe,
)
DROP_COLUMNS = StatelessOpSpec(
    name="drop_columns",
    step_model=DropColumnsStep,
    description="Remove columns that carry no usable signal, such as identifiers or constants.",
    apply=_drop_columns,
)
DROP_MISSING_TARGET = StatelessOpSpec(
    name="drop_missing_target",
    step_model=DropMissingTargetStep,
    description="Remove rows whose target value is missing.",
    apply=_drop_missing_target,
)
