"""Ops that learn nothing from the data, so they are safe to run before the split.

Every op returns a new frame with a fresh RangeIndex when rows were removed, and never
modifies its input.
"""

from collections.abc import Sequence

import pandas as pd

from prep_agent.ops.base import OpContext, OpError, StatelessOpSpec, StatementCode
from prep_agent.ops.code import Assign, Import, call, ref
from prep_agent.plan import CastToCategoryStep, DedupeStep, DropColumnsStep, DropMissingTargetStep


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


def _category_text(value: object) -> str:
    # A whole-number column with a gap is stored as floats, and str(1.0) would make "1.0" and
    # "1" different categories depending on whether the column happened to have a gap.
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _cast_to_category(df: pd.DataFrame, step: CastToCategoryStep, ctx: OpContext) -> pd.DataFrame:
    """Write the values of the columns as text, so that they are categories and not numbers.

    Missing values stay missing. A float with no fractional part is written without the decimal
    point (1.0 becomes "1", not "1.0"), so a whole-number column gives the same categories
    whether it is stored as integers or, because of missing values, as floats. Any other float
    keeps its fraction (2.25 becomes "2.25"), and text stays as it is.
    """
    _require_columns(df, step.columns, "cast_to_category")
    if ctx.target in step.columns:
        raise OpError(f"cast_to_category: cannot cast the target column {ctx.target!r}")
    out = df.copy()
    for column in step.columns:
        out[column] = df[column].map(_category_text, na_action="ignore")
    return out


# A chained call is written as two statements: the generated file then formats the same way
# whatever the length of the arguments.
def _dedupe_code(step: DedupeStep) -> StatementCode:
    return StatementCode(
        (
            Assign("df", call("df.drop_duplicates", subset=step.subset, keep=step.keep)),
            Assign("df", call("df.reset_index", drop=True)),
        )
    )


def _drop_columns_code(step: DropColumnsStep) -> StatementCode:
    return StatementCode((Assign("df", call("df.drop", columns=step.columns)),))


def _drop_missing_target_code(step: DropMissingTargetStep) -> StatementCode:
    return StatementCode(
        (
            Assign("df", call("df.dropna", subset=[ref("TARGET")])),
            Assign("df", call("df.reset_index", drop=True)),
        )
    )


# pipeline.py stands alone, so it carries a copy of the two functions above. The tests keep the
# copy equal to them.
_CATEGORY_TEXT_HELPER = '''\
def category_text(value: object) -> str:
    """The text of a category value: a float with no fractional part loses its ".0"."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)'''

_CAST_TO_CATEGORY_HELPER = '''\
def cast_to_category(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Write the columns as text and keep missing values missing."""
    out = df.copy()
    for column in columns:
        out[column] = df[column].map(category_text, na_action="ignore")
    return out'''


def _cast_to_category_code(step: CastToCategoryStep) -> StatementCode:
    return StatementCode(
        (Assign("df", call("cast_to_category", ref("df"), step.columns)),),
        (Import("pandas", alias="pd"),),
        (_CATEGORY_TEXT_HELPER, _CAST_TO_CATEGORY_HELPER),
    )


DEDUPE = StatelessOpSpec(
    name="dedupe",
    step_model=DedupeStep,
    description="Remove duplicate rows over all columns or a subset, keeping the first or last.",
    apply=_dedupe,
    codegen=_dedupe_code,
)
DROP_COLUMNS = StatelessOpSpec(
    name="drop_columns",
    step_model=DropColumnsStep,
    description="Remove columns that carry no usable signal, such as identifiers or constants.",
    apply=_drop_columns,
    codegen=_drop_columns_code,
)
DROP_MISSING_TARGET = StatelessOpSpec(
    name="drop_missing_target",
    step_model=DropMissingTargetStep,
    description="Remove rows whose target value is missing.",
    apply=_drop_missing_target,
    codegen=_drop_missing_target_code,
)
CAST_TO_CATEGORY = StatelessOpSpec(
    name="cast_to_category",
    step_model=CastToCategoryStep,
    description=(
        "Turn columns of numbers that are really categories (hour, month, weekday, a type or "
        "code number) into text, so they can be encoded. Missing values stay missing."
    ),
    apply=_cast_to_category,
    codegen=_cast_to_category_code,
)
