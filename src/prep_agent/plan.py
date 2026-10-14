"""Schema of the preparation plan that the planner (LLM) produces.

A plan is pure data: a target, a task type, a model family and an ordered list of steps. Each
step is its own pydantic model, told apart by its `op` literal, and every step carries a
non-empty `reason` that ends up in the report. All models reject unknown fields, so a plan that
does not match the schema exactly fails validation instead of being silently reinterpreted.

This module only checks the shape of a plan. Whether the columns exist, whether a step fits the
column's type, and whether stateful steps come after the split are checked by the validators
that see the profile and the step order.
"""

from typing import Annotated, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    StringConstraints,
    model_validator,
)


def _not_blank(name: str) -> str:
    if not name.strip():
        raise ValueError("column name must not be blank")
    return name


def _distinct(names: list[str]) -> list[str]:
    if len(set(names)) != len(names):
        raise ValueError("column names must be distinct")
    return names


# Column names are matched against the data exactly, so they are never stripped or altered.
ColumnName = Annotated[str, StringConstraints(min_length=1), AfterValidator(_not_blank)]
ColumnList = Annotated[list[ColumnName], Field(min_length=1), AfterValidator(_distinct)]
Reason = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
FillValue = StrictInt | StrictFloat | StrictStr


class _Step(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: Reason = Field(description="Why this step is needed. Shown in the report.")


class DedupeStep(_Step):
    """Remove duplicate rows."""

    op: Literal["dedupe"]
    subset: ColumnList | None = Field(
        default=None, description="Columns that define a duplicate. null means all columns."
    )
    keep: Literal["first", "last"]


class DropColumnsStep(_Step):
    """Remove columns that carry no usable signal (identifiers, constants, leaks)."""

    op: Literal["drop_columns"]
    columns: ColumnList


class DropMissingTargetStep(_Step):
    """Remove rows whose target value is missing."""

    op: Literal["drop_missing_target"]


class CastToCategoryStep(_Step):
    """Treat numbers that stand for categories (hour, month, codes) as categories."""

    op: Literal["cast_to_category"]
    columns: ColumnList


class SplitStep(_Step):
    """Split rows into train and test sets. Everything after this step is fit on train only."""

    op: Literal["split"]
    test_size: float = Field(default=0.2, gt=0, lt=1)
    stratify: bool = Field(description="Keep the target's class proportions in both parts.")
    random_state: int = Field(default=42, ge=0)


class ImputeStep(_Step):
    """Fill missing values with a statistic learned on the train set."""

    op: Literal["impute"]
    columns: ColumnList
    strategy: Literal["mean", "median", "most_frequent", "constant"]
    fill_value: FillValue | None = Field(
        default=None, description="Required for strategy 'constant', not allowed otherwise."
    )
    add_indicator: bool = Field(
        default=False, description="Also add a 0/1 column marking which values were missing."
    )

    @model_validator(mode="after")
    def _check_fill_value(self) -> Self:
        if self.strategy == "constant" and self.fill_value is None:
            raise ValueError("fill_value is required when strategy is 'constant'")
        if self.strategy != "constant" and self.fill_value is not None:
            raise ValueError("fill_value is only allowed when strategy is 'constant'")
        return self


class EncodeStep(_Step):
    """Turn categorical columns into numbers using categories learned on the train set."""

    op: Literal["encode"]
    columns: ColumnList
    method: Literal["onehot", "ordinal"]
    max_categories: int | None = Field(
        default=None,
        ge=2,
        description="Onehot only: keep at most this many columns per source column, "
        "grouping the rarest categories together.",
    )

    @model_validator(mode="after")
    def _check_max_categories(self) -> Self:
        if self.max_categories is not None and self.method != "onehot":
            raise ValueError("max_categories is only allowed when method is 'onehot'")
        return self


class ScaleStep(_Step):
    """Rescale numeric columns using statistics learned on the train set."""

    op: Literal["scale"]
    columns: ColumnList
    method: Literal["standard", "minmax", "robust"]


Step = Annotated[
    DedupeStep
    | DropColumnsStep
    | DropMissingTargetStep
    | CastToCategoryStep
    | SplitStep
    | ImputeStep
    | EncodeStep
    | ScaleStep,
    Field(discriminator="op"),
]


class Plan(BaseModel):
    """An ordered preparation plan for one dataset and target."""

    model_config = ConfigDict(extra="forbid")

    target: ColumnName
    task_type: Literal["classification", "regression"]
    model_family: Literal["tree", "linear", "any"]
    steps: list[Step]
