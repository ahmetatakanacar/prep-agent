"""The ground truth of a planted dataset: what is wrong with each column and what a plan should do.

Every fault type is labelled with the decision that handles it. A type no op can handle is
`measurable=False` on its columns: the data is in the set, but no score is computed from it,
because a plan has no way to get it right or wrong.
"""

from typing import Literal, Self, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

FaultType = Literal[
    "leak_linear",
    "leak_nonlinear",
    "leak_categorical",
    "id_sequential",
    "id_uuid",
    "pii_email",
    "pii_phone",
    "pii_name",
    "duplicate_exact",
    "duplicate_scaled",
    "numeric_category",
    "constant",
    "sentinel",
]
Role = Literal["must_drop", "must_keep", "either"]
OpName = Literal["cast_to_category"]

FAULT_TYPES: tuple[str, ...] = get_args(FaultType)

# Types whose column has to be dropped for the plan to count as having found the fault.
DROP_FAULTS = frozenset(
    {
        "leak_linear",
        "leak_nonlinear",
        "leak_categorical",
        "id_sequential",
        "id_uuid",
        "pii_email",
        "pii_phone",
        "pii_name",
        "constant",
    }
)
DUPLICATE_FAULTS = frozenset({"duplicate_exact", "duplicate_scaled"})

# Faults the op registry cannot fix, with the reason. `sentinel`: turning -999 into a missing
# value would need an op that replaces values, and there is none; dropping or keeping the column
# are both defensible, so neither is scored.
UNMEASURABLE_FAULTS: dict[str, str] = {
    "sentinel": "no op turns a special value into a missing value",
}


class ColumnTruth(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    role: Role
    fault: FaultType | None = Field(
        default=None, description="The fault planted in the column; null for a plain column."
    )
    expected_op: OpName | None = Field(
        default=None, description="An op a good plan applies to the column."
    )
    avoid_op: OpName | None = Field(
        default=None, description="An op that would be a mistake on the column."
    )
    duplicate_of: str | None = Field(
        default=None, description="For a duplicate: the column it copies (the original)."
    )
    measurable: bool = True

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.fault in DROP_FAULTS and self.role != "must_drop":
            raise ValueError(f"{self.fault} column {self.name!r} must have role must_drop")
        if self.fault in DUPLICATE_FAULTS:
            if self.role != "either" or self.duplicate_of is None:
                raise ValueError("a duplicate has role either and names the column it copies")
        elif self.duplicate_of is not None:
            raise ValueError("duplicate_of is only for duplicate faults")
        if self.fault == "numeric_category" and self.expected_op != "cast_to_category":
            raise ValueError("a numeric_category column expects cast_to_category")
        if self.expected_op is not None and self.role != "must_keep":
            raise ValueError("a column with an expected op must be kept")
        if self.fault in UNMEASURABLE_FAULTS and self.measurable:
            raise ValueError(f"{self.fault} is not measurable: {UNMEASURABLE_FAULTS[self.fault]}")
        if self.fault not in UNMEASURABLE_FAULTS and not self.measurable:
            raise ValueError(f"{self.fault} is measurable")
        return self


class DatasetTruth(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str
    seed: int
    target: str
    task_type: Literal["classification", "regression"]
    n_rows: int
    columns: list[ColumnTruth] = Field(description="The feature columns, in file order.")

    @model_validator(mode="after")
    def _check(self) -> Self:
        names = [c.name for c in self.columns]
        if len(set(names)) != len(names) or self.target in names:
            raise ValueError("column names must be distinct and different from the target")
        for column in self.columns:
            if column.duplicate_of is not None and column.duplicate_of not in names:
                raise ValueError(f"{column.name!r} copies unknown column {column.duplicate_of!r}")
        return self

    def column(self, name: str) -> ColumnTruth:
        return next(c for c in self.columns if c.name == name)
