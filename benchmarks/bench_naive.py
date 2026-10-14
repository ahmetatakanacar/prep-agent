"""The naive comparison plan: what a fixed, rule-based default would do without any judgement.

It answers one question for the benchmark: is the agent better than a default that never looks
at what a column means? The rules are the same for every dataset:

1. drop the columns the profile marks as identifiers or constants (never the target);
2. drop the rows with a missing target, but only if there are any, because the split cannot
   stratify on a missing target;
3. split 80/20 with seed 42, stratified for classification;
4. impute numeric columns that have missing values with the median and every other column that
   has missing values with the most frequent value;
5. one-hot encode every column that is not numeric, datetime and text columns included, since
   a fixed rule has no other way to handle them. A column with more than
   `HIGH_CARDINALITY_CAP` categories keeps only its most frequent ones (the rest are grouped)
   so that a name-like column cannot blow up the table.

Steps that would change nothing are left out so that the plan validates without any issue.
"""

from typing import Literal

from pandas.api import types as pdt

from prep_agent.plan import (
    DropColumnsStep,
    DropMissingTargetStep,
    EncodeStep,
    ImputeStep,
    Plan,
    SplitStep,
    Step,
)
from prep_agent.profiler import ColumnProfile, DatasetProfile
from prep_agent.validators import UNINFORMATIVE_TYPES, UNINFORMATIVE_WARNINGS

# The profiler's own threshold for calling a categorical column high-cardinality.
HIGH_CARDINALITY_CAP = 50


def naive_plan(profile: DatasetProfile, task_type: Literal["classification", "regression"]) -> Plan:
    """Build the rule-based plan for a profile that carries its target."""
    if profile.target is None:
        raise ValueError("the profile needs a target column to build a naive plan")
    target = profile.target
    target_profile = next(c for c in profile.columns if c.name == target)
    features = [c for c in profile.columns if c.name != target]

    dropped = [c.name for c in features if _uninformative(c)]
    kept = [c for c in features if c.name not in dropped]
    numeric = [c for c in kept if _is_numeric(c)]
    other = [c for c in kept if not _is_numeric(c)]

    steps: list[Step] = []
    if dropped:
        steps.append(
            DropColumnsStep(
                op="drop_columns",
                columns=dropped,
                reason="Naive rule: the profile marks these columns as identifiers or constants.",
            )
        )
    if target_profile.n_missing:
        steps.append(
            DropMissingTargetStep(
                op="drop_missing_target",
                reason="Naive rule: rows without a target value cannot be used or stratified on.",
            )
        )
    steps.append(
        SplitStep(
            op="split",
            test_size=0.2,
            stratify=task_type == "classification",
            random_state=42,
            reason="Naive rule: 80/20 split with seed 42, stratified for classification.",
        )
    )
    steps.extend(_imputation(numeric, other))
    steps.extend(_encoding(other))
    return Plan(target=target, task_type=task_type, model_family="any", steps=steps)


def _imputation(numeric: list[ColumnProfile], other: list[ColumnProfile]) -> list[Step]:
    steps: list[Step] = []
    numeric_missing = [c.name for c in numeric if c.n_missing]
    if numeric_missing:
        steps.append(
            ImputeStep(
                op="impute",
                columns=numeric_missing,
                strategy="median",
                reason="Naive rule: numeric columns with missing values get the median.",
            )
        )
    other_missing = [c.name for c in other if c.n_missing]
    if other_missing:
        steps.append(
            ImputeStep(
                op="impute",
                columns=other_missing,
                strategy="most_frequent",
                reason="Naive rule: other columns with missing values get the most frequent value.",
            )
        )
    return steps


def _encoding(other: list[ColumnProfile]) -> list[Step]:
    steps: list[Step] = []
    plain = [c.name for c in other if c.n_unique <= HIGH_CARDINALITY_CAP]
    capped = [c.name for c in other if c.n_unique > HIGH_CARDINALITY_CAP]
    if plain:
        steps.append(
            EncodeStep(
                op="encode",
                columns=plain,
                method="onehot",
                reason="Naive rule: every non-numeric column is one-hot encoded.",
            )
        )
    if capped:
        steps.append(
            EncodeStep(
                op="encode",
                columns=capped,
                method="onehot",
                max_categories=HIGH_CARDINALITY_CAP,
                reason=(
                    "Naive rule: one-hot encode, keeping only the "
                    f"{HIGH_CARDINALITY_CAP} most frequent categories of columns with more."
                ),
            )
        )
    return steps


def _uninformative(column: ColumnProfile) -> bool:
    return column.inferred_type in UNINFORMATIVE_TYPES or bool(
        UNINFORMATIVE_WARNINGS.intersection(column.warnings)
    )


def _is_numeric(column: ColumnProfile) -> bool:
    # The same notion of numeric as the plan validator: 0/1 columns stored as numbers count.
    if column.inferred_type == "numeric":
        return True
    return column.inferred_type == "binary" and bool(pdt.is_numeric_dtype(column.pandas_dtype))
