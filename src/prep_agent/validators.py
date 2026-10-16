"""Deterministic plan validator: checks a plan against the dataset profile before it runs.

The validator walks the steps in order and simulates, for every column, whether it still exists,
its type and whether it still contains missing values:

- `drop_columns` removes the columns. Rows that differed only in them become duplicates.
- `dedupe` removes the duplicates; one over all columns is a step without effect when the
  profile counted none and no dropped column could have created some.
- `cast_to_category` makes the column categorical: encode is then allowed, mean and median
  impute and scale are errors, and the missing values stay.
- `impute` clears the missing values.
- `encode` with `ordinal` keeps the column under the same name and makes it numeric.
- `encode` with `onehot` consumes the column: the names of the output columns are only known
  after fitting, so later steps cannot refer to it.
- `drop_missing_target` clears the missing values of the target. A split that still sees
  missing target values is an error (`missing_target_at_split`): the rows would end up in
  train and test without a label.
- `scale` changes nothing. The extra columns of `impute` with `add_indicator` are not tracked.

Leakage protection lives here and not in the planner: the kind of every step (stateless, split
or stateful) is read from the op registry and compared with the position of the split. Order is
judged relative to the first split; a plan without a split only gets `invalid_split_count`.

Issues come out in a fixed order: plan-level checks, then the steps in plan order (the order
check of a step first, then its columns in the order the step lists them), then the end-of-plan
checks, one rule at a time with columns in profile order.

The task type is judged from the profile of the target: a regression plan needs a numeric dtype
(error), and a classification plan with a numeric target of more distinct values than the
profiler's `regression_min_unique` gets a warning, since that is where the profile itself would
hint at regression.
"""

from dataclasses import dataclass
from typing import Literal

from pandas.api import types as pdt
from pydantic import BaseModel, ConfigDict

from prep_agent.ops import get_op
from prep_agent.plan import (
    CastToCategoryStep,
    DedupeStep,
    DropColumnsStep,
    DropMissingTargetStep,
    EncodeStep,
    ImputeStep,
    Plan,
    ScaleStep,
    SplitStep,
    Step,
)
from prep_agent.profiler import DEFAULT_CONFIG, ColumnProfile, DatasetProfile, InferredType

Severity = Literal["error", "warning"]
IssueCode = Literal[
    # errors
    "target_mismatch",
    "target_not_in_profile",
    "model_family_mismatch",
    "invalid_split_count",
    "stateless_after_split",
    "stateful_before_split",
    "unknown_column",
    "target_in_step",
    "impute_non_numeric",
    "fill_value_type_mismatch",
    "scale_non_numeric",
    "encode_numeric",
    "ordinal_with_missing",
    "stratify_regression",
    "missing_target_at_split",
    # an error for a regression task with a non-numeric target, a warning in the other direction
    "task_type_mismatch",
    # warnings
    "cast_non_numeric",
    "onehot_before_impute",
    "ordinal_for_linear",
    "scale_for_tree",
    "repeated_op",
    "impute_without_missing",
    "dedupe_without_duplicates",
    "drop_missing_target_without_missing",
    "uninformative_column_kept",
    "missing_values_remain",
    "unencoded_column_remains",
    "stratify_disabled",
    # raised by the agent when a valid plan fails on the data, not by validate_plan
    "execution_failed",
    # result errors (raised by result_validator, not by validate_plan)
    "empty_split",
    "column_mismatch",
    "target_altered",
    # result warnings
    "train_test_overlap",
    "nan_in_output",
    "non_numeric_output",
    "possible_target_leakage",
    "class_distribution_shift",
]

# The severity of an issue is a property of its code, defined once here. Two exceptions pass the
# severity explicitly. `fill_value_type_mismatch`: a string fill value for a numeric column makes
# the step fail, so that case is an error, while a number in a text column only changes the
# values, so that case is a warning. `task_type_mismatch`: a regression task on a target that is
# not numeric cannot work, so that case is an error, while a classification task on a numeric
# target with many values may just be a regression task, so that case is a warning. The entries
# below are the error severity.
SEVERITY: dict[IssueCode, Severity] = {
    "target_mismatch": "error",
    "target_not_in_profile": "error",
    "model_family_mismatch": "error",
    "invalid_split_count": "error",
    "stateless_after_split": "error",
    "stateful_before_split": "error",
    "unknown_column": "error",
    "target_in_step": "error",
    "impute_non_numeric": "error",
    "fill_value_type_mismatch": "error",
    "scale_non_numeric": "error",
    "encode_numeric": "error",
    "ordinal_with_missing": "error",
    "stratify_regression": "error",
    "missing_target_at_split": "error",
    "task_type_mismatch": "error",
    "cast_non_numeric": "warning",
    "onehot_before_impute": "warning",
    "ordinal_for_linear": "warning",
    "scale_for_tree": "warning",
    "repeated_op": "warning",
    "impute_without_missing": "warning",
    "dedupe_without_duplicates": "warning",
    "drop_missing_target_without_missing": "warning",
    "uninformative_column_kept": "warning",
    "missing_values_remain": "warning",
    "unencoded_column_remains": "warning",
    "stratify_disabled": "warning",
    "execution_failed": "error",
    "empty_split": "error",
    "column_mismatch": "error",
    "target_altered": "error",
    "train_test_overlap": "warning",
    "nan_in_output": "warning",
    "non_numeric_output": "warning",
    "possible_target_leakage": "warning",
    "class_distribution_shift": "warning",
}

# Type and warning sets the rules are built on.
NUMERIC_ONLY_IMPUTE_STRATEGIES = frozenset({"mean", "median"})
UNENCODED_TYPES = frozenset({"categorical", "text", "datetime"})
UNINFORMATIVE_TYPES = frozenset({"id_like", "constant"})
UNINFORMATIVE_WARNINGS = frozenset({"possible_id", "constant"})
LOW_CARDINALITY_WARNING = "low_cardinality_numeric"


class ValidationIssue(BaseModel):
    """One finding. `message` says what is wrong and how to fix it."""

    model_config = ConfigDict(frozen=True)

    code: IssueCode
    severity: Severity
    step_index: int | None = None
    column: str | None = None
    message: str


def has_errors(issues: list[ValidationIssue]) -> bool:
    return any(issue.severity == "error" for issue in issues)


def partition_issues(
    issues: list[ValidationIssue],
) -> tuple[list[ValidationIssue], list[ValidationIssue]]:
    """Return (errors, warnings), each in the original order."""
    errors = [issue for issue in issues if issue.severity == "error"]
    warnings = [issue for issue in issues if issue.severity == "warning"]
    return errors, warnings


def validate_plan(
    plan: Plan, profile: DatasetProfile, expected_model_family: str | None = None
) -> list[ValidationIssue]:
    """Check `plan` against `profile` without running anything. Inputs are not modified.

    `expected_model_family` is the family the user asked for, if any; a plan with another one
    gets `model_family_mismatch`. Without it, the plan's family is not questioned.
    """
    return _Validation(plan, profile, expected_model_family).run()


@dataclass
class _Column:
    inferred_type: InferredType
    numeric: bool
    has_missing: bool
    # The profile saw few distinct whole numbers: a hint that the numbers may be categories.
    few_distinct_numbers: bool = False


def _starts_numeric(column: ColumnProfile) -> bool:
    # A binary column counts as numeric only when it is stored as numbers (0/1), not as yes/no.
    if column.inferred_type == "numeric":
        return True
    return column.inferred_type == "binary" and bool(pdt.is_numeric_dtype(column.pandas_dtype))


def _needs_encoding(column: _Column) -> bool:
    if column.inferred_type in UNENCODED_TYPES:
        return True
    return column.inferred_type == "binary" and not column.numeric


class _Validation:
    def __init__(
        self, plan: Plan, profile: DatasetProfile, expected_model_family: str | None
    ) -> None:
        self.plan = plan
        self.profile = profile
        self.expected_model_family = expected_model_family
        self.issues: list[ValidationIssue] = []
        self.columns = {
            c.name: _Column(
                c.inferred_type,
                _starts_numeric(c),
                c.n_missing > 0,
                LOW_CARDINALITY_WARNING in c.warnings,
            )
            for c in profile.columns
        }
        # Whether rows that equal each other over all columns may still exist at this point.
        self.duplicates_possible = profile.n_duplicate_rows > 0
        self.gone: dict[str, str] = {}
        self.applied: set[tuple[str, str]] = set()
        self.kinds = [get_op(step.op).kind for step in plan.steps]
        self.first_split = self.kinds.index("split") if "split" in self.kinds else None

    def run(self) -> list[ValidationIssue]:
        self._check_target_matches_profile()
        self._check_target_exists()
        self._check_task_type()
        self._check_model_family()
        self._check_split_count()
        for index, step in enumerate(self.plan.steps):
            self._check_order(index, step)
            self._simulate(index, step)
        self._check_end_of_plan()
        return self.issues

    def _add(
        self,
        code: IssueCode,
        message: str,
        step_index: int | None = None,
        column: str | None = None,
        severity: Severity | None = None,
    ) -> None:
        self.issues.append(
            ValidationIssue(
                code=code,
                severity=severity or SEVERITY[code],
                step_index=step_index,
                column=column,
                message=message,
            )
        )

    # --- plan level ---

    def _check_target_matches_profile(self) -> None:
        profile_target = self.profile.target
        if profile_target is not None and profile_target != self.plan.target:
            self._add(
                "target_mismatch",
                f"the plan's target {self.plan.target!r} differs from the target of the "
                f"profile, {profile_target!r}, which is the one the user specified; set the "
                "plan's 'target' to the user's target.",
                column=self.plan.target,
            )

    def _check_target_exists(self) -> None:
        target = self.plan.target
        if target not in {c.name for c in self.profile.columns}:
            self._add(
                "target_not_in_profile",
                f"target {target!r} is not a column of the dataset; set 'target' to one of: "
                f"{', '.join(self.columns)}.",
                column=target,
            )

    def _check_task_type(self) -> None:
        target = next((c for c in self.profile.columns if c.name == self.plan.target), None)
        if target is None:
            return
        numeric = bool(pdt.is_numeric_dtype(target.pandas_dtype))
        if self.plan.task_type == "regression" and not numeric:
            self._add(
                "task_type_mismatch",
                f"the task is regression but the target {target.name!r} is not numeric "
                f"({target.inferred_type}); set 'task_type' to 'classification', or use a "
                "numeric target.",
                column=target.name,
            )
        elif (
            self.plan.task_type == "classification"
            and numeric
            and target.n_unique > DEFAULT_CONFIG.regression_min_unique
        ):
            self._add(
                "task_type_mismatch",
                f"the task is classification but the target {target.name!r} is numeric with "
                f"{target.n_unique} distinct values, which looks more like regression; set "
                "'task_type' to 'regression' unless these values are class labels.",
                column=target.name,
                severity="warning",
            )

    def _check_model_family(self) -> None:
        expected = self.expected_model_family
        if expected is not None and expected != self.plan.model_family:
            self._add(
                "model_family_mismatch",
                f"the plan's model_family {self.plan.model_family!r} differs from "
                f"{expected!r}, which is the one the user specified; set the plan's "
                "'model_family' to the user's model family and choose the steps for it.",
            )

    def _check_split_count(self) -> None:
        splits = [i for i, kind in enumerate(self.kinds) if kind == "split"]
        if not splits:
            self._add(
                "invalid_split_count",
                "the plan has no split step; add exactly one 'split' step after the stateless "
                "steps (dedupe, drop_columns, drop_missing_target) and before any impute, encode "
                "or scale step.",
            )
        elif len(splits) > 1:
            self._add(
                "invalid_split_count",
                f"the plan has {len(splits)} split steps (steps {splits}); keep exactly one.",
                step_index=splits[1],
            )

    # --- one step ---

    def _check_order(self, index: int, step: Step) -> None:
        kind = self.kinds[index]
        if self.first_split is None:
            return
        if kind == "stateless" and index > self.first_split:
            self._add(
                "stateless_after_split",
                f"stateless step {step.op!r} comes after the split (step {self.first_split}); "
                "move it before the split so it is applied to the whole dataset consistently.",
                step_index=index,
            )
        elif kind == "stateful" and index < self.first_split:
            self._add(
                "stateful_before_split",
                f"{step.op!r} learns from data but comes before the split (step "
                f"{self.first_split}), which leaks test data into training; move it after the "
                "split so it is fit on train only.",
                step_index=index,
            )

    def _simulate(self, index: int, step: Step) -> None:
        if isinstance(step, DedupeStep):
            self._dedupe(index, step)
        elif isinstance(step, DropColumnsStep):
            for name in self._resolve(index, step.columns):
                del self.columns[name]
                self.gone[name] = f"dropped by drop_columns at step {index}"
                self.duplicates_possible = True
        elif isinstance(step, CastToCategoryStep):
            self._cast(index, step)
        elif isinstance(step, DropMissingTargetStep):
            self._drop_missing_target(index)
        elif isinstance(step, SplitStep):
            self._check_split(index, step)
        elif isinstance(step, ImputeStep):
            self._impute(index, step)
        elif isinstance(step, EncodeStep):
            self._encode(index, step)
        elif isinstance(step, ScaleStep):
            self._scale(index, step)

    def _resolve(self, index: int, names: list[str], allow_target: bool = False) -> list[str]:
        """Return the names a step can act on; report the target and unknown columns."""
        usable = []
        for name in names:
            if name == self.plan.target and not allow_target:
                self._add(
                    "target_in_step",
                    f"the target {name!r} must not be changed by this step; remove it from "
                    "'columns'.",
                    step_index=index,
                    column=name,
                )
            elif name not in self.columns:
                self._add(
                    "unknown_column",
                    self._unknown_column_message(name),
                    step_index=index,
                    column=name,
                )
            else:
                usable.append(name)
        return usable

    def _unknown_column_message(self, name: str) -> str:
        if name not in self.gone:
            return f"column {name!r} does not exist in the dataset; use a column from the profile."
        reason = self.gone[name]
        if "one-hot" in reason:
            return (
                f"column {name!r} was {reason}; the names of one-hot output columns are only "
                "known after fitting, so no later step can refer to it. Apply every other step "
                "to this column before the one-hot encode."
            )
        return f"column {name!r} was {reason} and cannot be used afterwards; remove the reference."

    def _note_repeat(self, index: int, step: Step, name: str) -> None:
        if (step.op, name) in self.applied:
            self._add(
                "repeated_op",
                f"{step.op!r} is applied to column {name!r} more than once; keep a single step.",
                step_index=index,
                column=name,
            )
        self.applied.add((step.op, name))

    def _check_split(self, index: int, step: SplitStep) -> None:
        target = self.columns.get(self.plan.target)
        if target is not None and target.has_missing and index == self.first_split:
            self._add(
                "missing_target_at_split",
                f"the target {self.plan.target!r} still has missing values at the split, so "
                "rows without a label would end up in train and test; add a "
                "drop_missing_target step before the split.",
                step_index=index,
                column=self.plan.target,
            )
        if self.plan.task_type == "regression" and step.stratify:
            self._add(
                "stratify_regression",
                "stratify is on for a regression task, but a continuous target has no classes "
                "to stratify on; set stratify to false.",
                step_index=index,
            )
        if self.plan.task_type == "classification" and not step.stratify:
            self._add(
                "stratify_disabled",
                "stratify is off for a classification task, so class proportions can drift "
                "between train and test; set stratify to true unless classes are too small.",
                step_index=index,
            )

    def _dedupe(self, index: int, step: DedupeStep) -> None:
        self._resolve(index, step.subset or [], allow_target=True)
        # A subset can match rows that differ elsewhere, which the profile's count of identical
        # rows says nothing about, so only a dedupe over all columns can be called pointless.
        if step.subset is None and not self.duplicates_possible:
            self._add(
                "dedupe_without_duplicates",
                "the profile shows no duplicate rows, and nothing earlier in the plan can create "
                "some, so this dedupe removes nothing on this data; remove the step unless it is "
                "meant for duplicates in new data.",
                step_index=index,
            )
        self.duplicates_possible = False

    def _drop_missing_target(self, index: int) -> None:
        target = self.columns.get(self.plan.target)
        if target is None:
            return
        if not target.has_missing:
            self._add(
                "drop_missing_target_without_missing",
                f"the target {self.plan.target!r} has no missing values in the profile or after "
                "the earlier steps, so this step removes no rows on this data; remove it unless "
                "it is meant for missing targets in new data.",
                step_index=index,
                column=self.plan.target,
            )
        target.has_missing = False

    def _cast(self, index: int, step: CastToCategoryStep) -> None:
        for name in self._resolve(index, step.columns):
            self._note_repeat(index, step, name)
            column = self.columns[name]
            if not column.numeric:
                self._add(
                    "cast_non_numeric",
                    f"column {name!r} is already {column.inferred_type}, not a number, so "
                    "casting it to category changes nothing useful; remove it from 'columns'.",
                    step_index=index,
                    column=name,
                )
            column.inferred_type = "categorical"
            column.numeric = False
            column.few_distinct_numbers = False

    def _impute(self, index: int, step: ImputeStep) -> None:
        for name in self._resolve(index, step.columns):
            self._note_repeat(index, step, name)
            column = self.columns[name]
            # A warning, not an error: the step may be there on purpose for data that arrives later.
            if not column.has_missing:
                self._add(
                    "impute_without_missing",
                    f"column {name!r} has no missing values in the profile or after the earlier "
                    "steps, so imputing it changes nothing on this data; remove it from "
                    "'columns' unless the step is meant for missing values in new data.",
                    step_index=index,
                    column=name,
                )
            if step.strategy in NUMERIC_ONLY_IMPUTE_STRATEGIES and not column.numeric:
                self._add(
                    "impute_non_numeric",
                    f"strategy {step.strategy!r} needs a numeric column but {name!r} is "
                    f"{column.inferred_type}; use 'most_frequent' or 'constant'.",
                    step_index=index,
                    column=name,
                )
            if step.strategy == "constant":
                self._check_fill_value(index, step, name, column)
            column.has_missing = False

    def _check_fill_value(self, index: int, step: ImputeStep, name: str, column: _Column) -> None:
        is_text = isinstance(step.fill_value, str)
        if column.numeric and is_text:
            self._add(
                "fill_value_type_mismatch",
                f"fill_value {step.fill_value!r} is a string but column {name!r} is numeric, so "
                "the imputer fails; use a number or impute with 'mean', 'median' or "
                "'most_frequent'.",
                step_index=index,
                column=name,
            )
        elif not column.numeric and not is_text:
            self._add(
                "fill_value_type_mismatch",
                f"fill_value {step.fill_value!r} is a number but column {name!r} is "
                f"{column.inferred_type}; the number would be mixed into its categories. Use a "
                "string fill_value or 'most_frequent'.",
                step_index=index,
                column=name,
                severity="warning",
            )

    def _encode(self, index: int, step: EncodeStep) -> None:
        if step.method == "ordinal" and self.plan.model_family == "linear":
            self._add(
                "ordinal_for_linear",
                "ordinal encoding gives categories a false order that a linear model will "
                "use; prefer method 'onehot'.",
                step_index=index,
            )
        for name in self._resolve(index, step.columns):
            self._note_repeat(index, step, name)
            column = self.columns[name]
            if column.numeric:
                self._add(
                    "encode_numeric",
                    f"column {name!r} is already numeric ({column.inferred_type}) and cannot be "
                    f"encoded; {self._encode_numeric_advice(column)}",
                    step_index=index,
                    column=name,
                )
            elif step.method == "ordinal":
                if column.has_missing:
                    self._add(
                        "ordinal_with_missing",
                        f"column {name!r} still has missing values; impute it in an earlier "
                        "step before the ordinal encode.",
                        step_index=index,
                        column=name,
                    )
                column.inferred_type = "numeric"
                column.numeric = True
            else:
                if column.has_missing:
                    self._add(
                        "onehot_before_impute",
                        f"column {name!r} still has missing values, so they become their own "
                        "category; impute it first unless that is intended.",
                        step_index=index,
                        column=name,
                    )
                del self.columns[name]
                self.gone[name] = f"one-hot encoded at step {index}"

    @staticmethod
    def _encode_numeric_advice(column: _Column) -> str:
        if column.few_distinct_numbers:
            return (
                "it has few distinct values, so if they are categories (an hour, a month, a "
                "code), add a cast_to_category step for it before the split and keep it here; "
                "if they are real quantities, remove it from 'columns'."
            )
        return "remove it from 'columns'."

    def _scale(self, index: int, step: ScaleStep) -> None:
        if self.plan.model_family == "tree":
            self._add(
                "scale_for_tree",
                "tree models do not need scaling; remove this step.",
                step_index=index,
            )
        for name in self._resolve(index, step.columns):
            self._note_repeat(index, step, name)
            column = self.columns[name]
            if not column.numeric:
                self._add(
                    "scale_non_numeric",
                    f"column {name!r} is {column.inferred_type}, not numeric, and cannot be "
                    "scaled; encode it instead or remove it from 'columns'.",
                    step_index=index,
                    column=name,
                )

    # --- end of plan ---

    def _check_end_of_plan(self) -> None:
        target = self.plan.target
        profiled = {c.name: c for c in self.profile.columns}

        for name in self.columns:
            column = profiled[name]
            if name == target:
                continue
            if column.inferred_type in UNINFORMATIVE_TYPES or UNINFORMATIVE_WARNINGS.intersection(
                column.warnings
            ):
                self._add(
                    "uninformative_column_kept",
                    f"column {name!r} looks like an identifier or a constant "
                    f"({column.inferred_type}, warnings: {column.warnings}) and is still in the "
                    "data at the end of the plan; drop it with drop_columns.",
                    column=name,
                )

        # A target that still has missing values is `missing_target_at_split`, not this warning.
        for name, column in self.columns.items():
            if column.has_missing and name != target:
                self._add(
                    "missing_values_remain",
                    f"column {name!r} still has missing values at the end of the plan; add an "
                    "impute step after the split.",
                    column=name,
                )

        for name, column in self.columns.items():
            if name != target and _needs_encoding(column):
                self._add(
                    "unencoded_column_remains",
                    f"column {name!r} is {column.inferred_type} and is still not numeric at the "
                    "end of the plan; encode it or drop it.",
                    column=name,
                )
