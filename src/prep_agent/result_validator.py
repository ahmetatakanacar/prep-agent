"""Result validator: checks the train and test frames the executor produced.

Rules, in the order their issues are reported (columns in frame order):

- `empty_split` (error): train or test has no rows.
- `column_mismatch` (error): train and test differ in columns or in their order.
- `target_altered` (error): the target in the output differs from the target right after the
  split, so something in the pipeline touched it.
- `train_test_overlap`: a row (features and target) is in both train and test. It is judged on
  the frames right after the split (stateless steps applied, stateful ones not), because the
  pipeline can make distinct rows look alike (one-hot grouping, imputation) without that
  being leakage. An error when the plan has a dedupe step, which should have prevented it, and
  a warning otherwise.
- `nan_in_output` (warning): a feature column still has missing values.
- `non_numeric_output` (warning): a feature column is not numeric.
- `possible_target_leakage` (warning): for a numeric target, a feature whose absolute
  correlation with the target in train is above `LEAKAGE_CORRELATION_THRESHOLD`.
- `class_distribution_shift` (warning): classification without stratify, and some class share
  differs between train and test by more than `CLASS_SHIFT_THRESHOLD`.

The first two errors make the other rules meaningless, so they are reported alone. Rows are
compared by a 64-bit row hash, so a match is not a proof of equality (the odds of a false
match are negligible for in-memory tables).
"""

from typing import TYPE_CHECKING

import pandas as pd
from pandas.api import types as pdt

from prep_agent.plan import DedupeStep, Plan, SplitStep
from prep_agent.validators import SEVERITY, IssueCode, Severity, ValidationIssue

if TYPE_CHECKING:
    from prep_agent.executor import ExecutionResult

LEAKAGE_CORRELATION_THRESHOLD = 0.98
CLASS_SHIFT_THRESHOLD = 0.10


def validate_result(result: "ExecutionResult", plan: Plan) -> list[ValidationIssue]:
    """Check the executor output against `plan`. Nothing is modified."""
    train, test, target = result.train, result.test, plan.target

    if train.empty or test.empty:
        empty = [name for name, frame in (("train", train), ("test", test)) if frame.empty]
        return [_issue("empty_split", f"the {' and '.join(empty)} set has no rows.")]
    if list(train.columns) != list(test.columns):
        return [
            _issue(
                "column_mismatch",
                f"train columns {list(train.columns)} and test columns {list(test.columns)} "
                "differ in names or order.",
            )
        ]

    issues: list[ValidationIssue] = []
    issues += _check_target(result, target)
    issues += _check_overlap(result.split_train, result.split_test, plan)
    features = [c for c in train.columns if c != target]
    issues += _check_missing(train, test, features)
    issues += _check_numeric(train, test, features)
    issues += _check_leakage(train, target, features)
    issues += _check_class_shift(train, test, plan)
    return issues


def _issue(
    code: IssueCode, message: str, column: str | None = None, severity: Severity | None = None
) -> ValidationIssue:
    return ValidationIssue(
        code=code, severity=severity or SEVERITY[code], column=column, message=message
    )


def _check_target(result: "ExecutionResult", target: str) -> list[ValidationIssue]:
    issues = []
    for name, output, split in (
        ("train", result.train, result.split_train),
        ("test", result.test, result.split_test),
    ):
        if target not in output.columns or not _same_values(output[target], split[target]):
            issues.append(
                _issue(
                    "target_altered",
                    f"the target {target!r} in the {name} output differs from its values right "
                    "after the split; no step may change the target.",
                    column=target,
                )
            )
    return issues


def _same_values(actual: pd.Series, expected: pd.Series) -> bool:
    return actual.reset_index(drop=True).equals(expected.reset_index(drop=True))


def _check_overlap(
    split_train: pd.DataFrame, split_test: pd.DataFrame, plan: Plan
) -> list[ValidationIssue]:
    train_hashes = set(pd.util.hash_pandas_object(split_train, index=False))
    n_shared = sum(h in train_hashes for h in pd.util.hash_pandas_object(split_test, index=False))
    if n_shared == 0:
        return []
    has_dedupe = any(isinstance(step, DedupeStep) for step in plan.steps)
    return [
        _issue(
            "train_test_overlap",
            f"{n_shared} test row(s) also appear in train (features and target equal at the "
            "split); "
            + (
                "the plan has a dedupe step, but rows became equal after it, for example "
                "through a later drop_columns; move the dedupe after the steps that drop "
                "columns."
                if has_dedupe
                else "add a dedupe step before the split if duplicates are not intended."
            ),
            severity="error" if has_dedupe else "warning",
        )
    ]


def _check_missing(
    train: pd.DataFrame, test: pd.DataFrame, features: list[str]
) -> list[ValidationIssue]:
    issues = []
    for name in features:
        n_train, n_test = int(train[name].isna().sum()), int(test[name].isna().sum())
        if n_train or n_test:
            issues.append(
                _issue(
                    "nan_in_output",
                    f"feature {name!r} still has missing values (train: {n_train}, test: "
                    f"{n_test}); impute it with an impute step after the split.",
                    column=name,
                )
            )
    return issues


def _check_numeric(
    train: pd.DataFrame, test: pd.DataFrame, features: list[str]
) -> list[ValidationIssue]:
    return [
        _issue(
            "non_numeric_output",
            f"feature {name!r} is not numeric (train dtype {train[name].dtype}); encode or "
            "drop it.",
            column=name,
        )
        for name in features
        if not (pdt.is_numeric_dtype(train[name]) and pdt.is_numeric_dtype(test[name]))
    ]


def _check_leakage(train: pd.DataFrame, target: str, features: list[str]) -> list[ValidationIssue]:
    # Only numeric targets are checked; a text target would first have to be encoded.
    if not pdt.is_numeric_dtype(train[target]):
        return []
    y = train[target].astype("float64")
    if y.nunique() < 2:
        return []
    issues = []
    for name in features:
        # A constant column has no correlation, and computing it would divide by zero.
        if not pdt.is_numeric_dtype(train[name]) or train[name].nunique() < 2:
            continue
        correlation = train[name].astype("float64").corr(y)
        if pd.notna(correlation) and abs(correlation) > LEAKAGE_CORRELATION_THRESHOLD:
            issues.append(
                _issue(
                    "possible_target_leakage",
                    f"feature {name!r} has a correlation of {correlation:.3f} with the target "
                    f"in train (threshold {LEAKAGE_CORRELATION_THRESHOLD}); check that it is "
                    "not derived from the target.",
                    column=name,
                )
            )
    return issues


def _check_class_shift(
    train: pd.DataFrame, test: pd.DataFrame, plan: Plan
) -> list[ValidationIssue]:
    split = next((s for s in plan.steps if isinstance(s, SplitStep)), None)
    if plan.task_type != "classification" or split is None or split.stratify:
        return []
    target = plan.target
    share_train = train[target].value_counts(normalize=True)
    share_test = test[target].value_counts(normalize=True)
    gap = share_train.sub(share_test, fill_value=0).abs()
    worst = gap.idxmax()
    if gap[worst] <= CLASS_SHIFT_THRESHOLD:
        return []
    return [
        _issue(
            "class_distribution_shift",
            f"the share of class {worst!r} differs by {gap[worst]:.2f} between train "
            f"({share_train.get(worst, 0):.2f}) and test ({share_test.get(worst, 0):.2f}), "
            f"above {CLASS_SHIFT_THRESHOLD}; set stratify to true.",
            column=target,
        )
    ]
