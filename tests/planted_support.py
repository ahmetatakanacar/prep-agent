"""Plans of known quality for the planted datasets, shared by the scorer and end-to-end tests."""

import pandas as pd

from planted_truth import DatasetTruth
from prep_agent.plan import Plan

MAX_ONEHOT = 10


def build_plan(frame: pd.DataFrame, truth: DatasetTruth, drop: list[str], cast: list[str]) -> dict:
    """A plan that drops and casts the given columns and prepares the rest sensibly.

    The remaining columns are imputed and one-hot encoded, so the plan validates whatever the
    drop and cast lists are, as long as one feature column is left.
    """
    features = [c.name for c in truth.columns if c.name not in drop]
    missing = [name for name in features if frame[name].isna().any()]
    numeric = [n for n in missing if pd.api.types.is_numeric_dtype(frame[n]) and n not in cast]
    other = [n for n in missing if n not in numeric]
    encode = [n for n in features if n in cast or not pd.api.types.is_numeric_dtype(frame[n])]

    steps: list[dict] = []
    if drop:
        steps.append({"op": "drop_columns", "columns": drop, "reason": "Planted faults."})
    if cast:
        steps.append({"op": "cast_to_category", "columns": cast, "reason": "Codes, not numbers."})
    steps.append(
        {
            "op": "split",
            "test_size": 0.2,
            "stratify": truth.task_type == "classification",
            "random_state": 42,
            "reason": "Hold out a test set.",
        }
    )
    if numeric:
        steps.append(_impute(numeric, "median"))
    if other:
        steps.append(_impute(other, "most_frequent"))
    if encode:
        steps.append(
            {
                "op": "encode",
                "columns": encode,
                "method": "onehot",
                "max_categories": MAX_ONEHOT,
                "reason": "Text columns need numbers.",
            }
        )
    return {
        "target": truth.target,
        "task_type": truth.task_type,
        "model_family": "any",
        "steps": steps,
    }


def _impute(columns: list[str], strategy: str) -> dict:
    return {"op": "impute", "columns": columns, "strategy": strategy, "reason": "Gaps."}


def perfect_decisions(truth: DatasetTruth) -> tuple[list[str], list[str]]:
    """The drops and casts of a flawless plan: every must_drop column, the copy of each
    duplicate pair, and every column with an expected op."""
    drop = [c.name for c in truth.columns if c.role == "must_drop"]
    drop += [c.name for c in truth.columns if c.duplicate_of]
    cast = [c.name for c in truth.columns if c.expected_op == "cast_to_category"]
    return drop, cast


def plan_of(frame: pd.DataFrame, truth: DatasetTruth, drop: list[str], cast: list[str]) -> Plan:
    return Plan.model_validate(build_plan(frame, truth, drop, cast))


def perfect_plan(frame: pd.DataFrame, truth: DatasetTruth) -> Plan:
    return plan_of(frame, truth, *perfect_decisions(truth))


def leak_blind_plan(frame: pd.DataFrame, truth: DatasetTruth) -> Plan:
    """The perfect plan, except that it keeps every column derived from the target."""
    drop, cast = perfect_decisions(truth)
    leaks = {c.name for c in truth.columns if c.fault and c.fault.startswith("leak_")}
    return plan_of(frame, truth, [name for name in drop if name not in leaks], cast)


def drop_all_plan(frame: pd.DataFrame, truth: DatasetTruth) -> Plan:
    """Drops every feature but one plain numeric column."""
    keep = next(
        c.name
        for c in truth.columns
        if c.role == "must_keep"
        and c.fault is None
        and pd.api.types.is_numeric_dtype(frame[c.name])
    )
    return plan_of(frame, truth, [c.name for c in truth.columns if c.name != keep], [])
