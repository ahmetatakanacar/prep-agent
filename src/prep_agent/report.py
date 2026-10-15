"""Report: the human-readable report.md of one run.

Everything in the report is derived from data the run already has (the plan, the validator
issues, the baseline scores, the run metadata). No LLM writes any of it, so the same inputs
always give the same text. The layout lives in `templates/report.md.j2`; this module prepares
plain values for it.
"""

from typing import TYPE_CHECKING

from jinja2 import Environment, PackageLoader, StrictUndefined

from prep_agent.agent import AgentTrace
from prep_agent.baseline import SMALL_TRAIN_ROWS, BaselineResult, metric_label
from prep_agent.executor import ExecutionResult
from prep_agent.llm.base import format_usage
from prep_agent.plan import (
    DedupeStep,
    DropColumnsStep,
    DropMissingTargetStep,
    Plan,
    SplitStep,
    Step,
)
from prep_agent.validators import ValidationIssue

if TYPE_CHECKING:
    from prep_agent.runs import RunInfo

SHA_LENGTH = 12
_PARAMETERS_SKIPPED = {"op", "reason", "columns", "subset"}


def _cell(value: object) -> str:
    """Make a value safe inside a markdown table cell."""
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


_ENV = Environment(
    loader=PackageLoader("prep_agent", "templates"),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=True,
    autoescape=False,
)
_ENV.filters["cell"] = _cell


def step_columns(step: Step, target: str) -> str:
    """The columns a step works on, as shown to the user."""
    if isinstance(step, DedupeStep):
        return ", ".join(step.subset) if step.subset else "all columns"
    if isinstance(step, DropMissingTargetStep):
        return target
    if isinstance(step, SplitStep):
        return ""
    return ", ".join(step.columns)


def render_report(
    info: "RunInfo",
    *,
    plan: Plan | None,
    result: ExecutionResult | None,
    plan_issues: list[ValidationIssue],
    trace: AgentTrace | None,
    baseline: BaselineResult | None,
    small_train_rows: int = SMALL_TRAIN_ROWS,
) -> str:
    """Render report.md for one run.

    `plan_issues` is used when there is no `result` (the plan never executed); a result brings
    its own issues. `trace` is None for runs with a hand-written plan. A failed run gets a
    failure section and no baseline, whatever `baseline` holds; for a successful run `baseline`
    None means the baseline was switched off. Scores computed on fewer than `small_train_rows`
    train rows come with a warning.
    """
    failed = info.status == "failed"
    result_issues = result.result_issues if result is not None else []
    if result is not None:
        plan_issues = result.plan_issues
    return _ENV.get_template("report.md.j2").render(
        info=info,
        failed=failed,
        failure_reason=_failure_reason(info, result_issues) if failed else None,
        input_name=info.input_path.rsplit("/", 1)[-1],
        sha=info.input_sha256[:SHA_LENGTH],
        date=info.created_at.strftime("%Y-%m-%d %H:%M UTC"),
        llm=_llm_facts(trace),
        goal=trace.goal if trace is not None else None,
        plan=plan,
        plan_rows=_plan_rows(plan) if plan is not None else [],
        dropped=_dropped_columns(plan) if plan is not None else [],
        qa=trace.qa_history if trace is not None else [],
        has_result=result is not None,
        plan_issues=plan_issues,
        result_issues=result_issues,
        baseline=None if failed else baseline,
        baseline_table=_baseline_table(baseline) if baseline is not None else None,
        small_data=baseline is not None
        and baseline.skipped_reason is None
        and baseline.n_rows < small_train_rows,
    )


def _failure_reason(info: "RunInfo", result_issues: list[ValidationIssue]) -> str:
    if info.error:
        return info.error
    errors = [issue for issue in result_issues if issue.severity == "error"]
    if errors:
        codes = ", ".join(dict.fromkeys(issue.code for issue in errors))
        return f"result validation found {len(errors)} error(s): {codes}"
    return "no reason was recorded"


def _llm_facts(trace: AgentTrace | None) -> dict[str, str]:
    if trace is None:
        return {
            "model": "none (no LLM was used; the plan was written by hand)",
            "tokens": "none",
        }
    if not trace.calls:
        return {"model": "none (no LLM call completed)", "tokens": "none"}
    usage = trace.total_usage
    tokens = "not reported by the provider" if usage is None else format_usage(usage)
    return {"model": trace.model or "none", "tokens": tokens}


def _plan_rows(plan: Plan) -> list[dict[str, str]]:
    return [
        {
            "op": step.op,
            "columns": step_columns(step, plan.target),
            "parameters": _parameters(step),
            "reason": step.reason,
        }
        for step in plan.steps
    ]


def _parameters(step: Step) -> str:
    values = step.model_dump(exclude=_PARAMETERS_SKIPPED, exclude_none=True)
    return ", ".join(f"{name}={_format_value(value)}" for name, value in values.items())


def _format_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _dropped_columns(plan: Plan) -> list[str]:
    return [
        column
        for step in plan.steps
        if isinstance(step, DropColumnsStep)
        for column in step.columns
    ]


def _baseline_table(baseline: BaselineResult) -> dict | None:
    scored = [model for model in baseline.models if model.scores]
    if not scored:
        return None
    rows = [
        {
            "name": model.name,
            "cells": [
                f"{model.scores[metric].mean:.4f} ± {model.scores[metric].std:.4f}"
                for metric in baseline.metrics
            ],
        }
        for model in scored
    ]
    return {"headers": [metric_label(metric) for metric in baseline.metrics], "rows": rows}
