"""The result payload of a finished run: the artifacts of its run folder plus what the UI needs to
tell the story of it.

Two things are derived here and are not in the run folder. The trace keeps only the codes of the
errors a rejected plan was sent back with, so the validator is run again on every rejected plan
to get the full findings (step, column, message); the validators are deterministic, so this
repeats what the agent saw. And the dropped columns are classified from the profile's warnings
and the wording of the reasons, because the plan itself does not say why a column was dropped.
"""

import json
import re
from dataclasses import asdict
from pathlib import Path

from app.sessions import Outcome, Session
from prep_agent.agent import AgentTrace, LLMCallRecord
from prep_agent.baseline import (
    HIGHER_IS_BETTER,
    NOT_CONVERGED_NOTE,
    BaselineResult,
    metric_label,
)
from prep_agent.ops import get_op
from prep_agent.plan import (
    DedupeStep,
    DropColumnsStep,
    DropMissingTargetStep,
    Plan,
    SplitStep,
    Step,
)
from prep_agent.profiler import DatasetProfile, profile_dataframe
from prep_agent.validators import ValidationIssue, partition_issues, validate_plan

RESULT_CODES = frozenset(
    {
        "empty_split",
        "column_mismatch",
        "target_altered",
        "train_test_overlap",
        "nan_in_output",
        "non_numeric_output",
        "possible_target_leakage",
        "class_distribution_shift",
    }
)
# What the metrics can reach, so the chart can draw a sensible axis: [lowest, highest], None
# for open ends.
METRIC_DOMAIN = {
    "accuracy": (0.0, 1.0),
    "f1_macro": (0.0, 1.0),
    "roc_auc": (0.0, 1.0),
    "r2": (None, 1.0),
    "rmse": (0.0, None),
    "mae": (0.0, None),
}
LEAKAGE_WORDS = re.compile(
    r"leak|derived from the target|after the outcome|at prediction|future"
    r"|gives? (?:the )?target away",
    re.IGNORECASE,
)
# Types no op can turn into model input, so a plan can only drop such a column.
UNSUPPORTED_TYPES = frozenset({"datetime", "text"})
DOWNLOADABLE_SUFFIXES = {".csv", ".json", ".md", ".py", ".joblib"}


def build_result(session: Session) -> dict:
    outcome = session.outcome
    assert outcome is not None and session.run_dir is not None and session.request is not None
    run_dir = session.run_dir
    plan = Plan.model_validate(outcome.plan) if outcome.plan is not None else None
    failed = session.status == "failed"
    return {
        "status": "failed" if failed else "ok",
        "error": session.error,
        "goal": session.request.goal,
        "run": _read_json(run_dir / "run.json"),
        "plan": None if plan is None else _plan_view(plan),
        "issues": [_issue_view(issue) for issue in outcome.issues],
        "revisions": _revisions(session, outcome),
        "dropped": [] if plan is None else _dropped(plan, outcome.profile, outcome.trace),
        "qa": [record.model_dump(mode="json") for record in outcome.trace.qa_history],
        "baseline": _baseline_view(outcome.baseline),
        "baseline_error": session.baseline_error,
        "llm": _llm_view(outcome.trace),
        "report_md": _read_text(run_dir / "report.md"),
        "files": _files(run_dir),
    }


# --- plan ---


def _plan_view(plan: Plan) -> dict:
    return {
        "target": plan.target,
        "task_type": plan.task_type,
        "model_family": plan.model_family,
        "steps": [_step_view(index, step, plan.target) for index, step in enumerate(plan.steps)],
    }


def _step_view(index: int, step: Step, target: str) -> dict:
    return {
        "index": index,
        "op": step.op,
        "kind": get_op(step.op).kind,
        "columns": _step_columns(step, target),
        "all_columns": isinstance(step, DedupeStep) and step.subset is None,
        "parameters": step.model_dump(
            mode="json", exclude={"op", "reason", "columns", "subset"}, exclude_none=True
        ),
        "reason": step.reason,
    }


def _step_columns(step: Step, target: str) -> list[str]:
    if isinstance(step, DedupeStep):
        return step.subset or []
    if isinstance(step, DropMissingTargetStep):
        return [target]
    if isinstance(step, SplitStep):
        return []
    return list(step.columns)


def _issue_view(issue: ValidationIssue | dict) -> dict:
    data = issue.model_dump(mode="json") if isinstance(issue, ValidationIssue) else dict(issue)
    data["stage"] = "result" if data["code"] in RESULT_CODES else "plan"
    return data


# --- self-correction ---


def _revisions(session: Session, outcome: Outcome) -> list[dict]:
    """Every plan that was sent back, with the findings it was sent back with and the plan that
    came next."""
    request = session.request
    assert request is not None
    profiles: dict[str | None, DatasetProfile] = {}

    def profile_for(plan: Plan) -> DatasetProfile:
        # The agent checks a plan against a profile that carries the target; without a target
        # from the user, that is the plan's own.
        if request.target is not None:
            key: str | None = request.target
        elif plan.target in {c.name for c in session.profile.columns}:
            key = plan.target
        else:
            return session.profile
        if key not in profiles:
            profiles[key] = profile_dataframe(session.df, target=key)
        return profiles[key]

    revisions = []
    previous: Plan | None = None
    previous_index = -1
    for call in outcome.trace.calls:
        if call.response_kind != "plan" or call.response is None or call.response.plan is None:
            continue
        plan = call.response.plan
        if previous is not None and call.issue_codes:
            findings = validate_plan(previous, profile_for(previous), request.model_family)
            errors, warnings = partition_issues(findings)
            revisions.append(
                {
                    "from_call": previous_index,
                    "to_call": call.index,
                    "rejected_issues": [_issue_view(i) for i in errors],
                    "warnings": [_issue_view(i) for i in warnings],
                    # Errors the plan was sent back with that a plan-only check cannot repeat
                    # (they came from running it).
                    "other_codes": [
                        code
                        for code in call.issue_codes
                        if code not in {issue.code for issue in errors}
                    ],
                    **_diff(previous, plan, errors),
                }
            )
        previous, previous_index = plan, call.index
    return revisions


def _step_key(step: Step) -> str:
    return json.dumps(
        step.model_dump(mode="json", exclude={"reason"}, exclude_none=True), sort_keys=True
    )


def _side(plan: Plan, index: int) -> str:
    kinds = [get_op(step.op).kind for step in plan.steps]
    if "split" not in kinds:
        return "before"
    first = kinds.index("split")
    return "split" if index == first else ("before" if index < first else "after")


def _overlap(a: Step, b: Step) -> int:
    return len(set(_step_columns(a, "")) & set(_step_columns(b, "")))


def _diff(rejected: Plan, corrected: Plan, errors: list[ValidationIssue]) -> dict:
    """Pair the steps of the rejected plan with those of the corrected one.

    A step that is the same apart from its reason is kept, or moved when it changed sides of
    the split. A step of the same op that shares columns with a left-over one has changed. What
    stays unpaired was removed, or is new.
    """
    free_corrected = list(range(len(corrected.steps)))
    pairs: dict[int, tuple[int, str]] = {}
    for r_index, r_step in enumerate(rejected.steps):
        for c_index in free_corrected:
            if _step_key(r_step) == _step_key(corrected.steps[c_index]):
                same_side = _side(rejected, r_index) == _side(corrected, c_index)
                pairs[r_index] = (c_index, "kept" if same_side else "moved")
                free_corrected.remove(c_index)
                break
    for r_index, r_step in enumerate(rejected.steps):
        if r_index in pairs:
            continue
        candidates = [
            c_index for c_index in free_corrected if corrected.steps[c_index].op == r_step.op
        ]
        if not candidates:
            continue
        best = max(candidates, key=lambda c: (_overlap(r_step, corrected.steps[c]), -c))
        has_columns = bool(_step_columns(r_step, ""))
        if _overlap(r_step, corrected.steps[best]) > 0 or not has_columns:
            pairs[r_index] = (best, "changed")
            free_corrected.remove(best)

    by_corrected = {c: (r, fate) for r, (c, fate) in pairs.items()}
    target = rejected.target
    rejected_steps = []
    for r_index, r_step in enumerate(rejected.steps):
        view = _step_view(r_index, r_step, target)
        view["issues"] = [_issue_view(i) for i in errors if i.step_index == r_index]
        fate = pairs.get(r_index)
        view["fate"] = fate[1] if fate else "removed"
        view["to_index"] = fate[0] if fate else None
        rejected_steps.append(view)
    corrected_steps = []
    for c_index, c_step in enumerate(corrected.steps):
        view = _step_view(c_index, c_step, corrected.target)
        origin = by_corrected.get(c_index)
        view["origin"] = origin[1] if origin else "new"
        view["from_index"] = origin[0] if origin else None
        corrected_steps.append(view)
    return {
        "rejected_steps": rejected_steps,
        "corrected_steps": corrected_steps,
        "plan_level_issues": [_issue_view(i) for i in errors if i.step_index is None],
    }


# --- dropped columns ---


def _dropped(plan: Plan, profile: DatasetProfile, trace: AgentTrace) -> list[dict]:
    """The drop steps with their columns. A step carries one reason for all its columns, so the
    columns are grouped under it instead of repeating the reason on each."""
    profiled = {column.name: column for column in profile.columns}
    groups = []
    for index, step in enumerate(plan.steps):
        if not isinstance(step, DropColumnsStep):
            continue
        columns = []
        for name in step.columns:
            column = profiled.get(name)
            asked = [
                record.model_dump(mode="json")
                for record in trace.qa_history
                if _mentions(name, record.question.question)
                or _mentions(name, record.question.reason)
            ]
            claimed = claims_leakage(name, step.columns, step.reason) or any(
                _has_leakage_words(
                    f"{qa['question']['question']} {qa['question']['reason']} {qa['answer']}"
                )
                for qa in asked
            )
            columns.append(
                {
                    "column": name,
                    "kinds": _drop_kinds(
                        column.warnings if column else [],
                        claimed,
                        column.inferred_type if column else None,
                        bool(asked),
                    ),
                    "inferred_type": column.inferred_type if column else None,
                    "missing_ratio": column.missing_ratio if column else None,
                    "n_unique": column.n_unique if column else None,
                    "asked": asked,
                }
            )
        columns.sort(key=_drop_rank)
        groups.append({"step_index": index, "reason": step.reason, "columns": columns})
    groups.sort(key=lambda group: min((_drop_rank(c) for c in group["columns"]), default=3))
    return groups


def _drop_rank(column: dict) -> int:
    priority = {"leakage": 0, "pii": 1}
    return min((priority.get(kind, 2) for kind in column["kinds"]), default=3)


def _mentions(column: str, text: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(column)}(?!\w)", text) is not None


def _has_leakage_words(text: str) -> bool:
    return LEAKAGE_WORDS.search(text) is not None


def claims_leakage(column: str, step_columns: list[str], reason: str) -> bool:
    """Whether `reason`, which one drop step gives for all its columns, says `column` leaks.

    The reason is read from each mention of the column to the next mention of another column
    of the step, or to the end of the sentence, so a leakage word about one column is not
    held against its neighbours. A column the reason does not name is only charged when it is
    the step's only column.
    """
    mentions = sorted(
        (m.start(), m.group(0))
        for other in step_columns
        for m in re.finditer(rf"(?<!\w){re.escape(other)}(?!\w)", reason)
    )
    own = [start for start, name in mentions if name == column]
    if not own:
        return len(step_columns) == 1 and _has_leakage_words(reason)
    for start in own:
        end = len(reason)
        for other_start, name in mentions:
            if other_start > start and name != column:
                end = min(end, other_start)
                break
        sentence_end = re.search(r"[.;]", reason[start:])
        if sentence_end:
            end = min(end, start + sentence_end.start())
        # The column's own name may contain the word (a column called `leaky`).
        if _has_leakage_words(reason[start:end].replace(column, " ")):
            return True
    return False


def _drop_kinds(
    warnings: list[str], claimed: bool, inferred_type: str | None, asked: bool
) -> list[str]:
    kinds = []
    if "possible_pii" in warnings:
        kinds.append("pii")
    # The profile cannot see leakage; it is whatever the agent or the question said it was.
    if claimed:
        kinds.append("leakage")
    if "possible_id" in warnings:
        kinds.append("identifier")
    if "constant" in warnings:
        kinds.append("constant")
    if "high_missing" in warnings:
        kinds.append("high_missing")
    if inferred_type in UNSUPPORTED_TYPES:
        kinds.append("unsupported_type")
    if asked:
        kinds.append("asked")
    return kinds


# --- baseline and usage ---


def _baseline_view(baseline: BaselineResult | None) -> dict | None:
    if baseline is None:
        return None
    return {
        "task_type": baseline.task_type,
        "n_rows": baseline.n_rows,
        "n_folds": baseline.n_folds,
        "random_state": baseline.random_state,
        "skipped_reason": baseline.skipped_reason,
        "metrics": [
            {
                "name": metric,
                "label": metric_label(metric),
                "higher_is_better": HIGHER_IS_BETTER[metric],
                "domain": METRIC_DOMAIN[metric],
            }
            for metric in baseline.metrics
        ],
        "models": [
            {
                "name": model.name,
                "is_reference": model.name.startswith("dummy"),
                "scores": {name: score.model_dump() for name, score in model.scores.items()},
                "skipped_reason": model.skipped_reason,
                "note": model.note,
                "not_converged": model.note == NOT_CONVERGED_NOTE,
            }
            for model in baseline.models
        ],
    }


def _llm_view(trace: AgentTrace) -> dict:
    usage = trace.total_usage
    return {
        "model": trace.model,
        "usage": None if usage is None else asdict(usage),
        "calls": [_call_view(call) for call in trace.calls],
    }


def _call_view(call: LLMCallRecord) -> dict:
    return {
        "index": call.index,
        "purpose": call.purpose,
        "response_kind": call.response_kind,
        "issue_codes": call.issue_codes,
        "warning_codes": call.warning_codes,
        "usage": None if call.usage is None else asdict(call.usage),
    }


# --- files ---


def _files(run_dir: Path) -> list[dict]:
    return [
        {"name": path.name, "size": path.stat().st_size}
        for path in sorted(run_dir.iterdir())
        if path.is_file() and path.suffix in DOWNLOADABLE_SUFFIXES
    ]


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _read_json(path: Path) -> dict | None:
    text = _read_text(path)
    return None if text is None else json.loads(text)
