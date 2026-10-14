"""Runs the benchmark: every plan on every dataset, scored the same way.

For each dataset the jobs are the reference plan, the naive plan and one agent run per model
and repeat. All of them end in the same place: the plan is validated and executed with the
project's own executor, and the prepared train split is scored with `evaluate_baselines`, so a
score depends on the plan and nothing else. The agent is run the way `prepare --non-interactive`
runs it, with the dataset's target and the model family "any", which is the family the
reference plans are written for.

A job never takes the benchmark down. A plan that is rejected is recorded as `failed` and
anything unexpected as `error`, both with the reason; the run goes on with the next job.
`results.json` is rewritten after every job, so an interrupted run keeps what it has.
"""

import os
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from pydantic import ValidationError
from sklearn.exceptions import ConvergenceWarning

from bench_cost import (
    MAX_QUESTION_ROUNDS,
    MAX_REVISIONS,
    ModelPrice,
    estimate_agent_run,
    price_of_usage,
    token_cost,
)
from bench_datasets import DatasetSpec, csv_path, reference_plan_path
from bench_naive import naive_plan
from bench_results import AgentMetrics, BenchmarkResults, JobResult, Method, QAEntry, RunConfig
from bench_summary import render_summary
from prep_agent.agent import AgentError, AgentTrace, LLMCallRecord, run_agent
from prep_agent.baseline import BaselineResult, evaluate_baselines
from prep_agent.executor import ExecutionError, ExecutionResult, PlanValidationError, execute
from prep_agent.llm.base import LLMClient
from prep_agent.plan import Plan
from prep_agent.planner import Question
from prep_agent.profiler import ProfilerError, load_csv, profile_dataframe
from prep_agent.validators import ValidationIssue, partition_issues

MODEL_FAMILY = "any"
RESULTS_FILE = "results.json"
SUMMARY_FILE = "summary.md"

ClientFactory = Callable[[str], LLMClient]
Progress = Callable[[JobResult, int, int], None]


class BenchmarkError(RuntimeError):
    """The benchmark cannot start, for a reason the user has to fix."""


class _JobError(Exception):
    """A job cannot run for a reason that is not about the plan; the message says why."""


@dataclass(frozen=True)
class Job:
    dataset: str
    method: Method
    model: str | None
    repeat: int


@dataclass(frozen=True)
class ModelEstimate:
    """The upper bound of what all agent runs of one model can use. `cost` is None when the
    model has no price."""

    model: str
    runs: int
    input_tokens: int
    output_tokens: int
    cost: float | None


@dataclass(frozen=True)
class RunPlan:
    """What a run would do, worked out without calling any model."""

    jobs: list[Job]
    estimates: list[ModelEstimate]
    # Datasets whose file is missing or unreadable; they are not in the estimates.
    not_downloaded: list[str]


def plan_jobs(specs: list[DatasetSpec], models: list[str], repeats: int) -> list[Job]:
    """The jobs in the order they run: dataset by dataset, fixed plans before agent runs."""
    jobs = []
    for spec in specs:
        jobs.append(Job(spec.name, "reference", None, 1))
        jobs.append(Job(spec.name, "naive", None, 1))
        for model in models:
            jobs.extend(Job(spec.name, "agent", model, repeat) for repeat in range(1, repeats + 1))
    return jobs


def plan_run(
    specs: list[DatasetSpec],
    models: list[str],
    repeats: int,
    datasets_dir: Path,
    prices: dict[str, ModelPrice],
) -> RunPlan:
    """The jobs of a run and the most the agent runs can use. Nothing is executed."""
    estimates_per_dataset = []
    not_downloaded = []
    for spec in specs:
        try:
            df = load_csv(csv_path(spec, datasets_dir))
            profile = profile_dataframe(df, target=spec.target)
        except ProfilerError:
            not_downloaded.append(spec.name)
            continue
        estimates_per_dataset.append(
            estimate_agent_run(profile, spec.goal, spec.target, MODEL_FAMILY)
        )

    input_tokens = repeats * sum(e.input_tokens for e in estimates_per_dataset)
    output_tokens = repeats * sum(e.output_tokens for e in estimates_per_dataset)
    estimates = [
        ModelEstimate(
            model=model,
            runs=repeats * len(specs),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=token_cost(input_tokens, output_tokens, prices[model])
            if model in prices
            else None,
        )
        for model in models
    ]
    return RunPlan(plan_jobs(specs, models, repeats), estimates, not_downloaded)


def _cost_or_none(input_tokens: int, output_tokens: int, price: ModelPrice | None) -> float | None:
    return None if price is None else token_cost(input_tokens, output_tokens, price)


def run_benchmark(
    specs: list[DatasetSpec],
    models: list[str],
    *,
    repeats: int,
    datasets_dir: Path,
    reference_dir: Path,
    out_dir: Path,
    prices: dict[str, ModelPrice],
    currency: str,
    client_factory: ClientFactory,
    estimated_max_cost: dict[str, float | None] | None = None,
    progress: Progress | None = None,
) -> BenchmarkResults:
    """Run every job and write results.json and summary.md to `out_dir`.

    Raises `BenchmarkError` before anything runs if `out_dir` already holds results. The
    clients are built first, one per model, so a missing API key stops the run before it starts.
    """
    out_dir = Path(out_dir)
    if (out_dir / RESULTS_FILE).exists():
        raise BenchmarkError(
            f"{out_dir} already holds results; choose another --out-dir or remove them"
        )
    clients = {model: client_factory(model) for model in models}
    out_dir.mkdir(parents=True, exist_ok=True)

    results = BenchmarkResults(
        started_at=datetime.now(UTC),
        config=RunConfig(
            models=list(models),
            repeats=repeats,
            datasets=[spec.name for spec in specs],
            currency=currency,
            prices={model: prices[model] for model in models if model in prices},
            estimated_max_cost=estimated_max_cost or {},
        ),
    )
    jobs = plan_jobs(specs, models, repeats)
    by_name = {spec.name: spec for spec in specs}
    frame_name, frame = None, None
    try:
        for index, job in enumerate(jobs, start=1):
            if job.dataset != frame_name:
                frame_name, frame = job.dataset, _load_frame(by_name[job.dataset], datasets_dir)
            results.jobs.append(
                _run_job(
                    job, by_name[job.dataset], frame, reference_dir, clients, prices.get(job.model)
                )
            )
            _save_results(results, out_dir)
            if progress is not None:
                progress(results.jobs[-1], index, len(jobs))
        results.finished_at = datetime.now(UTC)
    finally:
        _save_results(results, out_dir)
        (out_dir / SUMMARY_FILE).write_text(render_summary(results), encoding="utf-8", newline="\n")
    return results


def _load_frame(spec: DatasetSpec, datasets_dir: Path) -> pd.DataFrame | Exception:
    """The dataset's frame, or the error that made it unavailable (raised per job)."""
    # Raised again for every job of the dataset, where it is recorded.
    try:
        return load_csv(csv_path(spec, datasets_dir))
    except Exception as exc:
        return exc


def _run_job(
    job: Job,
    spec: DatasetSpec,
    frame: pd.DataFrame | Exception,
    reference_dir: Path,
    clients: dict[str, LLMClient],
    price: ModelPrice | None,
) -> JobResult:
    base = {
        "dataset": job.dataset,
        "task_type": spec.task_type,
        "method": job.method,
        "model": job.model,
        "repeat": job.repeat,
    }
    # Whatever a job raises must end up in its record, not stop the benchmark.
    try:
        if isinstance(frame, Exception):
            raise frame
        if job.method == "agent":
            return _agent_job(base, spec, frame, clients[job.model], price)
        return _fixed_plan_job(base, spec, frame, reference_dir)
    except _JobError as exc:
        return JobResult(**base, status="error", error=str(exc))
    except Exception as exc:
        return JobResult(**base, status="error", error=f"{type(exc).__name__}: {exc}")


def _agent_job(
    base: dict, spec: DatasetSpec, df: pd.DataFrame, llm: LLMClient, price: ModelPrice | None
) -> JobResult:
    started = time.perf_counter()
    try:
        result = run_agent(
            df,
            spec.goal,
            llm,
            _answer_recommended,
            target=spec.target,
            model_family=MODEL_FAMILY,
            max_question_rounds=MAX_QUESTION_ROUNDS,
            max_revisions=MAX_REVISIONS,
        )
    except AgentError as exc:
        errors, warnings = partition_issues(exc.issues)
        return JobResult(
            **base,
            status="failed",
            error=str(exc),
            error_codes=_codes(errors),
            warning_codes=_codes(warnings),
            agent=_agent_metrics(exc.trace, errors, price),
            plan=exc.plan,
            plan_seconds=time.perf_counter() - started,
        )
    seconds = time.perf_counter() - started
    _, warnings = partition_issues(result.issues)
    return _scored(
        base,
        result.plan,
        result.execution,
        warnings,
        plan_seconds=seconds,
        agent=_agent_metrics(result.trace, [], price),
    )


def _fixed_plan_job(
    base: dict, spec: DatasetSpec, df: pd.DataFrame, reference_dir: Path
) -> JobResult:
    started = time.perf_counter()
    profile = profile_dataframe(df, target=spec.target)
    if base["method"] == "naive":
        plan = naive_plan(profile, spec.task_type)
    else:
        plan = _load_reference(spec, reference_dir)
    try:
        execution = execute(df, plan, profile)
    except PlanValidationError as exc:
        errors, warnings = partition_issues(exc.issues)
        return JobResult(
            **base,
            status="failed",
            error=str(exc),
            error_codes=_codes(errors),
            warning_codes=_codes(warnings),
            plan=plan,
            plan_seconds=time.perf_counter() - started,
        )
    except ExecutionError as exc:
        return JobResult(
            **base,
            status="failed",
            error=str(exc),
            plan=plan,
            plan_seconds=time.perf_counter() - started,
        )
    seconds = time.perf_counter() - started
    _, warnings = partition_issues(execution.plan_issues)
    return _scored(base, plan, execution, warnings, plan_seconds=seconds, agent=None)


def _scored(
    base: dict,
    plan: Plan,
    execution: ExecutionResult,
    warnings: list[ValidationIssue],
    *,
    plan_seconds: float,
    agent: AgentMetrics | None,
) -> JobResult:
    """Score an executed plan. The result of a plan that passed the validators can still hold
    errors (the executor reports them), which makes the job `failed`."""
    result_errors, result_warnings = partition_issues(execution.result_issues)
    common = {
        "plan": plan,
        "agent": agent,
        "plan_seconds": plan_seconds,
        "warning_codes": _codes(warnings + result_warnings),
        "n_train": len(execution.train),
        "n_test": len(execution.test),
        "n_features": execution.train.shape[1] - 1,
    }
    if result_errors:
        return JobResult(
            **base,
            **common,
            status="failed",
            error="the result failed validation",
            error_codes=_codes(result_errors),
        )
    started = time.perf_counter()
    try:
        baseline = _evaluate(plan, execution)
    except Exception as exc:
        return JobResult(
            **base,
            **common,
            status="error",
            error=f"scoring failed: {type(exc).__name__}: {exc}",
        )
    return JobResult(
        **base,
        **common,
        status="ok",
        baseline=baseline,
        baseline_seconds=time.perf_counter() - started,
    )


def _evaluate(plan: Plan, execution: ExecutionResult) -> BaselineResult:
    # A plan that does not scale makes the baseline's logistic regression stop before it
    # converges. That is a fair part of what the score measures, and the warning would only
    # bury the progress lines.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return evaluate_baselines(
            execution.split_train, execution.pipeline, plan.target, plan.task_type
        )


def _load_reference(spec: DatasetSpec, reference_dir: Path) -> Plan:
    path = reference_plan_path(spec, reference_dir)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise _JobError(f"reference plan not found: {path}") from None
    except (OSError, UnicodeDecodeError) as exc:
        raise _JobError(f"could not read the reference plan {path}: {exc}") from exc
    try:
        return Plan.model_validate_json(text)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in err['loc']) or '<root>'}: {err['msg']}"
            for err in exc.errors()
        )
        raise _JobError(f"invalid reference plan {path}: {problems}") from None


def _answer_recommended(questions: list[Question]) -> dict[str, str]:
    return {question.id: question.recommended for question in questions}


def _agent_metrics(
    trace: AgentTrace, final_errors: list[ValidationIssue], price: ModelPrice | None
) -> AgentMetrics:
    usage = trace.total_usage
    return AgentMetrics(
        reported_model=trace.model,
        calls=len(trace.calls),
        revisions=sum(call.purpose == "revise" for call in trace.calls),
        questions=len(trace.qa_history),
        first_plan_error_codes=_first_plan_errors(trace.calls, final_errors),
        input_tokens=usage.input_tokens if usage else None,
        output_tokens=usage.output_tokens if usage else None,
        cost=price_of_usage(usage, price) if price is not None else None,
        qa=[QAEntry(question=r.question.question, answer=r.answer) for r in trace.qa_history],
    )


def _first_plan_errors(
    calls: list[LLMCallRecord], final_errors: list[ValidationIssue]
) -> list[str]:
    """The error codes the first plan was sent back with.

    The trace keeps the codes on the call that was made to fix them, so they are read from
    the call after the first plan. A first plan with no call after it was either accepted (no
    errors) or the plan the agent gave up on, whose errors are the final ones.
    """
    for position, call in enumerate(calls):
        if call.response_kind != "plan":
            continue
        if position + 1 < len(calls):
            return list(calls[position + 1].issue_codes)
        return _codes(final_errors)
    return []


def _codes(issues: list[ValidationIssue]) -> list[str]:
    return [issue.code for issue in issues]


def _save_results(results: BenchmarkResults, out_dir: Path) -> None:
    """Write results.json through a temporary file so a crash never leaves half a file."""
    path = out_dir / RESULTS_FILE
    partial = path.with_name(path.name + ".part")
    partial.write_text(results.model_dump_json(indent=2), encoding="utf-8", newline="\n")
    os.replace(partial, path)
