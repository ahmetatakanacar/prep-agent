"""Runs the agent on the planted datasets and scores every plan against the ground truth.

The runs are the product of datasets, variants (anonymous, named), answer modes, models and
repeats. They run repeat by repeat, so an interrupted run holds the same number of repeats for
everything it covers. Each run is validated, executed and scored the way the main benchmark does
it: the agent is given the dataset's target and the model family "any", and its plan is judged
by `planted_scorer.score_plan`. A run never takes the benchmark down: a plan the agent could not
make valid is recorded as `failed` and anything unexpected as `error`, both with the reason.
`results.json` is rewritten after every run, so an interrupted run keeps what it has.
"""

import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from bench_cost import (
    MAX_QUESTION_ROUNDS,
    MAX_REVISIONS,
    MAX_SCHEMA_RETRIES,
    ModelPrice,
    estimate_agent_run,
    max_cost,
    price_of_usage,
)
from bench_runner import ModelEstimate
from planted_generate import ALL_RECIPES, DATA_DIR, RECIPES, file_stem, load
from planted_modes import MODES, Answerer, question_rounds
from planted_recipes_d import ALL_D, TEST
from planted_results import AgentRecord, MarkedColumn, PlantedResults, RunConfig, RunResult
from planted_scorer import score_plan
from planted_summary import render_summary
from planted_truth import VARIANTS, DatasetTruth, Variant
from prep_agent.agent import AgentError, AgentTrace, run_agent
from prep_agent.llm.base import LLMClient
from prep_agent.profiler import profile_dataframe
from prep_agent.validators import partition_issues

MODEL_FAMILY = "any"
RESULTS_FILE = "results.json"
SUMMARY_FILE = "summary.md"
# The default run uses the first set; the second one is named explicitly.
DATASETS: tuple[str, ...] = tuple(recipe.name for recipe in RECIPES)
KNOWN_DATASETS: tuple[str, ...] = tuple(r.name for r in ALL_RECIPES) + tuple(r.name for r in ALL_D)
# Set D's test recipes are run once, at the end. They are refused unless the run says it is that
# run, so that a routine run cannot use them up.
TEST_DATASETS: tuple[str, ...] = tuple(recipe.name for recipe in TEST)

ClientFactory = Callable[[str], LLMClient]
Progress = Callable[[RunResult, int, int], None]


class PlantedRunError(RuntimeError):
    """The run cannot start, for a reason the user has to fix."""


@dataclass(frozen=True)
class Job:
    dataset: str
    variant: Variant
    mode: str
    model: str
    repeat: int


@dataclass(frozen=True)
class RunPlan:
    """What a run would do, worked out without calling any model."""

    jobs: list[Job]
    estimates: list[ModelEstimate]


def goal_for(truth: DatasetTruth) -> str:
    # The same sentence for both variants, apart from the target's name.
    return f"Prepare this table for predicting the column {truth.target}."


def check_selection(
    datasets: list[str], variants: list[str], modes: list[str], final_test: bool = False
) -> None:
    held_out = [name for name in datasets if name in TEST_DATASETS]
    if held_out and not final_test:
        raise PlantedRunError(
            f"{', '.join(held_out)} is held-out test data and is run once, at the end: "
            "pass --final-test for that run"
        )
    for kind, chosen, known in (
        ("dataset", datasets, KNOWN_DATASETS),
        ("variant", variants, VARIANTS),
        ("mode", modes, MODES),
    ):
        unknown = [name for name in chosen if name not in known]
        if unknown:
            raise PlantedRunError(
                f"unknown {kind} {', '.join(map(repr, unknown))}; known: {', '.join(known)}"
            )
        if not chosen:
            raise PlantedRunError(f"no {kind} selected")


def check_data(datasets: list[str], variants: list[str], data_dir: Path) -> None:
    """Refuse a run whose data files are missing, before any model is called."""
    missing = [
        f"{file_stem(name, variant)}.csv"  # type: ignore[arg-type]
        for name in datasets
        for variant in variants
        if not (data_dir / f"{file_stem(name, variant)}.csv").is_file()  # type: ignore[arg-type]
    ]
    if missing:
        raise PlantedRunError(
            f"data files not found in {data_dir}: {', '.join(missing[:4])}"
            f"{' ...' if len(missing) > 4 else ''}; generate them with planted_generate_d.py"
        )


def plan_jobs(
    datasets: list[str], variants: list[str], modes: list[str], models: list[str], repeats: int
) -> list[Job]:
    """The jobs in the order they run: repeat by repeat, then model, mode, variant, dataset."""
    return [
        Job(dataset, variant, mode, model, repeat)  # type: ignore[arg-type]
        for repeat in range(1, repeats + 1)
        for model in models
        for mode in modes
        for variant in variants
        for dataset in datasets
    ]


def plan_run(
    datasets: list[str],
    variants: list[str],
    modes: list[str],
    models: list[str],
    repeats: int,
    prices: dict[str, ModelPrice],
    data_dir: Path = DATA_DIR,
    final_test: bool = False,
) -> RunPlan:
    """The jobs of a run and the most the agent runs can use. Nothing is executed.

    The bound counts every run with the benchmark's full limits, `no_questions` runs included,
    which cannot use all of them; it is an upper bound, not a prediction.
    """
    check_selection(datasets, variants, modes, final_test)
    check_data(datasets, variants, data_dir)
    per_pass_input = per_pass_output = 0
    for dataset in datasets:
        for variant in variants:
            frame, truth = load(dataset, variant, data_dir)  # type: ignore[arg-type]
            profile = profile_dataframe(frame, target=truth.target)
            estimate = estimate_agent_run(profile, goal_for(truth), truth.target, MODEL_FAMILY)
            per_pass_input += estimate.input_tokens
            per_pass_output += estimate.output_tokens
    passes = repeats * len(modes)
    runs = passes * len(datasets) * len(variants)
    estimates = [
        ModelEstimate(
            model=model,
            runs=runs,
            input_tokens=passes * per_pass_input,
            output_tokens=passes * per_pass_output,
            cost=(
                max_cost(passes * per_pass_input, passes * per_pass_output, prices[model])
                if model in prices
                else None
            ),
        )
        for model in models
    ]
    return RunPlan(plan_jobs(datasets, variants, modes, models, repeats), estimates)


def run_planted(
    models: list[str],
    *,
    datasets: list[str],
    variants: list[str],
    modes: list[str],
    repeats: int,
    out_dir: Path,
    prices: dict[str, ModelPrice],
    currency: str,
    client_factory: ClientFactory,
    data_dir: Path = DATA_DIR,
    estimated_max_cost: dict[str, float | None] | None = None,
    progress: Progress | None = None,
    final_test: bool = False,
) -> PlantedResults:
    """Run every job and write results.json and summary.md to `out_dir`.

    Raises `PlantedRunError` before anything runs if `out_dir` already holds results or the
    selection is wrong. The clients are built first, one per model, so a missing API key stops
    the run before it starts.
    """
    check_selection(datasets, variants, modes, final_test)
    check_data(datasets, variants, data_dir)
    out_dir = Path(out_dir)
    if (out_dir / RESULTS_FILE).exists():
        raise PlantedRunError(
            f"{out_dir} already holds results; choose another --out-dir or remove them"
        )
    clients = {model: client_factory(model) for model in models}
    out_dir.mkdir(parents=True, exist_ok=True)

    results = PlantedResults(
        started_at=datetime.now(UTC),
        config=RunConfig(
            models=list(models),
            variants=list(variants),
            modes=list(modes),
            repeats=repeats,
            datasets=list(datasets),
            currency=currency,
            prices={model: prices[model] for model in models if model in prices},
            estimated_max_cost=estimated_max_cost or {},
        ),
    )
    jobs = plan_jobs(datasets, variants, modes, models, repeats)
    frames: dict[tuple[str, str], tuple[pd.DataFrame, DatasetTruth]] = {}
    try:
        for index, job in enumerate(jobs, start=1):
            key = (job.dataset, job.variant)
            if key not in frames:
                frames[key] = load(job.dataset, job.variant, data_dir)
            frame, truth = frames[key]
            results.runs.append(
                _run_job(job, frame, truth, clients[job.model], prices.get(job.model))
            )
            _save_results(results, out_dir)
            if progress is not None:
                progress(results.runs[-1], index, len(jobs))
        results.finished_at = datetime.now(UTC)
    finally:
        _save_results(results, out_dir)
        (out_dir / SUMMARY_FILE).write_text(render_summary(results), encoding="utf-8", newline="\n")
    return results


def _run_job(
    job: Job,
    frame: pd.DataFrame,
    truth: DatasetTruth,
    llm: LLMClient,
    price: ModelPrice | None,
) -> RunResult:
    base = {
        "dataset": job.dataset,
        "variant": job.variant,
        "mode": job.mode,
        "model": job.model,
        "repeat": job.repeat,
        "task_type": truth.task_type,
    }
    answerer = Answerer(job.mode, truth)
    started = time.perf_counter()
    # Whatever a job raises must end up in its record, not stop the run.
    try:
        result = run_agent(
            frame,
            goal_for(truth),
            llm,
            answerer,
            target=truth.target,
            model_family=MODEL_FAMILY,
            max_question_rounds=question_rounds(job.mode, MAX_QUESTION_ROUNDS),
            max_revisions=MAX_REVISIONS,
            max_schema_retries=MAX_SCHEMA_RETRIES,
        )
    except AgentError as exc:
        errors, warnings = partition_issues(exc.issues)
        return RunResult(
            **base,
            status="failed",
            error=str(exc),
            error_codes=[issue.code for issue in errors],
            warning_codes=[issue.code for issue in warnings],
            agent=_agent_record(exc.trace, answerer, price),
            marked=_marked(exc.profile, answerer, truth),
            plan=exc.plan,
            plan_seconds=time.perf_counter() - started,
        )
    except Exception as exc:
        return RunResult(
            **base,
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            plan_seconds=time.perf_counter() - started,
        )
    _, warnings = partition_issues(result.issues)
    return RunResult(
        **base,
        status="ok",
        warning_codes=[issue.code for issue in warnings],
        agent=_agent_record(result.trace, answerer, price),
        marked=_marked(result.profile, answerer, truth),
        plan=result.plan,
        score=score_plan(result.plan, truth),
        plan_seconds=time.perf_counter() - started,
    )


SUSPECTED_FROM = 0.9
HIGH_FROM = 0.6


def _marked(profile, answerer: Answerer, truth: DatasetTruth) -> list[MarkedColumn]:
    """The columns the profile related strongly to the target, with whether any question named
    them. The tiers are fixed here, so that runs of different code are measured the same way."""
    asked_text = " ".join(" ".join([q.question, *q.options]) for q in answerer.log)
    roles = {c.name: c for c in truth.columns}
    marked = []
    for column in profile.columns:
        score = max(column.target_association or 0.0, column.missing_association or 0.0)
        truth_column = roles.get(column.name)
        if score < HIGH_FROM or truth_column is None:
            continue
        kind = (
            "leak"
            if truth_column.role == "must_drop"
            else "legit"
            if truth_column.fault == "legit_strong"
            else "other"
        )
        marked.append(
            MarkedColumn(
                name=column.name,
                score=score,
                tier="suspected" if score >= SUSPECTED_FROM else "high",
                asked=_mentions(column.name, asked_text),
                truth=kind,
            )
        )
    return marked


def _mentions(name: str, text: str) -> bool:
    return re.search(rf"(?<![\w]){re.escape(name)}(?![\w])", text) is not None


def _agent_record(trace: AgentTrace, answerer: Answerer, price: ModelPrice | None) -> AgentRecord:
    usage = trace.total_usage
    return AgentRecord(
        reported_model=trace.model,
        calls=len(trace.calls),
        revisions=sum(call.purpose == "revise" for call in trace.calls),
        schema_retries=sum(call.purpose == "schema_retry" for call in trace.calls),
        questions=len(trace.qa_history),
        forced_answers=answerer.forced,
        input_tokens=usage.input_tokens if usage else None,
        output_tokens=usage.output_tokens if usage else None,
        cached_input_tokens=usage.cached_input_tokens if usage else None,
        cache_write_tokens=usage.cache_write_tokens if usage else None,
        cost=price_of_usage(usage, price) if price is not None else None,
        qa=list(answerer.log),
    )


def _save_results(results: PlantedResults, out_dir: Path) -> None:
    """Write results.json through a temporary file so a crash never leaves half a file."""
    path = out_dir / RESULTS_FILE
    partial = path.with_name(path.name + ".part")
    partial.write_text(results.model_dump_json(indent=2), encoding="utf-8", newline="\n")
    os.replace(partial, path)
