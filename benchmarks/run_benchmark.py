"""Run the benchmark: the agent, the reference plans and the naive plan on the OpenML datasets.

    uv run python benchmarks/download.py
    uv run python benchmarks/run_benchmark.py --models gpt-6-luna --dry-run

Without `--models` only the reference and naive plans run, which calls no model and costs
nothing. With models, the run shows the jobs and the most it can cost, refuses a model that has
no price in `prices.yaml`, and asks before it spends anything (`--yes` skips the question).
`--dry-run` stops after that listing.

    uv run python benchmarks/run_benchmark.py --summarize benchmarks/results/<id>/results.json

`--summarize` writes summary.md again from an existing results.json. It runs no plan, scores
nothing and calls no model, so a summary can be regenerated after the summary code changes.
"""

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated
from unittest import mock

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from bench_cost import (
    DEFAULT_PRICES,
    MAX_QUESTION_ROUNDS,
    MAX_REVISIONS,
    MAX_SCHEMA_RETRIES,
    PriceError,
    PriceTable,
    load_prices,
    max_calls_per_run,
    require_prices,
)
from bench_datasets import (
    BENCHMARKS_DIR,
    DEFAULT_DATASETS_DIR,
    DEFAULT_REFERENCE_DIR,
    DEFAULT_REGISTRY,
    RegistryError,
    load_registry,
    select_datasets,
)
from bench_results import BenchmarkResults, JobResult
from bench_runner import SUMMARY_FILE, BenchmarkError, RunPlan, plan_run, run_benchmark
from bench_summary import render_summary
from prep_agent.env import load_dotenv_file
from prep_agent.llm.base import LLMClient, LLMError
from prep_agent.llm.factory import create_llm_client

DEFAULT_RESULTS_DIR = BENCHMARKS_DIR / "results"
MODEL_VARIABLE = "PREP_AGENT_MODEL"

app = typer.Typer(
    add_completion=False, help="Benchmark the agent against reference and naive plans."
)


def create_client(model: str) -> LLMClient:
    """A client for the configured provider that uses `model`, built through the same factory
    as the CLI so the benchmark needs no provider-specific code."""
    with mock.patch.dict(os.environ, {MODEL_VARIABLE: model}):
        return create_llm_client()


@app.command()
def main(
    models: Annotated[
        str, typer.Option(help="Comma-separated model names; empty runs no agent.")
    ] = "",
    datasets: Annotated[
        str | None, typer.Option(help="Comma-separated dataset names; default: all.")
    ] = None,
    repeats: Annotated[int, typer.Option(min=1, help="Agent runs per dataset and model.")] = 1,
    out_dir: Annotated[
        Path | None,
        typer.Option(
            help="Directory for results.json and summary.md (with --summarize: for summary.md)."
        ),
    ] = None,
    summarize: Annotated[
        Path | None,
        typer.Option(
            help="Write summary.md again from this results.json and run nothing else. It goes "
            "next to the results file unless --out-dir is given."
        ),
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="List the jobs and the cost estimate, run nothing.")
    ] = False,
    yes: Annotated[
        bool, typer.Option("--yes", help="Do not ask before running the agent.")
    ] = False,
    prices: Annotated[Path, typer.Option(help="Model price file.")] = DEFAULT_PRICES,
    registry: Annotated[Path, typer.Option(help="Dataset registry file.")] = DEFAULT_REGISTRY,
    datasets_dir: Annotated[
        Path, typer.Option(help="Directory with the downloaded CSV files.")
    ] = DEFAULT_DATASETS_DIR,
    reference_dir: Annotated[
        Path, typer.Option(help="Directory with the reference plans.")
    ] = DEFAULT_REFERENCE_DIR,
) -> None:
    console = Console()
    if summarize is not None:
        _summarize(console, summarize, out_dir or summarize.parent)
        return
    load_dotenv_file()
    model_list = list(dict.fromkeys(_split(models)))
    try:
        specs = select_datasets(load_registry(registry), _split(datasets) or None)
        table = load_prices(prices) if model_list else PriceTable()
    except (RegistryError, PriceError) as exc:
        _fail(str(exc))

    run_plan = plan_run(specs, model_list, repeats, datasets_dir, table.models)
    _print_run_plan(console, run_plan, table.currency)
    try:
        require_prices(table, model_list, prices)
    except PriceError as exc:
        _fail(str(exc))

    if dry_run:
        console.print(Text("Dry run: no model was called and nothing was written."))
        return

    ceiling = {e.model: e.cost for e in run_plan.estimates}
    if model_list and not yes:
        total = sum(cost for cost in ceiling.values() if cost is not None)
        runs = sum(e.runs for e in run_plan.estimates)
        question = f"Run {runs} agent run(s) costing up to {_money(total)} {table.currency}?"
        if not typer.confirm(question, default=False):
            console.print(Text("Aborted: nothing was run."))
            raise typer.Exit(1)

    out_dir = out_dir or DEFAULT_RESULTS_DIR / datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    try:
        results = run_benchmark(
            specs,
            model_list,
            repeats=repeats,
            datasets_dir=datasets_dir,
            reference_dir=reference_dir,
            out_dir=out_dir,
            prices=table.models,
            currency=table.currency,
            client_factory=create_client,
            estimated_max_cost=ceiling,
            progress=lambda job, index, total: _print_progress(console, job, index, total),
        )
    except (BenchmarkError, LLMError) as exc:
        _fail(str(exc))

    counts = {
        status: sum(j.status == status for j in results.jobs)
        for status in ("ok", "failed", "error")
    }
    console.print(
        Text(f"Jobs: {counts['ok']} ok, {counts['failed']} failed, {counts['error']} error")
    )
    console.print(Text(f"Results: {out_dir / 'results.json'}"), soft_wrap=True)
    console.print(Text(f"Summary: {out_dir / 'summary.md'}"), soft_wrap=True)


def _summarize(console: Console, results_file: Path, out_dir: Path) -> None:
    try:
        results = BenchmarkResults.model_validate_json(results_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _fail(f"cannot read results from {results_file}: {exc}")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_file = out_dir / SUMMARY_FILE
    summary_file.write_text(render_summary(results), encoding="utf-8", newline="\n")
    console.print(Text(f"Summary: {summary_file}"), soft_wrap=True)


def _print_run_plan(console: Console, run_plan: RunPlan, currency: str) -> None:
    grouped: dict[tuple[str, str, str], int] = {}
    for job in run_plan.jobs:
        key = (job.dataset, job.method, job.model or "")
        grouped[key] = grouped.get(key, 0) + 1
    jobs = Table(title=f"{len(run_plan.jobs)} job(s)")
    for column in ("dataset", "method", "model"):
        jobs.add_column(column)
    jobs.add_column("runs", justify="right")
    for (dataset, method, model), runs in grouped.items():
        jobs.add_row(Text(dataset), Text(method), Text(model), Text(str(runs)))
    console.print(jobs)

    if not run_plan.estimates:
        console.print(Text("No agent runs: the reference and naive plans call no model."))
    else:
        estimates = Table(title="Estimated maximum use of the agent runs")
        estimates.add_column("model")
        estimates.add_column("runs", justify="right")
        estimates.add_column("input tokens", justify="right")
        estimates.add_column("output tokens", justify="right")
        estimates.add_column(f"cost ({currency})", justify="right")
        for estimate in run_plan.estimates:
            cost = "no price" if estimate.cost is None else _money(estimate.cost)
            estimates.add_row(
                Text(estimate.model),
                Text(str(estimate.runs)),
                Text(f"{estimate.input_tokens:,}"),
                Text(f"{estimate.output_tokens:,}"),
                Text(cost),
            )
        console.print(estimates)
        console.print(
            Text(
                "The estimate is an upper bound from the size of the prompts and the most calls "
                "the agent can make, not a prediction. One agent run makes at most "
                f"{max_calls_per_run()} LLM calls per run: {MAX_QUESTION_ROUNDS} question "
                f"rounds + {MAX_REVISIONS} revisions + the first plan, each repeated up to "
                f"{MAX_SCHEMA_RETRIES} time(s) when its answer does not match the schema."
            )
        )
    if run_plan.not_downloaded:
        console.print(
            Text(
                "Datasets not downloaded or unreadable, left out of the estimate: "
                f"{', '.join(run_plan.not_downloaded)} (run benchmarks/download.py)"
            )
        )


def _print_progress(console: Console, job: JobResult, index: int, total: int) -> None:
    seconds = (job.plan_seconds or 0) + (job.baseline_seconds or 0)
    detail = f" - {job.error}" if job.error and job.status != "ok" else ""
    console.print(
        Text(
            f"[{index}/{total}] {job.dataset} {job.label} #{job.repeat}: "
            f"{job.status} ({seconds:.0f}s){detail}"
        ),
        soft_wrap=True,
    )


def _split(value: str | None) -> list[str]:
    return [part.strip() for part in (value or "").split(",") if part.strip()]


def _money(value: float) -> str:
    return f"{value:.4f}"


def _fail(message: str):
    Console(stderr=True).print(Text.assemble(("Error: ", "red"), message))
    raise typer.Exit(1)


if __name__ == "__main__":
    app()
