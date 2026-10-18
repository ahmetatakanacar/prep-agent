"""Run the agent on the planted-fault datasets and score its plans against the ground truth.

    uv run python benchmarks/run_planted.py --models gpt-6-luna --dry-run

The run is the product of datasets, variants (anonymous, named), answer modes (recommended,
not_recommended, no_questions), models and repeats. It shows the jobs and the most they can
cost, refuses a model that has no price in `benchmarks/prices.yaml`, and asks before it spends
anything (`--yes` skips the question). `--dry-run` stops after that listing.

    uv run python benchmarks/run_planted.py --summarize benchmarks/results/<id>/results.json

`--summarize` writes summary.md again from an existing results.json and runs nothing.
"""

import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

# The planted modules import each other as siblings, like the bench_* modules in this directory.
sys.path.insert(0, str(Path(__file__).resolve().parent / "planted"))

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
from planted_generate import DATA_DIR
from planted_modes import MODES
from planted_results import PlantedResults, RunResult
from planted_runner import (
    DATASETS,
    SUMMARY_FILE,
    PlantedRunError,
    RunPlan,
    plan_run,
    run_planted,
)
from planted_summary import render_summary
from planted_truth import VARIANTS
from prep_agent.env import load_dotenv_file
from prep_agent.llm.base import LLMError
from run_benchmark import DEFAULT_RESULTS_DIR, create_client

app = typer.Typer(add_completion=False, help="Evaluate the agent on planted-fault datasets.")


@app.command()
def main(
    models: Annotated[str, typer.Option(help="Comma-separated model names.")] = "",
    variants: Annotated[
        str, typer.Option(help=f"Comma-separated variants; default: {','.join(VARIANTS)}.")
    ] = "",
    modes: Annotated[
        str, typer.Option(help=f"Comma-separated answer modes; default: {','.join(MODES)}.")
    ] = "",
    datasets: Annotated[
        str, typer.Option(help="Comma-separated dataset names; default: all.")
    ] = "",
    repeats: Annotated[int, typer.Option(min=1, help="Runs per dataset, variant and mode.")] = 1,
    out_dir: Annotated[
        Path | None,
        typer.Option(help="Directory for results.json and summary.md (with --summarize: summary)."),
    ] = None,
    summarize: Annotated[
        Path | None,
        typer.Option(help="Write summary.md again from this results.json and run nothing else."),
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="List the jobs and the cost estimate, run nothing.")
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", help="Do not ask before running.")] = False,
    prices: Annotated[Path, typer.Option(help="Model price file.")] = DEFAULT_PRICES,
    data_dir: Annotated[Path, typer.Option(help="Directory with the planted data.")] = DATA_DIR,
) -> None:
    console = Console()
    if summarize is not None:
        _summarize(console, summarize, out_dir or summarize.parent)
        return
    load_dotenv_file()
    model_list = list(dict.fromkeys(_split(models)))
    if not model_list:
        _fail("give at least one model with --models")
    dataset_list = _split(datasets) or list(DATASETS)
    variant_list = _split(variants) or list(VARIANTS)
    mode_list = _split(modes) or list(MODES)
    try:
        table = load_prices(prices)
        run_plan = plan_run(
            dataset_list, variant_list, mode_list, model_list, repeats, table.models, data_dir
        )
        require_prices(table, model_list, prices)
    except (PriceError, PlantedRunError) as exc:
        _fail(str(exc))
    _print_run_plan(console, run_plan, table, mode_list)

    if dry_run:
        console.print(Text("Dry run: no model was called and nothing was written."))
        return

    ceiling = {e.model: e.cost for e in run_plan.estimates}
    if not yes:
        total = sum(cost for cost in ceiling.values() if cost is not None)
        question = (
            f"Run {len(run_plan.jobs)} agent run(s) costing up to {_money(total)} {table.currency}?"
        )
        if not typer.confirm(question, default=False):
            console.print(Text("Aborted: nothing was run."))
            raise typer.Exit(1)

    out_dir = out_dir or DEFAULT_RESULTS_DIR / f"planted-{datetime.now(UTC):%Y%m%d-%H%M%S}"
    try:
        results = run_planted(
            model_list,
            datasets=dataset_list,
            variants=variant_list,
            modes=mode_list,
            repeats=repeats,
            out_dir=out_dir,
            prices=table.models,
            currency=table.currency,
            client_factory=create_client,
            data_dir=data_dir,
            estimated_max_cost=ceiling,
            progress=lambda run, index, total: _print_progress(console, run, index, total),
        )
    except (PlantedRunError, LLMError) as exc:
        _fail(str(exc))

    counts = {s: sum(r.status == s for r in results.runs) for s in ("ok", "failed", "error")}
    console.print(
        Text(f"Runs: {counts['ok']} ok, {counts['failed']} failed, {counts['error']} error")
    )
    console.print(Text(f"Results: {out_dir / 'results.json'}"), soft_wrap=True)
    console.print(Text(f"Summary: {out_dir / SUMMARY_FILE}"), soft_wrap=True)


def _summarize(console: Console, results_file: Path, out_dir: Path) -> None:
    try:
        results = PlantedResults.model_validate_json(results_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _fail(f"cannot read results from {results_file}: {exc}")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_file = out_dir / SUMMARY_FILE
    summary_file.write_text(render_summary(results), encoding="utf-8", newline="\n")
    console.print(Text(f"Summary: {summary_file}"), soft_wrap=True)


def _print_run_plan(
    console: Console, run_plan: RunPlan, table: PriceTable, modes: list[str]
) -> None:
    jobs = Table(title=f"{len(run_plan.jobs)} run(s)")
    for column in ("model", "variant", "mode"):
        jobs.add_column(column)
    jobs.add_column("runs", justify="right")
    grouped: dict[tuple[str, str, str], int] = {}
    for job in run_plan.jobs:
        key = (job.model, job.variant, job.mode)
        grouped[key] = grouped.get(key, 0) + 1
    for (model, variant, mode), runs in grouped.items():
        jobs.add_row(Text(model), Text(variant), Text(mode), Text(str(runs)))
    console.print(jobs)

    estimates = Table(title="Estimated maximum use")
    estimates.add_column("model")
    estimates.add_column("runs", justify="right")
    estimates.add_column("input tokens", justify="right")
    estimates.add_column("output tokens", justify="right")
    estimates.add_column(f"cost ({table.currency})", justify="right")
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
    note = (
        " The no_questions mode cannot use the question rounds." if "no_questions" in modes else ""
    )
    console.print(
        Text(
            "The estimate is an upper bound from the size of the prompts and the most calls the "
            f"agent can make, not a prediction: at most {max_calls_per_run()} LLM calls per run "
            f"({MAX_QUESTION_ROUNDS} question rounds + {MAX_REVISIONS} revisions + the first "
            f"plan, each repeated up to {MAX_SCHEMA_RETRIES} time(s) on a schema error)." + note
        )
    )


def _print_progress(console: Console, run: RunResult, index: int, total: int) -> None:
    detail = f" - {run.error}" if run.error and run.status != "ok" else ""
    forced = run.agent.forced_answers if run.agent else 0
    note = f" ({forced} forced answer(s))" if forced else ""
    console.print(
        Text(
            f"[{index}/{total}] {run.dataset} {run.variant} {run.mode} {run.model} "
            f"#{run.repeat}: {run.status}{note}{detail}"
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
