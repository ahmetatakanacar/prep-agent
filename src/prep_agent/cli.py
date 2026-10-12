"""Command-line interface for prep-agent."""

from pathlib import Path
from typing import Annotated, NoReturn

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table
from rich.text import Text

from prep_agent import __version__
from prep_agent.env import load_dotenv_file
from prep_agent.executor import ExecutionError, PlanValidationError, execute
from prep_agent.llm.base import LLMError
from prep_agent.llm.factory import create_llm_client
from prep_agent.llm.ping import PING_SYSTEM, PING_USER, PingReply
from prep_agent.plan import Plan
from prep_agent.profiler import DatasetProfile, ProfilerError, load_csv, profile_dataframe
from prep_agent.runs import RunExistsError, save_run
from prep_agent.validators import ValidationIssue, has_errors, validate_plan

app = typer.Typer(
    help="LLM-planned, validated data preparation for tabular ML.",
    no_args_is_help=True,
    add_completion=False,
)


def _print_version(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_print_version,
        is_eager=True,
        help="Show the version and exit.",
    ),
) -> None:
    """LLM-planned, validated data preparation for tabular ML."""
    load_dotenv_file()


@app.command("profile")
def profile_command(
    csv_path: Annotated[Path, typer.Argument(help="CSV file to profile.")],
    target: Annotated[str | None, typer.Option(help="Name of the target column.")] = None,
    out: Annotated[Path | None, typer.Option(help="Write the profile JSON to this file.")] = None,
    no_samples: Annotated[
        bool,
        typer.Option("--no-samples", help="Leave sample values and top values out of the profile."),
    ] = False,
) -> None:
    """Profile a CSV file and print a summary table."""
    try:
        profile = profile_dataframe(
            load_csv(csv_path), target=target, include_samples=not no_samples
        )
    except ProfilerError as exc:
        Console(stderr=True).print(Text.assemble(("Error: ", "red"), str(exc)))
        raise typer.Exit(1) from None

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(profile.model_dump_json(indent=2), encoding="utf-8", newline="\n")

    console = Console()
    console.print(Text(_summary_line(profile)))
    console.print(_columns_table(profile))
    if out is not None:
        console.print(Text(f"Profile written to {out}"))


@app.command("run")
def run_command(
    csv_path: Annotated[Path, typer.Argument(help="CSV file to prepare.")],
    plan_path: Annotated[Path, typer.Option("--plan", help="Preparation plan as a JSON file.")],
    out_dir: Annotated[Path, typer.Option(help="Directory that receives the run folder.")] = Path(
        "runs"
    ),
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Only validate the plan; run nothing and write nothing."),
    ] = False,
) -> None:
    """Run a hand-written plan on a CSV file and save the run artifacts."""
    console = Console()
    try:
        plan = _load_plan(plan_path)
        df = load_csv(csv_path)
        profile = profile_dataframe(df, target=plan.target)
    except (ProfilerError, PlanFileError) as exc:
        _fail(str(exc))

    if dry_run:
        issues = validate_plan(plan, profile)
        _print_issues(console, issues)
        if has_errors(issues):
            raise typer.Exit(1)
        console.print(Text("Dry run: the plan is valid, nothing was executed or written."))
        return

    try:
        result = execute(df, plan, profile)
    except PlanValidationError as exc:
        _print_issues(console, exc.issues)
        raise typer.Exit(1) from None
    except ExecutionError as exc:
        _fail(str(exc))

    _print_issues(console, result.plan_issues + result.result_issues)
    try:
        run_dir = save_run(out_dir, input_path=csv_path, profile=profile, plan=plan, result=result)
    except RunExistsError as exc:
        _fail(str(exc))

    train_shape = f"{len(result.train)} rows x {result.train.shape[1]} columns"
    test_shape = f"{len(result.test)} rows x {result.test.shape[1]} columns"
    if has_errors(result.result_issues):
        console.print(Text("Result validation failed; artifacts saved with status failed."))
        console.print(Text(f"Run folder: {run_dir}"), soft_wrap=True)
        raise typer.Exit(1)
    console.print(Text(f"Train: {train_shape} | Test: {test_shape}"))
    console.print(Text(f"Run folder: {run_dir}"), soft_wrap=True)


@app.command("llm-check")
def llm_check_command() -> None:
    """Send one small request to the configured LLM provider and report the outcome."""
    try:
        result = create_llm_client().generate(PING_SYSTEM, PING_USER, PingReply)
    except LLMError as exc:
        _fail(str(exc))

    usage = result.usage
    tokens = (
        "not reported" if usage is None else f"{usage.input_tokens} in / {usage.output_tokens} out"
    )
    Console().print(Text(f"LLM check passed | model: {result.model} | tokens: {tokens}"))


class PlanFileError(ValueError):
    """Raised when the plan file cannot be read or does not match the plan schema."""


def _load_plan(path: Path) -> Plan:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise PlanFileError(f"plan file not found: {path}") from None
    except (OSError, UnicodeDecodeError) as exc:
        raise PlanFileError(f"could not read plan file {path}: {exc}") from exc
    try:
        return Plan.model_validate_json(text)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in err['loc']) or '<root>'}: {err['msg']}"
            for err in exc.errors()
        )
        raise PlanFileError(f"invalid plan file {path}: {problems}") from None


def _fail(message: str) -> NoReturn:
    Console(stderr=True).print(Text.assemble(("Error: ", "red"), message))
    raise typer.Exit(1)


def _print_issues(console: Console, issues: list[ValidationIssue]) -> None:
    if not issues:
        console.print(Text("No issues found."))
        return
    table = Table()
    table.add_column("severity")
    table.add_column("code", no_wrap=True)
    table.add_column("step", justify="right")
    table.add_column("column")
    table.add_column("message")
    for issue in issues:
        style = "red" if issue.severity == "error" else "yellow"
        table.add_row(
            Text(issue.severity, style=style),
            Text(issue.code),
            Text("" if issue.step_index is None else str(issue.step_index)),
            Text(issue.column or ""),
            Text(issue.message),
        )
    console.print(table)


def _summary_line(profile: DatasetProfile) -> str:
    parts = [
        f"{profile.n_rows} rows",
        f"{profile.n_columns} columns",
        f"{profile.n_duplicate_rows} duplicate rows",
    ]
    if profile.target is not None:
        parts.append(f"target: {profile.target} (hint: {profile.target_task_hint or 'none'})")
    return " | ".join(parts)


def _columns_table(profile: DatasetProfile) -> Table:
    table = Table()
    table.add_column("column")
    table.add_column("type")
    table.add_column("missing", justify="right")
    table.add_column("unique", justify="right")
    table.add_column("warnings")
    for col in profile.columns:
        table.add_row(
            col.name,
            col.inferred_type,
            f"{col.missing_ratio:.1%}",
            str(col.n_unique),
            ", ".join(col.warnings),
        )
    return table
