"""Command-line interface for prep-agent."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from prep_agent import __version__
from prep_agent.profiler import DatasetProfile, ProfilerError, load_csv, profile_dataframe

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


@app.command("profile")
def profile_command(
    csv_path: Annotated[Path, typer.Argument(help="CSV file to profile.")],
    target: Annotated[str | None, typer.Option(help="Name of the target column.")] = None,
    out: Annotated[Path | None, typer.Option(help="Write the profile JSON to this file.")] = None,
) -> None:
    """Profile a CSV file and print a summary table."""
    try:
        profile = profile_dataframe(load_csv(csv_path), target=target)
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
