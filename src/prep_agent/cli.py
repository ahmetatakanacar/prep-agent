"""Command-line interface for prep-agent."""

import typer

from prep_agent import __version__

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
