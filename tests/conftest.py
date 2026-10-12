import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pytest
from typer.testing import CliRunner

from prep_agent.cli import app

# Wide enough that rich never wraps table rows or help text in tests.
TERMINAL_WIDTH = "200"

# typer 0.27 vendors click privately, so there is no public click.unstyle to use.
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


@dataclass(frozen=True)
class CliResult:
    exit_code: int
    output: str


@pytest.fixture
def invoke_cli() -> Callable[[Sequence[str]], CliResult]:
    """Run the CLI with a fixed terminal width and return ANSI-free output."""
    runner = CliRunner()
    env = {"COLUMNS": TERMINAL_WIDTH, "TERMINAL_WIDTH": TERMINAL_WIDTH}

    def invoke(args: Sequence[str]) -> CliResult:
        result = runner.invoke(app, list(args), env=env)
        return CliResult(exit_code=result.exit_code, output=ANSI_ESCAPE.sub("", result.output))

    return invoke
