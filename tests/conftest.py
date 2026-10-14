import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pytest
from typer.testing import CliRunner

from prep_agent import executor
from prep_agent.cli import app
from prep_agent.env import load_dotenv_file
from prep_agent.plan import DedupeStep, Plan
from prep_agent.validators import ValidationIssue

# Wide enough that rich never wraps table rows or help text in tests.
TERMINAL_WIDTH = "200"

# typer 0.27 vendors click privately, so there is no public click.unstyle to use.
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


@dataclass(frozen=True)
class CliResult:
    exit_code: int
    output: str


@pytest.fixture(autouse=True)
def no_dotenv_in_cli(monkeypatch):
    """Keep a developer's real .env (and its API key) out of every CLI test."""
    monkeypatch.setattr("prep_agent.cli.load_dotenv_file", lambda: None)


@pytest.fixture
def make_plan() -> Callable[..., Plan]:
    """Build a Plan from step dicts; every step gets a placeholder reason."""

    def build(*steps, target="t", task_type="classification", model_family="any") -> Plan:
        return Plan.model_validate(
            {
                "target": target,
                "task_type": task_type,
                "model_family": model_family,
                "steps": [{"reason": "test", **step} for step in steps],
            }
        )

    return build


@pytest.fixture
def invoke_cli() -> Callable[..., CliResult]:
    """Run the CLI with a fixed terminal width and return ANSI-free output."""
    runner = CliRunner()
    env = {"COLUMNS": TERMINAL_WIDTH, "TERMINAL_WIDTH": TERMINAL_WIDTH}

    def invoke(args: Sequence[str], input: str | None = None) -> CliResult:
        result = runner.invoke(app, list(args), input=input, env=env)
        return CliResult(exit_code=result.exit_code, output=ANSI_ESCAPE.sub("", result.output))

    return invoke


def pytest_collection_modifyitems(config, items):
    """Skip `llm` tests, which call the real API, unless they were explicitly asked for."""
    if os.environ.get("PREP_AGENT_RUN_LLM_TESTS") == "1":
        load_dotenv_file()
        if os.environ.get("OPENAI_API_KEY", "").strip():
            return
        reason = "OPENAI_API_KEY is not set"
    else:
        reason = "real API tests; set PREP_AGENT_RUN_LLM_TESTS=1 and OPENAI_API_KEY to run them"
    skip = pytest.mark.skip(reason=reason)
    for item in items:
        if "llm" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def overlap_error_with_dedupe(monkeypatch):
    """Make the result validator report a train_test_overlap error for every plan with a dedupe
    step. A dedupe removes the identical input records that could overlap, so real data no longer
    leaves a result error to provoke; the tests that follow what happens to one need this."""
    real = executor.validate_result

    def validate(result, plan):
        if any(isinstance(step, DedupeStep) for step in plan.steps):
            return [
                ValidationIssue(
                    code="train_test_overlap",
                    severity="error",
                    message="1 test row(s) are copies of a record that is also in train.",
                )
            ]
        return real(result, plan)

    monkeypatch.setattr(executor, "validate_result", validate)
