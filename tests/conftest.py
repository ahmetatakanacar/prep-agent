import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pytest
from typer.testing import CliRunner

from prep_agent import executor
from prep_agent.cli import app
from prep_agent.env import load_dotenv_file
from prep_agent.plan import Plan
from prep_agent.validators import SEVERITY, IssueCode, ValidationIssue

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
def inject_result_error(monkeypatch) -> Callable[..., None]:
    """Make the result validator report a chosen error, so that the agent and the CLI can be
    tested on what they do with a result error.

    The result errors (`empty_split`, `column_mismatch`, `target_altered`) are defences against
    bugs in the executor. No real plan on real data can reach them, so there is no input that
    would exercise the revision of a result error, the shared revision budget, the failed run
    that keeps its frames, or the CLI skipping the baseline. This replaces `validate_result` in
    the executor, the one place all of those paths get their result issues from, and leaves
    every other part of the run real.

    Call it as `inject_result_error(ops=("dedupe",))` to fail only plans that contain one of
    those ops (a plan without them is validated for real, which lets a revision succeed), or
    without `ops` to fail every plan. `code` is the error to report.
    """
    real = executor.validate_result

    def inject(*, ops: tuple[str, ...] | None = None, code: IssueCode = "target_altered") -> None:
        assert SEVERITY[code] == "error", f"{code} is not an error"

        def validate(result, plan):
            if ops is None or any(step.op in ops for step in plan.steps):
                return [
                    ValidationIssue(
                        code=code, severity="error", column=plan.target, message="injected by test"
                    )
                ]
            return real(result, plan)

        monkeypatch.setattr(executor, "validate_result", validate)

    return inject
