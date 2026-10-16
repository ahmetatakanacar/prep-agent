"""Command-line interface for prep-agent."""

import sys
from enum import StrEnum
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table
from rich.text import Text

from prep_agent import __version__
from prep_agent.agent import AgentError, AgentTrace, AskFn, run_agent
from prep_agent.baseline import (
    NOT_CONVERGED_EXPLANATION,
    NOT_CONVERGED_NOTE,
    BaselineResult,
    evaluate_baselines,
    metric_label,
)
from prep_agent.env import load_dotenv_file
from prep_agent.executor import ExecutionError, ExecutionResult, PlanValidationError, execute
from prep_agent.llm.base import LLMError
from prep_agent.llm.factory import create_llm_client
from prep_agent.llm.ping import PING_SYSTEM, PING_USER, PingReply
from prep_agent.plan import (
    EncodeStep,
    ImputeStep,
    Plan,
    ScaleStep,
    Step,
)
from prep_agent.planner import Question
from prep_agent.profiler import DatasetProfile, ProfilerError, load_csv, profile_dataframe
from prep_agent.report import step_columns
from prep_agent.runs import RunExistsError, save_run
from prep_agent.validators import ValidationIssue, has_errors, validate_plan

app = typer.Typer(
    help="LLM-planned, validated data preparation for tabular ML.",
    no_args_is_help=True,
    add_completion=False,
)


def ensure_utf8_output() -> None:
    """Make stdout and stderr UTF-8 so the arrows and box characters of the output survive.

    On Windows a stream that is redirected to a file or a pipe uses the ANSI code page, which
    cannot encode them and would end the command with a UnicodeEncodeError.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
        if reconfigure is not None and encoding != "utf8":
            reconfigure(encoding="utf-8", errors="replace")


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
    ensure_utf8_output()
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
    no_baseline: Annotated[
        bool, typer.Option("--no-baseline", help="Skip the baseline scores (cross-validation).")
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
    baseline = None if no_baseline else _compute_baseline(console, plan, result)
    try:
        run_dir = save_run(
            out_dir,
            input_path=csv_path,
            profile=profile,
            plan=plan,
            result=result,
            baseline=baseline,
        )
    except RunExistsError as exc:
        _fail(str(exc))

    train_shape = f"{len(result.train)} rows x {result.train.shape[1]} columns"
    test_shape = f"{len(result.test)} rows x {result.test.shape[1]} columns"
    if has_errors(result.result_issues):
        console.print(Text("Result validation failed; artifacts saved with status failed."))
        console.print(Text(f"Run folder: {run_dir}"), soft_wrap=True)
        raise typer.Exit(1)
    console.print(Text(f"Train: {train_shape} | Test: {test_shape}"))
    if baseline is not None:
        _print_baseline(console, baseline)
    console.print(Text(f"Run folder: {run_dir}"), soft_wrap=True)


CUSTOM_OPTION_LABEL = "Other (type your own)"


class ModelFamily(StrEnum):
    tree = "tree"
    linear = "linear"
    any = "any"


@app.command("prepare")
def prepare_command(
    csv_path: Annotated[Path, typer.Argument(help="CSV file to prepare.")],
    goal: Annotated[str, typer.Option(help="What the data is for, in plain language.")],
    target: Annotated[str | None, typer.Option(help="Name of the target column.")] = None,
    model_family: Annotated[
        ModelFamily | None, typer.Option(help="Model family the data is prepared for.")
    ] = None,
    out_dir: Annotated[Path, typer.Option(help="Directory that receives the run folder.")] = Path(
        "runs"
    ),
    non_interactive: Annotated[
        bool,
        typer.Option(
            "--non-interactive", help="Answer every question with the recommended option."
        ),
    ] = False,
    max_revisions: Annotated[
        int, typer.Option(min=0, help="How often a plan with errors may be sent back.")
    ] = 2,
    no_baseline: Annotated[
        bool, typer.Option("--no-baseline", help="Skip the baseline scores (cross-validation).")
    ] = False,
) -> None:
    """Have the LLM plan the preparation of a CSV file, then validate, run and save it."""
    console = Console()
    if not goal.strip():
        _fail("the goal must not be empty")
    try:
        df = load_csv(csv_path)
        llm = create_llm_client()
        console.print(Text("Profiling the data and planning..."))
        result = run_agent(
            df,
            goal,
            llm,
            _ask_recommended(console) if non_interactive else _ask_interactively(console),
            target=target,
            model_family=model_family.value if model_family else None,
            max_revisions=max_revisions,
        )
    except AgentError as exc:
        _report_agent_failure(console, exc, csv_path, out_dir)
    except (ProfilerError, LLMError) as exc:
        _fail(str(exc))

    _print_plan(console, result.plan)
    _print_issues(console, result.issues)
    baseline = None if no_baseline else _compute_baseline(console, result.plan, result.execution)
    try:
        run_dir = save_run(
            out_dir,
            input_path=csv_path,
            profile=result.profile,
            plan=result.plan,
            result=result.execution,
            trace=result.trace,
            baseline=baseline,
        )
    except RunExistsError as exc:
        _fail(str(exc))
    train, test = result.execution.train, result.execution.test
    console.print(Text(_usage_line(result.trace)))
    console.print(
        Text(
            f"Train: {len(train)} rows x {train.shape[1]} columns | "
            f"Test: {len(test)} rows x {test.shape[1]} columns"
        )
    )
    if baseline is not None:
        _print_baseline(console, baseline)
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


def _ask_recommended(console: Console) -> AskFn:
    def ask(questions: list[Question]) -> dict[str, str]:
        for question in questions:
            console.print(Text(f"{question.question} -> {question.recommended} (recommended)"))
        return {question.id: question.recommended for question in questions}

    return ask


def _ask_interactively(console: Console) -> AskFn:
    def ask(questions: list[Question]) -> dict[str, str]:
        answers = {}
        for question in questions:
            console.print(Text(question.question, style="bold"))
            console.print(Text(f"Why I ask: {question.reason}", style="dim"))
            for number, option in enumerate(question.options, start=1):
                suffix = " (recommended)" if option == question.recommended else ""
                console.print(Text(f"  {number}) {option}{suffix}"))
            if question.allow_custom:
                console.print(Text(f"  {len(question.options) + 1}) {CUSTOM_OPTION_LABEL}"))
            answers[question.id] = _read_choice(console, question)
        return answers

    return ask


def _read_choice(console: Console, question: Question) -> str:
    count = len(question.options)
    total = count + 1 if question.allow_custom else count
    default = question.options.index(question.recommended) + 1
    while True:
        raw = _read_line(console, f"Choose 1-{total} [Enter = {default}]: ")
        if not raw:
            return question.recommended
        if raw.isdecimal() and 1 <= int(raw) <= count:
            return question.options[int(raw) - 1]
        if raw.isdecimal() and int(raw) == total and question.allow_custom:
            return _read_custom_answer(console)
        console.print(Text(f"Enter a number from 1 to {total}, or press Enter for {default}."))


def _read_custom_answer(console: Console) -> str:
    while True:
        answer = _read_line(console, "Your answer: ")
        if answer:
            return answer
        console.print(Text("The answer must not be empty."))


def _read_line(console: Console, prompt: str) -> str:
    try:
        return console.input(Text(prompt)).strip()
    except EOFError:
        _fail("input ended while waiting for an answer")


def _report_agent_failure(
    console: Console, error: AgentError, csv_path: Path, out_dir: Path
) -> NoReturn:
    """Print what went wrong, save the failed run and exit 1."""
    Console(stderr=True).print(Text.assemble(("Error: ", "red"), str(error)))
    if error.plan is not None:
        _print_plan(console, error.plan)
        _print_issues(console, error.issues)
    try:
        run_dir = save_run(
            out_dir,
            input_path=csv_path,
            profile=error.profile,
            plan=error.plan,
            result=error.execution,
            plan_issues=error.issues,
            trace=error.trace,
            error=str(error),
        )
    except RunExistsError as exc:
        _fail(str(exc))
    console.print(Text(_usage_line(error.trace)))
    console.print(Text("The run failed; artifacts saved with status failed."))
    console.print(Text(f"Run folder: {run_dir}"), soft_wrap=True)
    raise typer.Exit(1)


def _compute_baseline(
    console: Console, plan: Plan, execution: ExecutionResult
) -> BaselineResult | None:
    """Score the baselines for a successful execution; None if the result has errors."""
    if has_errors(execution.result_issues):
        return None
    console.print(Text("Computing baseline scores..."))
    return evaluate_baselines(
        execution.split_train, execution.pipeline, plan.target, plan.task_type
    )


def _print_baseline(console: Console, baseline: BaselineResult) -> None:
    if baseline.skipped_reason is not None:
        console.print(Text(f"Baseline skipped: {baseline.skipped_reason}"))
        return
    console.print(
        Text(
            f"Baseline: {baseline.n_folds}-fold cross-validation on train, "
            "mean ± std (the dummy model is the reference)"
        )
    )
    scored = [model for model in baseline.models if model.scores]
    if scored:
        table = Table()
        table.add_column("model", no_wrap=True)
        for metric in baseline.metrics:
            table.add_column(metric_label(metric), justify="right")
        for model in scored:
            table.add_row(
                Text(f"{model.name} ({model.note})" if model.note else model.name),
                *(
                    Text(f"{model.scores[m].mean:.3f} ± {model.scores[m].std:.3f}")
                    for m in baseline.metrics
                ),
            )
        console.print(table)
    if any(model.note == NOT_CONVERGED_NOTE for model in baseline.models):
        console.print(Text(NOT_CONVERGED_EXPLANATION))
    for model in baseline.models:
        if model.skipped_reason is not None:
            console.print(Text(f"Skipped {model.name}: {model.skipped_reason}"))


def _usage_line(trace: AgentTrace) -> str:
    usage = trace.total_usage
    tokens = (
        "not reported" if usage is None else f"{usage.input_tokens} in / {usage.output_tokens} out"
    )
    count = len(trace.calls)
    calls = f"{count} LLM call{'' if count == 1 else 's'}"
    return f"Tokens: {tokens} | {calls} | model: {trace.model or 'none'}"


def _print_plan(console: Console, plan: Plan) -> None:
    console.print(
        Text(f"Plan: target {plan.target} | {plan.task_type} | model family {plan.model_family}")
    )
    table = Table()
    table.add_column("step", justify="right")
    table.add_column("op", no_wrap=True)
    table.add_column("columns")
    table.add_column("reason")
    for index, step in enumerate(plan.steps):
        table.add_row(
            Text(str(index)),
            Text(_op_label(step)),
            Text(step_columns(step, plan.target)),
            Text(step.reason),
        )
    console.print(table)


def _op_label(step: Step) -> str:
    if isinstance(step, ImputeStep):
        return f"{step.op} ({step.strategy})"
    if isinstance(step, EncodeStep | ScaleStep):
        return f"{step.op} ({step.method})"
    return step.op


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
