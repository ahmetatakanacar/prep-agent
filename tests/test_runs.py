import dataclasses
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import pytest

from prep_agent import __version__
from prep_agent.agent import AgentTrace, LLMCallRecord
from prep_agent.baseline import BaselineResult, BaselineScore, ModelBaseline
from prep_agent.codegen import generate_pipeline_script
from prep_agent.executor import execute
from prep_agent.llm.base import TokenUsage
from prep_agent.ops import OpContext, get_op
from prep_agent.plan import Plan
from prep_agent.planner import PlannerResponse, QARecord, Question
from prep_agent.profiler import DatasetProfile, load_csv, profile_dataframe
from prep_agent.runs import RunExistsError, RunInfo, save_run
from prep_agent.validators import ValidationIssue

FIXTURES = Path(__file__).parent / "fixtures"
CUSTOMERS_CSV = FIXTURES / "customers.csv"
CUSTOMERS_PLAN = FIXTURES / "customers_plan.json"

EXPECTED_FILES = {
    "profile.json",
    "plan.json",
    "train.csv",
    "test.csv",
    "pipeline.joblib",
    "pipeline.py",
    "issues.json",
    "run.json",
    "report.md",
}


@pytest.fixture
def executed():
    df = load_csv(CUSTOMERS_CSV)
    plan = Plan.model_validate_json(CUSTOMERS_PLAN.read_text(encoding="utf-8"))
    profile = profile_dataframe(df, target=plan.target)
    return df, plan, profile, execute(df, plan, profile)


def save(tmp_path, executed, **kwargs) -> Path:
    _, plan, profile, result = executed
    return save_run(
        tmp_path / "runs",
        input_path=CUSTOMERS_CSV,
        profile=profile,
        plan=plan,
        result=result,
        **kwargs,
    )


def test_save_run_writes_all_files(tmp_path, executed):
    run_dir = save(tmp_path, executed, run_id="r1")

    assert run_dir == tmp_path / "runs" / "r1"
    assert {p.name for p in run_dir.iterdir()} == EXPECTED_FILES


def test_json_files_can_be_read_back(tmp_path, executed):
    _, plan, profile, result = executed
    run_dir = save(tmp_path, executed, run_id="r1")

    def read(name: str) -> str:
        return (run_dir / name).read_text(encoding="utf-8")

    assert DatasetProfile.model_validate_json(read("profile.json")) == profile
    assert Plan.model_validate_json(read("plan.json")) == plan
    issues = json.loads(read("issues.json"))
    assert set(issues) == {"plan_issues", "result_issues"}
    assert [ValidationIssue.model_validate(i) for i in issues["plan_issues"]] == result.plan_issues
    assert [
        ValidationIssue.model_validate(i) for i in issues["result_issues"]
    ] == result.result_issues
    json.loads(read("run.json"))


def test_csv_files_match_the_result(tmp_path, executed):
    _, _, _, result = executed
    run_dir = save(tmp_path, executed, run_id="r1")

    pd.testing.assert_frame_equal(pd.read_csv(run_dir / "train.csv"), result.train)
    pd.testing.assert_frame_equal(pd.read_csv(run_dir / "test.csv"), result.test)
    assert b"\r\n" not in (run_dir / "train.csv").read_bytes()


def test_run_json_describes_the_run(tmp_path, executed):
    run_dir = save(tmp_path, executed, run_id="r1")

    info = RunInfo.model_validate_json((run_dir / "run.json").read_text(encoding="utf-8"))
    assert info.run_id == "r1"
    assert info.prep_agent_version == __version__
    assert info.input_sha256 == hashlib.sha256(CUSTOMERS_CSV.read_bytes()).hexdigest()
    assert info.input_path == CUSTOMERS_CSV.as_posix()
    assert (info.rows.before, info.rows.train, info.rows.test) == (20, 16, 4)
    assert (info.columns.before, info.columns.train, info.columns.test) == (7, 6, 6)
    assert info.target == "churned"
    assert info.status == "ok"
    assert isinstance(info.created_at, datetime)
    assert info.created_at.tzinfo is not None
    assert "stateful" in info.pipeline_joblib
    assert "pipeline.py" in info.pipeline_joblib
    assert "not generated" not in info.pipeline_joblib


def test_relative_input_path_is_stored_as_given(tmp_path, executed, monkeypatch):
    _, plan, profile, result = executed
    monkeypatch.chdir(FIXTURES)

    run_dir = save_run(
        tmp_path / "runs",
        input_path=Path("customers.csv"),
        profile=profile,
        plan=plan,
        result=result,
        run_id="r1",
    )

    info = RunInfo.model_validate_json((run_dir / "run.json").read_text(encoding="utf-8"))
    assert info.input_path == "customers.csv"
    assert info.input_sha256 == hashlib.sha256(CUSTOMERS_CSV.read_bytes()).hexdigest()


def test_status_is_failed_when_the_result_has_errors(tmp_path, executed):
    _, plan, profile, result = executed
    error = ValidationIssue(code="target_altered", severity="error", message="x")
    failed = dataclasses.replace(result, result_issues=[error])

    run_dir = save_run(
        tmp_path / "runs",
        input_path=CUSTOMERS_CSV,
        profile=profile,
        plan=plan,
        result=failed,
        run_id="r1",
    )

    info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert info["status"] == "failed"
    assert {p.name for p in run_dir.iterdir()} == EXPECTED_FILES - {"pipeline.py"}


def test_existing_run_is_never_overwritten(tmp_path, executed):
    run_dir = save(tmp_path, executed, run_id="r1")
    before = {p.name: p.read_bytes() for p in run_dir.iterdir()}

    with pytest.raises(RunExistsError):
        save(tmp_path, executed, run_id="r1")

    assert {p.name: p.read_bytes() for p in run_dir.iterdir()} == before


def test_existing_empty_directory_is_not_reused(tmp_path, executed):
    (tmp_path / "runs" / "r1").mkdir(parents=True)

    with pytest.raises(RunExistsError):
        save(tmp_path, executed, run_id="r1")

    assert list((tmp_path / "runs" / "r1").iterdir()) == []


def test_default_run_id_comes_from_the_timestamp(tmp_path, executed):
    run_dir = save(tmp_path, executed)

    assert re.fullmatch(r"\d{8}-\d{6}", run_dir.name)
    assert run_dir.parent == tmp_path / "runs"


@pytest.mark.parametrize("run_id", ["../escape", "a/b", "a\\b", "", ".hidden", "with space"])
def test_unsafe_run_id_is_rejected(tmp_path, executed, run_id):
    with pytest.raises(ValueError, match="run_id"):
        save(tmp_path, executed, run_id=run_id)

    assert not (tmp_path / "runs").exists()


def test_failed_write_leaves_no_partial_run(tmp_path, executed, monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("prep_agent.runs.joblib.dump", boom)

    with pytest.raises(OSError, match="disk full"):
        save(tmp_path, executed, run_id="r1")

    assert not (tmp_path / "runs" / "r1").exists()


def test_saved_pipeline_reproduces_the_executor_output(tmp_path, executed):
    df, plan, _, result = executed
    run_dir = save(tmp_path, executed, run_id="r1")

    # Rebuild the raw test features by hand: the saved pipeline holds the stateful steps only.
    ctx = OpContext(target=plan.target)
    frame = df
    for step in plan.steps:
        spec = get_op(step.op)
        if spec.kind == "stateless":
            frame = spec.apply(frame, step, ctx)
        elif spec.kind == "split":
            _, raw_test = spec.apply(frame, step, ctx)
    pipeline = joblib.load(run_dir / "pipeline.joblib")
    transformed = pipeline.transform(raw_test.drop(columns=plan.target))
    transformed[plan.target] = raw_test[plan.target]

    pd.testing.assert_frame_equal(transformed, result.test, check_exact=True)


def test_pipeline_py_is_generated_from_the_plan_and_the_run_facts(tmp_path, executed):
    _, plan, _, _ = executed
    run_dir = save(tmp_path, executed, run_id="r1")

    info = RunInfo.model_validate_json((run_dir / "run.json").read_text(encoding="utf-8"))
    expected = generate_pipeline_script(
        plan, run_id="r1", prep_agent_version=__version__, input_sha256=info.input_sha256
    )
    assert (run_dir / "pipeline.py").read_text(encoding="utf-8") == expected
    assert b"\r\n" not in (run_dir / "pipeline.py").read_bytes()


def test_the_saved_pipeline_py_reproduces_the_saved_train_and_test(tmp_path, executed):
    run_dir = save(tmp_path, executed, run_id="r1")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}

    done = subprocess.run(
        [sys.executable, str(run_dir / "pipeline.py"), str(CUSTOMERS_CSV), "--out-dir", "out"],
        cwd=elsewhere,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )

    assert done.returncode == 0, done.stderr
    for name in ("train.csv", "test.csv"):
        assert (elsewhere / "out" / name).read_bytes() == (run_dir / name).read_bytes()


def test_a_run_whose_result_failed_validation_gets_no_pipeline_py(tmp_path, executed):
    _, plan, profile, result = executed
    error = ValidationIssue(code="target_altered", severity="error", message="x")

    run_dir = save_run(
        tmp_path / "runs",
        input_path=CUSTOMERS_CSV,
        profile=profile,
        plan=plan,
        result=dataclasses.replace(result, result_issues=[error]),
        run_id="r1",
    )

    assert not (run_dir / "pipeline.py").exists()
    assert (run_dir / "train.csv").exists()


# --- runs that come from the agent ---

DEFAULT_USAGE = TokenUsage(120, 30)


def make_trace(*, usage: TokenUsage | None = DEFAULT_USAGE) -> AgentTrace:
    question = Question(
        id="family",
        question="Which model family?",
        options=["tree", "any"],
        recommended="any",
        reason="scaling depends on it",
    )
    plan = Plan.model_validate_json(CUSTOMERS_PLAN.read_text(encoding="utf-8"))
    call = LLMCallRecord(
        index=0,
        purpose="plan",
        model="fake-model",
        usage=usage,
        issue_codes=[],
        warning_codes=[],
        response_kind="plan",
        response=PlannerResponse(kind="plan", plan=plan),
    )
    return AgentTrace(
        goal="prepare for churn",
        calls=[call],
        qa_history=[QARecord(question=question, answer="any")],
    )


def test_a_trace_adds_the_trace_and_qa_files(tmp_path, executed):
    run_dir = save(tmp_path, executed, run_id="r1", trace=make_trace())

    assert {p.name for p in run_dir.iterdir()} == EXPECTED_FILES | {"llm_trace.json", "qa.json"}
    trace = json.loads((run_dir / "llm_trace.json").read_text(encoding="utf-8"))
    assert trace["model"] == "fake-model"
    assert trace["total_usage"] == {"input_tokens": 120, "output_tokens": 30}
    [call] = trace["calls"]
    assert call["purpose"] == "plan" and call["issue_codes"] == []
    assert call["response_kind"] == "plan"
    assert call["response"]["kind"] == "plan"
    [qa] = json.loads((run_dir / "qa.json").read_text(encoding="utf-8"))
    assert qa["question"]["id"] == "family" and qa["answer"] == "any"


def test_run_json_gets_goal_model_and_usage_from_the_trace(tmp_path, executed):
    run_dir = save(tmp_path, executed, run_id="r1", trace=make_trace())

    info = RunInfo.model_validate_json((run_dir / "run.json").read_text(encoding="utf-8"))
    assert info.goal == "prepare for churn"
    assert info.model == "fake-model"
    assert info.token_usage == TokenUsage(120, 30)


def test_usage_stays_empty_when_the_provider_reported_none(tmp_path, executed):
    run_dir = save(tmp_path, executed, run_id="r1", trace=make_trace(usage=None))

    info = RunInfo.model_validate_json((run_dir / "run.json").read_text(encoding="utf-8"))
    assert info.token_usage is None
    trace = json.loads((run_dir / "llm_trace.json").read_text(encoding="utf-8"))
    assert trace["total_usage"] is None


def test_a_run_without_a_trace_has_no_goal(tmp_path, executed):
    run_dir = save(tmp_path, executed, run_id="r1")

    info = RunInfo.model_validate_json((run_dir / "run.json").read_text(encoding="utf-8"))
    assert (info.goal, info.model, info.token_usage) == (None, None, None)


def test_a_failed_run_without_a_result_still_writes_plan_issues_and_trace(tmp_path, executed):
    _, plan, profile, _ = executed
    error = ValidationIssue(
        code="stateful_before_split", severity="error", step_index=0, message="move it"
    )

    run_dir = save_run(
        tmp_path / "runs",
        input_path=CUSTOMERS_CSV,
        profile=profile,
        plan=plan,
        result=None,
        plan_issues=[error],
        trace=make_trace(),
        run_id="r1",
    )

    assert {p.name for p in run_dir.iterdir()} == {
        "profile.json",
        "plan.json",
        "issues.json",
        "run.json",
        "llm_trace.json",
        "qa.json",
        "report.md",
    }
    info = RunInfo.model_validate_json((run_dir / "run.json").read_text(encoding="utf-8"))
    assert info.status == "failed"
    assert (info.rows.before, info.rows.train, info.rows.test) == (20, None, None)
    assert (info.columns.before, info.columns.train, info.columns.test) == (7, None, None)
    assert info.pipeline_joblib is None
    assert not (run_dir / "pipeline.py").exists()
    issues = json.loads((run_dir / "issues.json").read_text(encoding="utf-8"))
    assert [i["code"] for i in issues["plan_issues"]] == ["stateful_before_split"]
    assert issues["result_issues"] == []


def test_a_run_without_a_plan_writes_profile_trace_qa_and_run_json(tmp_path, executed):
    _, _, profile, _ = executed

    run_dir = save_run(
        tmp_path / "runs",
        input_path=CUSTOMERS_CSV,
        profile=profile,
        plan=None,
        result=None,
        trace=make_trace(),
        error="the LLM call failed: rate limited",
        run_id="r1",
    )

    assert {p.name for p in run_dir.iterdir()} == {
        "profile.json",
        "run.json",
        "llm_trace.json",
        "qa.json",
        "report.md",
    }
    info = RunInfo.model_validate_json((run_dir / "run.json").read_text(encoding="utf-8"))
    assert info.status == "failed"
    assert info.target is None
    assert info.error == "the LLM call failed: rate limited"
    assert info.goal == "prepare for churn"
    assert (info.rows.before, info.rows.train, info.rows.test) == (20, None, None)
    assert info.pipeline_joblib is None


def test_a_run_without_a_plan_and_without_calls_still_has_an_empty_trace(tmp_path, executed):
    _, _, profile, _ = executed
    trace = AgentTrace(goal="g", calls=[], qa_history=[])

    run_dir = save_run(
        tmp_path / "runs",
        input_path=CUSTOMERS_CSV,
        profile=profile,
        plan=None,
        result=None,
        trace=trace,
        error="boom",
        run_id="r1",
    )

    assert json.loads((run_dir / "llm_trace.json").read_text(encoding="utf-8")) == {
        "model": None,
        "total_usage": None,
        "calls": [],
    }
    assert json.loads((run_dir / "qa.json").read_text(encoding="utf-8")) == []
    info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert info["model"] is None and info["goal"] == "g"


def test_error_is_recorded_for_a_run_that_has_a_plan(tmp_path, executed):
    _, plan, profile, _ = executed

    run_dir = save_run(
        tmp_path / "runs",
        input_path=CUSTOMERS_CSV,
        profile=profile,
        plan=plan,
        result=None,
        plan_issues=[],
        error="the plan still has 1 error(s)",
        run_id="r1",
    )

    info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert info["error"] == "the plan still has 1 error(s)"
    assert info["target"] == "churned"


def test_a_successful_run_has_no_error(tmp_path, executed):
    run_dir = save(tmp_path, executed, run_id="r1")

    assert json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["error"] is None


# --- report and baseline ---


def make_baseline() -> BaselineResult:
    scores = {"accuracy": BaselineScore(mean=0.75, std=0.1)}
    return BaselineResult(
        task_type="classification",
        n_rows=16,
        random_state=42,
        n_folds=4,
        metrics=["accuracy"],
        models=[ModelBaseline(name="dummy_most_frequent", scores=scores)],
    )


def test_report_md_is_written_for_a_successful_run(tmp_path, executed):
    run_dir = save(tmp_path, executed, run_id="r1")

    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert report.startswith("# Preparation report")
    assert "customers.csv" in report
    assert "**Status:** ok" in report
    assert b"\r\n" not in (run_dir / "report.md").read_bytes()


def test_a_baseline_is_written_as_json_and_shown_in_the_report(tmp_path, executed):
    baseline = make_baseline()

    run_dir = save(tmp_path, executed, run_id="r1", baseline=baseline)

    assert {p.name for p in run_dir.iterdir()} == EXPECTED_FILES | {"baseline.json"}
    written = BaselineResult.model_validate_json(
        (run_dir / "baseline.json").read_text(encoding="utf-8")
    )
    assert written == baseline
    assert "dummy_most_frequent" in (run_dir / "report.md").read_text(encoding="utf-8")


def test_without_a_baseline_there_is_no_baseline_json_and_the_report_says_so(tmp_path, executed):
    run_dir = save(tmp_path, executed, run_id="r1")

    assert not (run_dir / "baseline.json").exists()
    assert "--no-baseline" in (run_dir / "report.md").read_text(encoding="utf-8")


def test_a_failed_run_reports_the_reason_and_has_no_baseline_section(tmp_path, executed):
    _, plan, profile, _ = executed

    run_dir = save_run(
        tmp_path / "runs",
        input_path=CUSTOMERS_CSV,
        profile=profile,
        plan=plan,
        result=None,
        plan_issues=[],
        error="the plan still has 1 error(s)",
        baseline=make_baseline(),
        run_id="r1",
    )

    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "the plan still has 1 error(s)" in report
    assert "## Baseline" not in report


def test_a_failed_run_without_a_plan_still_gets_a_report(tmp_path, executed):
    _, _, profile, _ = executed

    run_dir = save_run(
        tmp_path / "runs",
        input_path=CUSTOMERS_CSV,
        profile=profile,
        plan=None,
        result=None,
        trace=make_trace(),
        error="the LLM call failed: rate limited",
        run_id="r1",
    )

    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "the LLM call failed: rate limited" in report
    assert "prepare for churn" in report


def test_the_report_uses_the_trace_for_model_and_tokens(tmp_path, executed):
    run_dir = save(tmp_path, executed, run_id="r1", trace=make_trace())

    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "fake-model" in report
    assert "120 input / 30 output" in report
    assert "Which model family?" in report


def test_input_path_is_stored_with_forward_slashes(tmp_path, executed, monkeypatch):
    _, plan, profile, result = executed
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "customers.csv").write_bytes(CUSTOMERS_CSV.read_bytes())
    monkeypatch.chdir(tmp_path)

    run_dir = save_run(
        tmp_path / "runs",
        input_path=Path("data") / "customers.csv",
        profile=profile,
        plan=plan,
        result=result,
        run_id="r1",
    )

    info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert info["input_path"] == "data/customers.csv"
    assert "customers.csv" in (run_dir / "report.md").read_text(encoding="utf-8")
