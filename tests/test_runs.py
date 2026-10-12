import dataclasses
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import pytest

from prep_agent import __version__
from prep_agent.executor import execute
from prep_agent.ops import OpContext, get_op
from prep_agent.plan import Plan
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
    "issues.json",
    "run.json",
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
    assert Path(info.input_path) == CUSTOMERS_CSV.resolve()
    assert (info.rows.before, info.rows.train, info.rows.test) == (20, 16, 4)
    assert (info.columns.before, info.columns.train, info.columns.test) == (7, 6, 6)
    assert info.target == "churned"
    assert info.status == "ok"
    assert isinstance(info.created_at, datetime)
    assert info.created_at.tzinfo is not None
    assert "stateful" in info.pipeline_joblib


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
    assert {p.name for p in run_dir.iterdir()} == EXPECTED_FILES


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
