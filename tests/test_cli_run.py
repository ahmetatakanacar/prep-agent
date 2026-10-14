import json
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"
CSV = str(FIXTURES / "customers.csv")
PLAN = str(FIXTURES / "customers_plan.json")
INVALID_PLAN = str(FIXTURES / "customers_plan_invalid.json")


def run_args(tmp_path, *extra, plan=PLAN):
    # The baseline has its own tests (test_cli_baseline.py); skipping it keeps these fast.
    return [
        "run",
        CSV,
        "--plan",
        plan,
        "--out-dir",
        str(tmp_path / "runs"),
        "--no-baseline",
        *extra,
    ]


def test_successful_run_writes_artifacts(invoke_cli, tmp_path):
    result = invoke_cli(run_args(tmp_path))

    assert result.exit_code == 0, result.output
    [run_dir] = list((tmp_path / "runs").iterdir())
    assert str(run_dir) in result.output.replace("\n", "")
    assert "16" in result.output and "4" in result.output
    info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert info["status"] == "ok"
    assert (run_dir / "train.csv").exists()


def test_run_prints_issues_in_a_table(invoke_cli, tmp_path):
    result = invoke_cli(run_args(tmp_path, plan=INVALID_PLAN))

    assert result.exit_code == 1
    for header in ("severity", "code", "step", "column", "message"):
        assert header in result.output
    assert "stateful_before_split" in result.output
    assert "unknown_column" in result.output
    assert "no_such_column" in result.output


def test_warnings_do_not_fail_the_run(invoke_cli, tmp_path):
    plan = tmp_path / "plan.json"
    data = json.loads(Path(PLAN).read_text(encoding="utf-8"))
    data["steps"][3]["stratify"] = False
    plan.write_text(json.dumps(data), encoding="utf-8")

    result = invoke_cli(run_args(tmp_path, plan=str(plan)))

    assert result.exit_code == 0, result.output
    assert "stratify_disabled" in result.output
    assert len(list((tmp_path / "runs").iterdir())) == 1


def test_dry_run_validates_without_writing(invoke_cli, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = invoke_cli(["run", CSV, "--plan", PLAN, "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "dry run" in result.output.lower()
    assert list(tmp_path.iterdir()) == []


def test_dry_run_with_out_dir_does_not_create_it(invoke_cli, tmp_path):
    result = invoke_cli(run_args(tmp_path, "--dry-run"))

    assert result.exit_code == 0
    assert not (tmp_path / "runs").exists()


def test_dry_run_reports_plan_errors(invoke_cli, tmp_path):
    result = invoke_cli(run_args(tmp_path, "--dry-run", plan=INVALID_PLAN))

    assert result.exit_code == 1
    assert "stateful_before_split" in result.output
    assert not (tmp_path / "runs").exists()


def test_invalid_plan_writes_no_artifacts(invoke_cli, tmp_path):
    result = invoke_cli(run_args(tmp_path, plan=INVALID_PLAN))

    assert result.exit_code == 1
    assert not (tmp_path / "runs").exists()


def test_plan_that_does_not_match_the_schema_fails_cleanly(invoke_cli, tmp_path):
    plan = tmp_path / "bad.json"
    plan.write_text('{"target": "churned", "steps": [{"op": "shell", "cmd": "rm"}]}')

    result = invoke_cli(run_args(tmp_path, plan=str(plan)))

    assert result.exit_code == 1
    assert "plan" in result.output.lower()
    assert not (tmp_path / "runs").exists()


def test_unreadable_plan_fails_cleanly(invoke_cli, tmp_path):
    result = invoke_cli(run_args(tmp_path, plan=str(tmp_path / "missing.json")))

    assert result.exit_code == 1
    assert "not found" in result.output
    assert not (tmp_path / "runs").exists()


def test_missing_csv_fails_cleanly(invoke_cli, tmp_path):
    result = invoke_cli(
        ["run", str(tmp_path / "missing.csv"), "--plan", PLAN, "--out-dir", str(tmp_path / "r")]
    )

    assert result.exit_code == 1
    assert "not found" in result.output


def test_target_missing_from_the_csv_fails_cleanly(invoke_cli, tmp_path):
    csv = tmp_path / "other.csv"
    csv.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    result = invoke_cli(["run", str(csv), "--plan", PLAN, "--out-dir", str(tmp_path / "r")])

    assert result.exit_code == 1
    assert "churned" in result.output
    assert not (tmp_path / "r").exists()
