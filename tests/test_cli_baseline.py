import json
from pathlib import Path

from prep_agent.baseline import BaselineResult
from prep_agent.llm.base import TokenUsage
from prep_agent.llm.fake import FakeLLM

FIXTURES = Path(__file__).parent / "fixtures"
CSV = str(FIXTURES / "customers.csv")
PLAN = str(FIXTURES / "customers_plan.json")
VALID_PLAN = json.loads((FIXTURES / "customers_plan.json").read_text(encoding="utf-8"))
MODELS = ("dummy_most_frequent", "logistic_regression", "random_forest")


def run_args(tmp_path, *extra, csv=CSV, plan=PLAN):
    return ["run", csv, "--plan", plan, "--out-dir", str(tmp_path / "runs"), *extra]


def prepare_args(tmp_path, *extra, csv=CSV):
    return [
        "prepare",
        csv,
        "--goal",
        "prepare for churn prediction",
        "--out-dir",
        str(tmp_path / "runs"),
        "--non-interactive",
        *extra,
    ]


def use_llm(monkeypatch, plan) -> FakeLLM:
    llm = FakeLLM([{"kind": "plan", "plan": plan, "questions": None}], usage=TokenUsage(10, 5))
    monkeypatch.setattr("prep_agent.cli.create_llm_client", lambda: llm)
    return llm


def only_run(tmp_path) -> Path:
    [run_dir] = list((tmp_path / "runs").iterdir())
    return run_dir


def write_csv_and_plan(tmp_path, *, classes: list[int], stratify=False):
    csv = tmp_path / "small.csv"
    rows = ["x,t"] + [f"{i},{c}" for i, c in enumerate(classes)]
    csv.write_text("\n".join(rows) + "\n", encoding="utf-8")
    steps = [{"op": "split", "stratify": stratify, "reason": "r"}]
    plan = {"target": "t", "task_type": "classification", "model_family": "any", "steps": steps}
    plan_path = tmp_path / "small_plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    return str(csv), str(plan_path), plan


# --- run ---


def test_run_writes_report_and_baseline(invoke_cli, tmp_path):
    result = invoke_cli(run_args(tmp_path))

    assert result.exit_code == 0, result.output
    run_dir = only_run(tmp_path)
    baseline = BaselineResult.model_validate_json(
        (run_dir / "baseline.json").read_text(encoding="utf-8")
    )
    assert [model.name for model in baseline.models] == list(MODELS)
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "## Baseline" in report
    assert "logistic_regression" in report


def test_run_prints_the_baselines_as_a_table(invoke_cli, tmp_path):
    result = invoke_cli(run_args(tmp_path))

    for name in MODELS:
        assert name in result.output
    for header in ("model", "accuracy", "f1_macro", "roc_auc"):
        assert header in result.output
    assert "±" in result.output
    assert "fold" in result.output


def test_run_with_no_baseline_writes_no_baseline_json(invoke_cli, tmp_path):
    result = invoke_cli(run_args(tmp_path, "--no-baseline"))

    assert result.exit_code == 0, result.output
    run_dir = only_run(tmp_path)
    assert not (run_dir / "baseline.json").exists()
    assert "--no-baseline" in (run_dir / "report.md").read_text(encoding="utf-8")
    assert "dummy_most_frequent" not in result.output
    assert "Run folder" in result.output


def test_dry_run_computes_no_baseline(invoke_cli, tmp_path):
    result = invoke_cli(run_args(tmp_path, "--dry-run"))

    assert result.exit_code == 0
    assert "dummy_most_frequent" not in result.output
    assert not (tmp_path / "runs").exists()


def test_a_run_with_result_errors_writes_a_report_without_a_baseline(invoke_cli, tmp_path):
    csv = tmp_path / "dups.csv"
    rows = ["id,x,t"] + [f"{i},{(i // 2) % 2},{i % 2}" for i in range(40)]
    csv.write_text("\n".join(rows) + "\n", encoding="utf-8")
    steps = [
        {"op": "dedupe", "keep": "first", "reason": "r"},
        {"op": "drop_columns", "columns": ["id"], "reason": "r"},
        {"op": "split", "stratify": True, "reason": "r"},
    ]
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {"target": "t", "task_type": "classification", "model_family": "any", "steps": steps}
        ),
        encoding="utf-8",
    )

    result = invoke_cli(run_args(tmp_path, csv=str(csv), plan=str(plan)))

    assert result.exit_code == 1
    run_dir = only_run(tmp_path)
    assert not (run_dir / "baseline.json").exists()
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "**Status:** failed" in report
    assert "train_test_overlap" in report
    assert "## Baseline" not in report
    assert "dummy_most_frequent" not in result.output


def test_a_baseline_that_cannot_run_is_reported_not_fatal(invoke_cli, tmp_path):
    csv, plan, _ = write_csv_and_plan(tmp_path, classes=[1] + [0] * 39)

    result = invoke_cli(run_args(tmp_path, csv=csv, plan=plan))

    assert result.exit_code == 0, result.output
    assert "skipped" in result.output.lower()
    run_dir = only_run(tmp_path)
    baseline = json.loads((run_dir / "baseline.json").read_text(encoding="utf-8"))
    assert baseline["skipped_reason"]
    assert baseline["models"] == []
    assert baseline["skipped_reason"] in (run_dir / "report.md").read_text(encoding="utf-8")


def test_models_that_cannot_run_are_listed_as_skipped(invoke_cli, tmp_path):
    csv = tmp_path / "text.csv"
    rows = ["x,city,t"] + [f"{i},{'ab'[i % 2]},{i % 2}" for i in range(40)]
    csv.write_text("\n".join(rows) + "\n", encoding="utf-8")
    steps = [{"op": "split", "stratify": True, "reason": "r"}]
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {"target": "t", "task_type": "classification", "model_family": "any", "steps": steps}
        ),
        encoding="utf-8",
    )

    result = invoke_cli(run_args(tmp_path, csv=str(csv), plan=str(plan)))

    assert result.exit_code == 0, result.output
    assert "dummy_most_frequent" in result.output
    assert "logistic_regression: " in result.output
    assert "non-numeric columns: city" in result.output


# --- prepare ---


def test_prepare_writes_report_and_baseline(invoke_cli, monkeypatch, tmp_path):
    use_llm(monkeypatch, VALID_PLAN)

    result = invoke_cli(prepare_args(tmp_path, "--target", "churned"))

    assert result.exit_code == 0, result.output
    run_dir = only_run(tmp_path)
    assert (run_dir / "baseline.json").exists()
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "prepare for churn prediction" in report
    assert "**Model:** fake" in report
    assert "10 input / 5 output" in report
    assert "## Baseline" in report
    for name in MODELS:
        assert name in result.output


def test_prepare_with_no_baseline_writes_no_baseline_json(invoke_cli, monkeypatch, tmp_path):
    use_llm(monkeypatch, VALID_PLAN)

    result = invoke_cli(prepare_args(tmp_path, "--no-baseline", "--target", "churned"))

    assert result.exit_code == 0, result.output
    run_dir = only_run(tmp_path)
    assert not (run_dir / "baseline.json").exists()
    assert "--no-baseline" in (run_dir / "report.md").read_text(encoding="utf-8")
    assert "dummy_most_frequent" not in result.output


def test_a_failed_prepare_run_writes_a_report_with_the_reason(invoke_cli, monkeypatch, tmp_path):
    bad = json.loads(json.dumps(VALID_PLAN))
    bad["steps"].insert(0, bad["steps"].pop(4))
    llm = FakeLLM([{"kind": "plan", "plan": bad, "questions": None}])
    monkeypatch.setattr("prep_agent.cli.create_llm_client", lambda: llm)

    result = invoke_cli(prepare_args(tmp_path, "--max-revisions", "0"))

    assert result.exit_code == 1
    run_dir = only_run(tmp_path)
    assert not (run_dir / "baseline.json").exists()
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "**Status:** failed" in report
    assert "stateful_before_split" in report
    assert "## Baseline" not in report


def test_baseline_results_are_the_same_on_a_second_run(invoke_cli, tmp_path):
    invoke_cli(run_args(tmp_path))
    invoke_cli(run_args(tmp_path, "--out-dir", str(tmp_path / "again")))

    [first] = (tmp_path / "runs").iterdir()
    [second] = (tmp_path / "again").iterdir()
    assert (first / "baseline.json").read_text(encoding="utf-8") == (
        second / "baseline.json"
    ).read_text(encoding="utf-8")
