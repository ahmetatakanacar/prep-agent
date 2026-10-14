import json
from pathlib import Path

import numpy as np
import pandas as pd

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
    for header in ("model", "accuracy ↑", "f1_macro ↑", "roc_auc ↑"):
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


def test_a_run_with_result_errors_writes_a_report_without_a_baseline(
    invoke_cli, tmp_path, inject_result_error
):
    inject_result_error()
    csv = tmp_path / "small.csv"
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
    assert "target_altered" in report
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


# --- regression, end to end ---


def baseline_section(report: str) -> str:
    return report.split("## Baseline\n", 1)[1].split("\n## ", 1)[0]


def run_regression(invoke_cli, tmp_path, n_rows: int):
    rng = np.random.default_rng(7)
    x1 = rng.normal(size=n_rows)
    x2 = rng.normal(size=n_rows)
    x2[::10] = np.nan
    price = 3 * x1 - 2 * np.nan_to_num(x2) + rng.normal(scale=0.5, size=n_rows)
    csv = tmp_path / "prices.csv"
    pd.DataFrame({"x1": x1, "x2": x2, "price": price}).to_csv(csv, index=False)
    steps = [
        {"op": "split", "stratify": False, "reason": "hold out a test set"},
        {"op": "impute", "columns": ["x2"], "strategy": "median", "reason": "x2 has gaps"},
        {"op": "scale", "columns": ["x1", "x2"], "method": "standard", "reason": "linear model"},
    ]
    plan = tmp_path / "prices_plan.json"
    plan.write_text(
        json.dumps(
            {
                "target": "price",
                "task_type": "regression",
                "model_family": "linear",
                "steps": steps,
            }
        ),
        encoding="utf-8",
    )
    result = invoke_cli(run_args(tmp_path, csv=str(csv), plan=str(plan)))
    assert result.exit_code == 0, result.output
    return result, only_run(tmp_path)


def test_a_small_regression_run_reports_directions_dummy_and_small_data(invoke_cli, tmp_path):
    result, run_dir = run_regression(invoke_cli, tmp_path, n_rows=60)

    text = baseline_section((run_dir / "report.md").read_text(encoding="utf-8"))
    assert "| Model | r2 ↑ | rmse ↓ | mae ↓ |" in text
    for name in ("dummy_mean", "ridge", "random_forest"):
        assert f"| {name} |" in text
    assert "always predicts the training mean" in text
    assert "most frequent class" not in text
    assert "Higher is better" not in text
    assert "> **Small data:** only 48 training rows" in text
    for header in ("r2 ↑", "rmse ↓", "mae ↓"):
        assert header in result.output
    baseline = json.loads((run_dir / "baseline.json").read_text(encoding="utf-8"))
    assert baseline["task_type"] == "regression"
    assert baseline["n_rows"] == 48


def test_the_ridge_beats_the_dummy_in_the_regression_report(invoke_cli, tmp_path):
    _, run_dir = run_regression(invoke_cli, tmp_path, n_rows=60)

    baseline = BaselineResult.model_validate_json(
        (run_dir / "baseline.json").read_text(encoding="utf-8")
    )
    scores = {model.name: model.scores for model in baseline.models}
    assert scores["ridge"]["r2"].mean > 0.8
    assert scores["ridge"]["rmse"].mean < scores["dummy_mean"]["rmse"].mean


def test_a_regression_run_with_enough_rows_has_no_small_data_note(invoke_cli, tmp_path):
    _, run_dir = run_regression(invoke_cli, tmp_path, n_rows=150)

    text = baseline_section((run_dir / "report.md").read_text(encoding="utf-8"))
    assert "Small data" not in text
    assert "| Model | r2 ↑ | rmse ↓ | mae ↓ |" in text
