import dataclasses
from datetime import UTC, datetime
from pathlib import Path

import pytest

from prep_agent.agent import AgentTrace, LLMCallRecord
from prep_agent.baseline import (
    NOT_CONVERGED_NOTE,
    SMALL_TRAIN_ROWS,
    BaselineResult,
    BaselineScore,
    ModelBaseline,
)
from prep_agent.executor import execute
from prep_agent.llm.base import TokenUsage
from prep_agent.plan import Plan
from prep_agent.planner import PlannerResponse, QARecord, Question
from prep_agent.profiler import load_csv, profile_dataframe
from prep_agent.report import render_report
from prep_agent.runs import Counts, RunInfo
from prep_agent.validators import ValidationIssue

FIXTURES = Path(__file__).parent / "fixtures"
CUSTOMERS_CSV = FIXTURES / "customers.csv"
CUSTOMERS_PLAN = FIXTURES / "customers_plan.json"
SHA = "0123456789abcdef" * 4


@pytest.fixture(scope="module")
def executed():
    df = load_csv(CUSTOMERS_CSV)
    plan = Plan.model_validate_json(CUSTOMERS_PLAN.read_text(encoding="utf-8"))
    return plan, execute(df, plan, profile_dataframe(df, target=plan.target))


def make_info(**overrides) -> RunInfo:
    data = {
        "run_id": "r1",
        "created_at": datetime(2026, 10, 14, 9, 30, 12, tzinfo=UTC),
        "prep_agent_version": "0.1.0",
        "input_path": "data/customers.csv",
        "input_sha256": SHA,
        "target": "churned",
        "rows": Counts(before=20, train=16, test=4),
        "columns": Counts(before=7, train=6, test=6),
        "status": "ok",
        "pipeline_joblib": "note",
    }
    return RunInfo.model_validate({**data, **overrides})


DEFAULT_USAGE = TokenUsage(1200, 340)


def make_trace(*, usage: TokenUsage | None = DEFAULT_USAGE, qa=(), model="fake-model"):
    plan = Plan.model_validate_json(CUSTOMERS_PLAN.read_text(encoding="utf-8"))
    call = LLMCallRecord(
        index=0,
        purpose="plan",
        model=model,
        usage=usage,
        issue_codes=[],
        warning_codes=[],
        response_kind="plan",
        response=PlannerResponse(language="English", kind="plan", plan=plan),
    )
    return AgentTrace(goal="prepare for churn", calls=[call], qa_history=list(qa))


def make_baseline() -> BaselineResult:
    def scores(accuracy, f1):
        return {
            "accuracy": BaselineScore(mean=accuracy, std=0.05),
            "f1_macro": BaselineScore(mean=f1, std=0.0123),
        }

    return BaselineResult(
        task_type="classification",
        n_rows=16,
        random_state=42,
        n_folds=4,
        metrics=["accuracy", "f1_macro"],
        models=[
            ModelBaseline(name="dummy_most_frequent", scores=scores(0.6875, 0.4074)),
            ModelBaseline(name="logistic_regression", scores=scores(0.8125, 0.7)),
            ModelBaseline(name="random_forest", skipped_reason="RuntimeError: boom"),
        ],
    )


def render(executed, *, info=None, trace=None, baseline=None, **kwargs) -> str:
    plan, result = executed
    return render_report(
        info or make_info(),
        plan=kwargs.pop("plan", plan),
        result=kwargs.pop("result", result),
        plan_issues=kwargs.pop("plan_issues", []),
        trace=trace,
        baseline=baseline,
        **kwargs,
    )


def section(report: str, heading: str) -> str:
    """The text under `## heading`, up to the next `## `."""
    marker = f"## {heading}\n"
    assert marker in report, f"no section {heading!r}"
    body = report.split(marker, 1)[1]
    return body.split("\n## ", 1)[0]


def test_all_sections_are_there_for_a_full_run(executed):
    qa = QARecord(
        question=Question(
            id="family",
            question="Which model family?",
            options=["tree", "any"],
            recommended="any",
            reason="scaling depends on it",
        ),
        answer="any",
    )

    report = render(executed, trace=make_trace(qa=[qa]), baseline=make_baseline())

    headings = [line for line in report.splitlines() if line.startswith("#")]
    assert headings == [
        "# Preparation report",
        "## Data summary",
        "## Questions and answers",
        "## Plan",
        "## Validation",
        "## Baseline",
        "## Notes",
        "## Reproduce",
    ]


def test_the_header_names_goal_input_hash_date_model_and_tokens(executed):
    report = render(executed, trace=make_trace(), baseline=make_baseline())

    assert "**Goal:** prepare for churn" in report
    assert "customers.csv" in report
    assert "data/customers.csv" not in report
    assert "0123456789ab" in report
    assert SHA not in report
    assert "2026-10-14 09:30 UTC" in report
    assert "**Model:** fake-model" in report
    assert "1200 input / 340 output" in report


def test_a_run_without_an_llm_says_so(executed):
    report = render(executed, baseline=make_baseline())

    assert "**Goal:** not given" in report
    assert "no LLM was used" in report


def test_unreported_token_usage_is_stated(executed):
    report = render(executed, trace=make_trace(usage=None), baseline=make_baseline())

    assert "not reported" in report


def test_the_header_names_the_target_and_the_status(executed):
    report = render(executed, trace=make_trace(), baseline=make_baseline())

    assert "churned" in report.split("## Data summary")[0]
    assert "classification" in report.split("## Data summary")[0]
    assert "**Status:** ok" in report


# --- data summary ---


def test_data_summary_has_counts_and_dropped_columns(executed):
    summary = section(render(executed), "Data summary")

    assert "| Input | 20 | 7 |" in summary
    assert "| Train | 16 | 6 |" in summary
    assert "| Test | 4 | 6 |" in summary
    assert "| Prepared (train + test) | 20 | 6 |" in summary
    assert "customer_id, signup_date, country" in summary


def test_dropped_columns_say_none_when_no_step_drops_any(executed):
    plan, result = executed
    without_drop = Plan(
        target=plan.target,
        task_type=plan.task_type,
        model_family=plan.model_family,
        steps=[step for step in plan.steps if step.op != "drop_columns"],
    )

    summary = section(render(executed, plan=without_drop), "Data summary")

    assert "Dropped columns:** none" in summary


# --- questions ---


def test_questions_and_answers_are_listed(executed):
    qa = QARecord(
        question=Question(
            id="family",
            question="Which model family?",
            options=["tree", "any"],
            recommended="any",
            reason="scaling depends on it",
        ),
        answer="tree",
    )

    text = section(render(executed, trace=make_trace(qa=[qa])), "Questions and answers")

    assert "Which model family?" in text
    assert "tree" in text
    assert "scaling depends on it" in text


def test_no_questions_section_without_questions(executed):
    assert "## Questions and answers" not in render(executed, trace=make_trace())
    assert "## Questions and answers" not in render(executed)


# --- plan ---


def test_the_plan_table_has_every_step_with_columns_parameters_and_reason(executed):
    plan, _ = executed

    text = section(render(executed), "Plan")

    assert "| Step | Op | Columns | Parameters | Reason |" in text
    for index, step in enumerate(plan.steps):
        assert f"| {index} | {step.op} |" in text
        assert step.reason in text
    assert "customer_id, signup_date, country" in text
    assert "test_size=0.2, stratify=true, random_state=42" in text
    assert "strategy=median" in text
    assert "method=onehot" in text
    assert "method=robust" in text
    assert "keep=first" in text


def test_a_pipe_in_a_reason_does_not_break_the_table(executed):
    plan, result = executed
    steps = [step.model_copy(update={"reason": "a | b"}) for step in plan.steps]
    piped = plan.model_copy(update={"steps": steps})

    text = section(render(executed, plan=piped), "Plan")

    assert "a \\| b" in text
    for row in text.splitlines():
        if row.startswith("| ") and "a \\| b" in row:
            assert row.replace("\\|", "").count("|") == 6


# --- validation ---


def test_report_names_the_cache_counts_when_the_provider_reported_them(executed):
    usage = TokenUsage(1200, 340, cached_input_tokens=900, cache_write_tokens=100)

    report = render(executed, trace=make_trace(usage=usage))

    assert "1200 input (900 cached, 100 cache writes) / 340 output" in report


def test_no_issues_is_stated_explicitly():
    # The customers target has no missing values, so the fixture plan's drop_missing_target
    # step would be reported as pointless.
    df = load_csv(CUSTOMERS_CSV)
    plan = Plan.model_validate_json(CUSTOMERS_PLAN.read_text(encoding="utf-8"))
    plan = plan.model_copy(
        update={"steps": [s for s in plan.steps if s.op != "drop_missing_target"]}
    )
    clean = (plan, execute(df, plan, profile_dataframe(df, target=plan.target)))

    text = section(render(clean), "Validation")

    assert "Plan validation: no issues." in text
    assert "Result validation: no issues." in text


def test_issues_are_listed_with_severity_code_and_message(executed):
    plan, result = executed
    plan_issue = ValidationIssue(
        code="stratify_disabled", severity="warning", step_index=3, message="turn it on"
    )
    result_issue = ValidationIssue(
        code="nan_in_output", severity="warning", column="age", message="still missing"
    )
    with_issues = dataclasses.replace(
        result, plan_issues=[plan_issue], result_issues=[result_issue]
    )

    text = section(render(executed, result=with_issues), "Validation")

    assert "| warning | stratify_disabled |" in text
    assert "turn it on" in text
    assert "| warning | nan_in_output |" in text
    assert "still missing" in text
    assert "no issues" not in text


# --- baseline ---


def test_the_baseline_table_shows_mean_and_std_per_model_and_metric(executed):
    text = section(render(executed, baseline=make_baseline()), "Baseline")

    assert "| Model | accuracy ↑ | f1_macro ↑ |\n|---|---:|---:|\n| dummy_most_frequent" in text
    assert "| dummy_most_frequent | 0.6875 ± 0.0500 | 0.4074 ± 0.0123 |" in text
    assert "| logistic_regression | 0.8125 ± 0.0500 | 0.7000 ± 0.0123 |" in text
    assert "4-fold" in text
    assert "16 rows" in text


def test_a_model_that_did_not_converge_says_so_next_to_its_score(executed):
    baseline = make_baseline()
    baseline.models[1] = baseline.models[1].model_copy(update={"note": NOT_CONVERGED_NOTE})

    text = section(render(executed, baseline=baseline), "Baseline")

    assert "| logistic_regression (did not converge) | 0.8125 ± 0.0500 | 0.7000 ± 0.0123 |" in text
    assert "| dummy_most_frequent | 0.6875" in text
    assert "iteration limit" in text


def test_a_baseline_without_notes_has_no_convergence_remark(executed):
    text = section(render(executed, baseline=make_baseline()), "Baseline")

    assert "converge" not in text


def test_skipped_models_are_listed_with_their_reason(executed):
    text = section(render(executed, baseline=make_baseline()), "Baseline")

    assert "random_forest" in text.split("Skipped")[1]
    assert "RuntimeError: boom" in text


def make_regression_baseline(n_rows: int = 150) -> BaselineResult:
    return BaselineResult(
        task_type="regression",
        n_rows=n_rows,
        random_state=42,
        n_folds=5,
        metrics=["r2", "rmse", "mae"],
        models=[
            ModelBaseline(
                name="dummy_mean",
                scores={m: BaselineScore(mean=1.5, std=0.25) for m in ("r2", "rmse", "mae")},
            )
        ],
    )


def test_the_dummy_sentence_names_the_most_frequent_class_for_classification(executed):
    text = section(render(executed, baseline=make_baseline()), "Baseline")

    assert "always predicts the most frequent class" in text
    assert "training mean" not in text


def test_the_dummy_sentence_names_the_training_mean_for_regression(executed):
    text = section(render(executed, baseline=make_regression_baseline()), "Baseline")

    assert "always predicts the training mean" in text
    assert "most frequent class" not in text


def test_the_dummy_baseline_is_explained_as_the_reference(executed):
    text = section(render(executed, baseline=make_baseline()), "Baseline")

    assert "dummy" in text and "reference" in text


def test_a_skipped_baseline_gives_its_reason(executed):
    skipped = BaselineResult(
        task_type="classification",
        n_rows=16,
        random_state=42,
        n_folds=None,
        metrics=[],
        models=[],
        skipped_reason="the smallest class has only 1 sample(s)",
    )

    text = section(render(executed, baseline=skipped), "Baseline")

    assert "skipped" in text.lower()
    assert "the smallest class has only 1 sample(s)" in text
    assert "| Model |" not in text


def test_without_a_baseline_the_report_says_it_was_turned_off(executed):
    text = section(render(executed, baseline=None), "Baseline")

    assert "--no-baseline" in text
    assert "| Model |" not in text


def test_regression_metric_headers_show_which_direction_is_better(executed):
    baseline = BaselineResult(
        task_type="regression",
        n_rows=50,
        random_state=42,
        n_folds=5,
        metrics=["r2", "rmse", "mae"],
        models=[
            ModelBaseline(
                name="ridge",
                scores={m: BaselineScore(mean=1.5, std=0.25) for m in ("r2", "rmse", "mae")},
            )
        ],
    )

    text = section(render(executed, baseline=baseline), "Baseline")

    assert "| Model | r2 ↑ | rmse ↓ | mae ↓ |" in text
    assert "| ridge | 1.5000 ± 0.2500 | 1.5000 ± 0.2500 | 1.5000 ± 0.2500 |" in text
    assert "Higher is better" not in text
    assert "lower is better" not in text


# --- small data ---


def baseline_text(executed, *, n_rows, **kwargs) -> str:
    baseline = make_baseline().model_copy(update={"n_rows": n_rows})
    return section(render(executed, baseline=baseline, **kwargs), "Baseline")


def test_the_small_data_threshold_defaults_to_100():
    assert SMALL_TRAIN_ROWS == 100


def test_a_small_train_set_gets_a_warning_at_the_top_of_the_baseline_section(executed):
    text = baseline_text(executed, n_rows=SMALL_TRAIN_ROWS - 1)

    first = text.strip().splitlines()[0]
    assert "99 training rows" in first
    assert "not reliable" in first
    assert "between folds" in first


def test_a_train_set_at_the_threshold_gets_no_warning(executed):
    assert "not reliable" not in baseline_text(executed, n_rows=SMALL_TRAIN_ROWS)
    assert "not reliable" not in baseline_text(executed, n_rows=5000)


def test_the_threshold_can_be_changed(executed):
    assert "not reliable" in baseline_text(executed, n_rows=300, small_train_rows=500)
    assert "not reliable" not in baseline_text(executed, n_rows=16, small_train_rows=10)


def test_no_small_data_warning_without_scores(executed):
    skipped = make_baseline().model_copy(update={"skipped_reason": "too few rows", "models": []})

    assert "not reliable" not in section(render(executed, baseline=skipped), "Baseline")
    assert "not reliable" not in section(render(executed, baseline=None), "Baseline")


# --- notes ---


def test_the_notes_state_the_leakage_guarantees(executed):
    text = section(render(executed, baseline=make_baseline()), "Notes")

    assert "fit on the train set only" in text
    assert "test set was not touched" in text
    assert "cross-validation" in text and "train" in text


def test_the_baseline_note_is_left_out_when_there_is_no_baseline(executed):
    text = section(render(executed, baseline=None), "Notes")

    assert "fit on the train set only" in text
    assert "cross-validat" not in text


# --- failed runs ---


def test_a_failed_run_without_a_result_has_the_reason_and_no_baseline(executed):
    plan, _ = executed
    error = ValidationIssue(
        code="stateful_before_split", severity="error", step_index=4, message="move it"
    )
    info = make_info(
        status="failed",
        error="the plan still has 1 error(s) after 2 revision(s)",
        rows=Counts(before=20),
        columns=Counts(before=7),
        pipeline_joblib=None,
    )

    report = render(
        executed,
        info=info,
        result=None,
        plan_issues=[error],
        trace=make_trace(),
        baseline=make_baseline(),
    )

    assert "**Status:** failed" in report
    assert "the plan still has 1 error(s) after 2 revision(s)" in section(report, "Failure")
    assert "## Baseline" not in report
    assert "## Notes" not in report
    assert "| error | stateful_before_split |" in section(report, "Validation")
    assert "not run" in section(report, "Validation")
    assert "| Input | 20 | 7 |" in section(report, "Data summary")
    assert "| Train" not in report


def test_a_failed_run_with_result_errors_names_the_codes(executed):
    plan, result = executed
    error = ValidationIssue(code="target_altered", severity="error", message="target changed")
    failed = dataclasses.replace(result, result_issues=[error])

    report = render(executed, info=make_info(status="failed"), result=failed)

    assert "target_altered" in section(report, "Failure")
    assert "## Baseline" not in report
    assert "| Train | 16 | 6 |" in report


def test_a_run_that_never_got_a_plan_still_renders():
    info = make_info(
        status="failed",
        target=None,
        error="the LLM call failed: rate limited",
        rows=Counts(before=20),
        columns=Counts(before=7),
        pipeline_joblib=None,
    )

    report = render_report(
        info, plan=None, result=None, plan_issues=[], trace=make_trace(), baseline=None
    )

    assert "the LLM call failed: rate limited" in section(report, "Failure")
    assert "## Plan" not in report
    assert "no plan" in section(report, "Validation").lower()
    assert "## Baseline" not in report


def test_a_trace_without_calls_has_no_model(executed):
    trace = AgentTrace(goal="g", calls=[], qa_history=[])

    report = render(executed, trace=trace)

    assert "**Model:** none" in report
    assert "**Goal:** g" in report


def test_rendering_is_deterministic(executed):
    first = render(executed, trace=make_trace(), baseline=make_baseline())
    second = render(executed, trace=make_trace(), baseline=make_baseline())

    assert first == second
    assert first.endswith("\n") and not first.endswith("\n\n")
    assert "\r" not in first


def test_reproduce_explains_how_to_run_pipeline_py(executed):
    report = render(executed, baseline=make_baseline())

    text = section(report, "Reproduce")
    assert "pipeline.py" in text
    assert "python pipeline.py <input.csv> --out-dir <folder>" in text
    assert "train.csv" in text and "test.csv" in text
    assert "pandas and scikit-learn" in text
    assert "customers.csv" in text
    assert SHA[:12] in text


def test_a_failed_run_has_no_reproduce_section(executed):
    report = render(executed, info=make_info(status="failed", error="boom"))

    assert "## Reproduce" not in report
    assert "pipeline.py" not in report


def test_a_run_without_a_plan_has_no_reproduce_section(executed):
    report = render(executed, info=make_info(status="failed", error="boom"), plan=None, result=None)

    assert "## Reproduce" not in report


LANGUAGE_NOTE = (
    "Text the user wrote (the goal and answers) and the planner's questions and reasons are kept "
    "in the language they were written in; everything else in this report is in English."
)


def test_the_report_says_which_texts_keep_their_language_right_under_the_header_list(executed):
    with_trace = render(executed, trace=make_trace(), baseline=make_baseline())
    without_llm = render(executed, baseline=make_baseline())
    failed = render(executed, info=make_info(status="failed"), result=None, plan=None)

    for report in (with_trace, without_llm, failed):
        assert report.count(LANGUAGE_NOTE) == 1
        status_line = next(line for line in report.splitlines() if line.startswith("- **Status:**"))
        lines = report.splitlines()
        assert lines[lines.index(status_line) + 1 :][:2] == ["", LANGUAGE_NOTE]
