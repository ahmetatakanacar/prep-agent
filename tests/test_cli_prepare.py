import copy
import json
from pathlib import Path

import pytest

from prep_agent.llm.base import LLMAPIError, LLMConfigError, TokenUsage
from prep_agent.llm.fake import FakeLLM

FIXTURES = Path(__file__).parent / "fixtures"
CSV = str(FIXTURES / "customers.csv")
GOAL = "prepare for churn prediction"
VALID_PLAN = json.loads((FIXTURES / "customers_plan.json").read_text(encoding="utf-8"))


def plan_reply(plan: dict) -> dict:
    return {"kind": "plan", "plan": plan, "questions": None}


def questions_reply(*questions: dict) -> dict:
    return {"kind": "questions", "plan": None, "questions": list(questions)}


def family_question(qid: str = "family") -> dict:
    return {
        "id": qid,
        "question": "Which model family will you use?",
        "options": ["tree", "linear", "any"],
        "recommended": "any",
        "reason": "scaling depends on it",
    }


def plan_with_impute_before_split() -> dict:
    plan = copy.deepcopy(VALID_PLAN)
    plan["steps"].insert(0, plan["steps"].pop(4))
    return plan


def use_llm(monkeypatch, responses, **kwargs) -> FakeLLM:
    llm = FakeLLM(responses, **kwargs)
    monkeypatch.setattr("prep_agent.cli.create_llm_client", lambda: llm)
    return llm


def prepare_args(tmp_path, *extra, csv=CSV):
    # The baseline has its own tests (test_cli_baseline.py); skipping it keeps these fast.
    return [
        "prepare",
        csv,
        "--goal",
        GOAL,
        "--out-dir",
        str(tmp_path / "runs"),
        "--no-baseline",
        *extra,
    ]


def only_run(tmp_path) -> Path:
    [run_dir] = list((tmp_path / "runs").iterdir())
    return run_dir


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


# --- success ---


def test_prepare_runs_the_plan_and_writes_the_run(invoke_cli, monkeypatch, tmp_path):
    llm = use_llm(monkeypatch, [plan_reply(VALID_PLAN)], usage=TokenUsage(1200, 340))

    result = invoke_cli(prepare_args(tmp_path, "--target", "churned", "--non-interactive"))

    assert result.exit_code == 0, result.output
    run_dir = only_run(tmp_path)
    assert {p.name for p in run_dir.iterdir()} == {
        "profile.json",
        "plan.json",
        "train.csv",
        "test.csv",
        "pipeline.joblib",
        "pipeline.py",
        "issues.json",
        "run.json",
        "llm_trace.json",
        "qa.json",
        "report.md",
    }
    info = read_json(run_dir / "run.json")
    assert info["status"] == "ok"
    assert info["goal"] == GOAL
    assert info["model"] == "fake"
    assert info["token_usage"] == {"input_tokens": 1200, "output_tokens": 340}
    assert len(llm.calls) == 1


def test_prepare_prints_the_plan_issues_tokens_and_run_folder(invoke_cli, monkeypatch, tmp_path):
    use_llm(monkeypatch, [plan_reply(VALID_PLAN)], usage=TokenUsage(1200, 340))

    result = invoke_cli(prepare_args(tmp_path, "--target", "churned", "--non-interactive"))

    for header in ("step", "op", "columns", "reason"):
        assert header in result.output
    for op in ("drop_columns", "dedupe", "drop_missing_target", "split", "impute", "encode"):
        assert op in result.output
    assert "customer_id, signup_date, country" in result.output
    assert "Rows without a label cannot be used" in result.output
    assert "1200" in result.output and "340" in result.output
    assert "1 LLM call" in result.output
    assert str(only_run(tmp_path)) in result.output.replace("\n", "")


def test_prepare_says_so_when_the_provider_reports_no_tokens(invoke_cli, monkeypatch, tmp_path):
    use_llm(monkeypatch, [plan_reply(VALID_PLAN)])

    result = invoke_cli(prepare_args(tmp_path, "--target", "churned", "--non-interactive"))

    assert result.exit_code == 0, result.output
    assert "not reported" in result.output


def test_warnings_are_printed_and_do_not_fail(invoke_cli, monkeypatch, tmp_path):
    plan = copy.deepcopy(VALID_PLAN)
    plan["steps"][3]["stratify"] = False
    use_llm(monkeypatch, [plan_reply(plan)])

    result = invoke_cli(prepare_args(tmp_path, "--target", "churned", "--non-interactive"))

    assert result.exit_code == 0, result.output
    assert "stratify_disabled" in result.output


def test_target_and_model_family_reach_the_planner(invoke_cli, monkeypatch, tmp_path):
    llm = use_llm(monkeypatch, [plan_reply({**VALID_PLAN, "model_family": "tree"})])

    invoke_cli(
        prepare_args(tmp_path, "--target", "churned", "--model-family", "tree", "--non-interactive")
    )

    [call] = llm.calls
    assert GOAL in call.user
    assert "Target column: churned" in call.user
    assert "Model family: tree" in call.user


def test_without_target_or_family_the_prompt_says_unspecified(invoke_cli, monkeypatch, tmp_path):
    llm = use_llm(monkeypatch, [plan_reply(VALID_PLAN)])

    invoke_cli(prepare_args(tmp_path, "--non-interactive"))

    assert "Target column: not specified" in llm.calls[0].user
    assert "Model family: not specified" in llm.calls[0].user


# --- questions ---


def test_non_interactive_answers_with_the_recommended_option(invoke_cli, monkeypatch, tmp_path):
    llm = use_llm(monkeypatch, [questions_reply(family_question()), plan_reply(VALID_PLAN)])

    result = invoke_cli(prepare_args(tmp_path, "--non-interactive"))

    assert result.exit_code == 0, result.output
    assert "Answer: any" in llm.calls[1].user
    assert "Which model family will you use?" in result.output
    [qa] = read_json(only_run(tmp_path) / "qa.json")
    assert qa["question"]["id"] == "family" and qa["answer"] == "any"


def test_interactive_questions_show_numbered_options_and_mark_the_recommended(
    invoke_cli, monkeypatch, tmp_path
):
    use_llm(monkeypatch, [questions_reply(family_question()), plan_reply(VALID_PLAN)])

    result = invoke_cli(prepare_args(tmp_path), input="\n")

    assert result.exit_code == 0, result.output
    assert "Which model family will you use?" in result.output
    assert "scaling depends on it" in result.output
    assert "1) tree" in result.output
    assert "2) linear" in result.output
    assert "3) any (recommended)" in result.output
    assert result.output.count("(recommended)") == 1


def test_enter_picks_the_recommended_option(invoke_cli, monkeypatch, tmp_path):
    llm = use_llm(monkeypatch, [questions_reply(family_question()), plan_reply(VALID_PLAN)])

    invoke_cli(prepare_args(tmp_path), input="\n")

    assert "Answer: any" in llm.calls[1].user


def test_a_number_picks_that_option(invoke_cli, monkeypatch, tmp_path):
    llm = use_llm(monkeypatch, [questions_reply(family_question()), plan_reply(VALID_PLAN)])

    invoke_cli(prepare_args(tmp_path), input="2\n")

    assert "Answer: linear" in llm.calls[1].user


def test_several_questions_are_asked_one_after_another(invoke_cli, monkeypatch, tmp_path):
    llm = use_llm(
        monkeypatch,
        [questions_reply(family_question("a"), family_question("b")), plan_reply(VALID_PLAN)],
    )

    invoke_cli(prepare_args(tmp_path), input="1\n2\n")

    user = llm.calls[1].user
    assert "[a] Which model family will you use?\n  Answer: tree" in user
    assert "[b] Which model family will you use?\n  Answer: linear" in user


def target_question(**overrides) -> dict:
    data = {
        "id": "target",
        "question": "Which column is the target?",
        "options": ["churned", "plan"],
        "recommended": "churned",
        "reason": "the goal does not name it",
        "allow_custom": True,
    }
    return {**data, **overrides}


def test_a_custom_option_is_offered_last_when_allowed(invoke_cli, monkeypatch, tmp_path):
    use_llm(monkeypatch, [questions_reply(target_question()), plan_reply(VALID_PLAN)])

    result = invoke_cli(prepare_args(tmp_path), input="\n")

    assert "1) churned (recommended)" in result.output
    assert "2) plan" in result.output
    assert "3) Other (type your own)" in result.output
    assert "Choose 1-3" in result.output


def test_no_custom_option_when_not_allowed(invoke_cli, monkeypatch, tmp_path):
    use_llm(monkeypatch, [questions_reply(family_question()), plan_reply(VALID_PLAN)])

    result = invoke_cli(prepare_args(tmp_path), input="\n")

    assert "Other (type your own)" not in result.output
    assert "Choose 1-3" in result.output


def test_choosing_the_custom_option_asks_for_text_and_keeps_it_verbatim(
    invoke_cli, monkeypatch, tmp_path
):
    llm = use_llm(monkeypatch, [questions_reply(target_question()), plan_reply(VALID_PLAN)])

    result = invoke_cli(prepare_args(tmp_path), input="3\nmonthly_spend\n")

    assert result.exit_code == 0, result.output
    assert "Answer: monthly_spend" in llm.calls[1].user
    [qa] = read_json(only_run(tmp_path) / "qa.json")
    assert qa["answer"] == "monthly_spend"
    assert qa["question"]["allow_custom"] is True


def test_an_empty_custom_answer_is_asked_again(invoke_cli, monkeypatch, tmp_path):
    llm = use_llm(monkeypatch, [questions_reply(target_question()), plan_reply(VALID_PLAN)])

    result = invoke_cli(prepare_args(tmp_path), input="3\n\n   \nmonthly_spend\n")

    assert result.exit_code == 0, result.output
    assert result.output.count("The answer must not be empty") == 2
    assert "Answer: monthly_spend" in llm.calls[1].user


def test_ending_the_input_while_typing_a_custom_answer_fails_cleanly(
    invoke_cli, monkeypatch, tmp_path
):
    use_llm(monkeypatch, [questions_reply(target_question())])

    result = invoke_cli(prepare_args(tmp_path), input="3\n")

    assert result.exit_code == 1
    assert "input ended" in result.output


def test_non_interactive_uses_the_recommended_option_even_if_custom_is_allowed(
    invoke_cli, monkeypatch, tmp_path
):
    llm = use_llm(monkeypatch, [questions_reply(target_question()), plan_reply(VALID_PLAN)])

    result = invoke_cli(prepare_args(tmp_path, "--non-interactive"))

    assert result.exit_code == 0, result.output
    assert "Answer: churned" in llm.calls[1].user


def test_invalid_input_is_asked_again(invoke_cli, monkeypatch, tmp_path):
    llm = use_llm(monkeypatch, [questions_reply(family_question()), plan_reply(VALID_PLAN)])

    result = invoke_cli(prepare_args(tmp_path), input="9\nfoo\n0\n2\n")

    assert result.exit_code == 0, result.output
    assert result.output.count("Enter a number from 1 to 3") == 3
    assert "Answer: linear" in llm.calls[1].user


def test_ending_the_input_while_asked_fails_cleanly(invoke_cli, monkeypatch, tmp_path):
    use_llm(monkeypatch, [questions_reply(family_question())])

    result = invoke_cli(prepare_args(tmp_path), input="")

    assert result.exit_code == 1
    assert "input ended" in result.output
    assert not (tmp_path / "runs").exists()


# --- failures ---


def test_a_plan_that_stays_invalid_fails_with_a_saved_failed_run(invoke_cli, monkeypatch, tmp_path):
    bad = plan_with_impute_before_split()
    llm = use_llm(monkeypatch, [plan_reply(bad)] * 2)

    result = invoke_cli(
        prepare_args(tmp_path, "--non-interactive", "--max-revisions", "1", "--target", "churned")
    )

    assert result.exit_code == 1
    assert len(llm.calls) == 2
    assert "stateful_before_split" in result.output
    run_dir = only_run(tmp_path)
    assert read_json(run_dir / "run.json")["status"] == "failed"
    assert not (run_dir / "train.csv").exists()
    trace = read_json(run_dir / "llm_trace.json")
    assert [c["purpose"] for c in trace["calls"]] == ["plan", "revise"]
    assert trace["calls"][1]["issue_codes"] == ["stateful_before_split"]
    issues = read_json(run_dir / "issues.json")
    assert "stateful_before_split" in [i["code"] for i in issues["plan_issues"]]
    assert str(run_dir) in result.output.replace("\n", "")


def test_max_revisions_zero_stops_at_the_first_error(invoke_cli, monkeypatch, tmp_path):
    llm = use_llm(monkeypatch, [plan_reply(plan_with_impute_before_split())])

    result = invoke_cli(prepare_args(tmp_path, "--non-interactive", "--max-revisions", "0"))

    assert result.exit_code == 1
    assert len(llm.calls) == 1


def test_a_result_error_that_stays_saves_the_frames_as_failed(invoke_cli, monkeypatch, tmp_path):
    rows = ["id,x,t"] + [f"{i},{(i // 2) % 2},{i % 2}" for i in range(40)]
    csv = tmp_path / "dups.csv"
    csv.write_text("\n".join(rows) + "\n", encoding="utf-8")
    steps = [
        {"op": "dedupe", "keep": "first", "reason": "r"},
        {"op": "drop_columns", "columns": ["id"], "reason": "r"},
        {"op": "split", "stratify": True, "reason": "r"},
    ]
    plan = {"target": "t", "task_type": "classification", "model_family": "any", "steps": steps}
    use_llm(monkeypatch, [plan_reply(plan)])

    result = invoke_cli(
        prepare_args(tmp_path, "--non-interactive", "--max-revisions", "0", csv=str(csv))
    )

    assert result.exit_code == 1
    assert "train_test_overlap" in result.output
    run_dir = only_run(tmp_path)
    assert read_json(run_dir / "run.json")["status"] == "failed"
    assert (run_dir / "train.csv").exists()
    assert (run_dir / "llm_trace.json").exists()


def test_an_llm_failure_after_a_plan_still_saves_the_trace(invoke_cli, monkeypatch, tmp_path):
    use_llm(monkeypatch, [plan_reply(plan_with_impute_before_split()), LLMAPIError("rate limited")])

    result = invoke_cli(prepare_args(tmp_path, "--non-interactive"))

    assert result.exit_code == 1
    assert "rate limited" in result.output
    run_dir = only_run(tmp_path)
    assert read_json(run_dir / "run.json")["status"] == "failed"
    assert len(read_json(run_dir / "llm_trace.json")["calls"]) == 1


def test_an_llm_failure_before_any_plan_saves_a_failed_run_without_a_plan(
    invoke_cli, monkeypatch, tmp_path
):
    use_llm(monkeypatch, [LLMAPIError("rate limited")])

    result = invoke_cli(prepare_args(tmp_path, "--non-interactive"))

    assert result.exit_code == 1
    assert "rate limited" in result.output
    run_dir = only_run(tmp_path)
    assert {p.name for p in run_dir.iterdir()} == {
        "profile.json",
        "run.json",
        "llm_trace.json",
        "qa.json",
        "report.md",
    }
    info = read_json(run_dir / "run.json")
    assert info["status"] == "failed"
    assert "rate limited" in info["error"]
    assert str(run_dir) in result.output.replace("\n", "")


def test_questions_past_the_limit_save_the_questions_in_a_failed_run(
    invoke_cli, monkeypatch, tmp_path
):
    # Two question rounds are allowed; the planner keeps asking, also after being told not to.
    replies = [questions_reply(family_question(qid)) for qid in "abcd"]
    use_llm(monkeypatch, replies)

    result = invoke_cli(prepare_args(tmp_path, "--non-interactive"))

    assert result.exit_code == 1
    run_dir = only_run(tmp_path)
    assert not (run_dir / "plan.json").exists()
    assert [q["question"]["id"] for q in read_json(run_dir / "qa.json")] == ["a", "b"]
    calls = read_json(run_dir / "llm_trace.json")["calls"]
    assert {c["response"]["kind"] for c in calls} == {"questions"}


def test_a_missing_api_key_fails_cleanly(invoke_cli, monkeypatch, tmp_path):
    def broken():
        raise LLMConfigError("OPENAI_API_KEY is not set")

    monkeypatch.setattr("prep_agent.cli.create_llm_client", broken)

    result = invoke_cli(prepare_args(tmp_path, "--non-interactive"))

    assert result.exit_code == 1
    assert "OPENAI_API_KEY" in result.output
    assert not (tmp_path / "runs").exists()


def test_a_target_that_is_not_a_column_fails_before_any_llm_call(invoke_cli, monkeypatch, tmp_path):
    llm = use_llm(monkeypatch, [])

    result = invoke_cli(prepare_args(tmp_path, "--target", "nope", "--non-interactive"))

    assert result.exit_code == 1
    assert "nope" in result.output
    assert llm.calls == []
    assert not (tmp_path / "runs").exists()


def test_a_missing_csv_fails_cleanly(invoke_cli, monkeypatch, tmp_path):
    use_llm(monkeypatch, [])

    result = invoke_cli(prepare_args(tmp_path, "--non-interactive", csv=str(tmp_path / "x.csv")))

    assert result.exit_code == 1
    assert "not found" in result.output


def test_an_unknown_model_family_is_a_usage_error(invoke_cli, monkeypatch, tmp_path):
    use_llm(monkeypatch, [])

    result = invoke_cli(prepare_args(tmp_path, "--model-family", "svm", "--non-interactive"))

    assert result.exit_code == 2
    assert not (tmp_path / "runs").exists()


def test_a_negative_revision_limit_is_a_usage_error(invoke_cli, monkeypatch, tmp_path):
    use_llm(monkeypatch, [])

    result = invoke_cli(prepare_args(tmp_path, "--max-revisions", "-1", "--non-interactive"))

    assert result.exit_code == 2


def test_goal_is_required(invoke_cli, tmp_path):
    result = invoke_cli(["prepare", CSV, "--out-dir", str(tmp_path / "runs")])

    assert result.exit_code == 2


@pytest.mark.parametrize("goal", ["", "   "])
def test_a_blank_goal_is_rejected(invoke_cli, monkeypatch, tmp_path, goal):
    use_llm(monkeypatch, [])

    result = invoke_cli(["prepare", CSV, "--goal", goal, "--out-dir", str(tmp_path / "runs")])

    assert result.exit_code == 1
    assert "goal" in result.output
