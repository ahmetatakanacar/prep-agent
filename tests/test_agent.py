import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from prep_agent.agent import AgentError, RevisionBudgetError, run_agent
from prep_agent.llm.base import LLMAPIError, TokenUsage
from prep_agent.llm.fake import FakeLLM
from prep_agent.plan import Plan
from prep_agent.planner import PlannerResponse, build_system_prompt
from prep_agent.profiler import ProfilerError, load_csv

FIXTURES = Path(__file__).parent / "fixtures"
GOAL = "prepare for churn prediction"

VALID_PLAN = json.loads((FIXTURES / "customers_plan.json").read_text(encoding="utf-8"))
STEP_IMPUTE = 4


def plan_reply(plan: dict) -> dict:
    return {"kind": "plan", "plan": plan, "questions": None}


def questions_reply(*questions: dict) -> dict:
    return {"kind": "questions", "plan": None, "questions": list(questions)}


def question(qid: str = "family", **overrides) -> dict:
    data = {
        "id": qid,
        "question": f"Question {qid}?",
        "options": ["tree", "linear", "any"],
        "recommended": "any",
        "reason": "it changes the plan",
    }
    return {**data, **overrides}


def plan_with_impute_before_split() -> dict:
    plan = copy.deepcopy(VALID_PLAN)
    impute = plan["steps"].pop(STEP_IMPUTE)
    plan["steps"].insert(0, impute)
    return plan


def plan_with_target(target: str) -> dict:
    return {**copy.deepcopy(VALID_PLAN), "target": target}


class Asker:
    """A scripted ask_fn: records what it was asked and answers from a mapping or the default."""

    def __init__(self, answers: dict[str, str] | None = None) -> None:
        self.answers = answers or {}
        self.asked: list[list] = []

    def __call__(self, questions) -> dict[str, str]:
        self.asked.append(list(questions))
        return {q.id: self.answers.get(q.id, q.recommended) for q in questions}


@pytest.fixture
def df() -> pd.DataFrame:
    return load_csv(FIXTURES / "customers.csv")


# dedupe runs while `id` still tells the rows apart; dropping it afterwards recreates duplicates
# that the split spreads over train and test, which is an error only when the plan has a dedupe
# step. Leaving the dedupe out turns it into a warning, so that is the fix.
@pytest.fixture
def dups(tmp_path) -> pd.DataFrame:
    rows = ["id,x,t"] + [f"{i},{(i // 2) % 2},{i % 2}" for i in range(40)]
    csv = tmp_path / "dups.csv"
    csv.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return load_csv(csv)


def dups_plan(*ops: str) -> dict:
    steps = {
        "dedupe": {"op": "dedupe", "keep": "first", "reason": "r"},
        "drop_id": {"op": "drop_columns", "columns": ["id"], "reason": "r"},
        "split": {"op": "split", "stratify": True, "reason": "r"},
        "impute": {"op": "impute", "columns": ["x"], "strategy": "mean", "reason": "r"},
    }
    return {
        "target": "t",
        "task_type": "classification",
        "model_family": "any",
        "steps": [steps[op] for op in ops],
    }


RESULT_ERROR_PLAN = dups_plan("dedupe", "drop_id", "split")
FIXED_PLAN = dups_plan("drop_id", "split")
PLAN_ERROR_PLAN = dups_plan("impute", "split")


def run(df, responses, *, usage=None, ask=None, **kwargs):
    llm = FakeLLM(responses, usage=usage)
    ask = ask or Asker()
    kwargs.setdefault("target", "churned")
    result = run_agent(df, GOAL, llm, ask, **kwargs)
    return result, llm, ask


# --- the plain path ---


def test_a_valid_plan_runs_with_a_single_llm_call(df):
    result, llm, ask = run(df, [plan_reply(VALID_PLAN)])

    assert len(llm.calls) == 1
    assert ask.asked == []
    assert result.plan == Plan.model_validate(VALID_PLAN)
    assert len(result.execution.train) == 16 and len(result.execution.test) == 4
    assert [c.purpose for c in result.calls] == ["plan"]
    assert result.calls[0].issue_codes == []
    assert result.qa_history == []


def test_the_llm_gets_the_system_prompt_and_the_planner_schema(df):
    _, llm, _ = run(df, [plan_reply(VALID_PLAN)])

    [call] = llm.calls
    assert call.system == build_system_prompt()
    assert call.schema is PlannerResponse
    assert GOAL in call.user


def test_result_issues_are_the_final_plan_and_result_issues(df):
    result, _, _ = run(df, [plan_reply(VALID_PLAN)])

    assert result.issues == result.execution.plan_issues + result.execution.result_issues


def test_warnings_do_not_trigger_a_revision(df):
    plan = copy.deepcopy(VALID_PLAN)
    plan["steps"][3]["stratify"] = False

    result, llm, _ = run(df, [plan_reply(plan)])

    assert len(llm.calls) == 1
    assert "stratify_disabled" in [i.code for i in result.issues]


def test_unknown_target_is_reported_before_any_llm_call(df):
    llm = FakeLLM([])

    with pytest.raises(ProfilerError, match="nope"):
        run_agent(df, GOAL, llm, Asker(), target="nope")

    assert llm.calls == []


# --- questions ---


def test_questions_are_asked_and_the_answers_reach_the_next_prompt(df):
    ask = Asker({"family": "tree", "drop_age": "keep it"})
    asked_first = [
        question("family"),
        question(
            "drop_age",
            question="Drop the sparse age column?",
            options=["drop it", "keep it"],
            recommended="drop it",
        ),
    ]

    result, llm, _ = run(df, [questions_reply(*asked_first), plan_reply(VALID_PLAN)], ask=ask)

    assert len(ask.asked) == 1
    assert [q.id for q in ask.asked[0]] == ["family", "drop_age"]
    first, second = llm.calls
    assert "Answer:" not in first.user
    assert "[family] Question family?" in second.user
    assert "Answer: tree" in second.user
    assert "[drop_age] Drop the sparse age column?" in second.user
    assert "Answer: keep it" in second.user
    assert [(r.question.id, r.answer) for r in result.qa_history] == [
        ("family", "tree"),
        ("drop_age", "keep it"),
    ]
    assert [c.purpose for c in result.calls] == ["plan", "plan"]


def test_question_rounds_are_limited_and_the_last_call_may_not_ask(df):
    responses = [questions_reply(question("a")), plan_reply(VALID_PLAN)]

    _, llm, ask = run(df, responses, max_question_rounds=1)

    first, last = llm.calls
    assert "Do not ask any more questions" not in first.user
    assert "Do not ask any more questions" in last.user
    assert len(ask.asked) == 1


def test_two_rounds_of_questions_then_a_plan(df):
    responses = [
        questions_reply(question("a")),
        questions_reply(question("b")),
        plan_reply(VALID_PLAN),
    ]

    result, llm, ask = run(df, responses, max_question_rounds=2)

    assert len(ask.asked) == 2
    assert "2 round(s) left" in llm.calls[0].user
    assert "1 round(s) left" in llm.calls[1].user
    assert "Do not ask any more questions" in llm.calls[2].user
    assert [r.question.id for r in result.qa_history] == ["a", "b"]


def test_zero_question_rounds_never_ask(df):
    _, llm, ask = run(df, [plan_reply(VALID_PLAN)], max_question_rounds=0)

    assert "Do not ask any more questions" in llm.calls[0].user
    assert ask.asked == []


def test_questions_after_the_limit_are_an_error_not_a_loop(df):
    responses = [questions_reply(question("a")), questions_reply(question("b"))]

    with pytest.raises(AgentError, match="question") as caught:
        run(df, responses, max_question_rounds=1)

    assert len(caught.value.trace.calls) == 2
    assert caught.value.plan is None


def test_answers_that_do_not_cover_the_questions_are_rejected(df):
    def incomplete(questions):
        return {}

    with pytest.raises(AgentError, match="family"):
        run(df, [questions_reply(question("family"))], ask=incomplete)


def test_an_answer_outside_the_options_is_rejected(df):
    with pytest.raises(AgentError, match="not one of the options"):
        run(df, [questions_reply(question("family"))], ask=Asker({"family": "gradient boosting"}))


def test_a_planner_that_has_to_ask_for_the_target_gets_the_profile_of_its_choice(df):
    ask = Asker({"target": "churned"})
    responses = [
        questions_reply(
            question("target", options=["churned", "plan"], recommended="churned"),
        ),
        plan_reply(VALID_PLAN),
    ]

    result, llm, _ = run(df, responses, ask=ask, target=None)

    assert "Target column: not specified" in llm.calls[0].user
    assert result.profile.target == "churned"
    assert result.profile.target_task_hint == "classification"


# --- revisions ---


def test_a_plan_with_errors_is_revised_with_the_issue_codes(df):
    responses = [plan_reply(plan_with_impute_before_split()), plan_reply(VALID_PLAN)]

    result, llm, _ = run(df, responses)

    first, second = llm.calls
    assert "stateful_before_split" not in first.user
    assert "code=stateful_before_split step=0 column=none:" in second.user
    assert "Previous plan, rejected" in second.user
    assert [c.purpose for c in result.calls] == ["plan", "revise"]
    assert result.calls[0].issue_codes == []
    assert result.calls[1].issue_codes == ["stateful_before_split"]
    assert result.plan == Plan.model_validate(VALID_PLAN)


def test_a_different_model_family_than_the_users_is_revised(df):
    tree_plan = {**copy.deepcopy(VALID_PLAN), "model_family": "tree"}
    responses = [plan_reply(VALID_PLAN), plan_reply(tree_plan)]

    result, llm, _ = run(df, responses, model_family="tree")

    assert "model_family_mismatch" in llm.calls[1].user
    assert result.calls[1].issue_codes == ["model_family_mismatch"]
    assert result.plan.model_family == "tree"


def test_any_model_family_is_not_enforced_when_the_user_gave_none(df):
    plan = {**copy.deepcopy(VALID_PLAN), "model_family": "linear"}

    _, llm, _ = run(df, [plan_reply(plan)])

    assert len(llm.calls) == 1


def test_only_errors_are_sent_for_revision(df):
    plan = plan_with_impute_before_split()
    split = next(step for step in plan["steps"] if step["op"] == "split")
    split["stratify"] = False

    _, llm, _ = run(df, [plan_reply(plan), plan_reply(VALID_PLAN)])

    assert "stateful_before_split" in llm.calls[1].user
    assert "stratify_disabled" not in llm.calls[1].user


def test_a_different_target_than_the_users_is_revised(df):
    responses = [plan_reply(plan_with_target("plan")), plan_reply(VALID_PLAN)]

    result, llm, _ = run(df, responses, target="churned")

    assert "target_mismatch" in llm.calls[1].user
    assert result.plan.target == "churned"
    assert "target_mismatch" in result.calls[1].issue_codes


def test_a_result_error_is_revised_too(dups):
    result, llm, _ = run(dups, [plan_reply(RESULT_ERROR_PLAN), plan_reply(FIXED_PLAN)], target="t")

    assert "train_test_overlap" in llm.calls[1].user
    assert [c.purpose for c in result.calls] == ["plan", "revise"]
    assert result.calls[1].issue_codes == ["train_test_overlap"]
    assert result.plan == Plan.model_validate(FIXED_PLAN)


def test_a_revision_budget_that_runs_out_raises_with_plan_and_issues(df):
    bad = plan_with_impute_before_split()

    with pytest.raises(RevisionBudgetError) as caught:
        run(df, [plan_reply(bad), plan_reply(bad), plan_reply(bad)], max_revisions=2)

    error = caught.value
    assert len(error.trace.calls) == 3
    assert error.plan == Plan.model_validate(bad)
    assert "stateful_before_split" in [i.code for i in error.issues]
    assert error.execution is None
    assert "stateful_before_split" in str(error)
    assert "2 revision(s)" in str(error)


def test_zero_revisions_means_the_first_error_is_final(df):
    with pytest.raises(RevisionBudgetError) as caught:
        run(df, [plan_reply(plan_with_impute_before_split())], max_revisions=0)

    assert len(caught.value.trace.calls) == 1


def test_plan_and_result_revisions_share_one_budget(dups):
    # One revision is allowed: the first plan has a plan error, its fix has a result error.
    responses = [plan_reply(PLAN_ERROR_PLAN), plan_reply(RESULT_ERROR_PLAN)]

    with pytest.raises(RevisionBudgetError) as caught:
        run(dups, responses, target="t", max_revisions=1)

    error = caught.value
    assert [c.purpose for c in error.trace.calls] == ["plan", "revise"]
    assert error.execution is not None
    assert "train_test_overlap" in [i.code for i in error.issues]


def test_a_budget_failure_on_the_result_keeps_the_execution(dups):
    with pytest.raises(RevisionBudgetError) as caught:
        run(dups, [plan_reply(RESULT_ERROR_PLAN)], target="t", max_revisions=0)

    assert caught.value.execution is not None
    assert len(caught.value.execution.train) > 0
    assert caught.value.profile.n_rows == 40


# --- failures of the LLM layer ---


def test_an_llm_failure_becomes_an_agent_error_that_keeps_the_trace(df):
    responses = [plan_reply(plan_with_impute_before_split()), LLMAPIError("rate limited")]

    with pytest.raises(AgentError, match="rate limited") as caught:
        run(df, responses)

    error = caught.value
    assert len(error.trace.calls) == 1
    assert error.plan is not None
    assert "stateful_before_split" in [i.code for i in error.issues]


def test_an_llm_failure_on_the_first_call_has_no_plan(df):
    with pytest.raises(AgentError, match="rate limited") as caught:
        run(df, [LLMAPIError("rate limited")])

    assert caught.value.plan is None
    assert caught.value.trace.calls == []


# --- records ---


def test_token_usage_is_summed_over_all_calls(df):
    responses = [
        questions_reply(question("a")),
        plan_reply(plan_with_impute_before_split()),
        plan_reply(VALID_PLAN),
    ]

    result, _, _ = run(df, responses, usage=TokenUsage(100, 10))

    assert [c.usage for c in result.calls] == [TokenUsage(100, 10)] * 3
    assert result.total_usage == TokenUsage(300, 30)


def test_total_usage_is_none_when_the_provider_reports_none(df):
    result, _, _ = run(df, [plan_reply(VALID_PLAN)])

    assert result.total_usage is None


def test_call_records_hold_the_model_and_the_planner_response_only(df):
    result, _, _ = run(df, [plan_reply(VALID_PLAN)])

    [record] = result.calls
    assert record.model == "fake"
    assert result.model == "fake"
    assert record.response == PlannerResponse.model_validate(plan_reply(VALID_PLAN))
    assert set(json.loads(record.model_dump_json())) == {
        "index",
        "purpose",
        "model",
        "usage",
        "issue_codes",
        "response",
    }


def test_the_input_frame_is_not_modified(df):
    before = df.copy()

    run(df, [plan_reply(VALID_PLAN)])

    pd.testing.assert_frame_equal(df, before)


@pytest.mark.parametrize("kwargs", [{"max_revisions": -1}, {"max_question_rounds": -1}])
def test_negative_limits_are_rejected(df, kwargs):
    with pytest.raises(ValueError):
        run(df, [], **kwargs)
