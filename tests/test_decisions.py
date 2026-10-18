"""Marked columns the user chose to keep: the reading of the answers, the validator, the agent and
the report."""

import numpy as np
import pandas as pd
import pytest

from prep_agent.agent import AgentTrace, run_agent
from prep_agent.decisions import UserKept, kept_by_user
from prep_agent.executor import execute
from prep_agent.llm.fake import FakeLLM
from prep_agent.plan import Plan
from prep_agent.planner import QARecord, Question
from prep_agent.profiler import profile_dataframe
from prep_agent.report import render_report
from prep_agent.validators import validate_plan


def table(n: int = 300) -> pd.DataFrame:
    """`after` nearly determines `outcome` (suspected leakage), `strong` predicts it well
    (high association), `noise` and `x` do not."""
    rng = np.random.default_rng(11)
    outcome = (rng.random(n) < 0.4).astype(int)
    return pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "after": 4.0 * outcome + rng.normal(size=n),
            "strong": 2.3 * outcome + rng.normal(size=n),
            "noise": rng.normal(size=n),
            "outcome": outcome,
        }
    )


PROFILE = profile_dataframe(table(), target="outcome")


def ask(text: str, options: list[str], answer: str, qid: str = "q") -> QARecord:
    question = Question(
        id=qid, question=text, options=options, recommended=options[0], reason="open"
    )
    return QARecord(question=question, answer=answer)


def kept(*records: QARecord) -> dict[str, UserKept]:
    return {u.column: u for u in kept_by_user(list(records), PROFILE)}


def test_the_fixture_has_one_column_in_each_mark():
    marks = {
        c.name: [w for w in c.warnings if "leakage" in w or "association" in w]
        for c in PROFILE.columns
    }
    assert marks["after"] == ["suspected_target_leakage"]
    assert marks["strong"] == ["high_target_association"]
    assert marks["x"] == marks["noise"] == []


@pytest.mark.parametrize(
    "answer",
    ["Keep `after`", "Retain after as a feature", "Use after", "Include it", "Preserve the column"],
)
def test_an_answer_that_says_keep_is_a_decision_to_keep(answer):
    found = kept(ask("Is after known at prediction time?", ["Drop after", "Keep after"], answer))

    assert list(found) == ["after"]
    assert found["after"].tier == "suspected" and found["after"].answer == answer
    assert found["after"].score >= 0.9


def test_the_lower_mark_is_a_decision_too():
    found = kept(ask("Is strong known?", ["Drop strong", "Keep strong"], "Keep strong"))

    assert found["strong"].tier == "high" and 0.6 <= found["strong"].score < 0.9


@pytest.mark.parametrize(
    "answer",
    ["Drop after", "Remove after", "Exclude the column", "Treat as categorical", "Ask me later"],
)
def test_other_answers_are_not_a_decision_to_keep(answer):
    assert kept(ask("Is after known?", ["Drop after", "Keep after"], answer)) == {}


def test_an_answer_that_both_keeps_and_drops_is_not_read():
    question = "What about after and strong?"
    options = ["Keep after, drop strong", "Drop both"]

    assert kept(ask(question, options, "Keep after, drop strong")) == {}


def test_one_answer_covers_every_marked_column_the_question_names():
    found = kept(ask("Keep after and strong?", ["Keep both", "Drop both"], "Keep both"))

    assert set(found) == {"after", "strong"}


def test_a_column_named_only_in_the_options_counts():
    found = kept(
        ask("Which of these is known at prediction time?", ["Keep after", "Drop it"], "Keep after")
    )

    assert list(found) == ["after"]


def test_names_are_matched_as_whole_words():
    assert (
        kept(ask("What about after_hours?", ["Keep after_hours", "Drop it"], "Keep after_hours"))
        == {}
    )


def test_unmarked_columns_are_never_decisions():
    assert kept(ask("What about x and noise?", ["Keep both", "Drop both"], "Keep both")) == {}


def test_the_last_answer_about_a_column_decides():
    first = ask("Keep after?", ["Keep after", "Drop after"], "Keep after", qid="a")
    second = ask("Reconsider after?", ["Keep after", "Drop after"], "Drop after", qid="b")

    assert kept(first, second) == {}
    assert list(kept(second, first)) == ["after"]


def test_no_questions_no_decisions():
    assert kept_by_user([], PROFILE) == []


# --- the validator ---


def plan_keeping_everything() -> Plan:
    return Plan.model_validate(
        {
            "target": "outcome",
            "task_type": "classification",
            "model_family": "any",
            "steps": [{"op": "split", "stratify": True, "reason": "r"}],
        }
    )


def codes(issues):
    return {(i.code, i.column) for i in issues}


def test_a_marked_column_nobody_decided_is_a_warning():
    issues = validate_plan(plan_keeping_everything(), PROFILE)

    assert ("suspected_leakage_kept", "after") in codes(issues)
    assert ("high_association_kept", "strong") in codes(issues)


def test_a_marked_column_the_user_decided_is_not():
    issues = validate_plan(plan_keeping_everything(), PROFILE, decided_by_user={"after"})

    assert ("suspected_leakage_kept", "after") not in codes(issues)
    assert ("high_association_kept", "strong") in codes(issues)


def test_the_executor_passes_the_decision_on():
    plan = plan_keeping_everything()
    result = execute(table(), plan, PROFILE, decided_by_user={"after", "strong"})

    assert not [i for i in result.plan_issues if i.code.endswith("_kept")]


# --- the agent ---


def reply(plan: dict) -> dict:
    return {"language": "English", "kind": "plan", "plan": plan, "questions": None}


def question_reply(text: str, options: list[str]) -> dict:
    question = {
        "id": "q",
        "question": text,
        "options": options,
        "recommended": options[0],
        "reason": "open",
    }
    return {"language": "English", "kind": "questions", "plan": None, "questions": [question]}


def plan_dict(*, drop: list[str]) -> dict:
    steps = [{"op": "drop_columns", "columns": drop, "reason": "r"}] if drop else []
    steps.append({"op": "split", "stratify": True, "reason": "r"})
    return {
        "target": "outcome",
        "task_type": "classification",
        "model_family": "any",
        "steps": steps,
    }


def run(responses, answer: str):
    llm = FakeLLM(responses)
    return run_agent(
        table(),
        "prepare",
        llm,
        lambda questions: {q.id: answer for q in questions},
        target="outcome",
        model_family="any",
    )


ASK_AFTER = question_reply(
    "Is `after` known when the prediction is made?", ["Drop after", "Keep after"]
)


def test_a_column_the_user_kept_when_asked_is_a_decision_in_the_trace_and_not_a_warning():
    result = run([ASK_AFTER, reply(plan_dict(drop=["strong"]))], "Keep after")

    assert [u.column for u in result.trace.user_kept] == ["after"]
    assert result.trace.user_kept[0].answer == "Keep after"
    assert not [i for i in result.issues if i.code == "suspected_leakage_kept"]


def test_a_marked_column_nobody_asked_about_keeps_its_warning_and_is_no_decision():
    result = run([reply(plan_dict(drop=["strong"]))], "Keep after")

    assert result.trace.user_kept == []
    assert [i.column for i in result.issues if i.code == "suspected_leakage_kept"] == ["after"]


def test_a_column_the_user_kept_but_the_plan_dropped_is_not_listed():
    result = run([ASK_AFTER, reply(plan_dict(drop=["after", "strong"]))], "Keep after")

    assert result.trace.user_kept == []


def test_a_column_the_user_decided_to_drop_is_no_decision_to_keep():
    result = run([ASK_AFTER, reply(plan_dict(drop=["after", "strong"]))], "Drop after")

    assert result.trace.user_kept == []


# --- the report ---


def render(trace: AgentTrace | None) -> str:
    from datetime import UTC, datetime

    from prep_agent.runs import Counts, RunInfo

    plan = plan_dict(drop=["strong"])
    df = table()
    parsed = Plan.model_validate(plan)
    execution = execute(df, parsed, PROFILE, decided_by_user={"after"})
    info = RunInfo(
        run_id="r1",
        created_at=datetime(2026, 10, 18, 9, 0, tzinfo=UTC),
        prep_agent_version="0.1.0",
        input_path="data/t.csv",
        input_sha256="0" * 64,
        target="outcome",
        rows=Counts(before=300, train=240, test=60),
        columns=Counts(before=5, train=3, test=3),
        status="ok",
        pipeline_joblib="note",
    )
    return render_report(
        info,
        plan=parsed,
        result=execution,
        plan_issues=[],
        trace=trace,
        baseline=None,
    )


def trace_with(user_kept) -> AgentTrace:
    return AgentTrace(goal="g", calls=[], qa_history=[], user_kept=user_kept)


def test_the_report_lists_the_columns_kept_by_the_users_decision():
    kept_after = UserKept(
        column="after",
        tier="suspected",
        score=0.97,
        question="Is `after` known when the prediction is made?",
        answer="Keep after",
    )
    report = render(trace_with([kept_after]))

    text = report.split("## Marked columns kept by your decision\n", 1)[1].split("\n## ", 1)[0]
    assert "**after**" in text and "almost determines the target" in text and "0.97" in text
    assert "Keep after" in text and "on your word" in text


def test_the_report_has_no_such_section_without_a_decision():
    assert "kept by your decision" not in render(trace_with([]))
    assert "kept by your decision" not in render(None)
