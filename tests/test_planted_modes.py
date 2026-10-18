import pytest

from planted_modes import MODES, Answerer, QuestionsNotAllowedError, question_rounds
from prep_agent.planner import Question


def ask(qid="a", options=("keep it", "drop it", "ask later"), recommended="drop it") -> Question:
    return Question(
        id=qid,
        question=f"Question {qid}?",
        options=list(options),
        recommended=recommended,
        reason="open",
    )


def test_the_modes_are_the_three_that_were_asked_for():
    assert MODES == ("recommended", "not_recommended", "no_questions")


def test_recommended_picks_the_recommended_option():
    answerer = Answerer("recommended")
    answers = answerer([ask("a"), ask("b", recommended="keep it")])
    assert answers == {"a": "drop it", "b": "keep it"}
    assert [item.forced for item in answerer.log] == [False, False]


def test_not_recommended_picks_the_first_other_option():
    answerer = Answerer("not_recommended")
    answers = answerer([ask("a"), ask("b", recommended="keep it")])
    assert answers == {"a": "keep it", "b": "drop it"}
    assert all(item.answer != item.recommended for item in answerer.log)
    assert answerer.forced == 0


def test_not_recommended_with_one_option_takes_it_and_says_so():
    only = Question.model_construct(
        id="only",
        question="Only one?",
        options=["yes"],
        recommended="yes",
        reason="open",
        allow_custom=False,
    )
    answerer = Answerer("not_recommended")
    assert answerer([only, ask("b")]) == {"only": "yes", "b": "keep it"}
    assert [item.forced for item in answerer.log] == [True, False]
    assert answerer.forced == 1


def test_every_answer_is_logged_with_its_question_and_options():
    answerer = Answerer("recommended")
    answerer([ask("a")])
    [item] = answerer.log
    assert item.id == "a" and item.question == "Question a?"
    assert item.options == ["keep it", "drop it", "ask later"]
    assert item.recommended == item.answer == "drop it"


def test_no_questions_refuses_to_answer():
    answerer = Answerer("no_questions")
    with pytest.raises(QuestionsNotAllowedError):
        answerer([ask()])
    assert answerer.log == []


def test_an_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown mode"):
        Answerer("agree")


def test_only_no_questions_takes_the_question_rounds_away():
    assert question_rounds("recommended", 2) == 2
    assert question_rounds("not_recommended", 2) == 2
    assert question_rounds("no_questions", 2) == 0
