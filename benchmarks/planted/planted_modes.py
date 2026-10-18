"""How a run answers the agent's questions, without a person.

Three modes, because the questions are where the agent hands decisions to the user and an eval
has to say what the user did with them:

- `recommended`: pick the option the agent recommends. This is what the main benchmark does.
- `not_recommended`: pick an option the agent did not recommend (the first one, in the order the
  agent listed them). This is a user who disagrees with the agent: if the agent asked whether to
  keep a column that leaks the target and the answer is "keep", the plan has to carry that out.
  When a question has no other option (the schema allows two or more, so only a hand-built
  question can) the recommended one is chosen and the answer is marked `forced`.
- `no_questions`: the agent may not ask. The agent already has a limit on question rounds; with
  the limit at zero its prompt says "Do not ask any more questions" and it must decide alone. No
  product code changes.

An answer log is kept for every run, so a result can show what the agent asked and what it was
told.
"""

from typing import Literal, get_args

from pydantic import BaseModel

from prep_agent.planner import Question

Mode = Literal["recommended", "not_recommended", "no_questions"]
MODES: tuple[str, ...] = get_args(Mode)


class AnsweredQuestion(BaseModel):
    id: str
    question: str
    options: list[str]
    recommended: str
    answer: str
    # The question had no option but the recommended one, so there was nothing else to pick.
    forced: bool = False


class QuestionsNotAllowedError(RuntimeError):
    """The agent asked although its question limit is zero. That is a bug in the agent loop."""


class Answerer:
    """Callable that the agent loop uses as its `ask_fn`; keeps the log of what it answered."""

    def __init__(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; known modes: {', '.join(MODES)}")
        self.mode = mode
        self.log: list[AnsweredQuestion] = []

    def __call__(self, questions: list[Question]) -> dict[str, str]:
        if self.mode == "no_questions":
            raise QuestionsNotAllowedError("the agent asked a question in no_questions mode")
        answers = {}
        for question in questions:
            answer, forced = self._choose(question)
            answers[question.id] = answer
            self.log.append(
                AnsweredQuestion(
                    id=question.id,
                    question=question.question,
                    options=list(question.options),
                    recommended=question.recommended,
                    answer=answer,
                    forced=forced,
                )
            )
        return answers

    def _choose(self, question: Question) -> tuple[str, bool]:
        if self.mode == "recommended":
            return question.recommended, False
        others = [option for option in question.options if option != question.recommended]
        if others:
            return others[0], False
        return question.recommended, True

    @property
    def forced(self) -> int:
        return sum(item.forced for item in self.log)


def question_rounds(mode: str, limit: int) -> int:
    """The number of question rounds the agent gets in `mode`, given the benchmark's limit."""
    return 0 if mode == "no_questions" else limit
