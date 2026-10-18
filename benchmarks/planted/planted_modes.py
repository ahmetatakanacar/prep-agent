"""How a run answers the agent's questions, without a person.

Four modes, because the questions are where the agent hands decisions to the user and an eval
has to say what the user did with them:

- `recommended`: pick the option the agent recommends. This is what the main benchmark does.
- `not_recommended`: pick an option the agent did not recommend (the first one, in the order the
  agent listed them). This is a user who disagrees with the agent: if the agent asked whether to
  keep a column that leaks the target and the answer is "keep", the plan has to carry that out.
  When a question has no other option (the schema allows two or more, so only a hand-built
  question can) the recommended one is chosen and the answer is marked `forced`.
- `oracle`: an informed user. Each question is answered from the ground truth of the dataset: a
  column that is a leak, an identifier, personal data or a constant is dropped, a column that is
  a real feature (a plain column, or a strong one that is known before the outcome) is kept, a
  number that is a category is cast and a count is left a number. The score of such a run is
  what the plan can be when the user knows the answer, so it measures what the agent does with
  good answers and how well it asks, not what it does alone. A question about no column of the
  truth, or about columns with opposite verdicts, gets the agent's own recommendation, and
  the log says so (`verdict`).
- `no_questions`: the agent may not ask. The agent already has a limit on question rounds; with
  the limit at zero its prompt says "Do not ask any more questions" and it must decide alone. No
  product code changes.

An answer log is kept for every run, so a result can show what the agent asked and what it was
told.
"""

import re
from collections import Counter
from typing import Literal, get_args

from pydantic import BaseModel

from planted_truth import DatasetTruth
from prep_agent.planner import Question

Mode = Literal["recommended", "not_recommended", "no_questions", "oracle"]
MODES: tuple[str, ...] = get_args(Mode)


class AnsweredQuestion(BaseModel):
    id: str
    question: str
    options: list[str]
    recommended: str
    answer: str
    # The question had no option but the recommended one, so there was nothing else to pick.
    forced: bool = False
    # Oracle mode: what the ground truth wants for the columns the question is about ("drop",
    # "keep", "categorical", "numeric"), "mixed" when they disagree and "none" when it has no
    # opinion; None in the other modes.
    verdict: str | None = None


class QuestionsNotAllowedError(RuntimeError):
    """The agent asked although its question limit is zero. That is a bug in the agent loop."""


class Answerer:
    """Callable that the agent loop uses as its `ask_fn`; keeps the log of what it answered."""

    def __init__(self, mode: str, truth: DatasetTruth | None = None) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; known modes: {', '.join(MODES)}")
        if mode == "oracle" and truth is None:
            raise ValueError("the oracle mode needs the ground truth of the dataset")
        self.mode = mode
        self.truth = truth
        self.log: list[AnsweredQuestion] = []

    def __call__(self, questions: list[Question]) -> dict[str, str]:
        if self.mode == "no_questions":
            raise QuestionsNotAllowedError("the agent asked a question in no_questions mode")
        answers = {}
        for question in questions:
            verdict = self._verdict(question) if self.mode == "oracle" else None
            answer, forced = self._choose(question, verdict)
            answers[question.id] = answer
            self.log.append(
                AnsweredQuestion(
                    id=question.id,
                    question=question.question,
                    options=list(question.options),
                    recommended=question.recommended,
                    answer=answer,
                    forced=forced,
                    verdict=verdict,
                )
            )
        return answers

    def _choose(self, question: Question, verdict: str | None) -> tuple[str, bool]:
        if self.mode == "recommended":
            return question.recommended, False
        if self.mode == "oracle":
            return _option_for(verdict, question), False
        others = [option for option in question.options if option != question.recommended]
        if others:
            return others[0], False
        return question.recommended, True

    def _verdict(self, question: Question) -> str:
        """What the ground truth wants for the columns the question names."""
        assert self.truth is not None
        text = " ".join([question.question, *question.options])
        wanted = []
        for column in self.truth.columns:
            if not re.search(rf"(?<![\w]){re.escape(column.name)}(?![\w])", text):
                continue
            if column.role == "must_drop":
                wanted.append("drop")
            elif column.expected_op == "cast_to_category":
                wanted.append("categorical")
            elif column.avoid_op == "cast_to_category":
                wanted.append("numeric")
            elif column.role == "must_keep":
                wanted.append("keep")
        if not wanted:
            return "none"
        counts = Counter(wanted)
        (best, n), *rest = counts.most_common()
        return best if not rest or n > rest[0][1] else "mixed"

    @property
    def forced(self) -> int:
        return sum(item.forced for item in self.log)


_DROP = re.compile(r"\b(drop|remove|exclude|discard|delete|omit|ignore|leave out)\b", re.I)
_KEEP = re.compile(r"\b(keep|retain|use|include|preserve)\b", re.I)
_CATEGORICAL = re.compile(r"categor|\bcast\b|\blabels?\b", re.I)
_NUMERIC = re.compile(r"numeric|\bcounts?\b|\bnumbers?\b|continuous|quantit|measure", re.I)


def _option_for(verdict: str | None, question: Question) -> str:
    """The option that carries out `verdict`, or the recommended one if there is none."""
    wanted = {
        "drop": lambda o: _DROP.search(o) is not None,
        "keep": lambda o: _DROP.search(o) is None and _KEEP.search(o) is not None,
        "categorical": lambda o: _CATEGORICAL.search(o) is not None,
        "numeric": lambda o: _NUMERIC.search(o) is not None and _CATEGORICAL.search(o) is None,
    }.get(verdict or "")
    if wanted is not None:
        for option in question.options:
            if wanted(option):
                return option
    return question.recommended


def question_rounds(mode: str, limit: int) -> int:
    """The number of question rounds the agent gets in `mode`, given the benchmark's limit."""
    return 0 if mode == "no_questions" else limit
