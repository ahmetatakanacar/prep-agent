"""Marked columns that the user decided to keep when asked.

A column the profile marks as strongly related to the target (`suspected_target_leakage` or
`high_target_association`) that stays in the plan is a warning of the validator, because nobody
decided it. If the planner asked about the column and the user answered that it stays, somebody
did: the warning would only repeat the user's answer back to them. The report says instead that
the column was kept by the user's decision, with the question and the answer, so that the
decision stays visible. A marked column that stays without a question keeps its warning.

The answer is free text (the planner writes the options), so what counts as a decision to keep
is deliberately narrow: the answer says keep, retain, use, include or preserve, and says nothing
of dropping, removing or excluding. An answer that does both ("keep A, drop B") is not read, and
the warning stays.
"""

import re
from typing import Literal

from pydantic import BaseModel

from prep_agent.planner import QARecord
from prep_agent.profiler import DatasetProfile

TIERS: dict[str, Literal["suspected", "high"]] = {
    "suspected_target_leakage": "suspected",
    "high_target_association": "high",
}
_DROP = re.compile(r"\b(drop|remove|exclude|discard|delete|omit|ignore|leave out)\b", re.I)
_KEEP = re.compile(r"\b(keep|retain|use|include|preserve)\b", re.I)


class UserKept(BaseModel):
    """A marked column that stays in the plan because the user said so."""

    column: str
    tier: Literal["suspected", "high"]
    score: float | None
    question: str
    answer: str


def kept_by_user(qa_history: list[QARecord], profile: DatasetProfile) -> list[UserKept]:
    """The marked columns of `profile` that a question named and an answer decided to keep.

    A column named in several questions is decided by the last of them.
    """
    marked = {
        c.name: (tier, c)
        for c in profile.columns
        for warning, tier in TIERS.items()
        if warning in c.warnings
    }
    decided: dict[str, UserKept | None] = {}
    for record in qa_history:
        question = record.question
        text = " ".join([question.question, *question.options])
        answer = record.answer
        keeps = _KEEP.search(answer) is not None and _DROP.search(answer) is None
        for name, (tier, column) in marked.items():
            if not _mentions(name, text):
                continue
            decided[name] = (
                UserKept(
                    column=name,
                    tier=tier,
                    score=max(column.target_association or 0.0, column.missing_association or 0.0),
                    question=question.question,
                    answer=answer,
                )
                if keeps
                else None
            )
    return [kept for kept in decided.values() if kept is not None]


def _mentions(name: str, text: str) -> bool:
    return re.search(rf"(?<![\w]){re.escape(name)}(?![\w])", text) is not None
