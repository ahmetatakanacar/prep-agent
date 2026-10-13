"""The planner's side of the conversation: what the LLM may answer with.

On every call the planner answers with a `PlannerResponse`: either a complete plan or a list of
questions for the user. The plan model itself is the one from `prep_agent.plan` and is not
touched here.

The prompts are Jinja2 templates in `prep_agent/prompts/`. The system prompt lists the ops from
the registry, so a new op shows up in it without editing the template. The user prompt is the
only place where data-derived content (the profile) reaches the LLM.
"""

from dataclasses import dataclass
from typing import Annotated, Literal, Self

from jinja2 import Environment, PackageLoader, StrictUndefined
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from prep_agent.ops import list_ops
from prep_agent.plan import Plan
from prep_agent.profiler import DatasetProfile
from prep_agent.validators import ValidationIssue

MIN_OPTIONS = 2
MAX_OPTIONS = 5

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


def _distinct(options: list[str]) -> list[str]:
    if len(set(options)) != len(options):
        raise ValueError("options must be distinct")
    return options


OptionList = Annotated[
    list[Text],
    Field(min_length=MIN_OPTIONS, max_length=MAX_OPTIONS),
    AfterValidator(_distinct),
]


class Question(BaseModel):
    """One question for the user about a decision the data and the goal leave open."""

    model_config = ConfigDict(extra="forbid")

    id: Text = Field(description="Short identifier, unique within one answer, such as 'target'.")
    question: Text
    options: OptionList = Field(description="The answers the user can pick from.")
    recommended: Text = Field(description="The option you would pick; must be one of options.")
    reason: Text = Field(description="Why this cannot be decided from the profile and the goal.")
    allow_custom: bool = Field(
        default=False,
        description="True if the user may type an answer of their own instead of an option.",
    )

    @model_validator(mode="after")
    def _check_recommended(self) -> Self:
        if self.recommended not in self.options:
            raise ValueError("recommended must be one of options")
        return self


class PlannerResponse(BaseModel):
    """Either a plan or questions. `kind` says which one is filled; the other stays null."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["plan", "questions"]
    plan: Plan | None = Field(default=None, description="Set when kind is 'plan'.")
    questions: list[Question] | None = Field(
        default=None, description="Set when kind is 'questions'."
    )

    @model_validator(mode="after")
    def _check_kind(self) -> Self:
        if self.kind == "plan":
            if self.plan is None:
                raise ValueError("kind 'plan' needs a plan")
            if self.questions:
                raise ValueError("kind 'plan' must not carry questions")
            # Strict mode makes the model write every field, so an empty list means "none".
            self.questions = None
        else:
            if not self.questions:
                raise ValueError("kind 'questions' needs at least one question")
            if self.plan is not None:
                raise ValueError("kind 'questions' must not carry a plan")
            ids = [question.id for question in self.questions]
            if len(set(ids)) != len(ids):
                raise ValueError("question ids must be distinct")
        return self


class QARecord(BaseModel):
    """A question the planner asked and the answer the user gave."""

    question: Question
    answer: str


@dataclass(frozen=True)
class Revision:
    """A rejected plan and the validator errors the planner has to fix."""

    plan: Plan
    issues: list[ValidationIssue]


_ENV = Environment(
    loader=PackageLoader("prep_agent", "prompts"),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=True,
    # Prompts are plain text, not HTML.
    autoescape=False,
)


def build_system_prompt() -> str:
    return _ENV.get_template("system.j2").render(ops=list_ops())


def build_user_prompt(
    *,
    goal: str,
    profile: DatasetProfile,
    target: str | None,
    model_family: str | None,
    qa_history: list[QARecord],
    revision: Revision | None,
    question_rounds_left: int,
    plan_only: bool = False,
) -> str:
    """Render the user message. `question_rounds_left == 0` tells the planner to stop asking;
    `plan_only` adds that its previous answer asked questions anyway and was refused."""
    return _ENV.get_template("user.j2").render(
        goal=goal,
        target=target,
        model_family=model_family,
        profile_json=profile.model_dump_json(exclude_none=True),
        qa_history=qa_history,
        revision=revision,
        previous_plan_json=revision.plan.model_dump_json(exclude_none=True) if revision else "",
        question_rounds_left=question_rounds_left,
        plan_only=plan_only,
    )
