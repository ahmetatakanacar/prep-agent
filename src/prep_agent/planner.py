"""The planner's side of the conversation: what the LLM may answer with.

On every call the planner answers with a `PlannerResponse`: either a complete plan or a list of
questions for the user. The plan model itself is the one from `prep_agent.plan` and is not
touched here.
"""

from typing import Annotated, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from prep_agent.plan import Plan

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
