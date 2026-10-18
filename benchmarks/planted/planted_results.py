"""The shape of results.json for a planted-fault run: one record per agent run.

A run is one dataset in one variant, answered in one mode by one model, once per repeat. Statuses
are those of the main benchmark: `ok` means the plan was validated, executed and scored; `failed`
means the agent could not deliver a valid plan (it ran out of revisions or the LLM call failed),
so there is no score; `error` is anything that is not a verdict on the plan.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from bench_cost import ModelPrice
from planted_modes import AnsweredQuestion
from planted_scorer import Score
from planted_truth import Variant
from prep_agent.plan import Plan

Status = Literal["ok", "failed", "error"]


class AgentRecord(BaseModel):
    """What an agent run cost and how it got to its plan."""

    reported_model: str | None
    calls: int
    revisions: int
    schema_retries: int = 0
    # The number of questions the agent asked, and how many had no other option to pick.
    questions: int
    forced_answers: int = 0
    # None when the provider reported no usage.
    input_tokens: int | None
    output_tokens: int | None
    cached_input_tokens: int | None = None
    cache_write_tokens: int | None = None
    cost: float | None
    qa: list[AnsweredQuestion]


class MarkedColumn(BaseModel):
    """A column the profile related strongly to the target, and what became of the question.

    `score` is the larger of its value and missing-value association; `tier` is "suspected" from
    0.9 and "high" from 0.6, whatever the code under test calls them. `truth` is what the ground
    truth says it is: a "leak" (must be dropped), "legit" (a strong feature known before the
    outcome) or "other".
    """

    name: str
    score: float
    tier: Literal["suspected", "high"]
    asked: bool
    truth: Literal["leak", "legit", "other"]


class RunResult(BaseModel):
    dataset: str
    variant: Variant
    mode: str
    model: str
    repeat: int
    task_type: Literal["classification", "regression"]
    status: Status
    error: str | None = None
    error_codes: list[str] = []
    warning_codes: list[str] = []
    agent: AgentRecord | None = None
    plan: Plan | None = None
    marked: list[MarkedColumn] = []
    # Marked columns that stay in the plan because the user said so when asked (they get no
    # warning from the validator; the report lists them).
    user_kept: list[str] = []
    # Only for `ok` runs.
    score: Score | None = None
    plan_seconds: float | None = None


class RunConfig(BaseModel):
    models: list[str]
    variants: list[str]
    modes: list[str]
    repeats: int
    datasets: list[str]
    currency: str
    prices: dict[str, ModelPrice]
    # Upper bound of the cost that was approved, per model; None where it could not be computed.
    estimated_max_cost: dict[str, float | None] = {}


class PlantedResults(BaseModel):
    started_at: datetime
    finished_at: datetime | None = None
    config: RunConfig
    runs: list[RunResult] = []
