"""The shape of results.json: what one benchmark run records, job by job.

A job is one plan on one dataset: the agent with one model (once per repeat), the hand-written
reference plan, or the naive plan. Reference and naive plans are deterministic, so they run
once whatever the number of repeats.

Statuses: `ok` means the plan ran, passed the validators and was scored. `failed` means the plan
or its result was rejected (for the agent: it ran out of revisions or the LLM call failed), with
the validator codes in `error_codes`. `error` is anything that is not a verdict on the plan: a
missing file, a crash, an exception that was not expected. Every status is kept in the results.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from bench_cost import ModelPrice
from prep_agent.baseline import BaselineResult
from prep_agent.plan import Plan

Method = Literal["agent", "reference", "naive"]
Status = Literal["ok", "failed", "error"]
TaskType = Literal["classification", "regression"]


class QAEntry(BaseModel):
    question: str
    answer: str


class AgentMetrics(BaseModel):
    """What an agent run cost and how it got to its plan."""

    # The model name the provider reported, which can be more specific than the requested one.
    reported_model: str | None
    calls: int
    revisions: int
    # Calls repeated because the answer did not match the schema; not part of `revisions`.
    schema_retries: int = 0
    questions: int
    # Validator error codes of the first plan; empty if it was accepted.
    first_plan_error_codes: list[str]
    # None when the provider reported no usage.
    input_tokens: int | None
    output_tokens: int | None
    # Parts of input_tokens; None when the provider did not report them.
    cached_input_tokens: int | None = None
    cache_write_tokens: int | None = None
    cost: float | None
    qa: list[QAEntry]


class JobResult(BaseModel):
    dataset: str
    task_type: TaskType
    method: Method
    # The model asked for; None for the reference and naive plans.
    model: str | None
    repeat: int
    status: Status
    error: str | None = None
    # Error codes of the last plan and its result (failed jobs), and warning codes of the last
    # plan that was validated.
    error_codes: list[str] = []
    warning_codes: list[str] = []
    agent: AgentMetrics | None = None
    plan: Plan | None = None
    # The dataset's known leaky columns (see datasets.yaml), the ones the plan does not drop and
    # how many those are. The count is None when there is no plan to look at.
    leaky_columns: list[str] = []
    kept_leaky_columns: list[str] = []
    leaky_columns_kept: int | None = None
    n_train: int | None = None
    n_test: int | None = None
    n_features: int | None = None
    baseline: BaselineResult | None = None
    # Planning (agent run, or validation and execution) and scoring, in seconds.
    plan_seconds: float | None = None
    baseline_seconds: float | None = None

    @property
    def label(self) -> str:
        return f"agent: {self.model}" if self.method == "agent" else self.method


class RunConfig(BaseModel):
    models: list[str]
    repeats: int
    datasets: list[str]
    currency: str
    prices: dict[str, ModelPrice]
    # Upper bound of the cost the user approved, per model; None where it could not be computed.
    estimated_max_cost: dict[str, float | None] = {}


class BenchmarkResults(BaseModel):
    started_at: datetime
    finished_at: datetime | None = None
    config: RunConfig
    jobs: list[JobResult] = []
