"""Model prices and the cost estimate that guards a benchmark run.

Prices come from `prices.yaml`, which the user fills in: the project does not guess what a model
costs. A model without a price cannot be benchmarked, so a run can never spend money that was
not priced first.

The estimate is an upper bound by construction, not a prediction. The agent makes at most
`MAX_QUESTION_ROUNDS + MAX_REVISIONS + 1` calls (every question round and every revision costs
one call, plus the first plan), and the estimate assumes it makes all of them. The size of a
prompt is taken from the real prompt text at `CHARS_PER_TOKEN` characters per token, which
over-counts for English and JSON; later calls add the previous plan and the validator's answer.
The output of a call is assumed to be as large as a plan for the dataset's width, with a
generous base for the answer's other fields. A model that bills hidden reasoning tokens as
output can still exceed it, which is why the real cost is measured and reported separately.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from bench_datasets import BENCHMARKS_DIR
from prep_agent.llm.base import TokenUsage
from prep_agent.planner import build_system_prompt, build_user_prompt
from prep_agent.profiler import DatasetProfile

DEFAULT_PRICES = BENCHMARKS_DIR / "prices.yaml"

# The limits the benchmark gives the agent; the runner passes them to `run_agent` and the
# estimate counts with them, so the two cannot drift apart.
MAX_QUESTION_ROUNDS = 2
MAX_REVISIONS = 2

CHARS_PER_TOKEN = 3.0
OUTPUT_BASE_TOKENS = 1500
OUTPUT_TOKENS_PER_COLUMN = 60
_PER_MILLION = 1_000_000


class PriceError(ValueError):
    """The price file is missing or wrong, or a requested model has no price."""


class ModelPrice(BaseModel):
    """What a model costs per million tokens, in the currency of the file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_per_million: float = Field(ge=0)
    output_per_million: float = Field(ge=0)


class PriceTable(BaseModel):
    model_config = ConfigDict(extra="forbid")

    currency: str = "USD"
    models: dict[str, ModelPrice] = Field(default_factory=dict)


def load_prices(path: Path = DEFAULT_PRICES) -> PriceTable:
    path = Path(path)
    if not path.exists():
        raise PriceError(f"price file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise PriceError(f"could not read the price file {path}: {exc}") from exc
    try:
        return PriceTable.model_validate(raw or {})
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        raise PriceError(f"invalid price file {path}: {problems}") from None


def require_prices(table: PriceTable, models: list[str], path: Path) -> dict[str, ModelPrice]:
    """The prices of `models`; raises, naming every model that has none."""
    missing = [model for model in models if model not in table.models]
    if missing:
        raise PriceError(
            f"no price for {', '.join(repr(m) for m in missing)}; add "
            f"{'it' if len(missing) == 1 else 'them'} to {path} before benchmarking"
        )
    return {model: table.models[model] for model in models}


def price_of_usage(usage: TokenUsage | None, price: ModelPrice) -> float | None:
    """The cost of one run's token use; None when the provider reported no usage."""
    if usage is None:
        return None
    return (
        usage.input_tokens * price.input_per_million
        + usage.output_tokens * price.output_per_million
    ) / _PER_MILLION


@dataclass(frozen=True)
class RunEstimate:
    """The most one agent run on one dataset can use."""

    calls: int
    first_prompt_tokens: int
    input_tokens: int
    output_tokens: int

    def cost(self, price: ModelPrice) -> float:
        return (
            self.input_tokens * price.input_per_million
            + self.output_tokens * price.output_per_million
        ) / _PER_MILLION


def estimate_agent_run(
    profile: DatasetProfile, goal: str, target: str | None, model_family: str | None
) -> RunEstimate:
    """Upper bound of the tokens one agent run can use. Builds the prompts, calls nothing."""
    system = build_system_prompt()
    user = build_user_prompt(
        goal=goal,
        profile=profile,
        target=target,
        model_family=model_family,
        qa_history=[],
        revision=None,
        question_rounds_left=MAX_QUESTION_ROUNDS,
    )
    first_prompt = math.ceil((len(system) + len(user)) / CHARS_PER_TOKEN)
    output_per_call = OUTPUT_BASE_TOKENS + OUTPUT_TOKENS_PER_COLUMN * profile.n_columns
    calls = MAX_QUESTION_ROUNDS + MAX_REVISIONS + 1
    # A later call repeats the first prompt and adds the previous plan and the validator's
    # answer to it, each about the size of a plan.
    later_call = first_prompt + 2 * output_per_call
    return RunEstimate(
        calls=calls,
        first_prompt_tokens=first_prompt,
        input_tokens=first_prompt + (calls - 1) * later_call,
        output_tokens=calls * output_per_call,
    )
