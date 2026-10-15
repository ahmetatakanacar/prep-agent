"""Model prices and the cost estimate that guards a benchmark run.

Prices come from `prices.yaml`, which the user fills in: the project does not guess what a model
costs. A model without a price cannot be benchmarked, so a run can never spend money that was
not priced first.

The estimate is an upper bound by construction, not a prediction. The agent makes at most
`MAX_QUESTION_ROUNDS + MAX_REVISIONS + 1` planner calls (every question round and every revision
costs one call, plus the first plan), and each of them is repeated up to `MAX_SCHEMA_RETRIES`
times when its answer does not match the schema. The estimate assumes it makes all of them. A
repeated call sends its prompt again with the validation error, at most `SCHEMA_ERROR_LIMIT`
characters, added. The size of a prompt is taken from the real prompt text at
`CHARS_PER_TOKEN` characters per token, which over-counts for English and JSON; later calls add
the previous plan and the validator's answer. The output of a call is assumed to be as large as
a plan for the dataset's width, with a generous base for the answer's other fields. A model that
bills hidden reasoning tokens as output can still exceed it, which is why the real cost is
measured and reported separately.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from bench_datasets import BENCHMARKS_DIR
from prep_agent.agent import MAX_SCHEMA_RETRIES, SCHEMA_ERROR_LIMIT
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
    """What a model costs per million tokens, in the currency of the file.

    The cache prices are optional because not every provider bills them. Where one is missing,
    the tokens it would apply to are charged the input price.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_per_million: float = Field(ge=0)
    output_per_million: float = Field(ge=0)
    cached_input_per_million: float | None = Field(default=None, ge=0)
    cache_write_per_million: float | None = Field(default=None, ge=0)

    @property
    def has_cache_prices(self) -> bool:
        return self.cached_input_per_million is not None or self.cache_write_per_million is not None

    @property
    def highest_input_per_million(self) -> float:
        """What a token of input can cost at most: cache writes are dearer than input."""
        return max(self.input_per_million, self.cache_write_per_million or 0.0)


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


def token_cost(
    input_tokens: int,
    output_tokens: int,
    price: ModelPrice,
    cached_input_tokens: int | None = None,
    cache_write_tokens: int | None = None,
) -> float:
    """Cached and cache-write tokens are part of `input_tokens` and are charged their own
    price; the rest of the input is charged the input price. A cache count that is unknown, or
    has no price, leaves its tokens in the ordinary input, which is the formula without caches."""
    cached = (cached_input_tokens or 0) if price.cached_input_per_million is not None else 0
    written = (cache_write_tokens or 0) if price.cache_write_per_million is not None else 0
    ordinary = max(input_tokens - cached - written, 0)
    return (
        ordinary * price.input_per_million
        + cached * (price.cached_input_per_million or 0.0)
        + written * (price.cache_write_per_million or 0.0)
        + output_tokens * price.output_per_million
    ) / _PER_MILLION


def max_cost(input_tokens: int, output_tokens: int, price: ModelPrice) -> float:
    """The most these tokens can cost: every input token at the dearest input price."""
    return (
        input_tokens * price.highest_input_per_million + output_tokens * price.output_per_million
    ) / _PER_MILLION


def price_of_usage(usage: TokenUsage | None, price: ModelPrice) -> float | None:
    """The cost of one run's token use; None when the provider reported no usage."""
    if usage is None:
        return None
    return token_cost(
        usage.input_tokens,
        usage.output_tokens,
        price,
        usage.cached_input_tokens,
        usage.cache_write_tokens,
    )


@dataclass(frozen=True)
class RunEstimate:
    """The most one agent run on one dataset can use."""

    calls: int
    first_prompt_tokens: int
    input_tokens: int
    output_tokens: int

    def cost(self, price: ModelPrice) -> float:
        # An upper bound has to assume the dearest input price, not the likely mix.
        return max_cost(self.input_tokens, self.output_tokens, price)


def max_calls_per_run() -> int:
    """The most LLM calls one agent run can make, repeated calls included."""
    return (MAX_QUESTION_ROUNDS + MAX_REVISIONS + 1) * (1 + MAX_SCHEMA_RETRIES)


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
    retry_notice = build_user_prompt(
        goal=goal,
        profile=profile,
        target=target,
        model_family=model_family,
        qa_history=[],
        revision=None,
        question_rounds_left=MAX_QUESTION_ROUNDS,
        schema_error="x" * SCHEMA_ERROR_LIMIT,
    )
    first_prompt = math.ceil((len(system) + len(user)) / CHARS_PER_TOKEN)
    notice = math.ceil((len(retry_notice) - len(user)) / CHARS_PER_TOKEN)
    output_per_call = OUTPUT_BASE_TOKENS + OUTPUT_TOKENS_PER_COLUMN * profile.n_columns
    planner_calls = MAX_QUESTION_ROUNDS + MAX_REVISIONS + 1
    attempts = 1 + MAX_SCHEMA_RETRIES
    # A later call repeats the first prompt and adds the previous plan and the validator's
    # answer to it, each about the size of a plan.
    later_call = first_prompt + 2 * output_per_call
    # Every attempt of a planner call sends that call's prompt; each repeat adds the notice.
    first_call_input = attempts * first_prompt + MAX_SCHEMA_RETRIES * notice
    later_call_input = attempts * later_call + MAX_SCHEMA_RETRIES * notice
    calls = planner_calls * attempts
    return RunEstimate(
        calls=calls,
        first_prompt_tokens=first_prompt,
        input_tokens=first_call_input + (planner_calls - 1) * later_call_input,
        output_tokens=calls * output_per_call,
    )
