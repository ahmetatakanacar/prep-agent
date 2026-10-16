import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

import bench_cost
from bench_cost import (
    DEFAULT_PRICES,
    MAX_QUESTION_ROUNDS,
    MAX_REVISIONS,
    MAX_SCHEMA_RETRIES,
    ModelPrice,
    PriceError,
    estimate_agent_run,
    load_prices,
    max_cost,
    price_of_usage,
    require_prices,
    token_cost,
)
from prep_agent import agent
from prep_agent.agent import AgentError, run_agent
from prep_agent.llm.base import TokenUsage
from prep_agent.llm.fake import FakeLLM
from prep_agent.profiler import load_csv, profile_dataframe

FIXTURES = Path(__file__).parent / "fixtures"


def _write(tmp_path, data) -> object:
    path = tmp_path / "prices.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _model(**overrides) -> dict:
    return {"input_per_million": 2.0, "output_per_million": 8.0} | overrides


class TestLoadPrices:
    def test_reads_prices_and_currency(self, tmp_path):
        table = load_prices(_write(tmp_path, {"currency": "EUR", "models": {"m": _model()}}))
        assert table.currency == "EUR"
        assert table.models == {"m": ModelPrice(input_per_million=2.0, output_per_million=8.0)}

    def test_an_empty_file_is_a_valid_table_without_models(self, tmp_path):
        assert load_prices(_write(tmp_path, {"currency": "USD", "models": {}})).models == {}

    def test_currency_defaults_to_usd(self, tmp_path):
        assert load_prices(_write(tmp_path, {"models": {}})).currency == "USD"

    def test_a_missing_file_is_reported(self, tmp_path):
        with pytest.raises(PriceError, match="not found"):
            load_prices(tmp_path / "nope.yaml")

    def test_cache_prices_are_optional_and_read_when_given(self, tmp_path):
        entry = _model(cached_input_per_million=0.5, cache_write_per_million=2.5)
        table = load_prices(_write(tmp_path, {"models": {"cached": entry, "plain": _model()}}))
        cached, plain = table.models["cached"], table.models["plain"]
        assert (cached.cached_input_per_million, cached.cache_write_per_million) == (0.5, 2.5)
        assert (plain.cached_input_per_million, plain.cache_write_per_million) == (None, None)

    @pytest.mark.parametrize(
        "models",
        [
            {"m": _model(cached_input_per_million=-0.1)},
            {"m": _model(cache_write_per_million=-1)},
            {"m": _model(cached_input_per_million="cheap")},
        ],
    )
    def test_a_bad_cache_price_is_reported(self, tmp_path, models):
        with pytest.raises(PriceError, match="invalid price file"):
            load_prices(_write(tmp_path, {"models": models}))

    @pytest.mark.parametrize(
        "models",
        [
            {"m": _model(input_per_million=-1)},
            {"m": {"input_per_million": 1.0}},
            {"m": _model(surprise=1)},
            {"m": _model(output_per_million="cheap")},
            {"m": None},
            ["m"],
        ],
    )
    def test_a_bad_price_entry_is_reported(self, tmp_path, models):
        with pytest.raises(PriceError, match="invalid price file"):
            load_prices(_write(tmp_path, {"models": models}))

    def test_invalid_yaml_is_reported(self, tmp_path):
        path = tmp_path / "prices.yaml"
        path.write_text("models: [unclosed", encoding="utf-8")
        with pytest.raises(PriceError, match="could not read"):
            load_prices(path)

    def test_the_shipped_file_is_well_formed(self):
        table = load_prices(DEFAULT_PRICES)
        raw = yaml.safe_load(DEFAULT_PRICES.read_text(encoding="utf-8"))
        assert table.currency
        for name, entry in raw["models"].items():
            for field in ("input_per_million", "output_per_million"):
                value = entry[field]
                assert isinstance(value, int | float) and not isinstance(value, bool), (name, field)
                assert value > 0, (name, field)


class TestRequirePrices:
    def test_returns_the_prices_of_the_requested_models(self, tmp_path):
        table = load_prices(_write(tmp_path, {"models": {"a": _model(), "b": _model()}}))
        assert set(require_prices(table, ["a"], tmp_path / "prices.yaml")) == {"a"}

    def test_names_every_model_without_a_price_and_the_file_to_edit(self, tmp_path):
        path = _write(tmp_path, {"models": {"a": _model()}})
        with pytest.raises(PriceError) as caught:
            require_prices(load_prices(path), ["a", "x", "y"], path)
        message = str(caught.value)
        assert "x" in message
        assert "y" in message
        assert str(path) in message
        assert "'a'" not in message

    def test_no_models_need_no_prices(self, tmp_path):
        table = load_prices(_write(tmp_path, {"models": {}}))
        assert require_prices(table, [], tmp_path / "prices.yaml") == {}


class TestPriceOfUsage:
    PRICE = ModelPrice(input_per_million=2.0, output_per_million=8.0)

    def test_charges_input_and_output_separately(self):
        usage = TokenUsage(input_tokens=500_000, output_tokens=250_000)
        assert price_of_usage(usage, self.PRICE) == pytest.approx(1.0 + 2.0)

    def test_unreported_usage_has_no_cost(self):
        assert price_of_usage(None, self.PRICE) is None


class TestCachedUsageCost:
    PLAIN = ModelPrice(input_per_million=2.0, output_per_million=8.0)
    CACHED = ModelPrice(
        input_per_million=2.0,
        output_per_million=8.0,
        cached_input_per_million=0.5,
        cache_write_per_million=2.5,
    )
    USAGE = TokenUsage(
        input_tokens=1_000_000,
        output_tokens=250_000,
        cached_input_tokens=400_000,
        cache_write_tokens=100_000,
    )

    def test_each_kind_of_input_is_charged_at_its_own_price(self):
        # 500k ordinary at 2.0, 400k cached at 0.5, 100k written at 2.5, 250k output at 8.0.
        expected = 1.0 + 0.2 + 0.25 + 2.0
        assert price_of_usage(self.USAGE, self.CACHED) == pytest.approx(expected)

    def test_cached_tokens_are_part_of_the_input_not_added_to_it(self):
        usage = TokenUsage(1_000_000, 0, cached_input_tokens=1_000_000)
        assert price_of_usage(usage, self.CACHED) == pytest.approx(0.5)

    def test_usage_without_cache_counts_is_charged_the_old_way(self):
        usage = TokenUsage(input_tokens=500_000, output_tokens=250_000)
        assert price_of_usage(usage, self.CACHED) == pytest.approx(1.0 + 2.0)

    def test_a_model_without_cache_prices_uses_the_old_formula_for_cached_usage(self):
        assert price_of_usage(self.USAGE, self.PLAIN) == pytest.approx(2.0 + 2.0)

    def test_without_a_cache_write_price_the_written_tokens_are_ordinary_input(self):
        price = ModelPrice(
            input_per_million=2.0, output_per_million=8.0, cached_input_per_million=0.5
        )
        # 600k ordinary (the written 100k included) at 2.0, 400k cached at 0.5.
        assert price_of_usage(self.USAGE, price) == pytest.approx(1.2 + 0.2 + 2.0)

    def test_without_a_cached_input_price_the_cached_tokens_are_ordinary_input(self):
        price = ModelPrice(
            input_per_million=2.0, output_per_million=8.0, cache_write_per_million=2.5
        )
        # 900k ordinary (the cached 400k included) at 2.0, 100k written at 2.5.
        assert price_of_usage(self.USAGE, price) == pytest.approx(1.8 + 0.25 + 2.0)

    def test_counts_that_exceed_the_input_never_make_the_cost_negative(self):
        usage = TokenUsage(100, 0, cached_input_tokens=300, cache_write_tokens=0)
        assert price_of_usage(usage, self.CACHED) >= 0

    def test_token_cost_takes_the_cache_counts_as_keywords(self):
        cost = token_cost(
            1_000_000, 0, self.CACHED, cached_input_tokens=1_000_000, cache_write_tokens=0
        )
        assert cost == pytest.approx(0.5)

    def test_cache_prices_are_reported_as_set_or_not(self):
        assert self.CACHED.has_cache_prices
        assert not self.PLAIN.has_cache_prices


def _profile(n_columns=5, n_rows=60):
    df = pd.DataFrame({f"c{i}": range(n_rows) for i in range(n_columns - 1)})
    df["y"] = [0, 1] * (n_rows // 2)
    return profile_dataframe(df, target="y")


class TestEstimateAgentRun:
    def estimate(self, profile=None, goal="Predict y."):
        return estimate_agent_run(profile or _profile(), goal, "y", "any")

    def test_the_call_count_is_the_most_the_agent_can_make(self):
        planner_calls = MAX_QUESTION_ROUNDS + MAX_REVISIONS + 1
        assert self.estimate().calls == planner_calls * (1 + MAX_SCHEMA_RETRIES)

    def test_the_schema_retry_limit_is_the_agents_not_a_copy(self):
        assert bench_cost.MAX_SCHEMA_RETRIES is agent.MAX_SCHEMA_RETRIES

    def test_a_higher_retry_limit_raises_calls_input_and_output(self, monkeypatch):
        before = self.estimate()
        monkeypatch.setattr(bench_cost, "MAX_SCHEMA_RETRIES", MAX_SCHEMA_RETRIES + 1)
        after = self.estimate()
        planner_calls = MAX_QUESTION_ROUNDS + MAX_REVISIONS + 1
        assert after.calls == planner_calls * (2 + MAX_SCHEMA_RETRIES)
        assert after.input_tokens > before.input_tokens
        assert after.output_tokens > before.output_tokens

    def test_without_retries_the_estimate_is_the_one_for_the_planner_calls_alone(self, monkeypatch):
        monkeypatch.setattr(bench_cost, "MAX_SCHEMA_RETRIES", 0)
        estimate = self.estimate()
        assert estimate.calls == MAX_QUESTION_ROUNDS + MAX_REVISIONS + 1
        later_call = estimate.first_prompt_tokens + 2 * (estimate.output_tokens // estimate.calls)
        assert estimate.input_tokens == (
            estimate.first_prompt_tokens + (estimate.calls - 1) * later_call
        )

    def test_a_retry_prompt_is_counted_with_the_error_it_carries(self):
        estimate = self.estimate()
        without_notice = estimate.calls * estimate.first_prompt_tokens
        assert estimate.input_tokens > without_notice

    def test_the_agent_in_its_worst_case_makes_exactly_the_estimated_calls(self):
        # Every planner call is first answered with something that breaks the schema; the
        # questions use up the question rounds, the broken plans the revision budget.
        plan = json.loads((FIXTURES / "customers_plan.json").read_text(encoding="utf-8"))
        plan["steps"].insert(0, plan["steps"].pop(4))
        question = {
            "id": "q",
            "question": "Q?",
            "options": ["a", "b"],
            "recommended": "a",
            "reason": "r",
        }
        broken = {
            "language": "English",
            "kind": "questions",
            "plan": None,
            "questions": [{**question, "recommended": "c"}],
        }
        asking = {"language": "English", "kind": "questions", "plan": None, "questions": [question]}
        bad_plan = {"language": "English", "kind": "plan", "plan": plan, "questions": None}
        script = []
        for reply in [asking] * MAX_QUESTION_ROUNDS + [bad_plan] * (MAX_REVISIONS + 1):
            script += [broken] * MAX_SCHEMA_RETRIES + [reply]
        llm = FakeLLM(script)

        with pytest.raises(AgentError) as caught:
            run_agent(
                load_csv(FIXTURES / "customers.csv").df,
                "goal",
                llm,
                lambda questions: {q.id: q.recommended for q in questions},
                target="churned",
                max_question_rounds=MAX_QUESTION_ROUNDS,
                max_revisions=MAX_REVISIONS,
                max_schema_retries=MAX_SCHEMA_RETRIES,
            )

        assert len(llm.calls) == len(caught.value.trace.calls) == self.estimate().calls
        assert llm.remaining == 0

    def test_input_covers_at_least_the_first_prompt_for_every_call(self):
        estimate = self.estimate()
        assert estimate.input_tokens >= estimate.calls * estimate.first_prompt_tokens

    def test_a_wider_dataset_costs_more_in_both_directions(self):
        small, wide = self.estimate(_profile(5)), self.estimate(_profile(40))
        assert wide.input_tokens > small.input_tokens
        assert wide.output_tokens > small.output_tokens

    def test_a_longer_goal_costs_more_input(self):
        assert self.estimate(goal="x " * 2000).input_tokens > self.estimate().input_tokens

    def test_the_estimate_is_deterministic(self):
        assert self.estimate() == self.estimate()

    def test_a_cache_write_price_above_the_input_price_raises_the_upper_bound(self):
        estimate = self.estimate()
        plain = ModelPrice(input_per_million=2.0, output_per_million=8.0)
        dear_writes = ModelPrice(
            input_per_million=2.0,
            output_per_million=8.0,
            cached_input_per_million=0.1,
            cache_write_per_million=2.5,
        )
        expected = (estimate.input_tokens * 2.5 + estimate.output_tokens * 8.0) / 1_000_000
        assert estimate.cost(dear_writes) == pytest.approx(expected)
        assert estimate.cost(dear_writes) > estimate.cost(plain)

    def test_cheaper_cache_prices_do_not_lower_the_upper_bound(self):
        estimate = self.estimate()
        plain = ModelPrice(input_per_million=2.0, output_per_million=8.0)
        cheap = ModelPrice(
            input_per_million=2.0,
            output_per_million=8.0,
            cached_input_per_million=0.1,
            cache_write_per_million=1.0,
        )
        assert estimate.cost(cheap) == pytest.approx(estimate.cost(plain))
        assert max_cost(1_000_000, 0, cheap) == pytest.approx(2.0)

    def test_the_cost_applies_the_price_to_both_token_counts(self):
        estimate = self.estimate()
        price = ModelPrice(input_per_million=1.0, output_per_million=3.0)
        expected = (estimate.input_tokens * 1.0 + estimate.output_tokens * 3.0) / 1_000_000
        assert estimate.cost(price) == pytest.approx(expected)


class TestShippedPrices:
    """The prices.yaml of the repository itself, read by the code that costs a run."""

    def test_luna_has_cache_prices_and_they_are_used_for_the_cache_tokens(self):
        luna = load_prices(DEFAULT_PRICES).models["gpt-6-luna"]
        assert luna.cached_input_per_million == 0.01
        assert luna.cache_write_per_million == 0.125

        usage = TokenUsage(
            input_tokens=1_000_000,
            output_tokens=100_000,
            cached_input_tokens=600_000,
            cache_write_tokens=200_000,
        )
        # 200k ordinary at 0.10, 600k cached at 0.01, 200k written at 0.125, 100k output at 0.50.
        expected = 0.02 + 0.006 + 0.025 + 0.05
        assert price_of_usage(usage, luna) == pytest.approx(expected)

    def test_every_listed_model_gets_a_cache_price_in_the_same_form(self):
        for name, entry in load_prices(DEFAULT_PRICES).models.items():
            assert entry.cached_input_per_million is not None, name
            assert entry.cache_write_per_million is not None, name
