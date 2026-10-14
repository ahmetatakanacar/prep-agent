import pandas as pd
import pytest
import yaml

from bench_cost import (
    DEFAULT_PRICES,
    MAX_QUESTION_ROUNDS,
    MAX_REVISIONS,
    ModelPrice,
    PriceError,
    estimate_agent_run,
    load_prices,
    price_of_usage,
    require_prices,
)
from prep_agent.llm.base import TokenUsage
from prep_agent.profiler import profile_dataframe


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

    def test_the_shipped_file_loads_and_leaves_the_prices_to_the_user(self):
        assert load_prices(DEFAULT_PRICES).models == {}


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


def _profile(n_columns=5, n_rows=60):
    df = pd.DataFrame({f"c{i}": range(n_rows) for i in range(n_columns - 1)})
    df["y"] = [0, 1] * (n_rows // 2)
    return profile_dataframe(df, target="y")


class TestEstimateAgentRun:
    def estimate(self, profile=None, goal="Predict y."):
        return estimate_agent_run(profile or _profile(), goal, "y", "any")

    def test_the_call_count_is_the_most_the_agent_can_make(self):
        assert self.estimate().calls == MAX_QUESTION_ROUNDS + MAX_REVISIONS + 1

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

    def test_the_cost_applies_the_price_to_both_token_counts(self):
        estimate = self.estimate()
        price = ModelPrice(input_per_million=1.0, output_per_million=3.0)
        expected = (estimate.input_tokens * 1.0 + estimate.output_tokens * 3.0) / 1_000_000
        assert estimate.cost(price) == pytest.approx(expected)
