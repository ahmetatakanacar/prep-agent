import json
import re

import pytest
import yaml
from typer.testing import CliRunner

import run_benchmark
from bench_cost import MAX_SCHEMA_RETRIES, max_calls_per_run
from benchmark_support import (
    answer,
    classification_plan,
    regression_plan,
    stub_baseline,
    write_workspace,
)
from prep_agent.llm.base import TokenUsage
from prep_agent.llm.fake import FakeLLM

PRICED = {
    "currency": "EUR",
    "models": {"fake-model": {"input_per_million": 2.0, "output_per_million": 8.0}},
}
ANSI = re.compile(r"\x1b\[[0-9;]*m")


class Setup:
    def __init__(self, root):
        self.root = root
        write_workspace(root)
        self.prices = root / "prices.yaml"
        self.prices.write_text(yaml.safe_dump(PRICED), encoding="utf-8")
        self.out = root / "out"
        self.built = []
        self.llm = FakeLLM(
            [answer(classification_plan()), answer(regression_plan())],
            usage=TokenUsage(1000, 200),
        )

    def client(self, model):
        self.built.append(model)
        return self.llm

    def invoke(self, *args, input=None):
        result = CliRunner().invoke(
            run_benchmark.app,
            [
                "--registry",
                str(self.root / "datasets.yaml"),
                "--datasets-dir",
                str(self.root / "datasets"),
                "--reference-dir",
                str(self.root / "reference"),
                "--prices",
                str(self.prices),
                "--out-dir",
                str(self.out),
                *args,
            ],
            input=input,
            env={"COLUMNS": "200"},
        )
        result.text = ANSI.sub("", result.output)
        return result


@pytest.fixture
def setup(tmp_path, monkeypatch) -> Setup:
    setup = Setup(tmp_path)
    monkeypatch.setattr(run_benchmark, "create_client", setup.client)
    monkeypatch.setattr("bench_runner.evaluate_baselines", stub_baseline)
    monkeypatch.setattr(run_benchmark, "load_dotenv_file", lambda: None)
    return setup


class TestDryRun:
    def test_lists_the_jobs_and_the_estimate_without_calling_a_model(self, setup):
        result = setup.invoke("--models", "fake-model", "--repeats", "2", "--dry-run")
        assert result.exit_code == 0, result.text
        assert setup.built == []
        assert setup.llm.calls == []
        assert not setup.out.exists()
        for expected in ("cls", "reg", "reference", "naive", "agent", "fake-model"):
            assert expected in result.text
        assert "estimated maximum" in result.text.lower()
        assert "EUR" in result.text
        assert "Dry run" in result.text

    def test_states_the_most_calls_one_agent_run_can_make(self, setup):
        result = setup.invoke("--models", "fake-model", "--dry-run")
        text = " ".join(result.text.split())
        calls = max_calls_per_run()
        assert f"at most {calls} LLM calls per run" in text
        assert f"each repeated up to {MAX_SCHEMA_RETRIES} time(s)" in text

    def test_the_estimate_is_the_one_a_real_run_asks_to_approve(self, setup):
        dry = setup.invoke("--models", "fake-model", "--dry-run")
        asked = setup.invoke("--models", "fake-model", input="n\n")
        figure = re.search(r"up to ([\d.]+) EUR", asked.text).group(1)
        assert figure in dry.text

    def test_lists_only_the_selected_datasets(self, setup):
        result = setup.invoke("--models", "fake-model", "--datasets", "reg", "--dry-run")
        assert result.exit_code == 0
        assert "cls" not in result.text

    def test_works_without_models_and_costs_nothing(self, setup):
        result = setup.invoke("--dry-run")
        assert result.exit_code == 0, result.text
        assert "No agent runs" in result.text

    def test_says_which_datasets_are_not_downloaded(self, setup):
        (setup.root / "datasets" / "reg.csv").unlink()
        result = setup.invoke("--models", "fake-model", "--dry-run")
        assert result.exit_code == 0
        assert "not downloaded" in result.text
        assert "reg" in result.text

    def test_a_model_without_a_price_is_refused_even_in_a_dry_run(self, setup):
        result = setup.invoke("--models", "fake-model,unpriced", "--dry-run")
        assert result.exit_code == 1
        assert "unpriced" in result.text
        assert str(setup.prices.name) in result.text
        assert setup.built == []


class TestPriceGuard:
    def test_a_model_without_a_price_does_not_run(self, setup):
        result = setup.invoke("--models", "unpriced", "--yes")
        assert result.exit_code == 1
        assert "no price for 'unpriced'" in result.text
        assert setup.built == []
        assert not setup.out.exists()

    def test_the_shipped_empty_price_file_blocks_every_model(self, setup):
        setup.prices.write_text("currency: USD\nmodels: {}\n", encoding="utf-8")
        result = setup.invoke("--models", "fake-model", "--yes")
        assert result.exit_code == 1
        assert setup.built == []

    def test_a_missing_price_file_is_an_error_when_models_are_given(self, setup):
        setup.prices.unlink()
        result = setup.invoke("--models", "fake-model", "--yes")
        assert result.exit_code == 1
        assert "not found" in result.text


class TestApproval:
    def test_declining_runs_nothing_and_writes_nothing(self, setup):
        result = setup.invoke("--models", "fake-model", input="n\n")
        assert result.exit_code == 1
        assert "Aborted" in result.text
        assert setup.built == []
        assert setup.llm.calls == []
        assert not setup.out.exists()

    def test_end_of_input_counts_as_declining(self, setup):
        result = setup.invoke("--models", "fake-model", input="")
        assert result.exit_code == 1
        assert setup.llm.calls == []
        assert not setup.out.exists()

    def test_the_prompt_names_the_runs_and_the_cost_ceiling(self, setup):
        result = setup.invoke("--models", "fake-model", "--repeats", "2", input="n\n")
        assert "4 agent run(s)" in result.text
        assert "EUR" in result.text

    def test_approving_runs_the_benchmark(self, setup):
        result = setup.invoke("--models", "fake-model", input="y\n")
        assert result.exit_code == 0, result.text
        assert setup.built == ["fake-model"]
        data = json.loads((setup.out / "results.json").read_text(encoding="utf-8"))
        assert len(data["jobs"]) == 6
        assert (setup.out / "summary.md").exists()
        assert "summary.md" in result.text

    def test_yes_skips_the_question(self, setup):
        result = setup.invoke("--models", "fake-model", "--yes")
        assert result.exit_code == 0, result.text
        assert "Proceed" not in result.text
        assert (setup.out / "results.json").exists()

    def test_the_approved_ceiling_is_recorded_in_the_results(self, setup):
        setup.invoke("--models", "fake-model", "--yes")
        data = json.loads((setup.out / "results.json").read_text(encoding="utf-8"))
        assert data["config"]["estimated_max_cost"]["fake-model"] > 0

    def test_without_models_nothing_needs_approval(self, setup):
        result = setup.invoke()
        assert result.exit_code == 0, result.text
        data = json.loads((setup.out / "results.json").read_text(encoding="utf-8"))
        assert {j["method"] for j in data["jobs"]} == {"reference", "naive"}
        assert setup.built == []


class TestOptions:
    def test_unknown_datasets_are_refused(self, setup):
        result = setup.invoke("--datasets", "nope", "--yes")
        assert result.exit_code == 1
        assert "unknown dataset" in result.text
        assert not setup.out.exists()

    def test_datasets_selects_a_subset(self, setup):
        result = setup.invoke("--models", "fake-model", "--datasets", "reg", "--yes")
        assert result.exit_code == 0, result.text
        data = json.loads((setup.out / "results.json").read_text(encoding="utf-8"))
        assert {j["dataset"] for j in data["jobs"]} == {"reg"}

    def test_repeats_must_be_at_least_one(self, setup):
        assert setup.invoke("--repeats", "0", "--dry-run").exit_code == 2

    def test_models_are_split_on_commas_and_deduplicated(self, setup, monkeypatch):
        entry = {"input_per_million": 1, "output_per_million": 1}
        setup.prices.write_text(
            yaml.safe_dump({"models": {"a": entry, "b": entry}}), encoding="utf-8"
        )
        built = []

        def client(model):
            built.append(model)
            return FakeLLM([answer(regression_plan())])

        monkeypatch.setattr(run_benchmark, "create_client", client)
        result = setup.invoke("--models", "a, b,a", "--yes", "--datasets", "reg")
        assert result.exit_code == 0, result.text
        assert built == ["a", "b"]

    def test_existing_results_in_the_out_dir_stop_the_run(self, setup):
        setup.out.mkdir()
        (setup.out / "results.json").write_text("{}", encoding="utf-8")
        result = setup.invoke("--models", "fake-model", "--yes")
        assert result.exit_code == 1
        assert "already holds results" in result.text
        assert setup.llm.calls == []

    def test_a_client_that_cannot_be_built_is_reported(self, setup, monkeypatch):
        from prep_agent.llm.base import LLMConfigError

        def refuse(model):
            raise LLMConfigError("OPENAI_API_KEY is not set")

        monkeypatch.setattr(run_benchmark, "create_client", refuse)
        result = setup.invoke("--models", "fake-model", "--yes")
        assert result.exit_code == 1
        assert "OPENAI_API_KEY" in result.text
        assert not (setup.out / "results.json").exists()


class TestSummarize:
    @pytest.fixture
    def finished(self, setup) -> Setup:
        assert setup.invoke("--models", "fake-model", "--yes").exit_code == 0
        return setup

    @pytest.fixture
    def offline(self, setup, monkeypatch):
        """Anything that would run or score a plan, call a model or read the datasets fails."""

        def refuse(*args, **kwargs):
            raise AssertionError("--summarize must not run anything")

        for name in ("run_benchmark", "create_client", "load_registry", "load_prices"):
            monkeypatch.setattr(run_benchmark, name, refuse)
        monkeypatch.setattr("bench_runner.evaluate_baselines", refuse)

    def test_rewrites_the_summary_from_the_results_file(self, finished, offline):
        original = (finished.out / "summary.md").read_text(encoding="utf-8")
        (finished.out / "summary.md").write_text("stale", encoding="utf-8")

        result = finished.invoke("--summarize", str(finished.out / "results.json"))

        assert result.exit_code == 0, result.text
        assert (finished.out / "summary.md").read_text(encoding="utf-8") == original
        assert "summary.md" in result.text

    def test_writes_next_to_the_results_file_by_default(self, finished, offline):
        elsewhere = finished.root / "elsewhere"
        elsewhere.mkdir()
        (elsewhere / "results.json").write_bytes((finished.out / "results.json").read_bytes())

        result = CliRunner().invoke(
            run_benchmark.app, ["--summarize", str(elsewhere / "results.json")]
        )

        assert result.exit_code == 0, result.output
        assert (elsewhere / "summary.md").exists()

    def test_out_dir_chooses_where_the_summary_goes(self, finished, offline):
        results_file = finished.out / "results.json"
        before = results_file.read_bytes()
        target = finished.root / "copy"

        result = CliRunner().invoke(
            run_benchmark.app, ["--summarize", str(results_file), "--out-dir", str(target)]
        )

        assert result.exit_code == 0, result.output
        assert (target / "summary.md").exists()
        assert results_file.read_bytes() == before

    def test_a_missing_results_file_is_reported(self, setup, offline):
        result = setup.invoke("--summarize", str(setup.root / "nope.json"))
        assert result.exit_code == 1
        assert "nope.json" in result.text

    def test_a_file_that_is_not_a_results_file_is_reported(self, setup, offline):
        bad = setup.root / "bad.json"
        bad.write_text('{"jobs": 3}', encoding="utf-8")
        result = setup.invoke("--summarize", str(bad))
        assert result.exit_code == 1
        assert "bad.json" in result.text
        assert not (setup.out / "summary.md").exists()


def test_the_default_client_selects_the_model_through_the_environment(monkeypatch):
    seen = {}

    def factory():
        import os

        seen["model"] = os.environ.get("PREP_AGENT_MODEL")
        return "client"

    monkeypatch.setattr(run_benchmark, "create_llm_client", factory)
    monkeypatch.setenv("PREP_AGENT_MODEL", "original")
    assert run_benchmark.create_client("other") == "client"
    assert seen["model"] == "other"
    import os

    assert os.environ["PREP_AGENT_MODEL"] == "original"
