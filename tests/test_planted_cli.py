import re

import pytest
import yaml
from typer.testing import CliRunner

import run_planted
from planted_results import PlantedResults
from planted_support import RoutedLLM

ANSI = re.compile(r"\x1b\[[0-9;]*m")
PRICES = {
    "currency": "EUR",
    "models": {"m": {"input_per_million": 1.0, "output_per_million": 2.0}},
}
SMALL = ["--datasets", "planted-08,planted-01", "--variants", "anonymous", "--repeats", "1"]


@pytest.fixture
def cli(tmp_path, monkeypatch):
    prices = tmp_path / "prices.yaml"
    prices.write_text(yaml.safe_dump(PRICES), encoding="utf-8")
    built = []

    def create_client(model):
        built.append(model)
        return RoutedLLM()

    monkeypatch.setattr(run_planted, "create_client", create_client)
    # The tests must not read the developer's .env.
    monkeypatch.setattr(run_planted, "load_dotenv_file", lambda: None)

    def invoke(*args, input=None):
        result = CliRunner().invoke(
            run_planted.app, ["--prices", str(prices), *args], input=input, terminal_width=200
        )
        return result, ANSI.sub("", result.output)

    invoke.out = tmp_path / "out"
    invoke.built = built
    return invoke


def test_a_dry_run_lists_the_runs_and_the_ceiling_and_calls_nothing(cli):
    result, output = cli("--models", "m", *SMALL, "--dry-run")
    assert result.exit_code == 0
    assert "6 run(s)" in output
    assert "recommended" in output and "not_recommended" in output and "no_questions" in output
    assert "Estimated maximum use" in output and "no price" not in output
    assert "Dry run: no model was called and nothing was written." in output
    assert cli.built == []


def test_the_default_scope_is_every_dataset_variant_and_mode(cli):
    result, output = cli("--models", "m", "--repeats", "3", "--dry-run")
    assert "144 run(s)" in output


def test_a_model_without_a_price_is_refused(cli):
    result, output = cli("--models", "unpriced", *SMALL, "--dry-run")
    assert result.exit_code == 1
    assert "no price for 'unpriced'" in output


def test_a_run_needs_a_model(cli):
    result, output = cli(*SMALL)
    assert result.exit_code == 1 and "give at least one model" in output


def test_an_unknown_mode_is_refused(cli):
    result, output = cli("--models", "m", "--modes", "agree", "--dry-run")
    assert result.exit_code == 1 and "unknown mode 'agree'" in output


def test_declining_the_confirmation_runs_nothing(cli):
    result, output = cli("--models", "m", *SMALL, "--out-dir", str(cli.out), input="n\n")
    assert result.exit_code == 1
    assert "costing up to" in output and "Aborted: nothing was run." in output
    assert not cli.out.exists() and cli.built == []


def test_a_confirmed_run_writes_results_and_a_summary(cli):
    result, output = cli("--models", "m", *SMALL, "--out-dir", str(cli.out), "--yes")
    assert result.exit_code == 0, output
    assert "Runs: 6 ok, 0 failed, 0 error" in output
    assert "[6/6]" in output
    results = PlantedResults.model_validate_json((cli.out / "results.json").read_text("utf-8"))
    assert len(results.runs) == 6 and results.config.estimated_max_cost["m"] > 0
    assert "| model | variant | mode |" in (cli.out / "summary.md").read_text("utf-8")


def test_summarize_rewrites_the_summary_from_the_results(cli, tmp_path):
    cli("--models", "m", *SMALL, "--out-dir", str(cli.out), "--yes")
    (cli.out / "summary.md").unlink()
    result, output = cli("--summarize", str(cli.out / "results.json"))
    assert result.exit_code == 0
    assert (cli.out / "summary.md").read_text("utf-8").startswith("# Planted-fault")
    elsewhere = tmp_path / "elsewhere"
    cli("--summarize", str(cli.out / "results.json"), "--out-dir", str(elsewhere))
    assert (elsewhere / "summary.md").is_file()


def test_summarize_reports_an_unreadable_file(cli, tmp_path):
    bad = tmp_path / "results.json"
    bad.write_text("not json", encoding="utf-8")
    result, output = cli("--summarize", str(bad))
    assert result.exit_code == 1 and "cannot read results" in output


def test_the_held_out_test_recipes_need_the_final_test_flag(cli):
    result, output = cli("--models", "m", "--datasets", "planted-t01", "--dry-run")
    assert result.exit_code == 1 and "held-out test data" in output and "--final-test" in output
    result, output = cli("--models", "m", "--datasets", "planted-t01", "--dry-run", "--final-test")
    assert result.exit_code == 1 and "data files not found" in output
    assert cli.built == []
