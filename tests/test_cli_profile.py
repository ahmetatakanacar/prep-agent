import json
from pathlib import Path

from prep_agent.profiler import CsvOptions, DatasetProfile

FIXTURE = Path(__file__).parent / "fixtures" / "customers.csv"
FIXTURE_TR = Path(__file__).parent / "fixtures" / "customers_tr.csv"


def test_profile_prints_summary_table(invoke_cli):
    result = invoke_cli(["profile", str(FIXTURE), "--target", "churned"])

    assert result.exit_code == 0
    for name in ("customer_id", "age", "plan", "signup_date", "monthly_spend", "churned"):
        assert name in result.output
    assert "id_like" in result.output
    assert "constant" in result.output
    assert "classification" in result.output


def test_profile_writes_json_file(invoke_cli, tmp_path):
    out = tmp_path / "nested" / "profile.json"

    result = invoke_cli(["profile", str(FIXTURE), "--target", "churned", "--out", str(out)])

    assert result.exit_code == 0
    assert out.exists()
    profile = DatasetProfile.model_validate_json(out.read_text(encoding="utf-8"))
    assert profile.n_rows == 20
    assert profile.target == "churned"
    assert {c.name for c in profile.columns} >= {"customer_id", "age", "plan"}
    json.loads(out.read_text(encoding="utf-8"))


def test_profile_without_out_writes_no_file(invoke_cli, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = invoke_cli(["profile", str(FIXTURE)])

    assert result.exit_code == 0
    assert list(tmp_path.iterdir()) == []


def test_profile_missing_file_fails_with_message(invoke_cli, tmp_path):
    result = invoke_cli(["profile", str(tmp_path / "missing.csv")])

    assert result.exit_code == 1
    assert "not found" in result.output


def test_profile_unknown_target_fails_with_message(invoke_cli):
    result = invoke_cli(["profile", str(FIXTURE), "--target", "nope"])

    assert result.exit_code == 1
    assert "target column 'nope' not found" in result.output


def test_profile_json_contains_samples_by_default(invoke_cli, tmp_path):
    out = tmp_path / "profile.json"

    invoke_cli(["profile", str(FIXTURE), "--out", str(out)])

    profile = DatasetProfile.model_validate_json(out.read_text(encoding="utf-8"))
    plan = next(c for c in profile.columns if c.name == "plan")
    assert plan.sample_values
    assert plan.top_values


def test_profile_no_samples_flag_removes_values_from_json(invoke_cli, tmp_path):
    out = tmp_path / "profile.json"

    result = invoke_cli(["profile", str(FIXTURE), "--no-samples", "--out", str(out)])

    assert result.exit_code == 0
    text = out.read_text(encoding="utf-8")
    profile = DatasetProfile.model_validate_json(text)
    assert all(c.sample_values == [] for c in profile.columns)
    assert all(not c.top_values for c in profile.columns)
    assert "enterprise" not in text
    assert "2023-01-05" not in text


def test_profile_json_records_the_csv_settings_that_were_found(invoke_cli, tmp_path):
    out = tmp_path / "profile.json"

    result = invoke_cli(["profile", str(FIXTURE_TR), "--out", str(out)])

    assert result.exit_code == 0, result.output
    profile = DatasetProfile.model_validate_json(out.read_text(encoding="utf-8"))
    assert profile.csv == CsvOptions(sep=";", decimal=",", encoding="cp1254")
    assert profile.n_rows == 20
    assert [c.inferred_type for c in profile.columns if c.name == "monthly_spend"] == ["numeric"]


def test_profile_warns_when_the_file_is_not_utf8(invoke_cli):
    result = invoke_cli(["profile", str(FIXTURE_TR)])

    assert result.exit_code == 0, result.output
    assert "Warning" in result.output
    assert "cp1254" in result.output


def test_profile_of_a_utf8_file_has_no_warning(invoke_cli):
    result = invoke_cli(["profile", str(FIXTURE)])

    assert "Warning" not in result.output
