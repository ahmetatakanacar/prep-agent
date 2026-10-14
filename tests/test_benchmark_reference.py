import pytest

from bench_datasets import (
    DEFAULT_DATASETS_DIR,
    DEFAULT_REFERENCE_DIR,
    DEFAULT_REGISTRY,
    csv_path,
    load_registry,
    reference_plan_path,
)
from bench_naive import naive_plan
from prep_agent.executor import execute
from prep_agent.plan import Plan
from prep_agent.profiler import load_csv, profile_dataframe
from prep_agent.validators import has_errors, validate_plan

SPECS = load_registry(DEFAULT_REGISTRY)
NAMES = [spec.name for spec in SPECS]
BY_NAME = {spec.name: spec for spec in SPECS}


def needs_data(name: str):
    present = csv_path(BY_NAME[name], DEFAULT_DATASETS_DIR).exists()
    return pytest.param(
        name,
        id=name,
        marks=pytest.mark.skipif(
            not present, reason="dataset not downloaded; run benchmarks/download.py"
        ),
    )


def load_plan(name: str) -> Plan:
    path = reference_plan_path(BY_NAME[name], DEFAULT_REFERENCE_DIR)
    return Plan.model_validate_json(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", NAMES)
class TestReferencePlanFiles:
    def test_the_plan_exists_and_matches_the_registry(self, name):
        plan = load_plan(name)
        assert plan.target == BY_NAME[name].target
        assert plan.task_type == BY_NAME[name].task_type

    def test_the_model_family_is_any(self, name):
        assert load_plan(name).model_family == "any"

    def test_every_step_gives_a_real_reason(self, name):
        for step in load_plan(name).steps:
            assert len(step.reason.split()) >= 6, f"{step.op}: {step.reason!r}"


def test_the_readme_marks_every_plan_as_a_draft_and_has_a_section_for_each():
    text = (DEFAULT_REFERENCE_DIR / "README.md").read_text(encoding="utf-8")
    assert "draft, needs human review" in text
    for name in NAMES:
        assert f"## {name}" in text, name


def test_there_is_a_plan_file_for_every_dataset_and_for_nothing_else():
    on_disk = {
        path.name.removesuffix(".plan.json") for path in DEFAULT_REFERENCE_DIR.glob("*.plan.json")
    }
    assert on_disk == set(NAMES)


@pytest.mark.parametrize("name", [needs_data(name) for name in NAMES])
class TestAgainstTheDownloadedData:
    @pytest.fixture
    def data(self, name):
        df = load_csv(csv_path(BY_NAME[name], DEFAULT_DATASETS_DIR))
        return df, profile_dataframe(df, target=BY_NAME[name].target)

    def test_the_reference_plan_passes_the_validator_without_errors(self, name, data):
        _, profile = data
        issues = validate_plan(load_plan(name), profile)
        assert not has_errors(issues), [(i.code, i.column) for i in issues]

    def test_the_reference_plan_runs_and_its_result_has_no_errors(self, name, data):
        df, profile = data
        result = execute(df, load_plan(name), profile)
        assert not has_errors(result.result_issues), [i.code for i in result.result_issues]

    def test_the_naive_plan_passes_the_validator_without_errors(self, name, data):
        _, profile = data
        plan = naive_plan(profile, BY_NAME[name].task_type)
        issues = validate_plan(plan, profile)
        assert not has_errors(issues), [(i.code, i.column) for i in issues]

    def test_the_naive_plan_runs_and_its_result_has_no_errors(self, name, data):
        df, profile = data
        plan = naive_plan(profile, BY_NAME[name].task_type)
        result = execute(df, plan, profile)
        assert not has_errors(result.result_issues), [i.code for i in result.result_issues]
