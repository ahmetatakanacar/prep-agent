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

    def test_the_reference_plan_drops_every_leaky_column(self, name):
        dropped = {
            column
            for step in load_plan(name).steps
            if step.op == "drop_columns"
            for column in step.columns
        }
        assert set(BY_NAME[name].leaky_columns) <= dropped

    def test_every_step_gives_a_real_reason(self, name):
        for step in load_plan(name).steps:
            assert len(step.reason.split()) >= 6, f"{step.op}: {step.reason!r}"


def test_the_readme_has_a_section_for_each_plan_and_calls_none_of_them_a_draft():
    text = (DEFAULT_REFERENCE_DIR / "README.md").read_text(encoding="utf-8")
    assert "draft" not in text.lower()
    for name in NAMES:
        assert f"## {name}" in text, name


def readme_section(name: str) -> str:
    text = (DEFAULT_REFERENCE_DIR / "README.md").read_text(encoding="utf-8")
    _, _, rest = text.partition(f"\n## {name}\n")
    return rest.split("\n## ", 1)[0]


@pytest.mark.parametrize("name", NAMES)
def test_every_readme_section_states_the_review_status_of_its_plan(name):
    [status] = [
        line for line in readme_section(name).splitlines() if line.startswith("**Review status:**")
    ]
    assert status.startswith("**Review status:** reviewed by human")


def steps_of(name: str, op: str) -> list:
    return [step for step in load_plan(name).steps if step.op == op]


def test_the_bike_sharing_plan_casts_hour_month_and_weekday_and_one_hot_encodes_them():
    plan = load_plan("bike-sharing-demand")

    assert [step.op for step in plan.steps] == [
        "drop_columns",
        "cast_to_category",
        "split",
        "encode",
        "scale",
    ]
    drop, cast, _, encode, scale = plan.steps
    assert drop.columns == ["feel_temp"]
    assert cast.columns == ["hour", "month", "weekday"]
    assert (encode.method, encode.columns) == (
        "onehot",
        ["season", "weather", "hour", "month", "weekday"],
    )
    assert (scale.method, scale.columns) == ("standard", ["temp", "humidity", "windspeed"])


def test_the_house_prices_plan_casts_mssubclass_and_does_not_scale_it():
    [cast] = steps_of("house-prices", "cast_to_category")
    assert cast.columns == ["MSSubClass"]
    onehot = [s for s in steps_of("house-prices", "encode") if "MSSubClass" in s.columns]
    assert [s.method for s in onehot] == ["onehot"]
    [scale] = steps_of("house-prices", "scale")
    assert "MSSubClass" not in scale.columns


def test_every_cast_comes_before_the_split_and_its_columns_are_encoded_after_it():
    for name in NAMES:
        steps = load_plan(name).steps
        ops = [step.op for step in steps]
        for index, step in enumerate(steps):
            if step.op != "cast_to_category":
                continue
            assert index < ops.index("split"), name
            encoded = {
                column
                for later in steps[ops.index("split") :]
                if later.op == "encode"
                for column in later.columns
            }
            assert set(step.columns) <= encoded, name


def test_the_adult_plan_removes_exact_copies_before_dropping_the_sampling_weight():
    plan = load_plan("adult")
    ops = [step.op for step in plan.steps]
    assert ops.index("dedupe") < ops.index("drop_columns") < ops.index("split")
    [dedupe] = steps_of("adult", "dedupe")
    assert (dedupe.subset, dedupe.keep) == (None, "first")
    assert "fnlwgt" in dedupe.reason


def test_the_employee_salaries_plan_keeps_underfilled_job_title_as_a_none_category():
    name = "employee-salaries"
    dropped = {c for step in steps_of(name, "drop_columns") for c in step.columns}
    assert "underfilled_job_title" not in dropped
    [fill] = [s for s in steps_of(name, "impute") if "underfilled_job_title" in s.columns]
    assert (fill.columns, fill.strategy, fill.fill_value) == (
        ["underfilled_job_title"],
        "constant",
        "None",
    )
    assert "hired" in fill.reason
    [encode] = [s for s in steps_of(name, "encode") if "underfilled_job_title" in s.columns]
    assert (encode.columns, encode.method, encode.max_categories) == (
        ["underfilled_job_title"],
        "onehot",
        10,
    )
    ops = [step.op for step in load_plan(name).steps]
    assert ops.index("split") < ops.index("impute") < ops.index("encode")


def test_the_employee_salaries_plan_adds_a_missing_indicator_for_overtime_pay():
    [fill] = [
        s for s in steps_of("employee-salaries", "impute") if "2016_overtime_pay" in s.columns
    ]
    assert (fill.strategy, fill.fill_value, fill.add_indicator) == ("constant", 0, True)
    assert "department" in fill.reason


def test_the_employee_salaries_plan_groups_positions_to_100_and_divisions_to_50():
    encodes = {
        tuple(s.columns): s.max_categories
        for s in steps_of("employee-salaries", "encode")
        if s.max_categories is not None
    }
    assert encodes[("employee_position_title",)] == 100
    assert encodes[("division",)] == 50


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

    def test_the_leaky_columns_are_columns_of_the_data(self, name, data):
        df, _ = data
        assert set(BY_NAME[name].leaky_columns) <= set(df.columns)

    def test_the_reference_plan_passes_the_validator_without_errors(self, name, data):
        _, profile = data
        issues = validate_plan(load_plan(name), profile)
        assert not has_errors(issues), [(i.code, i.column) for i in issues]

    def test_the_reference_plan_runs_and_its_result_has_no_errors(self, name, data):
        df, profile = data
        result = execute(df, load_plan(name), profile)
        assert not has_errors(result.result_issues), [i.code for i in result.result_issues]

    def test_the_cast_columns_are_numbers_the_profiler_hints_at(self, name, data):
        _, profile = data
        hinted = {c.name for c in profile.columns if "low_cardinality_numeric" in c.warnings}
        for step in steps_of(name, "cast_to_category"):
            assert set(step.columns) <= hinted

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
