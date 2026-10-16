from types import SimpleNamespace
from typing import get_args

import pandas as pd
import pytest

from prep_agent import validators
from prep_agent.plan import Plan
from prep_agent.profiler import DEFAULT_CONFIG, DatasetProfile, profile_dataframe
from prep_agent.validators import (
    SEVERITY,
    IssueCode,
    ValidationIssue,
    has_errors,
    partition_issues,
    validate_plan,
)

N_ROWS = 20


def make_frame() -> pd.DataFrame:
    rows = range(N_ROWS)
    return pd.DataFrame(
        {
            "customer_id": [i + 1 for i in rows],
            "age": [20.5 + i for i in rows],
            "income": [None if i % 5 == 0 else 1000.0 + 37 * i for i in rows],
            "city": [("ankara", "izmir", "bursa")[i % 3] for i in rows],
            "job": [None if i % 4 == 0 else ("nurse", "clerk", "pilot")[i % 3] for i in rows],
            "flag": [i % 2 for i in rows],
            "gender": [("f", "m")[i % 2] for i in rows],
            "signup": [f"2024-01-{i + 1:02d}" for i in rows],
            "notes": [f"customer note number {i} with a fairly long free text body" for i in rows],
            "country": ["tr"] * N_ROWS,
            "churn": [(i // 2) % 2 for i in rows],
            "price": [100.0 + 3.7 * i for i in rows],
            "segment": [("a", "b", "c")[i % 3] for i in rows],
        }
    )


PROFILE = profile_dataframe(make_frame(), target="churn")


def dedupe(subset=None):
    return {"op": "dedupe", "keep": "first", "subset": subset}


def drop(*columns):
    return {"op": "drop_columns", "columns": list(columns)}


def drop_missing_target():
    return {"op": "drop_missing_target"}


def cast(*columns):
    return {"op": "cast_to_category", "columns": list(columns)}


def split(stratify=True):
    return {"op": "split", "stratify": stratify}


def impute(columns, strategy="median", **extra):
    return {"op": "impute", "columns": list(columns), "strategy": strategy, **extra}


def encode(columns, method="onehot"):
    return {"op": "encode", "columns": list(columns), "method": method}


def scale(columns, method="standard"):
    return {"op": "scale", "columns": list(columns), "method": method}


def make_plan(*steps, target="churn", task_type="classification", model_family="any") -> Plan:
    return Plan.model_validate(
        {
            "target": target,
            "task_type": task_type,
            "model_family": model_family,
            "steps": [{"reason": "test", **step} for step in steps],
        }
    )


def validate(*steps, profile: DatasetProfile = PROFILE, **plan_kwargs) -> list[ValidationIssue]:
    return validate_plan(make_plan(*steps, **plan_kwargs), profile)


def find(issues: list[ValidationIssue], code: str) -> list[ValidationIssue]:
    return [i for i in issues if i.code == code]


def with_warning(profile: DatasetProfile, column: str, warning: str) -> DatasetProfile:
    columns = [
        c.model_copy(update={"warnings": [*c.warnings, warning]}) if c.name == column else c
        for c in profile.columns
    ]
    return profile.model_copy(update={"columns": columns})


# A plan that follows every rule: no errors and no warnings.
VALID_STEPS = [
    dedupe(),
    drop("customer_id", "country", "signup", "notes", "price", "segment"),
    drop_missing_target(),
    split(),
    impute(["income"], "median"),
    impute(["job"], "most_frequent"),
    encode(["city", "job", "gender"], "onehot"),
    scale(["age", "income"], "standard"),
]


def test_fixture_columns_have_the_expected_inferred_types():
    types = {c.name: c.inferred_type for c in PROFILE.columns}

    assert types == {
        "customer_id": "id_like",
        "age": "numeric",
        "income": "numeric",
        "city": "categorical",
        "job": "categorical",
        "flag": "binary",
        "gender": "binary",
        "signup": "datetime",
        "notes": "text",
        "country": "constant",
        "churn": "binary",
        "price": "numeric",
        "segment": "categorical",
    }


def test_valid_plan_has_no_issues():
    assert validate(*VALID_STEPS, profile=needing_cleanup()) == []


def test_every_issue_code_has_a_severity():
    assert set(SEVERITY) == set(get_args(IssueCode))
    assert set(SEVERITY.values()) == {"error", "warning"}


def test_issue_severity_comes_from_the_code():
    issues = validate(impute(["income"]), split())

    issue = find(issues, "stateful_before_split")[0]
    assert issue.severity == SEVERITY["stateful_before_split"] == "error"


# --- helpers ---------------------------------------------------------------------------------


def test_has_errors_and_partition_issues():
    issues = validate(split(), scale(["customer_id"]))
    errors, warnings = partition_issues(issues)

    assert has_errors(issues)
    assert errors
    assert warnings
    assert all(i.severity == "error" for i in errors)
    assert all(i.severity == "warning" for i in warnings)
    assert len(errors) + len(warnings) == len(issues)
    assert not has_errors(warnings)
    assert not has_errors([])


def test_partition_issues_keeps_the_original_order():
    issues = validate(split(), scale(["customer_id"]))
    errors, warnings = partition_issues(issues)

    assert errors == [i for i in issues if i.severity == "error"]
    assert warnings == [i for i in issues if i.severity == "warning"]


# --- errors ----------------------------------------------------------------------------------


def test_target_missing_from_profile_is_an_error():
    issues = validate(*VALID_STEPS, target="nope")

    [issue] = find(issues, "target_not_in_profile")
    assert issue.severity == "error"
    assert issue.column == "nope"
    assert find(validate(*VALID_STEPS), "target_not_in_profile") == []


def test_plan_target_different_from_the_profile_target_is_an_error():
    issues = validate(*VALID_STEPS, target="segment")

    [issue] = find(issues, "target_mismatch")
    assert issue.severity == "error"
    assert issue.step_index is None
    assert issue.column == "segment"
    assert "'churn'" in issue.message
    assert "user" in issue.message


def test_plan_target_equal_to_the_profile_target_is_not_a_mismatch():
    assert find(validate(*VALID_STEPS), "target_mismatch") == []


@pytest.mark.parametrize("target", ["churn", "segment", "nope"])
def test_target_mismatch_is_not_checked_when_the_profile_has_no_target(target):
    profile = profile_dataframe(make_frame())
    assert profile.target is None

    assert find(validate(split(), profile=profile, target=target), "target_mismatch") == []


def check_family(expected, plan_family="any") -> list[ValidationIssue]:
    plan = make_plan(*VALID_STEPS, model_family=plan_family)
    return validate_plan(plan, PROFILE, expected_model_family=expected)


def test_plan_model_family_different_from_the_expected_one_is_an_error():
    [issue] = find(check_family("tree", plan_family="linear"), "model_family_mismatch")

    assert issue.severity == "error"
    assert issue.step_index is None
    assert issue.column is None
    assert "'linear'" in issue.message and "'tree'" in issue.message
    assert "user" in issue.message


def test_any_is_not_accepted_in_place_of_a_specific_expected_family():
    assert find(check_family("tree", plan_family="any"), "model_family_mismatch")


@pytest.mark.parametrize("family", ["tree", "linear", "any"])
def test_plan_model_family_equal_to_the_expected_one_is_fine(family):
    assert find(check_family(family, plan_family=family), "model_family_mismatch") == []


@pytest.mark.parametrize("family", ["tree", "linear", "any"])
def test_model_family_is_not_checked_without_an_expected_one(family):
    plan = make_plan(*VALID_STEPS, model_family=family)

    assert find(validate_plan(plan, PROFILE), "model_family_mismatch") == []


def many_value_target() -> DatasetProfile:
    """A numeric target with 40 distinct values, more than the profiler's regression threshold."""
    frame = pd.DataFrame({"x": range(40), "score": [i * 1.5 for i in range(40)]})
    return profile_dataframe(frame, target="score")


@pytest.mark.parametrize("target", ["city", "gender", "notes", "signup"])
def test_regression_with_a_target_that_is_not_numeric_is_an_error(target):
    profile = profile_dataframe(make_frame(), target=target)

    issues = validate(split(stratify=False), target=target, task_type="regression", profile=profile)

    [issue] = find(issues, "task_type_mismatch")
    assert issue.severity == "error"
    assert issue.column == target
    assert issue.step_index is None
    assert "not numeric" in issue.message
    assert "classification" in issue.message


@pytest.mark.parametrize("target", ["age", "churn", "flag"])
def test_regression_with_a_numeric_target_is_fine(target):
    profile = profile_dataframe(make_frame(), target=target)

    issues = validate(split(stratify=False), target=target, task_type="regression", profile=profile)

    assert find(issues, "task_type_mismatch") == []


@pytest.mark.parametrize("target", ["city", "gender", "churn"])
def test_classification_with_a_non_numeric_or_binary_target_is_fine(target):
    profile = profile_dataframe(make_frame(), target=target)

    issues = validate(split(), target=target, profile=profile)

    assert find(issues, "task_type_mismatch") == []


def test_classification_with_a_numeric_target_of_many_values_is_a_warning():
    issues = validate(split(), target="score", profile=many_value_target())

    [issue] = find(issues, "task_type_mismatch")
    assert issue.severity == "warning"
    assert issue.column == "score"
    assert "40 distinct values" in issue.message
    assert "regression" in issue.message


def test_the_regression_threshold_of_the_profiler_decides_when_to_warn():
    at_threshold = profile_dataframe(make_frame(), target="age")
    assert at_threshold.columns[1].n_unique == DEFAULT_CONFIG.regression_min_unique

    issues = validate(split(), target="age", profile=at_threshold)

    assert find(issues, "task_type_mismatch") == []


def test_regression_with_a_numeric_target_of_many_values_is_fine():
    issues = validate(
        split(stratify=False), target="score", task_type="regression", profile=many_value_target()
    )

    assert find(issues, "task_type_mismatch") == []


def test_the_task_type_is_not_checked_when_the_target_is_not_a_column():
    issues = validate(split(), target="nope", task_type="regression")

    assert find(issues, "task_type_mismatch") == []


def test_impute_on_a_column_without_missing_values_is_a_warning():
    issues = validate(split(), impute(["age"]))

    [issue] = find(issues, "impute_without_missing")
    assert issue.severity == "warning"
    assert issue.step_index == 1
    assert issue.column == "age"
    assert "no missing values" in issue.message


def test_impute_on_a_column_with_missing_values_is_fine():
    issues = validate(split(), impute(["income"]), impute(["job"], "most_frequent"))

    assert find(issues, "impute_without_missing") == []


def test_only_the_columns_without_missing_values_are_flagged_in_step_order():
    issues = validate(split(), impute(["age", "income", "price"]))

    assert [i.column for i in find(issues, "impute_without_missing")] == ["age", "price"]


def test_imputing_a_column_twice_flags_the_second_step():
    issues = validate(split(), impute(["income"]), impute(["income"], "mean"))

    [issue] = find(issues, "impute_without_missing")
    assert issue.step_index == 2 and issue.column == "income"


def with_duplicates(n: int = 2) -> DatasetProfile:
    return PROFILE.model_copy(update={"n_duplicate_rows": n, "duplicate_ratio": n / N_ROWS})


def needing_cleanup() -> DatasetProfile:
    """A profile with duplicate rows and missing target values, which VALID_STEPS clean up."""
    return with_missing_target().model_copy(
        update={"n_duplicate_rows": 2, "duplicate_ratio": 2 / N_ROWS}
    )


def with_missing_target() -> DatasetProfile:
    frame = make_frame().assign(churn=lambda df: df["churn"].astype("float").mask(df.index == 3))
    return profile_dataframe(frame, target="churn")


def test_dedupe_on_data_without_duplicate_rows_is_a_warning():
    issues = validate(dedupe(), split())

    [issue] = find(issues, "dedupe_without_duplicates")
    assert issue.severity == "warning"
    assert issue.step_index == 0
    assert issue.column is None
    assert "no duplicate rows" in issue.message


def test_dedupe_on_data_with_duplicate_rows_is_fine():
    issues = validate(dedupe(), split(), profile=with_duplicates())

    assert find(issues, "dedupe_without_duplicates") == []


def test_dedupe_on_a_subset_is_not_flagged_because_rows_can_match_on_it_alone():
    issues = validate(dedupe(["customer_id"]), split())

    assert find(issues, "dedupe_without_duplicates") == []


def test_dropping_columns_first_can_create_duplicates_so_the_dedupe_after_it_is_fine():
    issues = validate(drop("customer_id"), dedupe(), split())

    assert find(issues, "dedupe_without_duplicates") == []


def test_a_second_dedupe_finds_nothing_left_to_remove():
    issues = validate(dedupe(), dedupe(), split(), profile=with_duplicates())

    assert [i.step_index for i in find(issues, "dedupe_without_duplicates")] == [1]


def test_dedupe_after_a_subset_dedupe_is_flagged_on_data_without_duplicates():
    issues = validate(dedupe(["customer_id"]), dedupe(), split())

    assert [i.step_index for i in find(issues, "dedupe_without_duplicates")] == [1]


def test_drop_missing_target_without_missing_target_values_is_a_warning():
    issues = validate(drop_missing_target(), split())

    [issue] = find(issues, "drop_missing_target_without_missing")
    assert issue.severity == "warning"
    assert issue.step_index == 0
    assert issue.column == "churn"
    assert "no missing values" in issue.message


def test_drop_missing_target_with_missing_target_values_is_fine():
    issues = validate(drop_missing_target(), split(), profile=with_missing_target())

    assert find(issues, "drop_missing_target_without_missing") == []


def test_a_second_drop_missing_target_is_flagged():
    issues = validate(
        drop_missing_target(), drop_missing_target(), split(), profile=with_missing_target()
    )

    assert [i.step_index for i in find(issues, "drop_missing_target_without_missing")] == [1]


def test_drop_missing_target_is_not_flagged_when_the_target_is_unknown():
    issues = validate(drop_missing_target(), split(), target="nope")

    assert find(issues, "drop_missing_target_without_missing") == []


def test_plan_without_a_split_is_an_error():
    issues = validate(dedupe(), drop("country"))

    [issue] = find(issues, "invalid_split_count")
    assert issue.severity == "error"
    assert issue.step_index is None


def test_plan_with_two_splits_is_an_error():
    issues = validate(dedupe(), split(), split())

    [issue] = find(issues, "invalid_split_count")
    assert issue.severity == "error"
    assert issue.step_index == 2


def test_plan_with_exactly_one_split_has_no_split_count_error():
    assert find(validate(dedupe(), split()), "invalid_split_count") == []


def test_leakage_impute_before_split_is_an_error():
    issues = validate(impute(["income"]), split())

    [issue] = find(issues, "stateful_before_split")
    assert issue.severity == "error"
    assert issue.step_index == 0


def test_leakage_scale_before_split_is_an_error():
    issues = validate(scale(["age"]), split())

    [issue] = find(issues, "stateful_before_split")
    assert issue.step_index == 0
    assert issue.severity == "error"


def test_leakage_encode_before_split_is_an_error():
    issues = validate(encode(["city"]), split())

    assert [i.step_index for i in find(issues, "stateful_before_split")] == [0]


def test_leakage_dedupe_after_split_is_an_error():
    issues = validate(split(), dedupe())

    [issue] = find(issues, "stateless_after_split")
    assert issue.severity == "error"
    assert issue.step_index == 1


def test_leakage_drop_columns_and_drop_missing_target_after_split_are_errors():
    issues = validate(split(), drop("country"), drop_missing_target())

    assert [i.step_index for i in find(issues, "stateless_after_split")] == [1, 2]


def test_leakage_cast_to_category_after_split_is_an_error():
    issues = validate(split(), cast("flag"))

    [issue] = find(issues, "stateless_after_split")
    assert issue.step_index == 1


def test_leakage_scaling_the_target_is_an_error():
    issues = validate(split(), scale(["churn"]))

    [issue] = find(issues, "target_in_step")
    assert issue.severity == "error"
    assert issue.column == "churn"
    assert issue.step_index == 1


def test_stateless_steps_before_and_stateful_steps_after_the_split_are_fine():
    issues = validate(dedupe(), drop("country"), split(), impute(["income"]), scale(["age"]))

    assert find(issues, "stateless_after_split") == []
    assert find(issues, "stateful_before_split") == []


def test_step_order_is_not_judged_when_there_is_no_split():
    issues = validate(dedupe(), impute(["income"]))

    assert find(issues, "stateless_after_split") == []
    assert find(issues, "stateful_before_split") == []
    assert len(find(issues, "invalid_split_count")) == 1


def test_step_kind_comes_from_the_registry(monkeypatch):
    real_get_op = validators.get_op

    def get_op(name):
        if name == "scale":
            return SimpleNamespace(kind="stateless")
        return real_get_op(name)

    monkeypatch.setattr(validators, "get_op", get_op)
    issues = validate(split(), scale(["age"]))

    assert [i.step_index for i in find(issues, "stateless_after_split")] == [1]


@pytest.mark.parametrize(
    "step",
    [
        impute(["nope"]),
        encode(["nope"]),
        scale(["nope"]),
        drop("nope"),
        dedupe(["nope"]),
        cast("nope"),
    ],
)
def test_reference_to_a_column_that_never_existed_is_an_error(step):
    stateless = step["op"] in ("dedupe", "drop_columns", "cast_to_category")
    issues = validate(step, split()) if stateless else validate(split(), step)

    [issue] = find(issues, "unknown_column")
    assert issue.severity == "error"
    assert issue.column == "nope"


def test_reference_to_a_dropped_column_is_an_error():
    issues = validate(drop("country"), split(), impute(["country"], "most_frequent"))

    [issue] = find(issues, "unknown_column")
    assert issue.column == "country"
    assert issue.step_index == 2
    assert "drop" in issue.message


def test_casting_a_dropped_column_is_an_error():
    issues = validate(drop("flag"), cast("flag"), split())

    [issue] = find(issues, "unknown_column")
    assert issue.column == "flag"
    assert issue.step_index == 1


def test_reference_to_a_onehot_encoded_column_is_an_error():
    issues = validate(split(), encode(["city"], "onehot"), scale(["city"]))

    [issue] = find(issues, "unknown_column")
    assert issue.column == "city"
    assert issue.step_index == 2
    assert "one-hot" in issue.message


def test_scaling_an_ordinal_encoded_column_is_not_an_error():
    issues = validate(split(), encode(["city"], "ordinal"), scale(["city"]))

    assert [i for i in issues if i.column == "city" and i.severity == "error"] == []


def test_reference_to_an_existing_column_is_not_unknown():
    assert find(validate(split(), scale(["age"])), "unknown_column") == []


@pytest.mark.parametrize(
    "step",
    [
        impute(["churn"], "most_frequent"),
        encode(["churn"]),
        scale(["churn"]),
        drop("churn"),
        cast("churn"),
    ],
)
def test_target_in_a_modifying_step_is_an_error(step):
    steps = (
        [step, split()] if step["op"] in ("drop_columns", "cast_to_category") else [split(), step]
    )
    issues = validate(*steps)

    [issue] = find(issues, "target_in_step")
    assert issue.severity == "error"
    assert issue.column == "churn"
    assert find(issues, "unknown_column") == []


def test_target_in_dedupe_subset_and_drop_missing_target_is_allowed():
    issues = validate(dedupe(["churn", "age"]), drop_missing_target(), split())

    assert find(issues, "target_in_step") == []


@pytest.mark.parametrize("strategy", ["mean", "median"])
@pytest.mark.parametrize("column", ["city", "gender", "signup"])
def test_mean_or_median_imputation_of_a_non_numeric_column_is_an_error(strategy, column):
    issues = validate(split(), impute([column], strategy))

    [issue] = find(issues, "impute_non_numeric")
    assert issue.severity == "error"
    assert issue.column == column
    assert issue.step_index == 1


@pytest.mark.parametrize("strategy", ["mean", "median"])
@pytest.mark.parametrize("column", ["age", "income", "flag"])
def test_mean_or_median_imputation_of_a_numeric_or_01_column_is_fine(strategy, column):
    assert find(validate(split(), impute([column], strategy)), "impute_non_numeric") == []


def test_most_frequent_and_constant_imputation_work_on_any_type():
    issues = validate(
        split(),
        impute(["job"], "most_frequent"),
        impute(["city"], "constant", fill_value="unknown"),
    )

    assert find(issues, "impute_non_numeric") == []


@pytest.mark.parametrize("column", ["age", "income", "flag"])
def test_string_fill_value_for_a_numeric_column_is_an_error(column):
    issues = validate(split(), impute([column], "constant", fill_value="unknown"))

    [issue] = find(issues, "fill_value_type_mismatch")
    assert issue.severity == "error"
    assert issue.column == column
    assert issue.step_index == 1


def test_string_fill_value_for_an_ordinal_encoded_column_is_an_error():
    issues = validate(
        split(), encode(["city"], "ordinal"), impute(["city"], "constant", fill_value="x")
    )

    [issue] = find(issues, "fill_value_type_mismatch")
    assert issue.severity == "error"
    assert issue.step_index == 2


@pytest.mark.parametrize("fill_value", [0, -1.5])
@pytest.mark.parametrize("column", ["age", "income", "flag"])
def test_numeric_fill_value_for_a_numeric_column_is_fine(column, fill_value):
    issues = validate(split(), impute([column], "constant", fill_value=fill_value))

    assert find(issues, "fill_value_type_mismatch") == []


@pytest.mark.parametrize("fill_value", [0, 2.5])
@pytest.mark.parametrize("column", ["city", "job", "gender", "notes"])
def test_numeric_fill_value_for_a_non_numeric_column_is_a_warning(column, fill_value):
    issues = validate(split(), impute([column], "constant", fill_value=fill_value))

    [issue] = find(issues, "fill_value_type_mismatch")
    assert issue.severity == "warning"
    assert issue.column == column
    assert issue.step_index == 1


@pytest.mark.parametrize("column", ["city", "job", "gender", "notes"])
def test_string_fill_value_for_a_non_numeric_column_is_fine(column):
    issues = validate(split(), impute([column], "constant", fill_value="unknown"))

    assert find(issues, "fill_value_type_mismatch") == []


def test_fill_value_type_is_only_checked_for_constant_imputation():
    issues = validate(split(), impute(["city"], "most_frequent"), impute(["age"], "mean"))

    assert find(issues, "fill_value_type_mismatch") == []


@pytest.mark.parametrize("column", ["city", "gender", "signup", "customer_id"])
def test_scaling_a_non_numeric_column_is_an_error(column):
    issues = validate(split(), scale([column]))

    [issue] = find(issues, "scale_non_numeric")
    assert issue.severity == "error"
    assert issue.column == column


@pytest.mark.parametrize("column", ["age", "income", "flag"])
def test_scaling_a_numeric_or_01_column_is_fine(column):
    assert find(validate(split(), scale([column])), "scale_non_numeric") == []


@pytest.mark.parametrize("method", ["onehot", "ordinal"])
@pytest.mark.parametrize("column", ["age", "flag"])
def test_encoding_a_numeric_column_is_an_error(method, column):
    issues = validate(split(), encode([column], method))

    [issue] = find(issues, "encode_numeric")
    assert issue.severity == "error"
    assert issue.column == column


def test_encode_numeric_message_suggests_cast_to_category_for_a_few_distinct_values():
    profile = with_warning(PROFILE, "age", "low_cardinality_numeric")

    issues = validate(split(), encode(["age"]), profile=profile)

    [issue] = find(issues, "encode_numeric")
    assert "cast_to_category" in issue.message
    assert "before the split" in issue.message


@pytest.mark.parametrize("column", ["age", "flag"])
def test_encode_numeric_message_does_not_suggest_a_cast_without_the_hint(column):
    issues = validate(split(), encode([column]))

    [issue] = find(issues, "encode_numeric")
    assert "cast_to_category" not in issue.message


def test_encode_numeric_message_does_not_suggest_a_cast_for_a_column_that_was_cast():
    profile = with_warning(PROFILE, "age", "low_cardinality_numeric")
    plan = make_plan(cast("age"), split(), encode(["age"], "ordinal"), encode(["age"], "ordinal"))

    issues = validate_plan(plan, profile)

    [issue] = find(issues, "encode_numeric")
    assert "cast_to_category" not in issue.message


@pytest.mark.parametrize("column", ["city", "gender", "signup"])
def test_encoding_a_non_numeric_column_is_fine(column):
    assert find(validate(split(), encode([column])), "encode_numeric") == []


def test_ordinal_encoding_of_an_ordinal_encoded_column_is_an_error():
    issues = validate(split(), encode(["city"], "ordinal"), encode(["city"], "ordinal"))

    assert [i.step_index for i in find(issues, "encode_numeric")] == [2]


def test_ordinal_encoding_a_column_with_missing_values_is_an_error():
    issues = validate(split(), encode(["job"], "ordinal"))

    [issue] = find(issues, "ordinal_with_missing")
    assert issue.severity == "error"
    assert issue.column == "job"
    assert issue.step_index == 1


def test_ordinal_encoding_after_imputation_is_fine():
    issues = validate(split(), impute(["job"], "most_frequent"), encode(["job"], "ordinal"))

    assert find(issues, "ordinal_with_missing") == []


def test_ordinal_encoding_a_column_without_missing_values_is_fine():
    assert find(validate(split(), encode(["city"], "ordinal")), "ordinal_with_missing") == []


def test_onehot_of_a_column_with_missing_values_is_not_an_error():
    issues = validate(split(), encode(["job"], "onehot"))

    assert [i for i in issues if i.column == "job" and i.severity == "error"] == []


# --- cast_to_category ------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["onehot", "ordinal"])
@pytest.mark.parametrize("column", ["age", "flag"])
def test_a_cast_numeric_column_can_be_encoded(method, column):
    issues = validate(cast(column), split(), encode([column], method))

    assert [i for i in issues if i.column == column] == []


@pytest.mark.parametrize("column", ["age", "income", "flag"])
def test_a_cast_column_can_no_longer_be_scaled(column):
    issues = validate(cast(column), split(), scale([column]))

    [issue] = find(issues, "scale_non_numeric")
    assert issue.severity == "error"
    assert issue.column == column
    assert issue.step_index == 2
    assert "categorical" in issue.message


@pytest.mark.parametrize("strategy", ["mean", "median"])
def test_a_cast_column_cannot_be_imputed_with_mean_or_median(strategy):
    issues = validate(cast("income"), split(), impute(["income"], strategy))

    [issue] = find(issues, "impute_non_numeric")
    assert issue.severity == "error"
    assert issue.column == "income"
    assert "categorical" in issue.message


def test_a_cast_column_can_be_imputed_with_the_most_frequent_value():
    issues = validate(
        cast("income"), split(), impute(["income"], "most_frequent"), encode(["income"])
    )

    assert [i for i in issues if i.column == "income"] == []


def test_a_cast_column_keeps_its_missing_values_until_they_are_imputed():
    issues = validate(cast("income"), split(), encode(["income"]))

    [issue] = find(issues, "onehot_before_impute")
    assert issue.column == "income"


def test_a_cast_column_with_missing_values_cannot_be_ordinal_encoded_before_imputing():
    issues = validate(cast("income"), split(), encode(["income"], "ordinal"))

    [issue] = find(issues, "ordinal_with_missing")
    assert issue.column == "income"


def test_a_cast_column_that_is_never_encoded_is_reported_at_the_end():
    issues = validate(cast("age"), split())

    [issue] = [i for i in find(issues, "unencoded_column_remains") if i.column == "age"]
    assert "categorical" in issue.message


@pytest.mark.parametrize("column", ["age", "income", "flag", "price"])
def test_casting_a_numeric_column_is_fine(column):
    issues = validate(cast(column), split())

    assert find(issues, "cast_non_numeric") == []


@pytest.mark.parametrize("column", ["city", "gender", "signup", "notes", "customer_id"])
def test_casting_a_column_that_is_not_numeric_is_a_warning(column):
    issues = validate(cast(column), split())

    [issue] = find(issues, "cast_non_numeric")
    assert issue.severity == "warning"
    assert issue.column == column
    assert issue.step_index == 0
    assert "already" in issue.message


def test_casting_the_same_column_twice_warns_about_the_repeat_and_the_second_cast():
    issues = validate(cast("age"), cast("age"), split())

    assert [i.step_index for i in find(issues, "repeated_op")] == [1]
    assert [i.step_index for i in find(issues, "cast_non_numeric")] == [1]


def test_a_plan_that_casts_before_the_split_and_encodes_after_it_has_no_issues():
    issues = validate(
        dedupe(),
        drop("customer_id", "country", "signup", "notes", "price", "segment"),
        drop_missing_target(),
        cast("age"),
        split(),
        impute(["income"], "median"),
        impute(["job"], "most_frequent"),
        encode(["city", "job", "gender", "age"], "onehot"),
        scale(["income"], "standard"),
        profile=needing_cleanup(),
    )

    assert issues == []


def test_stratified_split_for_regression_is_an_error():
    issues = validate(split(stratify=True), target="price", task_type="regression")

    [issue] = find(issues, "stratify_regression")
    assert issue.severity == "error"
    assert issue.step_index == 0


def test_unstratified_split_for_regression_is_fine():
    issues = validate(split(stratify=False), target="price", task_type="regression")

    assert find(issues, "stratify_regression") == []
    assert find(issues, "stratify_disabled") == []


# --- warnings --------------------------------------------------------------------------------


def test_onehot_before_imputing_a_column_with_missing_values_warns():
    issues = validate(split(), encode(["job"], "onehot"))

    [issue] = find(issues, "onehot_before_impute")
    assert issue.severity == "warning"
    assert issue.column == "job"
    assert issue.step_index == 1


def test_onehot_after_imputation_or_without_missing_values_does_not_warn():
    issues = validate(split(), impute(["job"], "most_frequent"), encode(["job", "city"]))

    assert find(issues, "onehot_before_impute") == []


def test_ordinal_encoding_for_a_linear_model_warns():
    issues = validate(split(), encode(["city"], "ordinal"), model_family="linear")

    [issue] = find(issues, "ordinal_for_linear")
    assert issue.severity == "warning"
    assert issue.step_index == 1


@pytest.mark.parametrize(
    ("method", "family"),
    [("ordinal", "any"), ("ordinal", "tree"), ("onehot", "linear")],
)
def test_ordinal_encoding_warning_needs_ordinal_and_a_linear_family(method, family):
    issues = validate(split(), encode(["city"], method), model_family=family)

    assert find(issues, "ordinal_for_linear") == []


def test_scaling_for_a_tree_model_warns():
    issues = validate(split(), scale(["age"]), model_family="tree")

    [issue] = find(issues, "scale_for_tree")
    assert issue.severity == "warning"
    assert issue.step_index == 1


@pytest.mark.parametrize("family", ["linear", "any"])
def test_scaling_for_other_model_families_does_not_warn(family):
    assert find(validate(split(), scale(["age"]), model_family=family), "scale_for_tree") == []


def test_tree_family_without_scaling_does_not_warn():
    assert find(validate(split(), model_family="tree"), "scale_for_tree") == []


def test_applying_the_same_op_twice_to_a_column_warns():
    issues = validate(split(), impute(["income"], "mean"), impute(["income"], "median"))

    [issue] = find(issues, "repeated_op")
    assert issue.severity == "warning"
    assert issue.column == "income"
    assert issue.step_index == 2


def test_repeated_scaling_of_a_column_warns():
    issues = validate(split(), scale(["age"]), scale(["age"], "minmax"))

    assert [i.column for i in find(issues, "repeated_op")] == ["age"]


def test_different_ops_or_different_columns_do_not_count_as_repeats():
    issues = validate(
        split(),
        impute(["income"]),
        scale(["income"]),
        impute(["flag"]),
        impute(["age"]),
    )

    assert find(issues, "repeated_op") == []


@pytest.mark.parametrize("column", ["customer_id", "country"])
def test_id_like_or_constant_column_left_at_the_end_warns(column):
    issues = validate(split())

    [issue] = [i for i in find(issues, "uninformative_column_kept") if i.column == column]
    assert issue.severity == "warning"
    assert issue.step_index is None


@pytest.mark.parametrize("column", ["customer_id", "country"])
def test_dropped_id_like_or_constant_column_does_not_warn(column):
    issues = validate(drop(column), split())

    assert [i for i in find(issues, "uninformative_column_kept") if i.column == column] == []


@pytest.mark.parametrize("warning", ["possible_id", "constant"])
def test_profile_warning_alone_marks_a_column_as_uninformative(warning):
    flagged = validate(split(), profile=with_warning(PROFILE, "age", warning))
    plain = validate(split())

    assert "age" in [i.column for i in find(flagged, "uninformative_column_kept")]
    assert "age" not in [i.column for i in find(plain, "uninformative_column_kept")]


def test_missing_values_left_at_the_end_warn():
    found = find(validate(split()), "missing_values_remain")

    assert [i.column for i in found] == ["income", "job"]
    assert all(i.severity == "warning" and i.step_index is None for i in found)


def test_imputed_columns_do_not_warn_about_missing_values():
    issues = validate(split(), impute(["income"]), impute(["job"], "most_frequent"))

    assert find(issues, "missing_values_remain") == []


def test_onehot_encoded_column_does_not_warn_about_missing_values_at_the_end():
    issues = validate(split(), encode(["job"]))

    assert "job" not in [i.column for i in find(issues, "missing_values_remain")]


def test_a_target_with_missing_values_is_not_reported_as_missing_values_remaining():
    for steps in ([split()], [split(), drop_missing_target()], [drop_missing_target(), split()]):
        issues = validate(*steps, profile=with_missing_target())

        assert "churn" not in [i.column for i in find(issues, "missing_values_remain")]


def test_split_with_a_missing_target_is_an_error():
    issues = validate(split(), profile=with_missing_target())

    [issue] = find(issues, "missing_target_at_split")
    assert issue.severity == "error"
    assert issue.step_index == 0
    assert issue.column == "churn"
    assert "drop_missing_target" in issue.message
    assert "before the split" in issue.message


def test_split_with_a_missing_target_is_an_error_without_stratify_too():
    issues = validate(split(stratify=False), profile=with_missing_target())

    assert [i.step_index for i in find(issues, "missing_target_at_split")] == [0]


def test_dropping_rows_with_a_missing_target_before_the_split_is_fine():
    issues = validate(drop_missing_target(), split(), profile=with_missing_target())

    assert find(issues, "missing_target_at_split") == []


def test_dropping_rows_with_a_missing_target_after_the_split_is_too_late():
    issues = validate(split(), drop_missing_target(), profile=with_missing_target())

    assert [i.step_index for i in find(issues, "missing_target_at_split")] == [0]
    assert [i.step_index for i in find(issues, "stateless_after_split")] == [1]


def test_a_target_without_missing_values_is_not_flagged_at_the_split():
    assert find(validate(split()), "missing_target_at_split") == []


def test_a_plan_with_two_splits_reports_the_missing_target_once():
    issues = validate(split(), split(), profile=with_missing_target())

    assert [i.step_index for i in find(issues, "missing_target_at_split")] == [0]


def test_a_missing_target_is_not_checked_when_the_target_is_not_a_column():
    issues = validate(split(), target="nope", profile=with_missing_target())

    assert find(issues, "missing_target_at_split") == []


def test_unencoded_categorical_text_and_datetime_columns_warn():
    issues = validate(split())

    flagged = {i.column: i for i in find(issues, "unencoded_column_remains")}
    assert {"city", "job", "notes", "signup"} <= set(flagged)
    assert all(i.severity == "warning" and i.step_index is None for i in flagged.values())


def test_unencoded_warning_skips_encoded_dropped_and_numeric_columns():
    issues = validate(
        drop("signup", "notes"),
        split(),
        encode(["city", "gender"], "onehot"),
        encode(["job"], "ordinal"),
    )

    flagged = {i.column for i in find(issues, "unencoded_column_remains")}
    assert flagged == {"segment"}


def test_unencoded_binary_text_column_warns():
    issues = validate(split())

    [issue] = [i for i in find(issues, "unencoded_column_remains") if i.column == "gender"]
    assert issue.severity == "warning"
    assert issue.step_index is None


@pytest.mark.parametrize("method", ["onehot", "ordinal"])
def test_encoded_binary_text_column_does_not_warn_as_unencoded(method):
    issues = validate(split(), encode(["gender"], method))

    assert "gender" not in {i.column for i in find(issues, "unencoded_column_remains")}


def test_binary_01_column_does_not_warn_as_unencoded():
    issues = validate(split())

    assert "flag" not in {i.column for i in find(issues, "unencoded_column_remains")}


def test_categorical_target_is_not_reported_as_unencoded():
    issues = validate(split(), target="segment")

    assert "segment" not in {i.column for i in find(issues, "unencoded_column_remains")}


def test_binary_text_target_is_not_reported_as_unencoded():
    issues = validate(split(), target="gender")

    assert "gender" not in {i.column for i in find(issues, "unencoded_column_remains")}


def test_unstratified_classification_split_warns():
    issues = validate(split(stratify=False))

    [issue] = find(issues, "stratify_disabled")
    assert issue.severity == "warning"
    assert issue.step_index == 0


def test_stratified_classification_split_does_not_warn():
    assert find(validate(split(stratify=True)), "stratify_disabled") == []


# --- determinism -----------------------------------------------------------------------------


def test_issue_order_is_plan_level_then_by_step_then_end_of_plan():
    issues = validate(
        scale(["churn"]),
        split(stratify=False),
        impute(["nope"]),
        target="churn",
    )

    assert [(i.code, i.step_index, i.column) for i in issues] == [
        ("stateful_before_split", 0, None),
        ("target_in_step", 0, "churn"),
        ("stratify_disabled", 1, None),
        ("unknown_column", 2, "nope"),
        ("uninformative_column_kept", None, "customer_id"),
        ("uninformative_column_kept", None, "country"),
        ("missing_values_remain", None, "income"),
        ("missing_values_remain", None, "job"),
        ("unencoded_column_remains", None, "city"),
        ("unencoded_column_remains", None, "job"),
        ("unencoded_column_remains", None, "gender"),
        ("unencoded_column_remains", None, "signup"),
        ("unencoded_column_remains", None, "notes"),
        ("unencoded_column_remains", None, "segment"),
    ]


def test_plan_level_issues_come_before_step_issues():
    issues = validate(scale(["age"]), target="nope")

    assert [i.code for i in issues[:3]] == [
        "target_mismatch",
        "target_not_in_profile",
        "invalid_split_count",
    ]


def test_model_family_mismatch_comes_between_the_target_checks_and_the_split_check():
    plan = make_plan(scale(["age"]), target="nope", model_family="any")

    issues = validate_plan(plan, PROFILE, expected_model_family="tree")

    assert [i.code for i in issues[:4]] == [
        "target_mismatch",
        "target_not_in_profile",
        "model_family_mismatch",
        "invalid_split_count",
    ]


def test_same_input_gives_the_same_issues_in_the_same_order():
    steps = [impute(["income"]), scale(["customer_id"]), split(stratify=False), dedupe()]

    first = validate(*steps)
    second = validate(*steps)

    assert first == second
    assert first, "scenario should produce issues"
    assert [i.model_dump() for i in first] == [i.model_dump() for i in second]


def test_validate_plan_does_not_modify_its_inputs():
    plan = make_plan(*VALID_STEPS)
    profile = PROFILE.model_copy(deep=True)
    plan_before = plan.model_dump()

    validate_plan(plan, profile)

    assert plan.model_dump() == plan_before
    assert profile == PROFILE
