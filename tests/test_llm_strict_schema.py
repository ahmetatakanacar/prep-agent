import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import BaseModel, Field

from prep_agent.llm.ping import PingReply
from prep_agent.llm.strict_schema import (
    UnsupportedSchemaError,
    drop_added_nulls,
    to_strict_schema,
)
from prep_agent.plan import Plan

FIXTURE_PLAN = Path(__file__).parent / "fixtures" / "customers_plan.json"


class Inner(BaseModel):
    label: str
    weight: float = 1.0


class Wrapper(BaseModel):
    name: str
    inner: Inner
    maybe_inner: Inner | None = None
    optional_inner: Inner = Field(default_factory=lambda: Inner(label="x"))
    count: int = 3
    note: str | None = None
    tags: list[str] = Field(default_factory=list)


def walk_schemas(node, path=()):
    """Yield (path, schema) for every schema node, skipping property-name and data containers."""
    if isinstance(node, list):
        for index, item in enumerate(node):
            yield from walk_schemas(item, (*path, index))
        return
    if not isinstance(node, dict):
        return
    yield path, node
    for key, value in node.items():
        if key in ("properties", "$defs"):
            for name, sub in value.items():
                yield from walk_schemas(sub, (*path, key, name))
        elif key not in ("enum", "const", "examples"):
            yield from walk_schemas(value, (*path, key))


def resolve_refs(schema):
    return [node["$ref"] for _, node in walk_schemas(schema) if "$ref" in node]


MODELS = [PingReply, Inner, Wrapper, Plan]


def test_the_walker_finds_the_keywords_in_the_unconverted_schema():
    found = {
        keyword
        for _, node in walk_schemas(Plan.model_json_schema())
        for keyword in ("oneOf", "discriminator", "default")
        if keyword in node
    }

    assert found == {"oneOf", "discriminator", "default"}


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.__name__)
def test_no_unsupported_keyword_survives_at_any_level(model):
    schema = to_strict_schema(model)

    for path, node in walk_schemas(schema):
        for keyword in ("oneOf", "discriminator", "default", "allOf"):
            assert keyword not in node, f"{keyword} left at {path}"


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.__name__)
def test_every_object_requires_all_properties_and_forbids_extras(model):
    schema = to_strict_schema(model)

    objects = [(p, n) for p, n in walk_schemas(schema) if "properties" in n]
    assert objects
    for path, node in objects:
        assert node["required"] == list(node["properties"]), path
        assert node["additionalProperties"] is False, path


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.__name__)
def test_result_is_a_valid_json_schema_and_every_ref_resolves(model):
    schema = to_strict_schema(model)

    Draft202012Validator.check_schema(schema)
    for ref in resolve_refs(schema):
        assert ref.startswith("#/$defs/")
        assert ref.removeprefix("#/$defs/") in schema["$defs"]


def test_defs_are_preserved():
    original = Plan.model_json_schema()

    schema = to_strict_schema(Plan)

    assert set(schema["$defs"]) == set(original["$defs"])


def test_the_pydantic_schema_is_not_mutated():
    before = json.dumps(Plan.model_json_schema(), sort_keys=True)

    to_strict_schema(Plan)

    assert json.dumps(Plan.model_json_schema(), sort_keys=True) == before


def test_discriminated_union_becomes_plain_any_of():
    schema = to_strict_schema(Plan)

    items = schema["properties"]["steps"]["items"]
    assert [member["$ref"] for member in items["anyOf"]] == [
        f"#/$defs/{name}"
        for name in (
            "DedupeStep",
            "DropColumnsStep",
            "DropMissingTargetStep",
            "CastToCategoryStep",
            "SplitStep",
            "ImputeStep",
            "EncodeStep",
            "ScaleStep",
        )
    ]


def test_optional_field_with_default_becomes_nullable():
    schema = to_strict_schema(Wrapper)

    count = schema["properties"]["count"]
    assert {"type": "null"} in count["anyOf"]
    assert {"type": "integer"} in count["anyOf"]


def test_nullable_nested_model_keeps_its_ref_and_gains_no_second_null():
    schema = to_strict_schema(Wrapper)

    members = schema["properties"]["maybe_inner"]["anyOf"]
    assert sum(1 for member in members if member == {"type": "null"}) == 1
    assert {"$ref": "#/$defs/Inner"} in members


def test_optional_nested_model_with_default_is_wrapped_in_any_of_with_null():
    schema = to_strict_schema(Wrapper)

    members = schema["properties"]["optional_inner"]["anyOf"]
    assert {"$ref": "#/$defs/Inner"} in members
    assert {"type": "null"} in members


def test_required_field_is_left_as_it_was():
    original = Wrapper.model_json_schema()["properties"]["name"]

    assert to_strict_schema(Wrapper)["properties"]["name"] == original


def test_description_stays_on_the_outer_schema_of_a_wrapped_field():
    class Described(BaseModel):
        size: int = Field(default=1, description="How big.")

    size = to_strict_schema(Described)["properties"]["size"]

    assert size["description"] == "How big."
    assert all("description" not in member for member in size["anyOf"])


def test_a_property_named_like_a_keyword_is_kept():
    class Odd(BaseModel):
        default: int = 1
        discriminator: str = "x"

    schema = to_strict_schema(Odd)

    assert set(schema["properties"]) == {"default", "discriminator"}
    assert schema["required"] == ["default", "discriminator"]


def test_free_form_mapping_is_rejected():
    class Loose(BaseModel):
        values: dict[str, int]

    with pytest.raises(UnsupportedSchemaError, match="values"):
        to_strict_schema(Loose)


# --- guard against the two schemas drifting apart -------------------------------------------


def step(op, **fields):
    return {"op": op, "reason": "because", **fields}


def plan_dict(*steps, target="t", task_type="classification", model_family="any"):
    return {
        "target": target,
        "task_type": task_type,
        "model_family": model_family,
        "steps": list(steps),
    }


PLANS = {
    "fixture": json.loads(FIXTURE_PLAN.read_text(encoding="utf-8")),
    "no_steps": plan_dict(),
    "dedupe_all_columns": plan_dict(step("dedupe", keep="first")),
    "dedupe_subset_last": plan_dict(step("dedupe", subset=["a", "b"], keep="last")),
    "drop_columns": plan_dict(step("drop_columns", columns=["a"])),
    "drop_missing_target": plan_dict(step("drop_missing_target")),
    "cast_to_category": plan_dict(step("cast_to_category", columns=["hour"])),
    "cast_several_then_onehot": plan_dict(
        step("cast_to_category", columns=["hour", "month", "weekday"]),
        step("split", stratify=False),
        step("encode", columns=["hour", "month", "weekday"], method="onehot"),
        task_type="regression",
    ),
    "split_defaults": plan_dict(step("split", stratify=False)),
    "split_custom": plan_dict(step("split", stratify=True, test_size=0.3, random_state=7)),
    "impute_median": plan_dict(step("impute", columns=["a"], strategy="median")),
    "impute_mean_indicator": plan_dict(
        step("impute", columns=["a"], strategy="mean", add_indicator=True)
    ),
    "impute_most_frequent": plan_dict(step("impute", columns=["a"], strategy="most_frequent")),
    "impute_constant_int": plan_dict(
        step("impute", columns=["a"], strategy="constant", fill_value=0)
    ),
    "impute_constant_float": plan_dict(
        step("impute", columns=["a"], strategy="constant", fill_value=0.5)
    ),
    "impute_constant_str": plan_dict(
        step("impute", columns=["a"], strategy="constant", fill_value="missing")
    ),
    "encode_onehot": plan_dict(step("encode", columns=["a"], method="onehot")),
    "encode_onehot_capped": plan_dict(
        step("encode", columns=["a"], method="onehot", max_categories=5)
    ),
    "encode_ordinal": plan_dict(step("encode", columns=["a", "b"], method="ordinal")),
    "scale_standard": plan_dict(step("scale", columns=["a"], method="standard")),
    "scale_minmax": plan_dict(step("scale", columns=["a"], method="minmax")),
    "scale_robust": plan_dict(step("scale", columns=["a"], method="robust")),
    "regression_tree": plan_dict(
        step("drop_columns", columns=["a"]),
        step("split", stratify=False),
        step("impute", columns=["b"], strategy="median"),
        task_type="regression",
        model_family="tree",
    ),
    "all_ops": plan_dict(
        step("dedupe", subset=["a"], keep="first"),
        step("drop_columns", columns=["id"]),
        step("drop_missing_target"),
        step("cast_to_category", columns=["d"]),
        step("split", stratify=True, test_size=0.25, random_state=1),
        step("impute", columns=["a"], strategy="constant", fill_value="none", add_indicator=True),
        step("encode", columns=["b"], method="onehot", max_categories=3),
        step("scale", columns=["c"], method="standard"),
        model_family="linear",
    ),
}


@pytest.fixture(scope="module")
def strict_validator():
    return Draft202012Validator(to_strict_schema(Plan))


@pytest.mark.parametrize("name", PLANS)
def test_every_valid_plan_passes_the_strict_schema(name, strict_validator):
    plan = Plan.model_validate(PLANS[name])

    # model_dump lists every field, including defaults: the shape strict mode forces on the LLM.
    strict_validator.validate(plan.model_dump(mode="json"))


@pytest.mark.parametrize("name", PLANS)
def test_a_plan_survives_the_strict_round_trip(name):
    plan = Plan.model_validate(PLANS[name])
    answer = json.dumps(plan.model_dump(mode="json"))

    restored = Plan.model_validate_json(json.dumps(drop_added_nulls(json.loads(answer), Plan)))

    assert restored == plan


def test_the_strict_schema_still_rejects_broken_plans(strict_validator):
    valid = Plan.model_validate(PLANS["all_ops"]).model_dump(mode="json")
    unknown_op = json.loads(json.dumps(valid))
    unknown_op["steps"][0]["op"] = "train_model"
    missing_field = json.loads(json.dumps(valid))
    del missing_field["steps"][0]["reason"]
    extra_field = json.loads(json.dumps(valid))
    extra_field["steps"][0]["code"] = "import os"
    bad_family = json.loads(json.dumps(valid))
    bad_family["model_family"] = "neural"

    for broken in (unknown_op, missing_field, extra_field, bad_family):
        assert not strict_validator.is_valid(broken)


# --- answers shaped like what strict mode forces the model to write ---------------------------


def llm_answer(*steps, **plan_fields):
    """Hand-written answer: every property present, unset optionals sent as null."""
    return {
        "target": "churned",
        "task_type": "classification",
        "model_family": "any",
        "steps": list(steps),
        **plan_fields,
    }


HAND_WRITTEN = {
    "split_with_nulls": llm_answer(
        {
            "op": "split",
            "reason": "hold out a test set",
            "test_size": None,
            "stratify": True,
            "random_state": None,
        }
    ),
    "impute_with_nulls": llm_answer(
        {
            "op": "impute",
            "reason": "fill gaps",
            "columns": ["age"],
            "strategy": "median",
            "fill_value": None,
            "add_indicator": None,
        }
    ),
    "encode_with_nulls": llm_answer(
        {
            "op": "encode",
            "reason": "categorical",
            "columns": ["plan"],
            "method": "onehot",
            "max_categories": None,
        }
    ),
    "cast_to_category": llm_answer(
        {"op": "cast_to_category", "reason": "an hour is a category", "columns": ["hour"]},
        {
            "op": "split",
            "reason": "hold out",
            "test_size": None,
            "stratify": False,
            "random_state": None,
        },
        {
            "op": "encode",
            "reason": "categorical now",
            "columns": ["hour"],
            "method": "onehot",
            "max_categories": None,
        },
    ),
    "dedupe_with_null_subset": llm_answer(
        {"op": "dedupe", "reason": "duplicates", "subset": None, "keep": "first"}
    ),
    "fully_specified": llm_answer(
        {
            "op": "split",
            "reason": "hold out",
            "test_size": 0.3,
            "stratify": False,
            "random_state": 3,
        },
        {
            "op": "impute",
            "reason": "constant",
            "columns": ["age"],
            "strategy": "constant",
            "fill_value": 0,
            "add_indicator": True,
        },
    ),
}


@pytest.mark.parametrize("name", HAND_WRITTEN)
def test_hand_written_answers_fit_the_strict_schema(name, strict_validator):
    strict_validator.validate(HAND_WRITTEN[name])


@pytest.mark.parametrize("name", HAND_WRITTEN)
def test_hand_written_answers_become_valid_plans(name):
    answer = HAND_WRITTEN[name]

    plan = Plan.model_validate_json(json.dumps(drop_added_nulls(answer, Plan)))

    assert plan.target == "churned"
    assert len(plan.steps) == len(answer["steps"])


def test_answers_without_nulls_validate_directly_against_plan():
    plan = Plan.model_validate_json(json.dumps(HAND_WRITTEN["fully_specified"]))

    assert plan.steps[0].test_size == 0.3


def test_null_for_a_defaulted_field_is_rejected_by_plan_without_the_cleanup():
    # This is why the cleanup exists: the schema tells the model null is fine for test_size, but
    # Plan itself only accepts a number there.
    with pytest.raises(ValueError, match="test_size"):
        Plan.model_validate_json(json.dumps(HAND_WRITTEN["split_with_nulls"]))


def test_cleanup_restores_defaults_for_dropped_nulls():
    plan = Plan.model_validate_json(
        json.dumps(drop_added_nulls(HAND_WRITTEN["split_with_nulls"], Plan))
    )

    assert plan.steps[0].test_size == 0.2
    assert plan.steps[0].random_state == 42


def test_cleanup_keeps_nulls_the_model_allows():
    answer = drop_added_nulls(HAND_WRITTEN["dedupe_with_null_subset"], Plan)

    assert answer["steps"][0]["subset"] is None


def test_cleanup_does_not_touch_nulls_it_cannot_attribute():
    bogus = llm_answer({"op": "no_such_op", "reason": "x", "test_size": None})

    assert drop_added_nulls(bogus, Plan) == bogus


def test_cleanup_leaves_required_nulls_for_validation_to_report():
    answer = llm_answer({"op": "scale", "reason": "x", "columns": None, "method": "robust"})

    assert drop_added_nulls(answer, Plan) == answer


def test_cleanup_works_on_nested_non_union_models():
    answer = {
        "name": "n",
        "inner": {"label": "l", "weight": None},
        "maybe_inner": {"label": "m", "weight": None},
        "optional_inner": None,
        "count": None,
        "note": None,
        "tags": None,
    }

    cleaned = drop_added_nulls(answer, Wrapper)

    assert cleaned == {
        "name": "n",
        "inner": {"label": "l"},
        "maybe_inner": {"label": "m"},
        "note": None,
    }
    assert Wrapper.model_validate(cleaned).count == 3


def test_cleanup_does_not_modify_its_input():
    answer = json.loads(json.dumps(HAND_WRITTEN["split_with_nulls"]))
    snapshot = json.loads(json.dumps(answer))

    drop_added_nulls(answer, Plan)

    assert answer == snapshot


def test_cleanup_passes_non_object_answers_through():
    assert drop_added_nulls([1, None], Plan) == [1, None]
    assert drop_added_nulls("text", Plan) == "text"
