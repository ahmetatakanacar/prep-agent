import pytest

from prep_agent.ops import OpError, OpRegistry, get_op, list_ops
from prep_agent.ops.base import StatefulOpSpec, StatelessOpSpec
from prep_agent.plan import DedupeStep, Plan

EXPECTED_KINDS = {
    "dedupe": "stateless",
    "drop_columns": "stateless",
    "drop_missing_target": "stateless",
    "split": "split",
    "impute": "stateful",
    "encode": "stateful",
    "scale": "stateful",
}

VALID_STEPS = {
    "dedupe": {"subset": None, "keep": "first"},
    "drop_columns": {"columns": ["a"]},
    "drop_missing_target": {},
    "split": {"stratify": False},
    "impute": {"columns": ["a"], "strategy": "mean"},
    "encode": {"columns": ["a"], "method": "onehot"},
    "scale": {"columns": ["a"], "method": "standard"},
}


def test_all_ops_are_registered_with_the_right_kind():
    assert {spec.name: spec.kind for spec in list_ops()} == EXPECTED_KINDS


def test_list_ops_order_is_stable():
    assert [s.name for s in list_ops()] == list(EXPECTED_KINDS)
    assert list_ops() == list_ops()


def test_get_op_returns_the_registered_spec():
    for spec in list_ops():
        assert get_op(spec.name) is spec


def test_get_op_unknown_name_raises_op_error_listing_known_ops():
    with pytest.raises(OpError, match="unknown op 'nope'.*dedupe"):
        get_op("nope")


def test_every_spec_has_a_description():
    for spec in list_ops():
        assert spec.description.strip(), spec.name
        assert len(spec.description) < 200, spec.name


def test_step_models_match_the_plan_schema():
    plan = Plan.model_validate(
        {
            "target": "y",
            "task_type": "classification",
            "model_family": "any",
            "steps": [
                {"op": name, "reason": "r", **params} for name, params in VALID_STEPS.items()
            ],
        }
    )

    assert {s.op for s in plan.steps} == set(EXPECTED_KINDS)
    for step in plan.steps:
        assert isinstance(step, get_op(step.op).step_model)


def test_spec_names_match_their_step_model_op():
    for spec in list_ops():
        assert spec.step_model.model_fields["op"].annotation.__args__ == (spec.name,)


def test_each_kind_provides_the_right_callable():
    for spec in list_ops():
        if spec.kind == "stateful":
            assert callable(spec.build_transformer)
            assert not hasattr(spec, "apply")
        else:
            assert callable(spec.apply)
            assert not hasattr(spec, "build_transformer")


def test_registering_the_same_name_twice_raises_op_error():
    registry = OpRegistry()
    spec = get_op("dedupe")
    registry.register(spec)

    with pytest.raises(OpError, match="already registered.*dedupe"):
        registry.register(spec)


def test_duplicate_name_with_a_different_spec_is_rejected_too():
    registry = OpRegistry()
    registry.register(get_op("dedupe"))
    imposter = StatelessOpSpec(
        name="dedupe",
        step_model=DedupeStep,
        description="another implementation",
        apply=lambda df, step, ctx: df,
    )

    with pytest.raises(OpError, match="already registered"):
        registry.register(imposter)


def test_spec_name_must_match_the_step_model_op():
    registry = OpRegistry()
    mismatched = StatelessOpSpec(
        name="dedupe_all",
        step_model=DedupeStep,
        description="wrong name",
        apply=lambda df, step, ctx: df,
    )

    with pytest.raises(OpError, match="dedupe_all"):
        registry.register(mismatched)


def test_a_fresh_registry_is_empty_and_independent_of_the_default_one():
    registry = OpRegistry()

    assert registry.list_ops() == []
    registry.register(get_op("dedupe"))
    assert [s.name for s in registry.list_ops()] == ["dedupe"]
    assert len(list_ops()) == len(EXPECTED_KINDS)


def test_stateful_spec_type():
    assert isinstance(get_op("impute"), StatefulOpSpec)
