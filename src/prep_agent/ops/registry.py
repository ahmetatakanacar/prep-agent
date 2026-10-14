"""The registry of ops. A plan step is only valid if its `op` is registered here."""

from typing import get_args

from prep_agent.ops.base import OpError, OpSpec
from prep_agent.ops.split import SPLIT
from prep_agent.ops.stateful import ENCODE, IMPUTE, SCALE
from prep_agent.ops.stateless import CAST_TO_CATEGORY, DEDUPE, DROP_COLUMNS, DROP_MISSING_TARGET


class OpRegistry:
    """Holds op specs by name, in registration order."""

    def __init__(self) -> None:
        self._ops: dict[str, OpSpec] = {}

    def register(self, spec: OpSpec) -> None:
        declared = get_args(spec.step_model.model_fields["op"].annotation)
        if declared != (spec.name,):
            raise OpError(
                f"op name {spec.name!r} does not match the op literal of its step model: {declared}"
            )
        if not callable(getattr(spec, "codegen", None)):
            raise OpError(f"op {spec.name!r} has no code generation for pipeline.py")
        if spec.name in self._ops:
            raise OpError(f"op is already registered: {spec.name!r}")
        self._ops[spec.name] = spec

    def get_op(self, name: str) -> OpSpec:
        try:
            return self._ops[name]
        except KeyError:
            known = ", ".join(self._ops)
            raise OpError(f"unknown op {name!r}; known ops: {known}") from None

    def list_ops(self) -> list[OpSpec]:
        return list(self._ops.values())


REGISTRY = OpRegistry()
for _spec in (
    DEDUPE,
    DROP_COLUMNS,
    DROP_MISSING_TARGET,
    CAST_TO_CATEGORY,
    SPLIT,
    IMPUTE,
    ENCODE,
    SCALE,
):
    REGISTRY.register(_spec)


def get_op(name: str) -> OpSpec:
    return REGISTRY.get_op(name)


def list_ops() -> list[OpSpec]:
    return REGISTRY.list_ops()
