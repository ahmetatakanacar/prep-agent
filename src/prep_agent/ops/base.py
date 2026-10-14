"""Shared types for ops: the context, the error, and one spec class per kind of op."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import pandas as pd
from pydantic import BaseModel
from sklearn.compose import ColumnTransformer

from prep_agent.ops.code import Assign, Expr, Import

OpKind = Literal["stateless", "split", "stateful"]


class OpError(ValueError):
    """Raised when an op cannot be applied or registered."""


@dataclass(frozen=True)
class OpContext:
    """Facts about the run that ops may need besides their own step."""

    target: str


@dataclass(frozen=True)
class StatementCode:
    """Code of a stateless or split op for the standalone pipeline.py.

    The statements work on `df` (a split op assigns `train` and `test` instead) and may refer to
    `TARGET`, the module constant that holds the target column. `imports` lists everything else
    the statements need. `helpers` are functions the statements call, as complete source text with
    no data in it; pipeline.py defines each distinct helper once, ahead of the steps.
    """

    statements: tuple[Assign, ...]
    imports: tuple[Import, ...] = ()
    helpers: tuple[str, ...] = ()


@dataclass(frozen=True)
class TransformerCode:
    """Code of a stateful op: one expression that builds the same unfitted transformer."""

    expression: Expr
    imports: tuple[Import, ...] = ()


@dataclass(frozen=True)
class OpSpec:
    """Registry entry shared by all ops. `description` is what the planner will be shown."""

    name: str
    step_model: type[BaseModel]
    description: str
    kind: ClassVar[OpKind]


@dataclass(frozen=True)
class StatelessOpSpec(OpSpec):
    """An op that learns nothing from the data. Runs before the split.

    `codegen` writes the same operation as source for pipeline.py. It is required, so an op
    cannot be added without it.
    """

    apply: Callable[[pd.DataFrame, Any, OpContext], pd.DataFrame]
    codegen: Callable[[Any], StatementCode]
    kind: ClassVar[OpKind] = "stateless"


@dataclass(frozen=True)
class SplitOpSpec(OpSpec):
    """The op that separates train from test. `codegen` is as for `StatelessOpSpec`."""

    apply: Callable[[pd.DataFrame, Any, OpContext], tuple[pd.DataFrame, pd.DataFrame]]
    codegen: Callable[[Any], StatementCode]
    kind: ClassVar[OpKind] = "split"


@dataclass(frozen=True)
class StatefulOpSpec(OpSpec):
    """An op that learns from data. Runs after the split and is fit on train only.

    `build_transformer` returns an unfitted transformer that takes and returns whole frames:
    the columns named in the step are transformed and every other column passes through.
    `codegen` returns the source of an expression that builds the same transformer.
    """

    build_transformer: Callable[[Any], ColumnTransformer]
    codegen: Callable[[Any], TransformerCode]
    kind: ClassVar[OpKind] = "stateful"
