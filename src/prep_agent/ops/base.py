"""Shared types for ops: the context, the error, and one spec class per kind of op."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import pandas as pd
from pydantic import BaseModel
from sklearn.compose import ColumnTransformer

OpKind = Literal["stateless", "split", "stateful"]


class OpError(ValueError):
    """Raised when an op cannot be applied or registered."""


@dataclass(frozen=True)
class OpContext:
    """Facts about the run that ops may need besides their own step."""

    target: str


@dataclass(frozen=True)
class OpSpec:
    """Registry entry shared by all ops. `description` is what the planner will be shown."""

    name: str
    step_model: type[BaseModel]
    description: str
    kind: ClassVar[OpKind]


@dataclass(frozen=True)
class StatelessOpSpec(OpSpec):
    """An op that learns nothing from the data. Runs before the split."""

    apply: Callable[[pd.DataFrame, Any, OpContext], pd.DataFrame]
    kind: ClassVar[OpKind] = "stateless"


@dataclass(frozen=True)
class SplitOpSpec(OpSpec):
    """The op that separates train from test."""

    apply: Callable[[pd.DataFrame, Any, OpContext], tuple[pd.DataFrame, pd.DataFrame]]
    kind: ClassVar[OpKind] = "split"


@dataclass(frozen=True)
class StatefulOpSpec(OpSpec):
    """An op that learns from data. Runs after the split and is fit on train only.

    `build_transformer` returns an unfitted transformer that takes and returns whole frames:
    the columns named in the step are transformed and every other column passes through.
    """

    build_transformer: Callable[[Any], ColumnTransformer]
    kind: ClassVar[OpKind] = "stateful"
