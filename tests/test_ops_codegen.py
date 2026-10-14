"""Each op's generated code must do what the op itself does.

The generated statements are executed here, in a namespace that holds only what the op
declared it imports, and compared with the result of the op's own implementation.
"""

import importlib

import numpy as np
import pandas as pd
import pytest
import sklearn

from prep_agent.ops import OpContext, get_op, list_ops
from prep_agent.ops.code import Import, render_statements
from prep_agent.plan import Plan

TARGET = "t"


def make_step(**step):
    plan = Plan.model_validate(
        {
            "target": TARGET,
            "task_type": "classification",
            "model_family": "any",
            "steps": [{"reason": "test", **step}],
        }
    )
    return plan.steps[0]


def namespace(imports: tuple[Import, ...]) -> dict:
    ns: dict = {"TARGET": TARGET}
    for item in imports:
        module = importlib.import_module(item.module)
        if item.name is None:
            ns[item.alias or item.module.split(".")[0]] = module
        else:
            ns[item.name] = getattr(module, item.name)
    return ns


def run_statements(code, **variables):
    ns = namespace(code.imports) | variables
    for helper in code.helpers:
        exec(helper, ns)
    exec("\n".join(render_statements(code.statements)), ns)
    return ns


def frame() -> pd.DataFrame:
    n = 30
    i = np.arange(n)
    return pd.DataFrame(
        {
            "a": np.where(i % 7 == 0, np.nan, i * 1.5),
            "b": [f"k{j % 4}" for j in i],
            "c": np.where(i % 5 == 0, None, [f"v{j % 3}" for j in i]),
            "t": np.where(i % 11 == 0, np.nan, (i % 2).astype(float)),
        }
    ).pipe(lambda d: pd.concat([d, d.iloc[:6]], ignore_index=True))


@pytest.mark.parametrize(
    "step",
    [
        {"op": "dedupe", "subset": None, "keep": "first"},
        {"op": "dedupe", "subset": None, "keep": "last"},
        {"op": "dedupe", "subset": ["b"], "keep": "first"},
        {"op": "dedupe", "subset": ["a", "b"], "keep": "last"},
        {"op": "drop_columns", "columns": ["a"]},
        {"op": "drop_columns", "columns": ["b", "c"]},
        {"op": "drop_missing_target"},
    ],
)
def test_stateless_code_matches_the_op(step):
    step = make_step(**step)
    spec = get_op(step.op)
    df = frame()

    expected = spec.apply(df.copy(), step, OpContext(target=TARGET))
    ns = run_statements(spec.codegen(step), df=df.copy())

    pd.testing.assert_frame_equal(ns["df"], expected)


HARD_COLUMN = 'x"); import os  #'


def cast_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "whole": [1, 2, 3, 1, 2, 3],
            "with_gaps": [1.0, np.nan, 20.0, 1.0, np.nan, 3.0],
            "fractions": [0.5, 1.0, 2.25, np.nan, 1.0, 0.5],
            "text": ["p", None, "q", "p", "r", None],
            "flag": [True, False, True, True, False, False],
            HARD_COLUMN: [5, 6, 7, 5, 6, 7],
            "t": [0, 1, 0, 1, 0, 1],
        }
    )


@pytest.mark.parametrize(
    "columns",
    [
        ["whole"],
        ["with_gaps"],
        ["fractions"],
        ["text", "flag"],
        [HARD_COLUMN],
        ["whole", "with_gaps", "fractions", "text", "flag", HARD_COLUMN],
    ],
)
def test_cast_to_category_code_matches_the_op(columns):
    step = make_step(op="cast_to_category", columns=columns)
    spec = get_op("cast_to_category")
    df = cast_frame()

    expected = spec.apply(df.copy(), step, OpContext(target=TARGET))
    ns = run_statements(spec.codegen(step), df=df.copy())

    pd.testing.assert_frame_equal(ns["df"], expected)


def test_cast_to_category_code_does_not_modify_the_input():
    step = make_step(op="cast_to_category", columns=["whole"])
    df = cast_frame()

    ns = run_statements(get_op("cast_to_category").codegen(step), df=df)

    pd.testing.assert_frame_equal(df, cast_frame())
    assert ns["df"] is not df


def test_dedupe_code_resets_the_index():
    step = make_step(op="dedupe", subset=None, keep="first")

    ns = run_statements(get_op("dedupe").codegen(step), df=frame())

    assert list(ns["df"].index) == list(range(len(ns["df"])))


@pytest.mark.parametrize(
    "step",
    [
        {"op": "split", "stratify": True},
        {"op": "split", "stratify": False},
        {"op": "split", "stratify": True, "test_size": 0.35, "random_state": 7},
        {"op": "split", "stratify": False, "test_size": 0.5, "random_state": 0},
    ],
)
def test_split_code_matches_the_op(step):
    step = make_step(**step)
    spec = get_op("split")
    df = frame().dropna(subset=[TARGET]).reset_index(drop=True)

    expected_train, expected_test = spec.apply(df.copy(), step, OpContext(target=TARGET))
    ns = run_statements(spec.codegen(step), df=df.copy())

    pd.testing.assert_frame_equal(ns["train"], expected_train)
    pd.testing.assert_frame_equal(ns["test"], expected_test)


STATEFUL_STEPS = [
    {"op": "impute", "columns": ["a"], "strategy": "mean"},
    {"op": "impute", "columns": ["a"], "strategy": "median", "add_indicator": True},
    {"op": "impute", "columns": ["c"], "strategy": "most_frequent"},
    {"op": "impute", "columns": ["a"], "strategy": "constant", "fill_value": -1},
    {"op": "impute", "columns": ["a"], "strategy": "constant", "fill_value": 2.5},
    {"op": "impute", "columns": ["c"], "strategy": "constant", "fill_value": "missing"},
    {
        "op": "impute",
        "columns": ["c"],
        "strategy": "constant",
        "fill_value": 'x"); import os  #',
        "add_indicator": True,
    },
    {"op": "encode", "columns": ["b"], "method": "onehot"},
    {"op": "encode", "columns": ["b"], "method": "onehot", "max_categories": 3},
    {"op": "encode", "columns": ["b", "c"], "method": "ordinal"},
    {"op": "scale", "columns": ["a"], "method": "standard"},
    {"op": "scale", "columns": ["a"], "method": "minmax"},
    {"op": "scale", "columns": ["a"], "method": "robust"},
]


@pytest.mark.parametrize("step", STATEFUL_STEPS)
def test_stateful_code_builds_the_same_transformer(step):
    step = make_step(**step)
    spec = get_op(step.op)
    x = frame().drop(columns=TARGET)
    train, test = x.iloc[:25], x.iloc[25:]

    built = spec.build_transformer(step)
    code = spec.codegen(step)
    generated = eval(code.expression.flat(), namespace(code.imports))

    with sklearn.config_context(print_changed_only=False):
        assert repr(generated) == repr(built)
    pd.testing.assert_frame_equal(generated.fit_transform(train), built.fit_transform(train))
    pd.testing.assert_frame_equal(generated.transform(test), built.transform(test))


def test_every_registered_op_has_code_generation():
    for spec in list_ops():
        assert callable(spec.codegen), spec.name
