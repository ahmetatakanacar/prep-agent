"""Ops that learn from data, so they must run after the split and be fit on train only.

Each builder returns an unfitted `ColumnTransformer` that transforms the step's columns and
passes every other column through, so a transformer can be fit on a whole train frame and
applied to a whole test frame. Output is a pandas DataFrame that keeps the index.
"""

import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

from prep_agent.ops.base import StatefulOpSpec, TransformerCode
from prep_agent.ops.code import Call, Expr, Import, call, ref
from prep_agent.plan import EncodeStep, ImputeStep, ScaleStep


def _on_columns(op: str, columns: list[str], transformer: TransformerMixin) -> ColumnTransformer:
    return ColumnTransformer(
        [(op, transformer, list(columns))],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def _build_imputer(step: ImputeStep) -> ColumnTransformer:
    # keep_empty_features keeps a column that is entirely missing in train, so the output
    # columns do not depend on the data.
    imputer = SimpleImputer(
        strategy=step.strategy,
        fill_value=step.fill_value,
        keep_empty_features=True,
    )
    transformers = [(step.op, imputer, list(step.columns))]
    if step.add_indicator:
        # SimpleImputer(add_indicator=True) would stack the indicator onto the imputed values,
        # and for text columns that makes an object column holding booleans, which no model
        # can read. A separate transformer keeps the indicators as 0/1 integers.
        indicator = Pipeline(
            [
                ("indicator", MissingIndicator(features="missing-only", error_on_new=False)),
                ("to_int", _to_int()),
            ]
        )
        transformers.append(("indicator", indicator, list(step.columns)))
    return ColumnTransformer(
        transformers, remainder="passthrough", verbose_feature_names_out=False
    ).set_output(transform="pandas")


def _to_int() -> FunctionTransformer:
    return FunctionTransformer(
        func=pd.DataFrame.astype, kw_args={"dtype": "int64"}, feature_names_out="one-to-one"
    )


def _build_encoder(step: EncodeStep) -> ColumnTransformer:
    if step.method == "onehot":
        # With infrequent_if_exist an unseen category falls into the infrequent group when
        # there is one and becomes all zeros otherwise, so transform never fails.
        encoder = OneHotEncoder(
            sparse_output=False,
            handle_unknown="infrequent_if_exist",
            max_categories=step.max_categories,
        )
    else:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    return _on_columns(step.op, step.columns, encoder)


def _build_scaler(step: ScaleStep) -> ColumnTransformer:
    scalers = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}
    return _on_columns(step.op, step.columns, scalers[step.method]())


# The generated code mirrors the builders above, parameter by parameter.
def _on_columns_code(op: str, columns: list[str], transformer: Expr) -> Call:
    return call(
        "ColumnTransformer",
        [(op, transformer, columns)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).then("set_output", transform="pandas")


_COLUMN_TRANSFORMER = Import("sklearn.compose", "ColumnTransformer")


def _impute_code(step: ImputeStep) -> TransformerCode:
    imputer = call(
        "SimpleImputer",
        strategy=step.strategy,
        fill_value=step.fill_value,
        keep_empty_features=True,
    )
    imports = (_COLUMN_TRANSFORMER, Import("sklearn.impute", "SimpleImputer"))
    transformers = [(step.op, imputer, step.columns)]
    if step.add_indicator:
        indicator = call(
            "Pipeline",
            [
                (
                    "indicator",
                    call("MissingIndicator", features="missing-only", error_on_new=False),
                ),
                (
                    "to_int",
                    call(
                        "FunctionTransformer",
                        func=ref("pd.DataFrame.astype"),
                        kw_args={"dtype": "int64"},
                        feature_names_out="one-to-one",
                    ),
                ),
            ],
        )
        transformers.append(("indicator", indicator, step.columns))
        imports += (
            Import("pandas", alias="pd"),
            Import("sklearn.impute", "MissingIndicator"),
            Import("sklearn.pipeline", "Pipeline"),
            Import("sklearn.preprocessing", "FunctionTransformer"),
        )
    expression = call(
        "ColumnTransformer",
        transformers,
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).then("set_output", transform="pandas")
    return TransformerCode(expression, imports)


def _encode_code(step: EncodeStep) -> TransformerCode:
    if step.method == "onehot":
        name = "OneHotEncoder"
        encoder = call(
            name,
            sparse_output=False,
            handle_unknown="infrequent_if_exist",
            max_categories=step.max_categories,
        )
    else:
        name = "OrdinalEncoder"
        encoder = call(name, handle_unknown="use_encoded_value", unknown_value=-1)
    return TransformerCode(
        _on_columns_code(step.op, step.columns, encoder),
        (_COLUMN_TRANSFORMER, Import("sklearn.preprocessing", name)),
    )


def _scale_code(step: ScaleStep) -> TransformerCode:
    names = {"standard": "StandardScaler", "minmax": "MinMaxScaler", "robust": "RobustScaler"}
    name = names[step.method]
    return TransformerCode(
        _on_columns_code(step.op, step.columns, call(name)),
        (_COLUMN_TRANSFORMER, Import("sklearn.preprocessing", name)),
    )


IMPUTE = StatefulOpSpec(
    name="impute",
    step_model=ImputeStep,
    description=(
        "Fill missing values with the mean, median, most frequent or a constant value, "
        "learned on train. Can add a missing-value indicator column."
    ),
    build_transformer=_build_imputer,
    codegen=_impute_code,
)
ENCODE = StatefulOpSpec(
    name="encode",
    step_model=EncodeStep,
    description=(
        "Encode categorical columns as one-hot or ordinal codes using categories learned on "
        "train. Categories unseen in train do not raise errors."
    ),
    build_transformer=_build_encoder,
    codegen=_encode_code,
)
SCALE = StatefulOpSpec(
    name="scale",
    step_model=ScaleStep,
    description="Rescale numeric columns (standard, minmax or robust) using statistics from train.",
    build_transformer=_build_scaler,
    codegen=_scale_code,
)
