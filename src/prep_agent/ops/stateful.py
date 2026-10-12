"""Ops that learn from data, so they must run after the split and be fit on train only.

Each builder returns an unfitted `ColumnTransformer` that transforms the step's columns and
passes every other column through, so a transformer can be fit on a whole train frame and
applied to a whole test frame. Output is a pandas DataFrame that keeps the index.
"""

from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

from prep_agent.ops.base import StatefulOpSpec
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
        add_indicator=step.add_indicator,
        keep_empty_features=True,
    )
    return _on_columns(step.op, step.columns, imputer)


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


IMPUTE = StatefulOpSpec(
    name="impute",
    step_model=ImputeStep,
    description=(
        "Fill missing values with the mean, median, most frequent or a constant value, "
        "learned on train. Can add a missing-value indicator column."
    ),
    build_transformer=_build_imputer,
)
ENCODE = StatefulOpSpec(
    name="encode",
    step_model=EncodeStep,
    description=(
        "Encode categorical columns as one-hot or ordinal codes using categories learned on "
        "train. Categories unseen in train do not raise errors."
    ),
    build_transformer=_build_encoder,
)
SCALE = StatefulOpSpec(
    name="scale",
    step_model=ScaleStep,
    description="Rescale numeric columns (standard, minmax or robust) using statistics from train.",
    build_transformer=_build_scaler,
)
