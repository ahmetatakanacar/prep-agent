"""The train/test split."""

import pandas as pd
from sklearn.model_selection import train_test_split

from prep_agent.ops.base import OpContext, OpError, SplitOpSpec
from prep_agent.plan import SplitStep


def _split(df: pd.DataFrame, step: SplitStep, ctx: OpContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    stratify = None
    if step.stratify:
        if ctx.target not in df.columns:
            raise OpError(f"split: target column {ctx.target!r} not found, cannot stratify")
        if df[ctx.target].isna().any():
            raise OpError(
                "split: cannot stratify while the target has missing values; "
                "drop them first with drop_missing_target"
            )
        stratify = df[ctx.target]
    try:
        train, test = train_test_split(
            df, test_size=step.test_size, random_state=step.random_state, stratify=stratify
        )
    except ValueError as exc:
        raise OpError(f"split failed: {exc}") from exc
    return train.reset_index(drop=True), test.reset_index(drop=True)


SPLIT = SplitOpSpec(
    name="split",
    step_model=SplitStep,
    description=(
        "Split rows into train and test sets, optionally stratified by the target. "
        "Every step after it is fit on train only."
    ),
    apply=_split,
)
