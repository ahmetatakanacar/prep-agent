"""Baseline evaluation: reference scores that show the prepared data is usable.

The agent does not choose a model. Two quick models per task (logistic regression and a random
forest for classification, ridge and a random forest for regression) and one dummy model that
ignores the features are scored by cross-validation on the train set. The dummy is the floor a
model has to beat; the others only show that the prepared features carry signal.

Cross-validation runs on the raw train split, the frame right after the split with the stateful
steps not applied yet, and not on the executor's prepared output. Each fold gets a freshly cloned,
unfitted copy of the stateful pipeline in front of the model, so imputers, encoders and scalers
learn their statistics from the training part of that fold only. Scoring the prepared output
instead would let every fold's validation rows shape those statistics, which is the leakage the
executor avoids at the split. The test set is never an input: the signature takes no test data.

A model that cannot run (the pipeline output still has missing values or text columns, say) is
recorded as skipped with the reason and the others carry on. When the target cannot be
cross-validated at all, the whole baseline is skipped with a reason.
"""

from typing import Literal

import numpy as np
import pandas as pd
from pandas.api import types as pdt
from pydantic import BaseModel
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

DEFAULT_FOLDS = 5
RANDOM_STATE = 42
_MIN_FOLDS = 2
# sklearn reports these as negated scores so that higher is always better.
_NEGATED = {"rmse", "mae"}
# The one place that says which way each metric reads; the report and the CLI take their arrows
# from here.
HIGHER_IS_BETTER = {
    "accuracy": True,
    "f1_macro": True,
    "roc_auc": True,
    "r2": True,
    "rmse": False,
    "mae": False,
}


class BaselineScore(BaseModel):
    """Mean and standard deviation (ddof 0, like `numpy.std`) of one metric over the folds."""

    mean: float
    std: float


class ModelBaseline(BaseModel):
    """One model's scores, or why it was skipped (then `scores` is empty)."""

    name: str
    scores: dict[str, BaselineScore] = {}
    skipped_reason: str | None = None


class BaselineResult(BaseModel):
    """The contents of baseline.json.

    `n_folds` is None and `models` empty when the whole baseline was skipped, with the reason
    in `skipped_reason`.
    """

    task_type: Literal["classification", "regression"]
    n_rows: int
    random_state: int
    n_folds: int | None
    metrics: list[str]
    models: list[ModelBaseline]
    skipped_reason: str | None = None


def metric_label(metric: str) -> str:
    """The metric name with an arrow: ↑ when higher is better, ↓ when lower is."""
    return f"{metric} {'↑' if HIGHER_IS_BETTER[metric] else '↓'}"


def evaluate_baselines(
    train: pd.DataFrame,
    pipeline: Pipeline,
    target: str,
    task_type: Literal["classification", "regression"],
    *,
    n_folds: int = DEFAULT_FOLDS,
    random_state: int = RANDOM_STATE,
) -> BaselineResult:
    """Cross-validate the baseline models on `train`, the split train frame (target included).

    `pipeline` is the executor's stateful pipeline; only its structure is used, every fold fits
    a clone of it. `n_folds` is lowered to what the target allows (the smallest class for
    classification, the row count for regression); below 2 the baseline is skipped.
    """
    if n_folds < _MIN_FOLDS:
        raise ValueError(f"n_folds must be at least {_MIN_FOLDS}")

    y = train[target]
    x = train.drop(columns=target)
    folds, reason = _choose_folds(y, task_type, n_folds)
    if folds is None:
        return BaselineResult(
            task_type=task_type,
            n_rows=len(train),
            random_state=random_state,
            n_folds=None,
            metrics=[],
            models=[],
            skipped_reason=reason,
        )

    if task_type == "classification":
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
        models = _classification_models(random_state)
        scoring = _classification_scoring(binary=y.nunique() == 2)
    else:
        cv = KFold(n_splits=folds, shuffle=True, random_state=random_state)
        models = _regression_models(random_state)
        scoring = _REGRESSION_SCORING

    outcomes = {
        name: _cross_validate(model, pipeline, x, y, cv, scoring) for name, model in models.items()
    }
    failures = [outcome for outcome in outcomes.values() if isinstance(outcome, Exception)]
    problems = _feature_problems(pipeline, x) if failures else None

    results = []
    for name, outcome in outcomes.items():
        if isinstance(outcome, Exception):
            results.append(ModelBaseline(name=name, skipped_reason=_explain(outcome, problems)))
        else:
            results.append(ModelBaseline(name=name, scores=_summarise(outcome, scoring)))
    return BaselineResult(
        task_type=task_type,
        n_rows=len(train),
        random_state=random_state,
        n_folds=folds,
        metrics=list(scoring),
        models=results,
    )


def _choose_folds(
    y: pd.Series, task_type: str, requested: int
) -> tuple[int, None] | tuple[None, str]:
    if y.empty:
        return None, "the train set has no rows"
    n_missing = int(y.isna().sum())
    if n_missing:
        return None, f"the target has {n_missing} missing value(s)"
    if task_type == "regression":
        folds = min(requested, len(y))
        if folds < _MIN_FOLDS:
            return None, f"at least {_MIN_FOLDS} rows are needed for cross-validation, got {len(y)}"
        return folds, None

    counts = y.value_counts()
    if len(counts) < _MIN_FOLDS:
        return None, "the target has only one class"
    smallest = int(counts.min())
    folds = min(requested, smallest)
    if folds < _MIN_FOLDS:
        return None, (
            f"the smallest class has only {smallest} sample(s); at least {_MIN_FOLDS} are "
            "needed for stratified cross-validation"
        )
    return folds, None


def _classification_models(random_state: int) -> dict[str, BaseEstimator]:
    return {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "random_forest": RandomForestClassifier(random_state=random_state),
    }


def _regression_models(random_state: int) -> dict[str, BaseEstimator]:
    return {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "ridge": Ridge(random_state=random_state),
        "random_forest": RandomForestRegressor(random_state=random_state),
    }


def _classification_scoring(*, binary: bool) -> dict:
    scoring = {
        "accuracy": "accuracy",
        # zero_division=0: a model that never predicts a class scores 0 for it, without a warning.
        "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
    }
    if binary:
        scoring["roc_auc"] = "roc_auc"
    return scoring


_REGRESSION_SCORING = {
    "r2": "r2",
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
}


def _cross_validate(
    model: BaseEstimator,
    pipeline: Pipeline,
    x: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold | KFold,
    scoring: dict,
) -> dict | Exception:
    estimator = Pipeline([("prepare", clone(pipeline)), ("model", model)])
    try:
        return cross_validate(estimator, x, y, cv=cv, scoring=scoring, error_score="raise")
    # Whatever a model raises must end up in the report, not crash the run.
    except Exception as exc:
        return exc


def _summarise(raw: dict, scoring: dict) -> dict[str, BaselineScore]:
    scores = {}
    for metric in scoring:
        values = -raw[f"test_{metric}"] if metric in _NEGATED else raw[f"test_{metric}"]
        scores[metric] = BaselineScore(mean=float(np.mean(values)), std=float(np.std(values)))
    return scores


def _feature_problems(pipeline: Pipeline, x: pd.DataFrame) -> str | None:
    """Describe what the prepared features contain that models cannot take, or None.

    Only used to word a skip reason. The pipeline is fit on all of `x` for that, and the output
    is inspected for dtypes and missing values only, never scored.
    """
    try:
        prepared = clone(pipeline).fit_transform(x)
    except Exception:
        return None
    if not isinstance(prepared, pd.DataFrame):
        return None
    problems = []
    non_numeric = [str(c) for c in prepared.columns if not pdt.is_numeric_dtype(prepared[c])]
    if non_numeric:
        problems.append(f"non-numeric columns: {', '.join(non_numeric)}")
    missing = [str(c) for c in prepared.columns if prepared[c].isna().any()]
    if missing:
        problems.append(f"missing values in: {', '.join(missing)}")
    return "; ".join(problems) or None


def _explain(error: Exception, problems: str | None) -> str:
    if problems is not None and isinstance(error, ValueError | TypeError):
        return f"the prepared features are not usable by this model ({problems})"
    return f"{type(error).__name__}: {error}"
