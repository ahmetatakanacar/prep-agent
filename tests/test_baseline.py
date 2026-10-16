import inspect
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from prep_agent.baseline import (
    HIGHER_IS_BETTER,
    BaselineResult,
    evaluate_baselines,
    metric_label,
)
from prep_agent.executor import execute
from prep_agent.plan import Plan
from prep_agent.profiler import load_csv, profile_dataframe

FIXTURES = Path(__file__).parent / "fixtures"
CUSTOMERS_CSV = FIXTURES / "customers.csv"
CUSTOMERS_PLAN = FIXTURES / "customers_plan.json"


def identity_pipeline() -> Pipeline:
    return Pipeline([("identity", FunctionTransformer())]).set_output(transform="pandas")


def classification_frame(n: int = 100, n_classes: int = 2, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    target = np.digitize(x1, np.quantile(x1, np.linspace(0, 1, n_classes + 1)[1:-1]))
    return pd.DataFrame({"x1": x1, "x2": x2, "t": target})


def regression_frame(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "t": 3 * x1 - 2 * x2 + rng.normal(scale=0.1, size=n)})


def evaluate(frame, task_type="classification", pipeline=None, **kwargs) -> BaselineResult:
    return evaluate_baselines(frame, pipeline or identity_pipeline(), "t", task_type, **kwargs)


def by_name(result: BaselineResult) -> dict:
    return {model.name: model for model in result.models}


# --- models and metrics ---


def test_binary_classification_reports_every_model_and_metric():
    result = evaluate(classification_frame())

    assert result.skipped_reason is None
    assert result.task_type == "classification"
    assert result.n_folds == 5
    assert result.metrics == ["accuracy", "f1_macro", "roc_auc"]
    assert [m.name for m in result.models] == [
        "dummy_most_frequent",
        "logistic_regression",
        "random_forest",
    ]
    for model in result.models:
        assert model.skipped_reason is None
        assert set(model.scores) == {"accuracy", "f1_macro", "roc_auc"}
        for score in model.scores.values():
            assert 0 <= score.mean <= 1
            assert score.std >= 0


def test_the_learning_models_beat_the_dummy_on_informative_data():
    models = by_name(evaluate(classification_frame()))

    dummy = models["dummy_most_frequent"].scores
    assert dummy["roc_auc"].mean == pytest.approx(0.5)
    for name in ("logistic_regression", "random_forest"):
        assert models[name].scores["accuracy"].mean > dummy["accuracy"].mean + 0.2


def test_multiclass_classification_has_no_roc_auc():
    result = evaluate(classification_frame(n_classes=3))

    assert result.metrics == ["accuracy", "f1_macro"]
    for model in result.models:
        assert set(model.scores) == {"accuracy", "f1_macro"}


def test_a_text_target_works():
    frame = classification_frame()
    frame["t"] = frame["t"].map({0: "no", 1: "yes"})

    result = evaluate(frame)

    assert all(model.skipped_reason is None for model in result.models)
    assert result.metrics == ["accuracy", "f1_macro", "roc_auc"]


def test_regression_reports_every_model_and_metric():
    result = evaluate(regression_frame(), "regression")

    assert result.metrics == ["r2", "rmse", "mae"]
    assert [m.name for m in result.models] == ["dummy_mean", "ridge", "random_forest"]
    models = by_name(result)
    for model in result.models:
        assert model.skipped_reason is None
        assert set(model.scores) == {"r2", "rmse", "mae"}
        assert model.scores["rmse"].mean >= model.scores["mae"].mean > 0
    assert models["dummy_mean"].scores["r2"].mean <= 0
    assert models["ridge"].scores["r2"].mean > 0.95
    assert models["ridge"].scores["rmse"].mean < models["dummy_mean"].scores["rmse"].mean


def test_the_signature_takes_no_test_data():
    parameters = inspect.signature(evaluate_baselines).parameters

    assert not [name for name in parameters if "test" in name]


def test_the_input_frame_and_pipeline_are_left_alone():
    frame = classification_frame()
    before = frame.copy()
    pipeline = identity_pipeline()

    evaluate(frame, pipeline=pipeline)

    pd.testing.assert_frame_equal(frame, before)
    assert not hasattr(pipeline.steps[0][1], "n_features_in_")


# --- metric direction ---


def test_the_direction_of_every_reported_metric_is_defined():
    reported = set(evaluate(classification_frame()).metrics)
    reported |= set(evaluate(classification_frame(n_classes=3)).metrics)
    reported |= set(evaluate(regression_frame(), "regression").metrics)

    assert reported <= set(HIGHER_IS_BETTER)


def test_error_metrics_are_lower_is_better_and_the_rest_higher():
    assert {m for m, higher in HIGHER_IS_BETTER.items() if not higher} == {"rmse", "mae"}


def test_metric_labels_carry_an_arrow():
    assert metric_label("accuracy") == "accuracy ↑"
    assert metric_label("r2") == "r2 ↑"
    assert metric_label("rmse") == "rmse ↓"
    assert metric_label("mae") == "mae ↓"


# --- no leakage inside cross-validation ---


class RecordingTransformer(TransformerMixin, BaseEstimator):
    """Remembers the row labels of every frame it was fit on."""

    fits: list[frozenset] = []

    def fit(self, X, y=None):
        type(self).fits.append(frozenset(X.index))
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return X


@pytest.fixture
def recorder():
    RecordingTransformer.fits = []
    return RecordingTransformer


def test_stateful_steps_are_fit_on_the_training_part_of_every_fold(recorder):
    frame = classification_frame(n=50)
    pipeline = Pipeline([("record", recorder())])
    everything = frozenset(frame.index)

    result = evaluate(frame, pipeline=pipeline)

    n_models = len(result.models)
    assert len(recorder.fits) == result.n_folds * n_models
    held_out = [everything - fit for fit in recorder.fits]
    assert all(held_out), "a fit saw every row"
    folds = set(held_out)
    assert len(folds) == result.n_folds
    for fold in folds:
        assert held_out.count(fold) == n_models
    assert frozenset().union(*folds) == everything
    assert sum(len(fold) for fold in folds) == len(frame)


def test_an_imputer_never_sees_the_held_out_rows():
    # The rows of the second half have a huge x1; a leak of their statistics into a fold that
    # trains on the first half would show in the learned mean.
    frame = classification_frame(n=60)
    frame["x1"] = frame["x1"].astype("float64")
    frame.loc[frame.index[30:], "x1"] = 1e6
    frame.loc[frame.index[::7], "x1"] = np.nan
    seen_means = []

    class Spy(SimpleImputer):
        def fit(self, X, y=None):
            super().fit(X, y)
            seen_means.append(float(self.statistics_[0]))
            return self

    spy = Spy(strategy="mean").set_output(transform="pandas")

    result = evaluate(frame[["x1", "t"]], pipeline=Pipeline([("impute", spy)]))

    assert result.n_folds == 5
    all_rows_mean = float(frame["x1"].mean())
    assert all(mean != pytest.approx(all_rows_mean) for mean in seen_means)


# --- folds ---


def test_fold_count_drops_to_the_smallest_class():
    frame = classification_frame(n=30)
    frame["t"] = [1] * 3 + [0] * 27

    result = evaluate(frame)

    assert result.skipped_reason is None
    assert result.n_folds == 3


def test_a_requested_fold_count_is_used_when_the_data_allows_it():
    assert evaluate(classification_frame(), n_folds=3).n_folds == 3


def test_a_single_sample_class_skips_the_baseline_with_a_reason():
    frame = classification_frame(n=30)
    frame["t"] = [1] + [0] * 29

    result = evaluate(frame)

    assert result.models == []
    assert result.n_folds is None
    assert "smallest class" in result.skipped_reason
    assert "1" in result.skipped_reason


def test_a_single_class_target_skips_the_baseline():
    frame = classification_frame(n=30)
    frame["t"] = 1

    result = evaluate(frame)

    assert result.models == []
    assert "one class" in result.skipped_reason


def test_a_regression_with_one_row_skips_the_baseline():
    result = evaluate(regression_frame(n=1), "regression")

    assert result.models == []
    assert "at least 2" in result.skipped_reason


def test_regression_folds_shrink_to_the_row_count():
    result = evaluate(regression_frame(n=4), "regression")

    assert result.n_folds == 4


def test_a_missing_target_skips_the_baseline():
    frame = regression_frame()
    frame.loc[0, "t"] = np.nan

    result = evaluate(frame, "regression")

    assert result.models == []
    assert "missing" in result.skipped_reason


# --- models that cannot run ---


def test_a_text_column_skips_the_models_that_need_numbers_and_keeps_the_dummy():
    frame = classification_frame()
    frame["city"] = ["a", "b"] * (len(frame) // 2)

    result = evaluate(frame)

    models = by_name(result)
    assert models["dummy_most_frequent"].skipped_reason is None
    assert models["dummy_most_frequent"].scores
    for name in ("logistic_regression", "random_forest"):
        assert models[name].scores == {}
        assert "non-numeric" in models[name].skipped_reason
        assert "city" in models[name].skipped_reason
    assert result.skipped_reason is None


def test_missing_values_skip_the_models_that_cannot_take_them_and_the_rest_run():
    frame = classification_frame()
    frame.loc[frame.index[::9], "x2"] = np.nan

    models = by_name(evaluate(frame))

    assert models["dummy_most_frequent"].skipped_reason is None
    assert models["logistic_regression"].scores == {}
    assert "missing values" in models["logistic_regression"].skipped_reason
    assert "x2" in models["logistic_regression"].skipped_reason


def test_a_failing_model_does_not_stop_the_others(monkeypatch):
    class Broken(BaseEstimator):
        def fit(self, X, y):
            raise RuntimeError("boom")

    import prep_agent.baseline as baseline

    original = baseline._classification_models

    def with_broken_forest(random_state):
        models = original(random_state)
        models["random_forest"] = Broken()
        return models

    monkeypatch.setattr(baseline, "_classification_models", with_broken_forest)

    models = by_name(evaluate(classification_frame()))

    assert "boom" in models["random_forest"].skipped_reason
    assert "RuntimeError" in models["random_forest"].skipped_reason
    assert models["logistic_regression"].scores
    assert models["dummy_most_frequent"].scores


# --- convergence warnings ---


@pytest.fixture
def slow_logistic_regression(monkeypatch):
    """Make the logistic regression stop after one iteration, which always raises a
    ConvergenceWarning; real data that fails to converge is hard to build on purpose."""
    import prep_agent.baseline as baseline

    original = baseline._classification_models

    def with_one_iteration(random_state):
        models = original(random_state)
        models["logistic_regression"] = LogisticRegression(max_iter=1, random_state=random_state)
        return models

    monkeypatch.setattr(baseline, "_classification_models", with_one_iteration)


def test_a_model_that_does_not_converge_keeps_its_scores_and_gets_a_note(slow_logistic_regression):
    models = by_name(evaluate(classification_frame()))

    slow = models["logistic_regression"]
    assert slow.skipped_reason is None
    assert set(slow.scores) == {"accuracy", "f1_macro", "roc_auc"}
    assert slow.note is not None
    assert "did not converge" in slow.note
    assert models["random_forest"].note is None
    assert models["dummy_most_frequent"].note is None


def test_convergence_warnings_do_not_reach_the_terminal(slow_logistic_regression):
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        evaluate(classification_frame())

    assert [w for w in seen if issubclass(w.category, ConvergenceWarning)] == []


def test_other_warnings_of_a_model_are_not_swallowed(monkeypatch):
    class Chatty(DummyClassifier):
        def fit(self, X, y, sample_weight=None):
            warnings.warn("something else", UserWarning, stacklevel=2)
            return super().fit(X, y, sample_weight)

    import prep_agent.baseline as baseline

    original = baseline._classification_models

    def with_chatty_dummy(random_state):
        models = original(random_state)
        models["dummy_most_frequent"] = Chatty(strategy="most_frequent")
        return models

    monkeypatch.setattr(baseline, "_classification_models", with_chatty_dummy)

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        models = by_name(evaluate(classification_frame()))

    assert any("something else" in str(w.message) for w in seen)
    assert models["dummy_most_frequent"].note is None


def test_a_regression_that_does_not_converge_gets_a_note_too(monkeypatch):
    import prep_agent.baseline as baseline

    original = baseline._regression_models

    def with_one_iteration(random_state):
        models = original(random_state)
        models["ridge"] = Ridge(solver="sag", max_iter=1, tol=1e-12, random_state=random_state)
        return models

    monkeypatch.setattr(baseline, "_regression_models", with_one_iteration)

    models = by_name(evaluate(regression_frame(), task_type="regression"))

    assert models["ridge"].note is not None
    assert models["ridge"].scores


def test_the_note_round_trips_through_json(slow_logistic_regression):
    result = evaluate(classification_frame())

    assert BaselineResult.model_validate_json(result.model_dump_json()) == result
    assert by_name(result)["logistic_regression"].note is not None


def test_a_model_without_a_note_has_none():
    assert all(model.note is None for model in evaluate(classification_frame()).models)


# --- determinism ---


def test_two_runs_give_identical_scores():
    frame = classification_frame()

    first = evaluate(frame)
    second = evaluate(frame)

    assert first.model_dump() == second.model_dump()


def test_a_different_seed_changes_the_folds_but_not_the_shape_of_the_result():
    frame = classification_frame()

    first = evaluate(frame, random_state=1)
    second = evaluate(frame, random_state=2)

    assert [m.name for m in first.models] == [m.name for m in second.models]
    assert first.model_dump() != second.model_dump()


def test_the_result_round_trips_through_json():
    result = evaluate(classification_frame())

    assert BaselineResult.model_validate_json(result.model_dump_json()) == result


# --- with the executor ---


def test_baseline_on_the_executor_output_of_the_fixture():
    df = load_csv(CUSTOMERS_CSV)
    plan = Plan.model_validate_json(CUSTOMERS_PLAN.read_text(encoding="utf-8"))
    execution = execute(df, plan, profile_dataframe(df, target=plan.target))

    result = evaluate_baselines(
        execution.split_train, execution.pipeline, plan.target, plan.task_type
    )

    assert result.skipped_reason is None
    assert 2 <= result.n_folds <= 5
    assert all(model.skipped_reason is None for model in result.models)
