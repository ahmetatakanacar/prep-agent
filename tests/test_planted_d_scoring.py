"""Set D, scoring: the false alarm rate and the levels in the scorer and the summary, the runner
on a dev dataset, and the guard around the test recipes. Only dev data is read."""

import pytest

from bench_cost import ModelPrice
from planted_generate import load
from planted_recipes_d import DEV, TEST
from planted_runner import PlantedRunError, check_selection, plan_run, run_planted
from planted_scorer import combine, score_plan
from planted_support import (
    RoutedLLM,
    drop_all_plan,
    perfect_decisions,
    perfect_plan,
    plan_of,
    run_through_agent,
)
from planted_truth import VARIANTS

DEV_NAMES = [r.name for r in DEV]
TEST_NAMES = [r.name for r in TEST]


# --- the scorer: false alarms and levels ---


@pytest.fixture(scope="module")
def d01():
    frame, truth = load("planted-d01")
    return frame, truth


def column_of(truth, fault):
    return [c for c in truth.columns if c.fault == fault]


def test_a_perfect_plan_keeps_every_legit_feature_and_drops_every_leak(d01):
    frame, truth = d01
    score = score_plan(perfect_plan(frame, truth), truth)
    assert score.false_alarm_rate == 0.0
    assert score.levels_found == score.levels_planted == {"0.85": 1, "0.60": 1}
    assert score.legit_planted == {"0.97": 1} and sum(score.legit_dropped.values()) == 0
    assert score.precision == score.recall == 1.0


def test_dropping_a_legit_strong_feature_is_a_false_alarm_and_a_wrong_drop(d01):
    frame, truth = d01
    drop, cast = perfect_decisions(truth)
    legit = column_of(truth, "legit_strong")[0]
    score = score_plan(plan_of(frame, truth, [*drop, legit.name], cast), truth)
    assert score.false_alarm_rate == 1.0 and score.legit_dropped == {"0.97": 1}
    assert score.over_dropped == [f"planted-d01/{legit.name}"]
    assert score.precision < 1.0 and score.recall == 1.0


def test_keeping_a_leak_is_counted_at_its_level(d01):
    frame, truth = d01
    drop, cast = perfect_decisions(truth)
    leak = next(c for c in column_of(truth, "leak_partial") if c.level == 0.6)
    score = score_plan(plan_of(frame, truth, [d for d in drop if d != leak.name], cast), truth)
    assert score.levels_found["0.60"] == 0 and score.levels_found["0.85"] == 1
    assert score.by_type["leak_partial"].detected == 1 and score.by_type["leak_partial"].n == 2
    assert score.false_alarm_rate == 0.0


def test_a_plan_that_drops_everything_has_every_false_alarm(d01):
    frame, truth = d01
    score = score_plan(drop_all_plan(frame, truth), truth)
    assert score.false_alarm_rate == 1.0


def test_there_is_no_false_alarm_rate_without_legit_features():
    frame, truth = load("planted-01")
    assert score_plan(perfect_plan(frame, truth), truth).false_alarm_rate is None


def test_levels_and_false_alarms_add_up_over_datasets():
    scores = []
    for recipe in DEV:
        frame, truth = load(recipe.name)
        scores.append(score_plan(perfect_plan(frame, truth), truth))
    total = combine(scores)
    assert sum(total.levels_planted.values()) == 12
    assert dict(total.levels_planted) == {"0.60": 3, "0.75": 3, "0.85": 3, "0.95": 3}
    assert sum(total.legit_planted.values()) == 9 and total.false_alarm_rate == 0.0
    assert total.legit_planted == {"0.97": 2, "0.93": 3, "0.85": 2, "0.75": 1, "0.60": 1}


def test_a_score_with_levels_survives_json(d01):
    frame, truth = d01
    score = score_plan(drop_all_plan(frame, truth), truth)
    from planted_scorer import Score

    assert Score.model_validate_json(score.model_dump_json()) == score


# --- end to end on dev with a scripted planner ---


@pytest.mark.parametrize("name", DEV_NAMES)
def test_a_perfect_plan_runs_on_every_dev_dataset_and_scores_full(name):
    for variant in VARIANTS:
        frame, truth = load(name, variant)
        result = run_through_agent(frame, truth, perfect_plan(frame, truth))
        score = score_plan(result.plan, truth)
        assert score.precision == score.recall == 1.0 and score.false_alarm_rate == 0.0


def test_the_runner_handles_a_dev_dataset_in_both_variants_and_all_modes(tmp_path):
    results = run_planted(
        ["m"],
        datasets=["planted-d01"],
        variants=list(VARIANTS),
        modes=["recommended", "not_recommended", "no_questions"],
        repeats=1,
        out_dir=tmp_path / "o",
        prices={"m": ModelPrice(input_per_million=1.0, output_per_million=2.0)},
        currency="USD",
        client_factory=lambda model: RoutedLLM(),
    )
    assert len(results.runs) == 6 and all(r.status == "ok" for r in results.runs)
    for run in results.runs:
        assert run.score.false_alarm_rate == 0.0
        assert run.score.levels_planted == {"0.85": 1, "0.60": 1}
    kept = [r for r in results.runs if r.mode == "not_recommended"]
    assert all(r.score.levels_found.total() < 2 for r in kept)
    summary = (tmp_path / "o" / "summary.md").read_text(encoding="utf-8")
    assert "legit strong dropped" in summary and "| 0 of 1 |" in summary
    assert "## Leaks of partial strength found, by level of association" in summary
    assert "## Legitimate strong features dropped, by level of association" in summary


# --- the guard around the test recipes ---


def test_the_test_recipes_are_refused_without_the_final_test_flag():
    with pytest.raises(PlantedRunError, match="held-out test data"):
        check_selection(["planted-t01"], ["anonymous"], ["recommended"])
    check_selection(["planted-t01"], ["anonymous"], ["recommended"], final_test=True)
    check_selection(["planted-d01"], ["anonymous"], ["recommended"])


def test_a_run_of_missing_data_is_refused_before_anything_happens(tmp_path):
    with pytest.raises(PlantedRunError, match="data files not found"):
        plan_run(["planted-d01"], ["anonymous"], ["recommended"], ["m"], 1, {}, tmp_path)
    with pytest.raises(PlantedRunError, match="data files not found"):
        run_planted(
            ["m"],
            datasets=["planted-t01"],
            variants=["anonymous"],
            modes=["recommended"],
            repeats=1,
            out_dir=tmp_path / "o",
            prices={},
            currency="USD",
            client_factory=lambda m: RoutedLLM(),
            data_dir=tmp_path,
            final_test=True,
        )
    assert not (tmp_path / "o").exists()


def test_the_test_recipes_are_known_to_the_runner_but_not_its_default():
    from planted_runner import DATASETS, KNOWN_DATASETS, TEST_DATASETS

    assert set(TEST_DATASETS) == set(TEST_NAMES) and set(DEV_NAMES) <= set(KNOWN_DATASETS)
    assert not set(DEV_NAMES + TEST_NAMES) & set(DATASETS)
