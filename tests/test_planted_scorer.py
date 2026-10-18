import pytest

from planted_generate import RECIPES, load
from planted_scorer import Score, combine, score_plan
from planted_support import (
    drop_all_plan,
    leak_blind_plan,
    perfect_decisions,
    perfect_plan,
    plan_of,
)
from planted_truth import FAULT_TYPES, UNMEASURABLE_FAULTS

NAMES = [recipe.name for recipe in RECIPES]
MEASURABLE = [f for f in FAULT_TYPES if f not in UNMEASURABLE_FAULTS]


def column_with(truth, fault):
    return next(c for c in truth.columns if c.fault == fault)


@pytest.fixture(scope="module")
def p03():
    return load("planted-03")


@pytest.fixture(scope="module")
def all_sets():
    return [load(name) for name in NAMES]


def test_a_perfect_plan_scores_everything(p03):
    frame, truth = p03
    score = score_plan(perfect_plan(frame, truth), truth)
    assert set(score.by_type) == {
        "leak_nonlinear",
        "id_sequential",
        "pii_phone",
        "duplicate_exact",
        "numeric_category",
    }
    assert all(t.rate == 1.0 for t in score.by_type.values())
    assert score.precision == score.recall == score.detection_rate == score.op_rate == 1.0
    assert not score.over_dropped and not score.missing_ops and not score.avoided_op_violations
    assert not score.pairs_dropped_both and not score.unknown_dropped


def test_a_plan_that_misses_the_leak_loses_recall_but_not_precision(p03):
    frame, truth = p03
    score = score_plan(leak_blind_plan(frame, truth), truth)
    assert score.by_type["leak_nonlinear"].rate == 0.0
    assert score.by_type["id_sequential"].rate == 1.0
    assert score.precision == 1.0
    assert score.recall == pytest.approx(3 / 4)
    assert score.detection_rate == pytest.approx(4 / 5)


def test_a_plan_that_drops_everything_loses_precision(p03):
    frame, truth = p03
    score = score_plan(drop_all_plan(frame, truth), truth)
    assert score.recall == 1.0
    assert score.precision is not None and score.precision < 0.5
    assert score.op_rate == 0.0
    # Every must_keep column but the one left in the plan is a wrong drop, the casted one included.
    assert f"{truth.dataset}/{column_with(truth, 'numeric_category').name}" in score.over_dropped
    assert len(score.over_dropped) >= 5


def test_a_plan_that_drops_nothing_has_no_precision(p03):
    frame, truth = p03
    score = score_plan(plan_of(frame, truth, [], []), truth)
    assert score.precision is None
    assert score.recall == 0.0
    assert score.detection_rate == 0.0


def test_a_dataset_without_required_drops_has_no_recall():
    frame, truth = load("planted-08")
    only_keep = truth.model_copy(update={"columns": [c for c in truth.columns if c.fault is None]})
    score = score_plan(plan_of(frame, only_keep, [], []), only_keep)
    assert score.recall is None and score.detection_rate is None and score.op_rate is None


def test_dropping_one_of_a_duplicate_pair_is_enough_whichever_it_is(p03):
    frame, truth = p03
    copy = column_with(truth, "duplicate_exact")
    drop, cast = perfect_decisions(truth)
    for dropped in (copy.name, copy.duplicate_of):
        swapped = [dropped if name == copy.name else name for name in drop]
        score = score_plan(plan_of(frame, truth, swapped, cast), truth)
        assert score.by_type["duplicate_exact"].rate == 1.0
        assert score.precision == score.recall == 1.0


def test_dropping_both_of_a_pair_is_a_wrong_drop_counted_once(p03):
    frame, truth = p03
    copy = column_with(truth, "duplicate_exact")
    drop, cast = perfect_decisions(truth)
    score = score_plan(plan_of(frame, truth, [*drop, copy.duplicate_of], cast), truth)
    assert score.by_type["duplicate_exact"].rate == 1.0
    assert score.pairs_dropped_both == [f"planted-03/{copy.name} + planted-03/{copy.duplicate_of}"]
    assert score.wrong_drops == 1
    assert score.precision == pytest.approx(4 / 5)
    assert score.recall == 1.0


def test_keeping_both_of_a_pair_misses_the_fault(p03):
    frame, truth = p03
    copy = column_with(truth, "duplicate_exact")
    drop, cast = perfect_decisions(truth)
    score = score_plan(plan_of(frame, truth, [n for n in drop if n != copy.name], cast), truth)
    assert score.by_type["duplicate_exact"].rate == 0.0
    assert score.precision == 1.0


def test_a_numeric_category_needs_the_cast_and_must_not_be_dropped(p03):
    frame, truth = p03
    code = column_with(truth, "numeric_category")
    drop, _ = perfect_decisions(truth)

    uncast = score_plan(plan_of(frame, truth, drop, []), truth)
    assert uncast.by_type["numeric_category"].rate == 0.0
    assert uncast.missing_ops == [f"planted-03/{code.name}"]
    assert uncast.op_rate == 0.0 and uncast.precision == 1.0

    dropped = score_plan(plan_of(frame, truth, [*drop, code.name], []), truth)
    assert dropped.by_type["numeric_category"].rate == 0.0
    assert dropped.over_dropped == [f"planted-03/{code.name}"]
    assert dropped.precision < 1.0


def test_casting_a_count_is_reported_and_dropping_it_is_a_wrong_drop(p03):
    frame, truth = p03
    decoy = next(c for c in truth.columns if c.avoid_op)
    drop, cast = perfect_decisions(truth)

    cast_score = score_plan(plan_of(frame, truth, drop, [*cast, decoy.name]), truth)
    assert cast_score.avoided_op_violations == [f"planted-03/{decoy.name}"]
    assert cast_score.precision == 1.0

    dropped = score_plan(plan_of(frame, truth, [*drop, decoy.name], cast), truth)
    assert dropped.over_dropped == [f"planted-03/{decoy.name}"]


def test_sentinel_columns_are_listed_but_never_scored():
    frame, truth = load("planted-02")
    sentinel = column_with(truth, "sentinel")
    drop, cast = perfect_decisions(truth)
    kept = score_plan(plan_of(frame, truth, drop, cast), truth)
    dropped = score_plan(plan_of(frame, truth, [*drop, sentinel.name], cast), truth)
    assert kept.unmeasurable == [f"planted-02/{sentinel.name} (sentinel)"]
    assert "sentinel" not in kept.by_type
    for score in (kept, dropped):
        assert score.precision == score.recall == score.detection_rate == 1.0
        assert not score.over_dropped


def test_columns_the_truth_does_not_know_are_reported(p03):
    frame, truth = p03
    drop, cast = perfect_decisions(truth)
    plan = plan_of(frame, truth, drop, cast)
    plan.steps[0].columns.append("ghost")
    assert score_plan(plan, truth).unknown_dropped == ["planted-03/ghost"]


def test_combine_adds_counts_before_taking_ratios(all_sets):
    perfect = [score_plan(perfect_plan(f, t), t) for f, t in all_sets]
    blind = [score_plan(leak_blind_plan(f, t), t) for f, t in all_sets]
    assert combine(perfect).recall == 1.0
    total = combine(blind)
    assert total.datasets == NAMES
    assert total.by_type["leak_linear"].rate == 0.0
    assert total.by_type["pii_email"].rate == 1.0
    assert total.precision == 1.0 and 0.5 < total.recall < 1.0
    assert total.recall == pytest.approx(
        sum(s.correct_drops for s in blind) / sum(s.required_drops for s in blind)
    )
    assert len(total.unmeasurable) == 3


def test_every_measurable_type_is_scored_across_the_set(all_sets):
    total = combine([score_plan(perfect_plan(f, t), t) for f, t in all_sets])
    assert sorted(total.by_type) == sorted(MEASURABLE)
    assert all(t.n >= 2 for t in total.by_type.values())


def test_a_score_survives_a_round_trip_through_json(p03):
    frame, truth = p03
    score = score_plan(leak_blind_plan(frame, truth), truth)
    again = Score.model_validate_json(score.model_dump_json())
    assert again == score
    assert again.by_type == score.by_type and again.precision == score.precision
