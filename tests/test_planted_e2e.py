"""The whole path on the planted datasets: FakeLLM answers with a plan, the agent validates and
executes it on the data, and the scorer judges the plan that came out."""

import pytest

from planted_generate import RECIPES, load
from planted_scorer import combine, score_plan
from planted_support import build_plan, drop_all_plan, leak_blind_plan, perfect_plan
from prep_agent.agent import run_agent
from prep_agent.llm.fake import FakeLLM
from prep_agent.plan import Plan

NAMES = [recipe.name for recipe in RECIPES]


def no_questions(questions):
    raise AssertionError("the scripted planner does not ask")


def run_with(frame, truth, plan: Plan):
    reply = {"language": "English", "kind": "plan", "plan": plan.model_dump(), "questions": None}
    llm = FakeLLM([reply])
    result = run_agent(
        frame, "prepare for modelling", llm, no_questions, target=truth.target, model_family="any"
    )
    assert llm.remaining == 0
    return result


@pytest.mark.parametrize("name", NAMES)
def test_a_perfect_plan_runs_and_gets_full_marks(name):
    frame, truth = load(name)
    result = run_with(frame, truth, perfect_plan(frame, truth))
    score = score_plan(result.plan, truth)
    assert score.precision == score.recall == score.detection_rate == 1.0
    assert score.op_rate in (1.0, None)
    assert not score.over_dropped and not score.avoided_op_violations
    train = result.execution.train
    assert not train.drop(columns=truth.target).isna().any().any()
    assert not any(
        c.fault and c.fault.startswith("leak_") for c in truth.columns if c.name in train
    )


@pytest.mark.parametrize("name", NAMES)
def test_a_plan_that_keeps_the_leaks_scores_low_on_them(name):
    frame, truth = load(name)
    result = run_with(frame, truth, leak_blind_plan(frame, truth))
    score = score_plan(result.plan, truth)
    leaks = [t for fault, t in score.by_type.items() if fault.startswith("leak_")]
    assert all(t.detected == 0 for t in leaks)
    assert score.precision in (1.0, None)
    if leaks:
        assert score.recall < 1.0
        assert score.detection_rate < 1.0


@pytest.mark.parametrize("name", NAMES)
def test_a_plan_that_drops_everything_loses_precision(name):
    frame, truth = load(name)
    result = run_with(frame, truth, drop_all_plan(frame, truth))
    score = score_plan(result.plan, truth)
    assert score.recall == 1.0
    assert score.precision < 0.6
    assert len(score.over_dropped) >= 4


def test_the_three_plans_are_ordered_over_the_whole_set():
    scores = {"perfect": [], "blind": [], "drop_all": []}
    for name in NAMES:
        frame, truth = load(name)
        for kind, make in (
            ("perfect", perfect_plan),
            ("blind", leak_blind_plan),
            ("drop_all", drop_all_plan),
        ):
            scores[kind].append(score_plan(run_with(frame, truth, make(frame, truth)).plan, truth))
    perfect, blind, drop_all = (combine(scores[k]) for k in ("perfect", "blind", "drop_all"))
    assert perfect.detection_rate == 1.0
    assert blind.detection_rate < perfect.detection_rate
    assert drop_all.precision < blind.precision == perfect.precision
    leak_types = [f for f in perfect.by_type if f.startswith("leak_")]
    assert len(leak_types) == 3
    assert all(blind.by_type[f].rate == 0.0 for f in leak_types)


def test_the_builder_helper_gives_valid_plans_for_any_drop_list():
    frame, truth = load("planted-01")
    plan = Plan.model_validate(build_plan(frame, truth, [], []))
    assert run_with(frame, truth, plan).plan == plan
