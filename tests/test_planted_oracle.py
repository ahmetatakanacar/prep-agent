"""The oracle answer mode, the plan accuracy, and the question metrics of a run."""

import pytest

from bench_cost import ModelPrice
from planted_generate import load
from planted_modes import Answerer
from planted_results import MarkedColumn
from planted_runner import run_planted
from planted_scorer import combine, score_plan
from planted_support import RoutedLLM, perfect_decisions, plan_of
from planted_truth import ColumnTruth, DatasetTruth
from prep_agent.planner import Question


def truth_of(*columns: ColumnTruth) -> DatasetTruth:
    return DatasetTruth(
        dataset="d",
        seed=1,
        target="y",
        task_type="classification",
        n_rows=100,
        columns=list(columns),
    )


TRUTH = truth_of(
    ColumnTruth(
        name="refund_date", role="must_drop", fault="leak_partial", level=0.9, reason="after"
    ),
    ColumnTruth(
        name="floor_area", role="must_keep", fault="legit_strong", level=0.9, reason="before"
    ),
    ColumnTruth(
        name="channel", role="must_keep", fault="numeric_category", expected_op="cast_to_category"
    ),
    ColumnTruth(name="rooms", role="must_keep", avoid_op="cast_to_category"),
    ColumnTruth(name="age", role="must_keep"),
    ColumnTruth(name="uuid", role="must_drop", fault="id_uuid"),
)


def ask(text, options, recommended=None, qid="q") -> Question:
    return Question(
        id=qid, question=text, options=options, recommended=recommended or options[0], reason="open"
    )


def oracle(question) -> tuple[str, str]:
    answerer = Answerer("oracle", TRUTH)
    answer = answerer([question])[question.id]
    return answer, answerer.log[0].verdict


def test_the_oracle_needs_the_truth():
    with pytest.raises(ValueError, match="needs the ground truth"):
        Answerer("oracle")


def test_the_oracle_drops_a_leak_the_agent_recommends_keeping():
    question = ask(
        "Is refund_date known when the prediction is made?",
        ["Keep refund_date", "Drop refund_date"],
        recommended="Keep refund_date",
    )
    assert oracle(question) == ("Drop refund_date", "drop")


def test_the_oracle_keeps_a_legitimate_feature_the_agent_recommends_dropping():
    question = ask(
        "floor_area is flagged as suspected target leakage. What should happen to it?",
        ["Drop floor_area", "Keep floor_area as a feature"],
        recommended="Drop floor_area",
    )
    assert oracle(question) == ("Keep floor_area as a feature", "keep")


def test_the_oracle_drops_identifiers_and_keeps_plain_columns():
    assert oracle(ask("What about uuid?", ["Keep uuid", "Remove uuid"])) == ("Remove uuid", "drop")
    assert oracle(ask("Use age?", ["Drop age", "Retain age"])) == ("Retain age", "keep")


def test_the_oracle_casts_codes_and_leaves_counts_alone():
    codes = ask(
        "Is channel a number or a category?",
        ["Keep channel as numeric", "Treat channel as categorical"],
    )
    assert oracle(codes) == ("Treat channel as categorical", "categorical")
    counts = ask(
        "Is rooms a count or a category?",
        ["Treat rooms as categorical", "Keep rooms as a numeric count"],
    )
    assert oracle(counts) == ("Keep rooms as a numeric count", "numeric")


def test_the_oracle_matches_whole_column_names_only():
    question = ask(
        "What about refund_date_extra?", ["Drop refund_date_extra", "Keep refund_date_extra"]
    )
    assert oracle(question) == ("Drop refund_date_extra", "none")


def test_a_question_about_no_known_column_gets_the_recommendation():
    question = ask("Which model will you use?", ["tree", "linear", "any"], recommended="any")
    assert oracle(question) == ("any", "none")


def test_columns_with_opposite_verdicts_get_the_recommendation_and_are_logged_as_mixed():
    question = ask("Drop uuid and floor_area?", ["Keep both", "Drop both"], recommended="Keep both")
    assert oracle(question) == ("Keep both", "mixed")


def test_the_majority_decides_between_several_columns():
    question = ask(
        "What about uuid, refund_date and age?", ["Keep all", "Drop all"], recommended="Keep all"
    )
    assert oracle(question) == ("Drop all", "drop")


def test_an_option_that_does_not_fit_the_verdict_falls_back_to_the_recommendation():
    question = ask(
        "What about refund_date?", ["Ask later", "Think about it"], recommended="Ask later"
    )
    assert oracle(question) == ("Ask later", "drop")


def test_the_other_modes_do_not_log_a_verdict():
    answerer = Answerer("recommended")
    answerer([ask("What about uuid?", ["Keep uuid", "Drop uuid"])])
    assert answerer.log[0].verdict is None


# --- plan accuracy ---


@pytest.fixture(scope="module")
def d01():
    return load("planted-d01")


def test_a_perfect_plan_has_full_accuracy_and_a_wrong_one_loses_it(d01):
    frame, truth = d01
    drop, cast = perfect_decisions(truth)
    perfect = score_plan(plan_of(frame, truth, drop, cast), truth)
    assert perfect.plan_accuracy == 1.0
    legit = next(c.name for c in truth.columns if c.fault == "legit_strong")
    wrong = score_plan(plan_of(frame, truth, [*drop, legit], cast), truth)
    assert wrong.plan_accuracy == pytest.approx(
        (perfect.correct_drops + perfect.keep_total - 1)
        / (perfect.required_drops + perfect.keep_total)
    )
    missed = score_plan(plan_of(frame, truth, drop[1:], cast), truth)
    assert missed.plan_accuracy < 1.0


def test_accuracy_adds_up_over_datasets(d01):
    frame, truth = d01
    score = score_plan(plan_of(frame, truth, *perfect_decisions(truth)), truth)
    total = combine([score, score])
    assert total.keep_total == 2 * score.keep_total and total.plan_accuracy == 1.0


def test_there_is_no_accuracy_without_decisions():
    assert score_plan_of_nothing().plan_accuracy is None


def score_plan_of_nothing():
    from planted_scorer import Score

    return Score(datasets=[])


# --- the oracle and the question metrics in a run ---


def run(tmp_path, dataset, variant, mode):
    return run_planted(
        ["m"],
        datasets=[dataset],
        variants=[variant],
        modes=[mode],
        repeats=1,
        out_dir=tmp_path / f"{variant}-{mode}",
        prices={"m": ModelPrice(input_per_million=1.0, output_per_million=2.0)},
        currency="USD",
        client_factory=lambda model: RoutedLLM(),
    )


@pytest.mark.parametrize("variant", ["anonymous", "named"])
def test_an_oracle_run_answers_from_the_truth_and_logs_the_verdict(tmp_path, variant):
    [r] = run(tmp_path, "planted-d01", variant, "oracle").runs
    assert r.status == "ok" and r.agent.questions == 1
    [item] = r.agent.qa
    assert item.verdict == "drop" and item.answer.startswith("Drop ")
    assert r.score.plan_accuracy == 1.0


def test_marked_columns_are_recorded_with_their_tier_truth_and_whether_they_were_asked(tmp_path):
    [r] = run(tmp_path, "planted-d01", "named", "recommended").runs
    by_name = {m.name: m for m in r.marked}
    deposit = by_name["deposit_policy"]
    assert deposit == MarkedColumn(
        name="deposit_policy", score=deposit.score, tier="suspected", asked=False, truth="legit"
    )
    status = by_name["reservation_status_note"]
    assert status.tier == "high" and status.truth == "leak" and status.asked
    assert all(m.score >= 0.6 for m in r.marked)


def test_a_run_without_questions_records_marks_that_were_not_asked(tmp_path):
    [r] = run(tmp_path, "planted-d01", "named", "no_questions").runs
    assert r.marked and not any(m.asked for m in r.marked)


def test_the_summary_has_the_accuracy_and_the_question_columns(tmp_path):
    run(tmp_path, "planted-d01", "named", "recommended")
    summary = (tmp_path / "named-recommended" / "summary.md").read_text(encoding="utf-8")
    assert "plan accuracy" in summary and "asked: suspected" in summary
    assert "## Questions about columns strongly related to the target" in summary
    assert "leaks, 0.6 to 0.9" in summary
    row = [line for line in summary.splitlines() if line.startswith("| m | named | recommended")][0]
    assert "0 of 1" in row and "1 of 2" in row
