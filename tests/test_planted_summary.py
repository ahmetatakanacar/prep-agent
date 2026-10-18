from datetime import UTC, datetime

import pytest

from bench_cost import ModelPrice
from planted_generate import load
from planted_results import AgentRecord, PlantedResults, RunConfig, RunResult
from planted_scorer import score_plan
from planted_summary import groups, render_summary
from planted_support import drop_all_plan, leak_blind_plan, perfect_decisions, perfect_plan, plan_of
from planted_truth import FAULT_TYPES

PRICE = ModelPrice(input_per_million=1.0, output_per_million=2.0)


def agent(cost, questions=0, calls=1) -> AgentRecord:
    return AgentRecord(
        reported_model="m",
        calls=calls,
        revisions=0,
        questions=questions,
        input_tokens=100,
        output_tokens=10,
        cost=cost,
        qa=[],
    )


def run(variant, mode, make_plan, *, repeat=1, cost=0.01, questions=0, calls=1, status="ok"):
    frame, truth = load("planted-03", variant)
    plan = make_plan(frame, truth) if make_plan else None
    return RunResult(
        dataset="planted-03",
        variant=variant,
        mode=mode,
        model="m",
        repeat=repeat,
        task_type="regression",
        status=status,
        error="the plan still has 1 error(s): unknown_column | ghost" if status != "ok" else None,
        agent=agent(cost, questions, calls),
        plan=plan,
        score=score_plan(plan, truth) if status == "ok" else None,
    )


def cast_the_decoy(frame, truth):
    drop, cast = perfect_decisions(truth)
    decoy = next(c.name for c in truth.columns if c.avoid_op)
    return plan_of(frame, truth, drop, [*cast, decoy])


def results(runs, variants=("anonymous", "named")) -> PlantedResults:
    return PlantedResults(
        started_at=datetime(2026, 10, 18, 9, 0, tzinfo=UTC),
        finished_at=datetime(2026, 10, 18, 9, 30, tzinfo=UTC),
        config=RunConfig(
            models=["m"],
            variants=list(variants),
            modes=["recommended", "not_recommended", "no_questions"],
            repeats=2,
            datasets=["planted-03"],
            currency="USD",
            prices={"m": PRICE},
        ),
        runs=runs,
    )


@pytest.fixture(scope="module")
def sample():
    return results(
        [
            run(
                "anonymous", "recommended", perfect_plan, repeat=1, cost=0.01, questions=1, calls=2
            ),
            run(
                "anonymous",
                "recommended",
                leak_blind_plan,
                repeat=2,
                cost=0.03,
                questions=1,
                calls=2,
            ),
            run(
                "anonymous",
                "recommended",
                None,
                repeat=2,
                cost=0.02,
                questions=1,
                calls=3,
                status="failed",
            ),
            run("named", "recommended", perfect_plan),
            run("anonymous", "not_recommended", drop_all_plan),
            run("named", "not_recommended", cast_the_decoy),
        ]
    )


def table(markdown: str, heading: str) -> list[dict[str, str]]:
    rows = []
    for line in markdown.split(f"## {heading}\n", 1)[1].splitlines():
        if line.startswith("|"):
            rows.append([cell.strip() for cell in line.strip("|").split("|")])
        elif rows:
            break
    return [dict(zip(rows[0], row, strict=True)) for row in rows[2:]]


def headline(sample, variant, mode):
    rows = table(render_summary(sample), "Results by model, variant and mode")
    [row] = [r for r in rows if r["variant"] == variant and r["mode"] == mode]
    return row


def test_a_failed_run_is_counted_but_never_scored(sample):
    row = headline(sample, "anonymous", "recommended")
    assert row["runs (ok)"] == "3 (2)"


def test_detection_precision_and_recall_add_up_the_counts_before_the_ratio(sample):
    row = headline(sample, "anonymous", "recommended")
    # 5 of 5 faults handled in the perfect run and 4 of 5 in the leak-blind one.
    assert row["detection"].startswith("0.90")
    assert row["drop precision"] == "1.00"
    # 4 of 4 needed drops in one run, 3 of 4 in the other.
    assert row["drop recall"] == "0.88"


def test_the_spread_between_repeats_is_shown(sample):
    assert headline(sample, "anonymous", "recommended")["detection"] == "0.90 (0.80-1.00)"
    assert headline(sample, "named", "recommended")["detection"] == "1.00"


def test_questions_calls_and_cost_are_averaged_over_every_run_with_a_record(sample):
    row = headline(sample, "anonymous", "recommended")
    assert row["questions/run"] == "1.0"
    assert row["calls/run"] == "2.3"
    assert row["cost/run (USD)"] == "0.0200"
    assert row["cost total (USD)"] == "0.0600"


def test_dropped_must_keep_columns_and_avoided_ops_are_counted(sample):
    dropped = headline(sample, "anonymous", "not_recommended")
    frame, truth = load("planted-03")
    lost = len(score_plan(drop_all_plan(frame, truth), truth).over_dropped)
    assert dropped["must_keep dropped"] == f"{lost} ({lost:.2f}/run)"
    assert dropped["drop precision"] != "1.00"
    cast = headline(sample, "named", "not_recommended")
    assert cast["avoided_op violations"] == "1 (1.00/run)"
    assert cast["must_keep dropped"] == "0 (0.00/run)"


def test_the_fault_table_has_counts_per_type_and_marks_what_is_not_scored(sample):
    rows = table(render_summary(sample), "Detection by fault type")
    [row] = [r for r in rows if r["variant"] == "anonymous" and r["mode"] == "recommended"]
    assert row["leak_nonlinear"] == "50% (1/2)"
    assert row["id_sequential"] == "100% (2/2)"
    assert row["leak_linear"] == "n/a"
    assert row["sentinel"] == "not scored"
    assert list(row)[3:] == [*FAULT_TYPES, "leak_partial"]


def test_the_variants_are_compared_for_the_same_model_and_mode(sample):
    rows = table(render_summary(sample), "Anonymous against named")
    [row] = [r for r in rows if r["mode"] == "recommended"]
    assert row["detection anonymous"] == "0.90" and row["detection named"] == "1.00"
    assert row["detection difference"] == "+10 pp"
    assert row["precision difference"] == "+0 pp"
    [other] = [r for r in rows if r["mode"] == "not_recommended"]
    assert other["model"] == "m"


def test_runs_without_a_score_are_listed_with_their_reason(sample):
    markdown = render_summary(sample)
    section = markdown.split("### Runs without a score\n", 1)[1]
    assert "failed" in section and "unknown_column / ghost" in section


def test_a_single_variant_has_no_comparison():
    markdown = render_summary(
        results([run("anonymous", "recommended", perfect_plan)], variants=("anonymous",))
    )
    assert "Anonymous against named" not in markdown


def test_modes_and_variants_are_separate_columns_in_a_fixed_order(sample):
    order = [(g.variant, g.mode) for g in groups(sample)]
    assert order == [
        ("anonymous", "recommended"),
        ("named", "recommended"),
        ("anonymous", "not_recommended"),
        ("named", "not_recommended"),
    ]
    header = render_summary(sample).split("## Results by model, variant and mode\n\n", 1)[1]
    assert header.startswith("| model | variant | mode |")


def test_unknown_cost_and_no_ok_runs_show_as_not_available():
    failed = run("anonymous", "recommended", None, status="failed", cost=None)
    row = headline(results([failed]), "anonymous", "recommended")
    assert row["runs (ok)"] == "1 (0)"
    assert row["detection"] == row["drop precision"] == row["must_keep dropped"] == "n/a"
    assert row["cost/run (USD)"] == row["cost total (USD)"] == "n/a"


def test_an_empty_result_renders():
    assert "Runs recorded: 0" in render_summary(results([]))
