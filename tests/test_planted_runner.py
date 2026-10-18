import pytest

from bench_cost import ModelPrice
from planted_generate import RECIPES, load
from planted_modes import MODES
from planted_results import PlantedResults
from planted_runner import (
    RESULTS_FILE,
    SUMMARY_FILE,
    PlantedRunError,
    check_selection,
    goal_for,
    plan_jobs,
    plan_run,
    run_planted,
)
from planted_support import RoutedLLM, build_plan, perfect_decisions
from prep_agent.llm.base import LLMAPIError, TokenUsage
from prep_agent.llm.fake import FakeLLM

PRICE = ModelPrice(input_per_million=1.0, output_per_million=2.0)
CALL_COST = (1000 * 1.0 + 200 * 2.0) / 1_000_000
DATASETS = ["planted-01", "planted-08"]
VARIANTS = ["anonymous", "named"]
SMALL = {"datasets": ["planted-08"], "variants": ["anonymous"], "modes": ["recommended"]}
ONE = {"datasets": ["planted-01"], "variants": ["anonymous"], "modes": ["recommended"]}


def run(out_dir, llm, *, datasets=DATASETS, variants=VARIANTS, modes=MODES, repeats=1, **kw):
    return run_planted(
        ["m"],
        datasets=datasets,
        variants=variants,
        modes=list(modes),
        repeats=repeats,
        out_dir=out_dir,
        prices={"m": PRICE},
        currency="USD",
        client_factory=lambda model: llm,
        **kw,
    )


@pytest.fixture(scope="module")
def full(tmp_path_factory):
    out = tmp_path_factory.mktemp("planted") / "out"
    llm = RoutedLLM()
    results = run(out, llm, repeats=2)
    return results, llm, out


def select(results, **match):
    return [
        r for r in results.runs if all(getattr(r, key) == value for key, value in match.items())
    ]


def test_jobs_run_repeat_by_repeat_over_the_whole_product():
    jobs = plan_jobs(DATASETS, VARIANTS, list(MODES), ["a", "b"], 2)
    assert len(jobs) == 2 * 2 * 3 * 2 * 2
    assert [j.repeat for j in jobs] == [1] * 24 + [2] * 24
    assert [j.model for j in jobs[:12]] == ["a"] * 12
    assert len(set(jobs)) == len(jobs)


@pytest.mark.parametrize(
    "selection, message",
    [
        ((["nope"], VARIANTS, MODES), "unknown dataset 'nope'"),
        ((DATASETS, ["blue"], MODES), "unknown variant 'blue'"),
        ((DATASETS, VARIANTS, ["agree"]), "unknown mode 'agree'"),
        (([], VARIANTS, MODES), "no dataset selected"),
    ],
)
def test_a_wrong_selection_is_refused_before_anything_runs(selection, message):
    with pytest.raises(PlantedRunError, match=message):
        check_selection(*selection)


def test_every_job_ends_with_a_score(full):
    results, _, _ = full
    assert len(results.runs) == 2 * 2 * 3 * 2
    assert all(
        r.status == "ok" and r.score is not None and r.plan is not None for r in results.runs
    )
    assert {r.repeat for r in results.runs} == {1, 2}


def test_results_and_summary_are_written_and_round_trip(full):
    results, _, out = full
    on_disk = PlantedResults.model_validate_json((out / RESULTS_FILE).read_text(encoding="utf-8"))
    assert on_disk == results
    assert results.finished_at is not None
    assert (out / SUMMARY_FILE).read_text(encoding="utf-8").startswith("# Planted-fault")


def test_recommended_mode_follows_the_recommendation_and_finds_every_fault(full):
    results, _, _ = full
    runs = select(results, mode="recommended", dataset="planted-01")
    assert len(runs) == 4
    for r in runs:
        assert r.agent.questions == 1 and r.agent.calls == 2
        [item] = r.agent.qa
        assert item.answer == item.recommended and item.answer.startswith("Drop ")
        assert r.score.detection_rate == 1.0 and r.score.recall == 1.0


def test_not_recommended_mode_overrules_the_agent_and_the_plan_follows(full):
    results, _, _ = full
    for r in select(results, mode="not_recommended", dataset="planted-01"):
        [item] = r.agent.qa
        assert item.answer.startswith("Keep ") and item.answer != item.recommended
        assert not item.forced
        assert r.score.by_type["leak_linear"].detected == 0
        assert r.score.by_type["id_sequential"].detected == 1
        assert r.score.precision == 1.0 and r.score.recall < 1.0


def test_no_questions_mode_gives_the_agent_no_chance_to_ask(full):
    results, llm, _ = full
    for r in select(results, mode="no_questions"):
        assert r.agent.questions == 0 and r.agent.calls == 1 and r.agent.qa == []
        assert r.score.detection_rate == 1.0
    # Eight no_questions runs, each told once that it may not ask; the other modes' prompts
    # always leave at least one round.
    assert len([u for u in llm.users if "Do not ask any more questions" in u]) == 8
    assert len(llm.users) == 32


def test_a_dataset_without_leaks_gets_no_questions_in_any_mode(full):
    results, _, _ = full
    for r in select(results, dataset="planted-08"):
        assert r.agent.questions == 0 and r.agent.calls == 1
        assert r.score.detection_rate == 1.0


def test_both_variants_score_the_same_for_a_planner_that_ignores_names(full):
    results, _, _ = full
    for r in select(results, variant="anonymous"):
        twin = select(results, variant="named", dataset=r.dataset, mode=r.mode, repeat=r.repeat)
        [twin] = twin
        assert twin.score.found == r.score.found and twin.score.planted == r.score.planted


def test_cost_is_priced_from_the_reported_usage(full):
    results, _, _ = full
    for r in results.runs:
        assert r.agent.input_tokens == 1000 * r.agent.calls
        assert r.agent.cost == pytest.approx(CALL_COST * r.agent.calls)


def test_the_agent_gets_the_target_and_a_goal_that_names_it(full):
    _, llm, _ = full
    anonymous = [u for u in llm.users if "predicting the column y." in u]
    named = [u for u in llm.users if "predicting the column abonelik_iptal." in u]
    assert anonymous and named
    _, truth = load("planted-01", "named")
    assert goal_for(truth) == "Prepare this table for predicting the column abonelik_iptal."


def reply_with_unknown_drop(dataset="planted-01"):
    frame, truth = load(dataset)
    drop, cast = perfect_decisions(truth)
    plan = build_plan(frame, truth, [*drop, "ghost"], cast)
    return {"language": "English", "kind": "plan", "plan": plan, "questions": None}


def test_a_plan_that_never_becomes_valid_is_a_failed_run_with_no_score(tmp_path):
    bad = reply_with_unknown_drop()
    llm = FakeLLM([bad, bad, bad], usage=TokenUsage(1000, 200))
    results = run(
        tmp_path / "o", llm, datasets=["planted-01"], variants=["anonymous"], modes=["recommended"]
    )
    [r] = results.runs
    assert r.status == "failed" and r.score is None
    assert r.error_codes == ["unknown_column"]
    assert r.plan is not None and r.agent.calls == 3
    assert r.agent.cost == pytest.approx(3 * CALL_COST)


def test_a_crash_is_recorded_and_the_run_goes_on(tmp_path):
    good = RoutedLLM()
    calls = []

    class Flaky:
        def generate(self, system, user, schema):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("boom")
            return good.generate(system, user, schema)

    results = run(
        tmp_path / "o",
        Flaky(),
        datasets=["planted-08"],
        variants=["anonymous"],
        modes=["recommended", "no_questions"],
    )
    first, second = results.runs
    assert first.status == "error" and first.error == "RuntimeError: boom"
    assert second.status == "ok"


def test_an_llm_failure_is_a_failed_run(tmp_path):
    llm = FakeLLM([LLMAPIError("rate limit")])
    [r] = run(
        tmp_path / "o", llm, datasets=["planted-08"], variants=["anonymous"], modes=["recommended"]
    ).runs
    assert r.status == "failed" and "rate limit" in r.error


def test_a_model_without_a_price_still_runs_with_unknown_cost(tmp_path):
    results = run_planted(
        ["free"],
        datasets=["planted-08"],
        variants=["anonymous"],
        modes=["recommended"],
        repeats=1,
        out_dir=tmp_path / "o",
        prices={},
        currency="USD",
        client_factory=lambda model: RoutedLLM(),
    )
    [r] = results.runs
    assert r.status == "ok" and r.agent.cost is None and r.agent.input_tokens == 1000


def test_an_existing_result_is_not_overwritten(tmp_path):
    out = tmp_path / "o"
    out.mkdir()
    (out / RESULTS_FILE).write_text("{}", encoding="utf-8")
    with pytest.raises(PlantedRunError, match="already holds results"):
        run(
            out, RoutedLLM(), datasets=["planted-08"], variants=["anonymous"], modes=["recommended"]
        )
    assert (out / RESULTS_FILE).read_text(encoding="utf-8") == "{}"


def test_a_client_that_cannot_be_built_stops_the_run_before_it_writes(tmp_path):
    def broken(model):
        raise PlantedRunError("no key")

    with pytest.raises(PlantedRunError, match="no key"):
        run_planted(
            ["m"],
            datasets=["planted-08"],
            variants=["anonymous"],
            modes=["recommended"],
            repeats=1,
            out_dir=tmp_path / "o",
            prices={"m": PRICE},
            currency="USD",
            client_factory=broken,
        )
    assert not (tmp_path / "o").exists()


def test_progress_is_reported_for_every_run(tmp_path):
    seen = []
    run(
        tmp_path / "o",
        RoutedLLM(),
        datasets=["planted-08"],
        variants=["anonymous", "named"],
        modes=["recommended"],
        progress=lambda r, index, total: seen.append((index, total, r.variant)),
    )
    assert seen == [(1, 2, "anonymous"), (2, 2, "named")]


def test_the_estimate_scales_with_repeats_modes_and_data():
    one = plan_run(DATASETS, VARIANTS, ["recommended"], ["m", "x"], 1, {"m": PRICE})
    three = plan_run(DATASETS, VARIANTS, list(MODES), ["m", "x"], 2, {"m": PRICE})
    assert [e.runs for e in one.estimates] == [4, 4]
    assert [e.runs for e in three.estimates] == [24, 24]
    m, x = three.estimates
    assert m.input_tokens == 6 * one.estimates[0].input_tokens
    assert m.cost is not None and x.cost is None
    assert len(three.jobs) == 48
    assert len(RECIPES) == 8


def test_the_second_set_can_be_selected_by_name_but_is_not_the_default():
    check_selection(["planted-b01", "planted-08"], VARIANTS, list(MODES))
    from planted_runner import DATASETS as DEFAULT

    assert "planted-b01" not in DEFAULT and len(DEFAULT) == 8
