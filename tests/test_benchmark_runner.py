import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import pytest
from sklearn.exceptions import ConvergenceWarning

import bench_runner
from bench_cost import MAX_SCHEMA_RETRIES, ModelPrice
from bench_datasets import DatasetSpec
from bench_naive import naive_plan
from bench_results import BenchmarkResults
from bench_runner import BenchmarkError, plan_jobs, plan_run, run_benchmark
from benchmark_support import (
    answer,
    classification_frame,
    classification_plan,
    questions_answer,
    regression_plan,
    stub_baseline,
    write_workspace,
)
from prep_agent.baseline import evaluate_baselines
from prep_agent.executor import execute
from prep_agent.llm.base import LLMAPIError, TokenUsage
from prep_agent.llm.fake import FakeLLM
from prep_agent.plan import Plan
from prep_agent.profiler import profile_dataframe

MODEL = "fake-model"
PRICE = ModelPrice(input_per_million=2.0, output_per_million=8.0)
USAGE = TokenUsage(input_tokens=1000, output_tokens=200)
CALL_COST = (1000 * 2.0 + 200 * 8.0) / 1_000_000


@dataclass
class Workspace:
    root: Path
    specs: list[DatasetSpec]

    @property
    def datasets_dir(self) -> Path:
        return self.root / "datasets"

    @property
    def reference_dir(self) -> Path:
        return self.root / "reference"

    @property
    def out_dir(self) -> Path:
        return self.root / "out"

    def run(self, llm=None, models=(MODEL,), repeats=1, factory=None, price=PRICE, **kwargs):
        factory = factory or (lambda model: llm)
        return run_benchmark(
            self.specs,
            list(models),
            repeats=repeats,
            datasets_dir=self.datasets_dir,
            reference_dir=self.reference_dir,
            out_dir=self.out_dir,
            prices={model: price for model in models},
            currency="USD",
            client_factory=factory,
            **kwargs,
        )


@pytest.fixture
def ws(tmp_path) -> Workspace:
    return Workspace(tmp_path, write_workspace(tmp_path))


@pytest.fixture
def real_baseline():
    """Ask for this fixture to score with the real cross-validation."""


@pytest.fixture(autouse=True)
def fast_baseline(request, monkeypatch):
    if "real_baseline" not in request.fixturenames:
        monkeypatch.setattr("bench_runner.evaluate_baselines", stub_baseline)


def good_llm(**kwargs) -> FakeLLM:
    return FakeLLM([answer(classification_plan()), answer(regression_plan())], **kwargs)


def job_of(results: BenchmarkResults, dataset: str, method: str, repeat: int = 1):
    [match] = [
        j
        for j in results.jobs
        if j.dataset == dataset and j.method == method and j.repeat == repeat
    ]
    return match


class TestPlanJobs:
    def test_each_dataset_gets_a_reference_a_naive_and_every_agent_run(self, ws):
        jobs = plan_jobs(ws.specs, ["a", "b"], repeats=2)
        per_dataset = [(j.method, j.model, j.repeat) for j in jobs if j.dataset == "cls"]
        assert per_dataset == [
            ("reference", None, 1),
            ("naive", None, 1),
            ("agent", "a", 1),
            ("agent", "a", 2),
            ("agent", "b", 1),
            ("agent", "b", 2),
        ]
        assert len(jobs) == 12

    def test_datasets_stay_together_in_registry_order(self, ws):
        assert [j.dataset for j in plan_jobs(ws.specs, ["a"], 1)] == ["cls"] * 3 + ["reg"] * 3

    def test_without_models_only_the_fixed_plans_run(self, ws):
        assert {j.method for j in plan_jobs(ws.specs, [], 3)} == {"reference", "naive"}


class TestEndToEnd:
    def test_runs_every_plan_and_writes_results_and_summary(self, ws, real_baseline):
        llm = good_llm(usage=USAGE)
        results = ws.run(llm)

        assert (ws.out_dir / "results.json").exists()
        assert (ws.out_dir / "summary.md").exists()
        assert len(results.jobs) == 6
        assert [j.status for j in results.jobs] == ["ok"] * 6
        assert all(
            j.baseline is not None and j.baseline.skipped_reason is None for j in results.jobs
        )
        assert all(j.plan is not None for j in results.jobs)
        assert llm.remaining == 0

        on_disk = BenchmarkResults.model_validate_json(
            (ws.out_dir / "results.json").read_text(encoding="utf-8")
        )
        assert on_disk == results
        summary = (ws.out_dir / "summary.md").read_text(encoding="utf-8")
        assert summary.startswith("# Benchmark summary")
        assert "### f1_macro" in summary
        assert "### r2" in summary
        assert "agent: fake-model" in summary

    def test_records_the_run_configuration(self, ws):
        results = ws.run(good_llm())
        config = results.config
        assert (config.models, config.repeats, config.datasets) == ([MODEL], 1, ["cls", "reg"])
        assert config.currency == "USD"
        assert config.prices == {MODEL: PRICE}
        assert results.finished_at is not None

    def test_records_sizes_and_timings(self, ws):
        job = job_of(ws.run(good_llm()), "cls", "naive")
        assert job.n_train + job.n_test == 80
        assert job.n_features is not None and job.n_features > 0
        assert job.plan_seconds > 0
        assert job.baseline_seconds is not None

    def test_the_baseline_is_what_the_baseline_module_gives(self, ws, real_baseline):
        ws.specs = ws.specs[:1]
        job = job_of(ws.run(models=()), "cls", "naive")
        df = classification_frame()
        profile = profile_dataframe(df, target="y")
        plan = naive_plan(profile, "classification")
        execution = execute(df, plan, profile)
        expected = evaluate_baselines(
            execution.split_train, execution.pipeline, plan.target, plan.task_type
        )
        assert job.baseline == expected

    def test_only_the_naive_and_reference_plans_run_without_models(self, ws):
        llm = good_llm()
        results = ws.run(llm, models=())
        assert {j.method for j in results.jobs} == {"reference", "naive"}
        assert llm.calls == []

    def test_every_repeat_is_its_own_agent_run(self, ws):
        llm = FakeLLM([answer(classification_plan())] * 2 + [answer(regression_plan())] * 2)
        results = ws.run(llm, repeats=2)
        agents = [j for j in results.jobs if j.method == "agent"]
        assert [(j.dataset, j.repeat) for j in agents] == [
            ("cls", 1),
            ("cls", 2),
            ("reg", 1),
            ("reg", 2),
        ]
        assert len([j for j in results.jobs if j.method == "naive"]) == 2

    def test_a_client_is_built_once_per_model_before_any_job_runs(self, ws):
        built = []

        def factory(model):
            built.append(model)
            return FakeLLM([answer(classification_plan()), answer(regression_plan())] * 1)

        ws.run(models=("a", "b"), factory=factory, repeats=1)
        assert built == ["a", "b"]

    def test_no_client_is_built_without_models(self, ws):
        def factory(model):
            raise AssertionError("a client was built")

        ws.run(models=(), factory=factory)

    def test_a_client_that_cannot_be_built_stops_before_anything_runs(self, ws):
        def factory(model):
            raise RuntimeError("no key")

        with pytest.raises(RuntimeError, match="no key"):
            ws.run(factory=factory)
        assert not (ws.out_dir / "results.json").exists()

    def test_refuses_to_overwrite_earlier_results(self, ws):
        ws.out_dir.mkdir()
        (ws.out_dir / "results.json").write_text("{}", encoding="utf-8")
        llm = good_llm()
        with pytest.raises(BenchmarkError, match="already holds results"):
            ws.run(llm)
        assert llm.calls == []
        assert (ws.out_dir / "results.json").read_text(encoding="utf-8") == "{}"

    def test_results_are_saved_after_every_job(self, ws):
        seen = []

        def progress(job, index, total):
            on_disk = json.loads((ws.out_dir / "results.json").read_text(encoding="utf-8"))
            seen.append((index, total, len(on_disk["jobs"]), job.dataset, job.method))

        ws.run(good_llm(), progress=progress)
        assert [s[:3] for s in seen] == [(i, 6, i) for i in range(1, 7)]
        assert seen[0][3:] == ("cls", "reference")


class TestAgentMetrics:
    def test_tokens_and_cost_come_from_the_reported_usage_and_the_price(self, ws):
        agent = job_of(ws.run(good_llm(usage=USAGE)), "cls", "agent").agent
        assert (agent.input_tokens, agent.output_tokens) == (1000, 200)
        assert agent.cost == pytest.approx(CALL_COST)
        assert agent.reported_model == "fake"
        assert agent.calls == 1

    def test_cache_counts_and_their_prices_enter_the_metrics(self, ws):
        usage = TokenUsage(1000, 200, cached_input_tokens=600, cache_write_tokens=100)
        price = ModelPrice(
            input_per_million=2.0,
            output_per_million=8.0,
            cached_input_per_million=0.5,
            cache_write_per_million=2.5,
        )

        agent = job_of(ws.run(good_llm(usage=usage), price=price), "cls", "agent").agent

        assert (agent.cached_input_tokens, agent.cache_write_tokens) == (600, 100)
        assert agent.cost == pytest.approx((300 * 2.0 + 600 * 0.5 + 100 * 2.5 + 200 * 8.0) / 1e6)

    def test_cache_counts_without_cache_prices_cost_the_old_way(self, ws):
        usage = TokenUsage(1000, 200, cached_input_tokens=600, cache_write_tokens=100)

        agent = job_of(ws.run(good_llm(usage=usage)), "cls", "agent").agent

        assert (agent.cached_input_tokens, agent.cache_write_tokens) == (600, 100)
        assert agent.cost == pytest.approx(CALL_COST)

    def test_cache_counts_are_unknown_when_the_provider_reported_none(self, ws):
        agent = job_of(ws.run(good_llm(usage=USAGE)), "cls", "agent").agent
        assert (agent.cached_input_tokens, agent.cache_write_tokens) == (None, None)

    def test_unreported_usage_leaves_tokens_and_cost_unknown(self, ws):
        agent = job_of(ws.run(good_llm()), "cls", "agent").agent
        assert (agent.input_tokens, agent.output_tokens, agent.cost) == (None, None, None)

    def test_a_revision_is_counted_with_the_errors_that_caused_it(self, ws):
        llm = FakeLLM(
            [answer(classification_plan(drop="nope")), answer(classification_plan())]
            + [answer(regression_plan())],
            usage=USAGE,
        )
        job = job_of(ws.run(llm), "cls", "agent")
        assert job.status == "ok"
        assert job.agent.revisions == 1
        assert job.agent.calls == 2
        assert job.agent.first_plan_error_codes == ["unknown_column"]
        assert job.agent.input_tokens == 2000
        assert job.agent.cost == pytest.approx(2 * CALL_COST)

    def test_a_schema_retry_is_counted_apart_from_revisions(self, ws):
        invalid = {"kind": "plan", "plan": None, "questions": None}
        llm = FakeLLM([invalid, answer(classification_plan()), answer(regression_plan())])
        job = job_of(ws.run(llm), "cls", "agent")
        assert job.status == "ok"
        assert (job.agent.calls, job.agent.revisions, job.agent.schema_retries) == (2, 0, 1)
        assert job.agent.first_plan_error_codes == []

    def test_a_rejected_revision_is_one_revision_and_one_retry(self, ws):
        invalid = {"kind": "plan", "plan": None, "questions": None}
        llm = FakeLLM(
            [
                answer(classification_plan(drop="nope")),
                invalid,
                answer(classification_plan()),
                answer(regression_plan()),
            ]
        )
        job = job_of(ws.run(llm), "cls", "agent")
        assert job.status == "ok"
        assert (job.agent.calls, job.agent.revisions, job.agent.schema_retries) == (3, 1, 1)
        assert job.agent.first_plan_error_codes == ["unknown_column"]

    def test_runs_without_retries_record_zero(self, ws):
        assert job_of(ws.run(good_llm()), "cls", "agent").agent.schema_retries == 0

    def test_the_agent_gets_the_retry_limit_of_the_estimate(self, ws, monkeypatch):
        seen = {}
        real = bench_runner.run_agent

        def spy(*args, **kwargs):
            seen.update(kwargs)
            return real(*args, **kwargs)

        monkeypatch.setattr(bench_runner, "run_agent", spy)
        ws.run(good_llm())
        assert seen["max_schema_retries"] == MAX_SCHEMA_RETRIES

    def test_a_plan_accepted_at_once_has_no_first_plan_errors(self, ws):
        agent = job_of(ws.run(good_llm()), "cls", "agent").agent
        assert (agent.revisions, agent.first_plan_error_codes) == (0, [])

    def test_questions_are_counted_and_answered_with_the_recommendation(self, ws):
        llm = FakeLLM(
            [questions_answer(), answer(classification_plan()), answer(regression_plan())]
        )
        job = job_of(ws.run(llm), "cls", "agent")
        assert job.agent.questions == 1
        assert [(q.question, q.answer) for q in job.agent.qa] == [("Keep outliers?", "no")]
        assert job.status == "ok"

    def test_the_warnings_of_the_final_plan_are_recorded(self, ws):
        plan = classification_plan()
        plan["steps"][0]["columns"] = ["income"]
        llm = FakeLLM([answer(plan), answer(regression_plan())])
        job = job_of(ws.run(llm), "cls", "agent")
        assert job.status == "ok"
        assert "uninformative_column_kept" in job.warning_codes

    def test_the_agent_is_given_the_goal_and_the_any_model_family(self, ws):
        llm = good_llm()
        ws.run(llm)
        first = llm.calls[0].user
        assert "Prepare cls for modelling." in first
        assert "Target column: y (use exactly this" in first
        assert "Model family: any (use exactly this" in first


class TestLeakyColumns:
    def mark(self, ws, *columns):
        ws.specs[0] = ws.specs[0].model_copy(update={"leaky_columns": list(columns)})

    def test_a_plan_that_keeps_a_leaky_column_is_flagged(self, ws):
        self.mark(ws, "income")
        results = ws.run(good_llm())
        for method in ("reference", "naive", "agent"):
            assert job_of(results, "cls", method).kept_leaky_columns == ["income"]

    def test_a_plan_that_drops_it_is_not(self, ws):
        self.mark(ws, "uid")
        results = ws.run(good_llm())
        for method in ("reference", "naive", "agent"):
            assert job_of(results, "cls", method).kept_leaky_columns == []

    def test_every_job_counts_how_many_of_the_leaky_columns_it_kept(self, ws):
        self.mark(ws, "income", "uid")
        results = ws.run(good_llm())
        for method in ("reference", "naive", "agent"):
            job = job_of(results, "cls", method)
            assert job.leaky_columns == ["income", "uid"]
            assert job.kept_leaky_columns == ["income"]
            assert job.leaky_columns_kept == 1

    def test_the_count_is_written_to_results_json(self, ws):
        self.mark(ws, "income", "uid")
        ws.run(good_llm())
        on_disk = json.loads((ws.out_dir / "results.json").read_text(encoding="utf-8"))
        cls_jobs = [j for j in on_disk["jobs"] if j["dataset"] == "cls"]
        assert [j["leaky_columns_kept"] for j in cls_jobs] == [1, 1, 1]
        assert [j["leaky_columns_kept"] for j in on_disk["jobs"] if j["dataset"] != "cls"] == [
            0,
            0,
            0,
        ]

    def test_a_job_without_a_plan_has_no_count_but_still_knows_the_leaky_columns(self, ws):
        self.mark(ws, "income")
        (ws.reference_dir / "cls.plan.json").write_text("{not json", encoding="utf-8")
        job = job_of(ws.run(good_llm()), "cls", "reference")
        assert job.status == "error"
        assert job.leaky_columns == ["income"]
        assert job.leaky_columns_kept is None

    def test_datasets_without_leaky_columns_are_never_flagged(self, ws):
        assert all(j.kept_leaky_columns == [] for j in ws.run(good_llm()).jobs)

    def test_a_failed_plan_is_flagged_too(self, ws):
        self.mark(ws, "income")
        bad = answer(classification_plan(drop="nope"))
        results = ws.run(FakeLLM([bad, bad, bad, answer(regression_plan())]))
        job = job_of(results, "cls", "agent")
        assert job.status == "failed"
        assert job.kept_leaky_columns == ["income"]


class TestFailuresStayLocal:
    def test_an_agent_that_runs_out_of_revisions_is_recorded_as_failed(self, ws):
        bad = answer(classification_plan(drop="nope"))
        llm = FakeLLM([bad, bad, bad, answer(regression_plan())], usage=USAGE)
        results = ws.run(llm)
        job = job_of(results, "cls", "agent")
        assert job.status == "failed"
        assert job.agent.revisions == 2
        assert job.error_codes == ["unknown_column"]
        assert job.agent.first_plan_error_codes == ["unknown_column"]
        assert "revision" in job.error
        assert job.baseline is None
        assert job.plan is not None
        assert job.agent.cost == pytest.approx(3 * CALL_COST)
        assert job_of(results, "reg", "agent").status == "ok"
        assert job_of(results, "cls", "naive").status == "ok"

    def test_a_failing_llm_call_is_recorded_as_failed(self, ws):
        llm = FakeLLM([LLMAPIError("rate limit"), answer(regression_plan())])
        results = ws.run(llm)
        job = job_of(results, "cls", "agent")
        assert job.status == "failed"
        assert "rate limit" in job.error
        assert job.plan is None
        assert job_of(results, "reg", "agent").status == "ok"

    def test_an_unexpected_exception_is_recorded_as_error_and_the_rest_carries_on(self, ws):
        llm = FakeLLM([RuntimeError("boom"), answer(regression_plan())])
        results = ws.run(llm)
        job = job_of(results, "cls", "agent")
        assert job.status == "error"
        assert "RuntimeError" in job.error
        assert "boom" in job.error
        assert [j.status for j in results.jobs if j.dataset == "reg"] == ["ok"] * 3
        assert (ws.out_dir / "summary.md").exists()

    def test_a_missing_dataset_file_fails_only_that_dataset(self, ws):
        (ws.datasets_dir / "cls.csv").unlink()
        results = ws.run(FakeLLM([answer(regression_plan())]))
        failed = [j for j in results.jobs if j.dataset == "cls"]
        assert [j.status for j in failed] == ["error"] * 3
        assert all("not found" in j.error for j in failed)
        assert [j.status for j in results.jobs if j.dataset == "reg"] == ["ok"] * 3

    def test_an_unexpected_error_while_loading_a_dataset_fails_only_that_dataset(
        self, ws, monkeypatch
    ):
        real = bench_runner.load_csv

        def flaky(path):
            if Path(path).name == "cls.csv":
                raise RuntimeError("disk on fire")
            return real(path)

        monkeypatch.setattr("bench_runner.load_csv", flaky)
        results = ws.run(FakeLLM([answer(regression_plan())]))
        broken = [j for j in results.jobs if j.dataset == "cls"]
        assert [j.status for j in broken] == ["error"] * 3
        assert all("disk on fire" in j.error for j in broken)
        assert [j.status for j in results.jobs if j.dataset == "reg"] == ["ok"] * 3

    def test_a_missing_reference_plan_fails_only_the_reference_job(self, ws):
        (ws.reference_dir / "cls.plan.json").unlink()
        results = ws.run(good_llm())
        job = job_of(results, "cls", "reference")
        assert job.status == "error"
        assert "reference plan" in job.error
        assert job_of(results, "cls", "naive").status == "ok"
        assert job_of(results, "cls", "agent").status == "ok"

    def test_a_reference_plan_that_is_not_valid_json_is_an_error(self, ws):
        (ws.reference_dir / "cls.plan.json").write_text("{not json", encoding="utf-8")
        job = job_of(ws.run(good_llm()), "cls", "reference")
        assert job.status == "error"
        assert "invalid reference plan" in job.error

    def test_a_reference_plan_the_validator_rejects_is_failed_with_its_codes(self, ws):
        bad = Plan.model_validate(classification_plan(drop="nope"))
        (ws.reference_dir / "cls.plan.json").write_text(bad.model_dump_json(), encoding="utf-8")
        job = job_of(ws.run(good_llm()), "cls", "reference")
        assert job.status == "failed"
        assert job.error_codes == ["unknown_column"]
        assert job.plan == bad
        assert job.baseline is None

    def test_a_crash_while_scoring_is_recorded_and_the_run_goes_on(self, ws, monkeypatch):
        def explode(*args, **kwargs):
            raise MemoryError("too big")

        monkeypatch.setattr("bench_runner.evaluate_baselines", explode)
        results = ws.run(good_llm())
        assert [j.status for j in results.jobs] == ["error"] * 6
        assert all("too big" in j.error for j in results.jobs)


class TestConvergenceWarnings:
    def test_a_baseline_that_does_not_converge_is_scored_without_noise(self, ws, monkeypatch):
        def noisy(train, pipeline, target, task_type, **kwargs):
            warnings.warn("did not converge", ConvergenceWarning, stacklevel=2)
            return stub_baseline(train, pipeline, target, task_type)

        monkeypatch.setattr("bench_runner.evaluate_baselines", noisy)
        results = ws.run(models=())
        assert [j.status for j in results.jobs] == ["ok"] * 4


class TestPlanRun:
    def test_estimates_every_model_from_the_downloaded_data(self, ws):
        run_plan = plan_run(
            ws.specs, ["a", "b"], repeats=2, datasets_dir=ws.datasets_dir, prices={"a": PRICE}
        )
        assert [(e.model, e.runs) for e in run_plan.estimates] == [("a", 4), ("b", 4)]
        first, second = run_plan.estimates
        assert first.input_tokens == second.input_tokens > 0
        assert first.output_tokens == second.output_tokens > 0
        assert first.cost == pytest.approx(
            (first.input_tokens * 2.0 + first.output_tokens * 8.0) / 1_000_000
        )
        assert second.cost is None

    def test_scales_with_the_number_of_repeats(self, ws):
        once = plan_run(ws.specs, ["a"], 1, ws.datasets_dir, {"a": PRICE}).estimates[0]
        twice = plan_run(ws.specs, ["a"], 2, ws.datasets_dir, {"a": PRICE}).estimates[0]
        assert twice.input_tokens == 2 * once.input_tokens

    def test_leaves_datasets_that_are_not_downloaded_out_and_says_so(self, ws):
        (ws.datasets_dir / "reg.csv").unlink()
        run_plan = plan_run(ws.specs, ["a"], 1, ws.datasets_dir, {"a": PRICE})
        assert run_plan.not_downloaded == ["reg"]
        full = plan_run(ws.specs, ["a"], 1, ws.datasets_dir, {"a": PRICE})
        assert run_plan.estimates[0].input_tokens < 2 * full.estimates[0].input_tokens

    def test_lists_the_jobs_and_needs_no_client(self, ws):
        run_plan = plan_run(ws.specs, ["a"], 1, ws.datasets_dir, {"a": PRICE})
        assert len(run_plan.jobs) == 6
