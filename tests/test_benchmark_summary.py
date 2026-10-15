from datetime import UTC, datetime

from bench_cost import ModelPrice
from bench_results import AgentMetrics, BenchmarkResults, JobResult, RunConfig
from bench_summary import render_summary
from prep_agent.baseline import BaselineResult, BaselineScore, ModelBaseline
from prep_agent.plan import Plan

CLASSIFIERS = ("dummy_most_frequent", "logistic_regression", "random_forest")
REGRESSORS = ("dummy_mean", "ridge", "random_forest")


def _scores(std, **means) -> dict:
    return {metric: BaselineScore(mean=mean, std=std) for metric, mean in means.items()}


def _baseline(task_type, linear: dict, forest: dict, std=0.01) -> BaselineResult:
    names = CLASSIFIERS if task_type == "classification" else REGRESSORS
    return BaselineResult(
        task_type=task_type,
        n_rows=100,
        random_state=42,
        n_folds=5,
        metrics=list(linear),
        models=[
            ModelBaseline(name=names[0], scores=_scores(std, **{m: 0.0 for m in linear})),
            ModelBaseline(name=names[1], scores=_scores(std, **linear)),
            ModelBaseline(name=names[2], scores=_scores(std, **forest)),
        ],
    )


def classification_baseline(linear_f1, forest_f1, linear_auc=None, forest_auc=None, std=0.01):
    linear, forest = {"f1_macro": linear_f1}, {"f1_macro": forest_f1}
    if linear_auc is not None:
        linear["roc_auc"], forest["roc_auc"] = linear_auc, forest_auc
    return _baseline("classification", linear, forest, std)


def regression_baseline(linear_r2, forest_r2, std=0.01):
    return _baseline("regression", {"r2": linear_r2}, {"r2": forest_r2}, std)


def job(dataset="d1", method="agent", model="m1", repeat=1, status="ok", **kwargs) -> JobResult:
    kwargs.setdefault("task_type", "classification")
    if method != "agent":
        model = None
    return JobResult(
        dataset=dataset, method=method, model=model, repeat=repeat, status=status, **kwargs
    )


def agent_metrics(**overrides) -> AgentMetrics:
    values = {
        "reported_model": "m1",
        "calls": 1,
        "revisions": 0,
        "questions": 0,
        "first_plan_error_codes": [],
        "input_tokens": 1000,
        "output_tokens": 200,
        "cost": 0.01,
        "qa": [],
    }
    return AgentMetrics(**(values | overrides))


def results(jobs, models=("m1",), repeats=1, datasets=("d1",)) -> BenchmarkResults:
    return BenchmarkResults(
        started_at=datetime(2026, 10, 14, 12, 0, tzinfo=UTC),
        finished_at=datetime(2026, 10, 14, 12, 30, tzinfo=UTC),
        config=RunConfig(
            models=list(models), repeats=repeats, datasets=list(datasets), currency="EUR", prices={}
        ),
        jobs=jobs,
    )


def table(markdown: str, heading: str) -> list[list[str]]:
    """The cells of the table under `heading`, header row first, separator row removed."""
    lines = markdown.splitlines()
    start = lines.index(heading) + 1
    rows = []
    for line in lines[start:]:
        if line.startswith("|"):
            rows.append([cell.strip() for cell in line.strip().strip("|").split(" | ")])
        elif rows:
            break
    return [row for row in rows if not set("".join(row)) <= set("-: ")]


def row(rows: list[list[str]], first_cell: str) -> list[str]:
    return next(r for r in rows if r[0] == first_cell)


class TestScores:
    def test_one_table_per_metric_with_linear_and_forest_in_each_cell(self):
        jobs = [
            job("d1", "reference", baseline=classification_baseline(0.6, 0.7, 0.8, 0.9)),
            job("d1", "naive", baseline=classification_baseline(0.5, 0.6, 0.7, 0.8)),
            job("d1", "agent", baseline=classification_baseline(0.65, 0.75, 0.85, 0.95)),
        ]
        md = render_summary(results(jobs))
        f1 = table(md, "### f1_macro ↑")
        assert f1[0] == ["dataset", "reference", "naive", "agent: m1"]
        assert row(f1, "d1") == ["d1", "0.600 / 0.700", "0.500 / 0.600", "0.650 / 0.750"]
        auc = table(md, "### roc_auc ↑")
        assert row(auc, "d1")[1:] == ["0.800 / 0.900", "0.700 / 0.800", "0.850 / 0.950"]

    def test_regression_datasets_only_appear_under_r2(self):
        jobs = [
            job("c", "naive", baseline=classification_baseline(0.5, 0.6)),
            job("r", "naive", task_type="regression", baseline=regression_baseline(0.4, 0.8)),
        ]
        md = render_summary(results(jobs, models=(), datasets=("c", "r")))
        assert [r[0] for r in table(md, "### r2 ↑")[1:]] == ["r"]
        assert [r[0] for r in table(md, "### f1_macro ↑")[1:]] == ["c"]
        assert "### roc_auc" not in md

    def test_failed_and_errored_jobs_are_named_instead_of_scored(self):
        jobs = [
            job("d1", "reference", status="failed", error_codes=["unknown_column"]),
            job("d1", "naive", status="error", error="boom"),
            job("d1", "agent", baseline=classification_baseline(0.6, 0.7)),
        ]
        cells = row(table(render_summary(results(jobs)), "### f1_macro ↑"), "d1")
        assert cells[1:] == ["failed", "error", "0.600 / 0.700"]

    def test_repeats_are_averaged_and_partial_success_is_marked(self):
        jobs = [
            job("d1", "agent", repeat=1, baseline=classification_baseline(0.6, 0.7)),
            job("d1", "agent", repeat=2, baseline=classification_baseline(0.8, 0.9)),
            job("d1", "agent", repeat=3, status="failed"),
        ]
        md = render_summary(results(jobs, repeats=3))
        assert row(table(md, "### f1_macro ↑"), "d1")[-1] == "0.700 / 0.800 (2/3 ok)"

    def test_a_skipped_baseline_leaves_no_score_table(self):
        skipped = BaselineResult(
            task_type="classification",
            n_rows=3,
            random_state=42,
            n_folds=None,
            metrics=[],
            models=[],
            skipped_reason="too small",
        )
        md = render_summary(results([job("d1", "naive", baseline=skipped)], models=()))
        assert "### f1_macro" not in md
        assert "No plan has scores yet." in md

    def test_methods_follow_the_models_of_the_run(self):
        jobs = [
            job("d1", "agent", model="b", baseline=classification_baseline(0.6, 0.7)),
            job("d1", "agent", model="a", baseline=classification_baseline(0.6, 0.7)),
        ]
        header = table(render_summary(results(jobs, models=("b", "a"))), "### f1_macro ↑")[0]
        assert header == ["dataset", "reference", "naive", "agent: b", "agent: a"]


TALLY = "better / worse / within noise"


class TestLeakyPlans:
    def test_a_score_of_a_plan_that_kept_a_leaky_column_is_marked(self):
        jobs = [
            job("d1", "reference", baseline=classification_baseline(0.6, 0.7)),
            job(
                "d1",
                "naive",
                baseline=classification_baseline(0.9, 0.95),
                kept_leaky_columns=["boat"],
            ),
        ]
        md = render_summary(results(jobs, models=()))
        cells = row(table(md, "### f1_macro ↑"), "d1")
        assert cells[1:3] == ["0.600 / 0.700", "0.900 / 0.950 †"]
        assert "† the plan kept a column that gives the target away" in md

    def test_the_footnote_is_left_out_when_no_plan_leaks(self):
        jobs = [job("d1", "naive", baseline=classification_baseline(0.6, 0.7))]
        assert "†" not in render_summary(results(jobs, models=()))

    def test_one_leaking_repeat_marks_the_cell(self):
        jobs = [
            job("d1", "agent", repeat=1, baseline=classification_baseline(0.6, 0.7)),
            job(
                "d1",
                "agent",
                repeat=2,
                baseline=classification_baseline(0.6, 0.7),
                kept_leaky_columns=["boat"],
            ),
        ]
        cell = row(table(render_summary(results(jobs, repeats=2)), "### f1_macro ↑"), "d1")[-1]
        assert cell.endswith(" †")

    def test_a_comparison_with_a_leaking_plan_is_marked_and_not_counted(self):
        jobs = [
            job(
                "leaky",
                "naive",
                baseline=classification_baseline(0.5, 0.95),
                kept_leaky_columns=["boat"],
            ),
            job("leaky", "agent", baseline=classification_baseline(0.5, 0.65)),
            job("clean", "naive", baseline=classification_baseline(0.5, 0.60)),
            job("clean", "agent", baseline=classification_baseline(0.5, 0.65)),
        ]
        md = render_summary(results(jobs, datasets=("leaky", "clean")))
        rows = table(md, "## Agent against the naive plan")
        assert row(rows, "leaky")[2] == "-0.300 †"
        assert row(rows, "clean")[2] == "+0.050"
        assert row(rows, TALLY)[2] == "1 / 0 / 0 (1 excluded †)"


class TestLeakyColumnsKept:
    HEADING = "## Leaky columns kept"

    def leaky_job(self, method, kept, dataset="d1", **kwargs):
        return job(
            dataset,
            method,
            leaky_columns=["boat", "body"],
            kept_leaky_columns=kept,
            leaky_columns_kept=len(kept),
            baseline=classification_baseline(0.6, 0.7),
            **kwargs,
        )

    def test_counts_the_kept_columns_of_every_method(self):
        jobs = [
            self.leaky_job("reference", []),
            self.leaky_job("naive", ["boat", "body"]),
            self.leaky_job("agent", ["body"]),
        ]

        rows = table(render_summary(results(jobs)), self.HEADING)

        assert rows[0] == ["dataset", "leaky columns", "reference", "naive", "agent: m1"]
        assert row(rows, "d1") == ["d1", "boat, body", "0 of 2", "2 of 2", "1 of 2"]

    def test_repeats_of_an_agent_are_added_up(self):
        jobs = [
            self.leaky_job("agent", ["body"], repeat=1),
            self.leaky_job("agent", [], repeat=2),
            self.leaky_job("agent", ["boat", "body"], repeat=3),
        ]

        cell = row(table(render_summary(results(jobs, repeats=3)), self.HEADING), "d1")[-1]

        assert cell == "3 of 6 (3 runs)"

    def test_a_method_without_a_plan_shows_a_dash(self):
        jobs = [
            self.leaky_job("reference", []),
            job("d1", "naive", status="error", leaky_columns=["boat", "body"], error="boom"),
        ]

        cells = row(table(render_summary(results(jobs, models=())), self.HEADING), "d1")

        assert cells[2:] == ["0 of 2", "–"]

    def test_only_datasets_with_leaky_columns_are_listed(self):
        jobs = [
            self.leaky_job("reference", []),
            job("d2", "reference", baseline=classification_baseline(0.6, 0.7)),
        ]

        rows = table(render_summary(results(jobs, models=(), datasets=("d1", "d2"))), self.HEADING)

        assert [r[0] for r in rows[1:]] == ["d1"]

    def test_the_section_is_left_out_when_no_dataset_has_leaky_columns(self):
        jobs = [job("d1", "reference", baseline=classification_baseline(0.6, 0.7))]

        assert self.HEADING not in render_summary(results(jobs, models=()))


class TestAgainstNaive:
    HEADING = "## Agent against the naive plan"

    def test_shows_the_difference_in_the_forest_score_of_the_main_metric(self):
        jobs = [
            job("d1", "naive", baseline=classification_baseline(0.5, 0.60)),
            job("d1", "agent", baseline=classification_baseline(0.5, 0.65)),
            job(
                "r",
                "naive",
                task_type="regression",
                baseline=regression_baseline(0.5, 0.80),
            ),
            job(
                "r",
                "agent",
                task_type="regression",
                baseline=regression_baseline(0.5, 0.75),
            ),
        ]
        rows = table(render_summary(results(jobs, datasets=("d1", "r"))), self.HEADING)
        assert rows[0] == ["dataset", "metric", "agent: m1"]
        assert row(rows, "d1") == ["d1", "f1_macro", "+0.050"]
        assert row(rows, "r") == ["r", "r2", "-0.050"]
        assert row(rows, TALLY)[2] == "1 / 1 / 0"

    def test_a_difference_below_the_larger_fold_spread_is_marked_and_not_counted(self):
        jobs = [
            job("small", "naive", baseline=classification_baseline(0.5, 0.600, std=0.01)),
            job("small", "agent", baseline=classification_baseline(0.5, 0.605, std=0.02)),
            job("equal", "naive", baseline=classification_baseline(0.5, 0.600, std=0.0)),
            job("equal", "agent", baseline=classification_baseline(0.5, 0.600, std=0.0)),
            job("up", "naive", baseline=classification_baseline(0.5, 0.60, std=0.01)),
            job("up", "agent", baseline=classification_baseline(0.5, 0.65, std=0.02)),
            job("down", "naive", baseline=classification_baseline(0.5, 0.65, std=0.02)),
            job("down", "agent", baseline=classification_baseline(0.5, 0.60, std=0.01)),
        ]
        md = render_summary(results(jobs, datasets=("small", "equal", "up", "down")))
        rows = table(md, self.HEADING)
        assert row(rows, "small")[2] == "+0.005 ≈"
        assert row(rows, "equal")[2] == "+0.000 ≈"
        assert row(rows, "up")[2] == "+0.050"
        assert row(rows, "down")[2] == "-0.050"
        assert row(rows, TALLY)[2] == "1 / 1 / 2"

    def test_the_larger_spread_of_the_two_plans_decides(self):
        jobs = [
            job("d1", "naive", baseline=classification_baseline(0.5, 0.60, std=0.001)),
            job("d1", "agent", baseline=classification_baseline(0.5, 0.65, std=0.06)),
        ]
        assert row(table(render_summary(results(jobs)), self.HEADING), "d1")[2] == "+0.050 ≈"

    def test_a_leaking_comparison_is_excluded_even_when_within_noise(self):
        jobs = [
            job(
                "d1",
                "naive",
                baseline=classification_baseline(0.5, 0.60),
                kept_leaky_columns=["boat"],
            ),
            job("d1", "agent", baseline=classification_baseline(0.5, 0.601)),
        ]
        rows = table(render_summary(results(jobs)), self.HEADING)
        assert row(rows, "d1")[2] == "+0.001 †"
        assert row(rows, TALLY)[2] == "0 / 0 / 0 (1 excluded †)"

    def test_the_spread_of_a_repeated_agent_is_the_mean_of_its_fold_spreads(self):
        jobs = [
            job("d1", "naive", baseline=classification_baseline(0.5, 0.60, std=0.01)),
            job("d1", "agent", repeat=1, baseline=classification_baseline(0.5, 0.64, std=0.02)),
            job("d1", "agent", repeat=2, baseline=classification_baseline(0.5, 0.66, std=0.06)),
        ]
        cell = row(table(render_summary(results(jobs, repeats=2)), self.HEADING), "d1")[2]
        assert cell == "+0.050"

    def test_a_dash_stands_in_when_either_side_has_no_score(self):
        jobs = [
            job("d1", "naive", status="failed"),
            job("d1", "agent", baseline=classification_baseline(0.5, 0.65)),
        ]
        assert row(table(render_summary(results(jobs)), self.HEADING), "d1")[2] == "–"

    def test_is_left_out_without_agent_runs(self):
        jobs = [job("d1", "naive", baseline=classification_baseline(0.5, 0.6))]
        assert self.HEADING not in render_summary(results(jobs, models=()))


class TestAgainstReference:
    HEADING = "## Agent against the reference plan"

    def test_compares_with_the_reference_plan_in_the_same_way(self):
        jobs = [
            job("up", "reference", baseline=classification_baseline(0.5, 0.60)),
            job("up", "agent", baseline=classification_baseline(0.5, 0.65)),
            job("down", "reference", baseline=classification_baseline(0.5, 0.70)),
            job("down", "agent", baseline=classification_baseline(0.5, 0.65)),
            job("tie", "reference", baseline=classification_baseline(0.5, 0.650)),
            job("tie", "agent", baseline=classification_baseline(0.5, 0.655)),
        ]
        rows = table(render_summary(results(jobs, datasets=("up", "down", "tie"))), self.HEADING)
        assert rows[0] == ["dataset", "metric", "agent: m1"]
        assert row(rows, "up") == ["up", "f1_macro", "+0.050"]
        assert row(rows, "down")[2] == "-0.050"
        assert row(rows, "tie")[2] == "+0.005 ≈"
        assert row(rows, TALLY)[2] == "1 / 1 / 1"

    def test_is_independent_of_the_naive_plan(self):
        jobs = [
            job("d1", "reference", baseline=classification_baseline(0.5, 0.60)),
            job("d1", "naive", baseline=classification_baseline(0.5, 0.90)),
            job("d1", "agent", baseline=classification_baseline(0.5, 0.65)),
        ]
        md = render_summary(results(jobs))
        assert row(table(md, self.HEADING), "d1")[2] == "+0.050"
        assert row(table(md, "## Agent against the naive plan"), "d1")[2] == "-0.250"

    def test_a_leaking_reference_is_marked_and_not_counted(self):
        jobs = [
            job(
                "d1",
                "reference",
                baseline=classification_baseline(0.5, 0.95),
                kept_leaky_columns=["boat"],
            ),
            job("d1", "agent", baseline=classification_baseline(0.5, 0.65)),
        ]
        rows = table(render_summary(results(jobs)), self.HEADING)
        assert row(rows, "d1")[2] == "-0.300 †"
        assert row(rows, TALLY)[2] == "0 / 0 / 0 (1 excluded †)"

    def test_a_dash_stands_in_when_the_reference_has_no_score(self):
        jobs = [
            job("d1", "reference", status="failed"),
            job("d1", "agent", baseline=classification_baseline(0.5, 0.65)),
        ]
        assert row(table(render_summary(results(jobs)), self.HEADING), "d1")[2] == "–"

    def test_is_left_out_without_agent_runs(self):
        jobs = [job("d1", "reference", baseline=classification_baseline(0.5, 0.6))]
        assert self.HEADING not in render_summary(results(jobs, models=()))


class TestRepeats:
    HEADING = "## Agent repeats"

    def repeated(self, scores, dataset="d1", model="m1", **kwargs):
        return [
            job(
                dataset,
                "agent",
                model=model,
                repeat=i,
                baseline=classification_baseline(0.5, forest),
                **kwargs,
            )
            for i, forest in enumerate(scores, start=1)
        ]

    def test_gives_success_rate_and_the_spread_of_the_score_across_runs(self):
        jobs = [
            *self.repeated([0.60, 0.70, 0.80]),
            job("d1", "agent", repeat=4, status="failed"),
        ]
        rows = table(render_summary(results(jobs, repeats=4)), self.HEADING)
        assert rows[0] == [
            "dataset",
            "model",
            "successful runs",
            "leaky columns caught",
            "score std across runs",
        ]
        assert row(rows, "d1") == ["d1", "m1", "3 of 4 (75%)", "–", "0.100"]

    def test_counts_the_caught_leaky_columns_of_every_run_separately(self):
        def leaky(repeat, kept):
            return job(
                "d1",
                "agent",
                repeat=repeat,
                leaky_columns=["boat", "body"],
                kept_leaky_columns=kept,
                leaky_columns_kept=len(kept),
                baseline=classification_baseline(0.5, 0.7),
            )

        jobs = [leaky(1, []), leaky(2, ["body"]), leaky(3, ["boat", "body"])]
        rows = table(render_summary(results(jobs, repeats=3)), self.HEADING)
        assert row(rows, "d1")[3] == "3 of 6 (50%)"

    def test_a_run_without_a_plan_is_left_out_of_the_leaky_count(self):
        jobs = [
            job(
                "d1",
                "agent",
                repeat=1,
                leaky_columns=["boat"],
                leaky_columns_kept=0,
                baseline=classification_baseline(0.5, 0.7),
            ),
            job("d1", "agent", repeat=2, status="failed", leaky_columns=["boat"]),
        ]
        rows = table(render_summary(results(jobs, repeats=2)), self.HEADING)
        assert row(rows, "d1")[3] == "1 of 1 (100%)"

    def test_one_row_per_dataset_and_model(self):
        jobs = [
            *self.repeated([0.6, 0.7], dataset="d1", model="a"),
            *self.repeated([0.6, 0.6], dataset="d1", model="b"),
            *self.repeated([0.6, 0.8], dataset="d2", model="a"),
        ]
        md = render_summary(results(jobs, models=("a", "b"), repeats=2, datasets=("d1", "d2")))
        rows = table(md, self.HEADING)[1:]
        assert [(r[0], r[1], r[4]) for r in rows] == [
            ("d1", "a", "0.071"),
            ("d1", "b", "0.000"),
            ("d2", "a", "0.141"),
        ]

    def test_the_spread_needs_two_scored_runs(self):
        jobs = [
            *self.repeated([0.7]),
            job("d1", "agent", repeat=2, status="failed"),
        ]
        rows = table(render_summary(results(jobs, repeats=2)), self.HEADING)
        assert row(rows, "d1")[4] == "n/a"

    def test_the_spread_is_taken_on_the_main_metric_of_the_task(self):
        jobs = [
            job(
                "r",
                "agent",
                task_type="regression",
                repeat=i,
                baseline=regression_baseline(0.5, r2),
            )
            for i, r2 in enumerate([0.2, 0.4], start=1)
        ]
        rows = table(render_summary(results(jobs, repeats=2, datasets=("r",))), self.HEADING)
        assert row(rows, "r")[4] == "0.141"

    def test_is_left_out_without_repeats(self):
        md = render_summary(results(self.repeated([0.6]), repeats=1))
        assert self.HEADING not in md

    def test_is_left_out_without_agent_runs(self):
        md = render_summary(results([job("d1", "naive")], models=(), repeats=3))
        assert self.HEADING not in md


def plan(drop=(), encode=None, cast=()) -> Plan:
    steps = [{"op": "drop_columns", "columns": [c], "reason": "r"} for c in drop]
    steps += [{"op": "cast_to_category", "columns": [c], "reason": "r"} for c in cast]
    steps.append(
        {"op": "split", "test_size": 0.2, "stratify": True, "random_state": 1, "reason": "r"}
    )
    for column, how in (encode or {}).items():
        method, max_categories = how if isinstance(how, tuple) else (how, None)
        step = {"op": "encode", "columns": [column], "method": method, "reason": "r"}
        steps.append(step | ({"max_categories": max_categories} if max_categories else {}))
    return Plan.model_validate(
        {"target": "y", "task_type": "classification", "model_family": "any", "steps": steps}
    )


class TestPlanDifferences:
    HEADING = "## Plan differences"
    HEADER = [
        "dataset",
        "agent run",
        "dropped by the reference only",
        "dropped by the agent only",
        "encoded or cast differently (reference vs agent)",
    ]

    def pair(self, reference: Plan, agent: Plan, dataset="d1", **kwargs):
        return [
            job(dataset, "reference", plan=reference),
            job(dataset, "agent", plan=agent, **kwargs),
        ]

    def test_lists_the_columns_each_side_drops_alone(self):
        jobs = self.pair(
            plan(drop=["id", "name", "boat"]),
            plan(drop=["id", "ticket", "cabin"]),
        )
        rows = table(render_summary(results(jobs)), self.HEADING)
        assert rows[0] == self.HEADER
        assert row(rows, "d1") == ["d1", "m1", "boat, name", "cabin, ticket", "–"]

    def test_lists_the_columns_encoded_or_cast_differently(self):
        jobs = self.pair(
            plan(encode={"city": "onehot", "grade": "ordinal", "sex": "onehot"}, cast=["pclass"]),
            plan(
                encode={"city": "ordinal", "grade": "ordinal", "sex": "onehot", "pclass": "onehot"}
            ),
        )
        cells = row(table(render_summary(results(jobs)), self.HEADING), "d1")
        assert cells[2:] == ["–", "–", "city (onehot vs ordinal), pclass (cast vs onehot)"]

    def test_a_column_only_one_side_encodes_counts_as_different(self):
        jobs = self.pair(plan(encode={"city": "onehot"}), plan())
        cells = row(table(render_summary(results(jobs)), self.HEADING), "d1")
        assert cells[4] == "city (onehot vs none)"

    def test_a_different_category_cap_counts_and_is_shown_with_the_method(self):
        jobs = self.pair(
            plan(encode={"city": ("onehot", 100), "job": ("onehot", 20), "sex": ("onehot", 10)}),
            plan(encode={"city": "onehot", "job": ("onehot", 50), "sex": ("onehot", 10)}),
        )
        cells = row(table(render_summary(results(jobs)), self.HEADING), "d1")
        assert cells[4] == "city (onehot(100) vs onehot), job (onehot(20) vs onehot(50))"

    def test_a_column_cast_and_encoded_shows_both_treatments(self):
        jobs = self.pair(
            plan(encode={"city": "onehot"}, cast=["city"]), plan(encode={"city": "ordinal"})
        )
        cells = row(table(render_summary(results(jobs)), self.HEADING), "d1")
        assert cells[4] == "city (cast + onehot vs ordinal)"

    def test_a_dropped_column_is_not_also_listed_as_treated_differently(self):
        jobs = self.pair(plan(encode={"city": "onehot"}), plan(drop=["city"]))
        cells = row(table(render_summary(results(jobs)), self.HEADING), "d1")
        assert cells[2:] == ["–", "city", "–"]

    def test_identical_handling_gives_dashes(self):
        same = plan(drop=["id"], encode={"city": "onehot"}, cast=["pclass"])
        jobs = self.pair(same, same)
        assert row(table(render_summary(results(jobs)), self.HEADING), "d1")[2:] == ["–"] * 3

    def test_one_row_per_agent_run_numbered_when_there_are_repeats(self):
        jobs = [
            job("d1", "reference", plan=plan(drop=["a"])),
            job("d1", "agent", model="m1", repeat=1, plan=plan()),
            job("d1", "agent", model="m1", repeat=2, plan=plan(drop=["a"])),
        ]
        rows = table(render_summary(results(jobs, repeats=2)), self.HEADING)[1:]
        assert [(r[0], r[1], r[2]) for r in rows] == [("d1", "m1 #1", "a"), ("d1", "m1 #2", "–")]

    def test_runs_without_a_plan_or_without_a_reference_are_left_out(self):
        jobs = [
            job("d1", "reference", plan=plan()),
            job("d1", "agent", repeat=1, status="failed", plan=plan(drop=["a"])),
            job("d2", "agent", plan=plan(drop=["a"])),
            job("d3", "reference", status="failed"),
            job("d3", "agent", plan=plan()),
        ]
        md = render_summary(results(jobs, datasets=("d1", "d2", "d3")))
        assert self.HEADING not in md

    def test_is_left_out_without_agent_runs(self):
        jobs = [job("d1", "reference", plan=plan())]
        assert self.HEADING not in render_summary(results(jobs, models=()))


def with_prices(res: BenchmarkResults, **prices: ModelPrice) -> BenchmarkResults:
    config = res.config.model_copy(update={"prices": prices})
    return res.model_copy(update={"config": config})


class TestCachePricingNote:
    PLAIN = ModelPrice(input_per_million=2.0, output_per_million=8.0)
    CACHED = ModelPrice(
        input_per_million=2.0,
        output_per_million=8.0,
        cached_input_per_million=0.5,
        cache_write_per_million=2.5,
    )

    def notes(self, **prices):
        res = results([job("d1", "agent", agent=agent_metrics())], models=tuple(prices))
        return render_summary(with_prices(res, **prices)).split("## Notes", 1)[1]

    def test_a_model_without_cache_prices_is_said_to_use_the_old_formula(self):
        notes = self.notes(m1=self.PLAIN)
        assert "m1: no cache prices in prices.yaml" in notes
        assert "counted at the input price" in notes

    def test_a_model_with_cache_prices_is_said_to_use_them(self):
        notes = self.notes(m1=self.CACHED)
        assert "m1: cached input and cache writes are priced separately" in notes

    def test_each_model_gets_its_own_statement(self):
        notes = self.notes(m1=self.CACHED, m2=self.PLAIN)
        assert "m1: cached input" in notes
        assert "m2: no cache prices" in notes

    def test_a_model_with_only_a_cached_input_price_is_described_exactly(self):
        half = ModelPrice(
            input_per_million=2.0, output_per_million=8.0, cached_input_per_million=0.5
        )
        notes = self.notes(m1=half)
        assert "m1: cached input is priced separately; cache writes are counted" in notes

    def test_a_run_without_agent_jobs_has_no_cache_statement(self):
        res = results([job("d1", "naive", baseline=classification_baseline(0.5, 0.6))], models=())
        assert "cache" not in render_summary(res).split("## Notes", 1)[1]


class TestAgentRuns:
    HEADING = "## Agent runs"

    def test_summarises_success_revisions_questions_tokens_cost_and_time(self):
        jobs = [
            job(
                "d1",
                "agent",
                repeat=1,
                agent=agent_metrics(revisions=2, questions=1, cost=0.02),
                plan_seconds=10.0,
                baseline_seconds=2.0,
            ),
            job(
                "d2",
                "agent",
                repeat=1,
                agent=agent_metrics(
                    revisions=0, questions=0, input_tokens=3000, output_tokens=400, cost=0.04
                ),
                plan_seconds=20.0,
                baseline_seconds=4.0,
            ),
            job("d3", "agent", status="failed", agent=agent_metrics(revisions=2, cost=0.03)),
        ]
        rows = table(render_summary(results(jobs, datasets=("d1", "d2", "d3"))), self.HEADING)
        header = rows[0]
        values = dict(zip(header, row(rows, "m1"), strict=True))
        assert values["runs"] == "3"
        assert values["ok"] == "2 (67%)"
        assert values["revisions"] == "1.3"
        assert values["questions"] == "0.3"
        assert values["input tokens"] == "1667"
        assert values["output tokens"] == "267"
        assert values["cost (EUR)"] == "0.0900"
        assert values["plan seconds"] == "15.0"

    def test_cost_is_marked_incomplete_when_some_runs_reported_no_usage(self):
        jobs = [
            job("d1", "agent", agent=agent_metrics(cost=0.02)),
            job("d1", "agent", repeat=2, agent=agent_metrics(cost=None, input_tokens=None)),
        ]
        rows = table(render_summary(results(jobs, repeats=2)), self.HEADING)
        assert (
            dict(zip(rows[0], row(rows, "m1"), strict=True))["cost (EUR)"] == "0.0200+ (1 unknown)"
        )

    def test_cost_and_tokens_are_not_available_without_any_usage(self):
        jobs = [
            job(
                "d1", "agent", agent=agent_metrics(cost=None, input_tokens=None, output_tokens=None)
            )
        ]
        rows = table(render_summary(results(jobs)), self.HEADING)
        values = dict(zip(rows[0], row(rows, "m1"), strict=True))
        assert (values["cost (EUR)"], values["input tokens"]) == ("n/a", "n/a")

    def test_errored_runs_count_as_runs_but_not_as_successes(self):
        jobs = [job("d1", "agent", status="error", error="boom")]
        rows = table(render_summary(results(jobs)), self.HEADING)
        values = dict(zip(rows[0], row(rows, "m1"), strict=True))
        assert (values["runs"], values["ok"]) == ("1", "0 (0%)")

    def test_says_so_without_agent_runs(self):
        md = render_summary(results([job("d1", "naive")], models=()))
        assert "No agent runs." in md


class TestIssueCodes:
    def test_counts_the_runs_a_code_appeared_in_most_frequent_first(self):
        jobs = [
            job(
                "d1",
                "agent",
                repeat=i,
                agent=agent_metrics(first_plan_error_codes=codes),
                warning_codes=warnings,
            )
            for i, (codes, warnings) in enumerate(
                [
                    (
                        ["unknown_column", "unknown_column", "stateless_after_split"],
                        ["scale_for_tree"],
                    ),
                    (["unknown_column"], []),
                    ([], ["scale_for_tree", "repeated_op"]),
                ],
                start=1,
            )
        ]
        md = render_summary(results(jobs, repeats=3))
        errors = table(md, "### First-plan errors of agent runs")
        assert errors[0] == ["code", "runs", "share of runs"]
        assert errors[1:] == [
            ["unknown_column", "2", "67%"],
            ["stateless_after_split", "1", "33%"],
        ]
        warnings = table(md, "### Final-plan warnings of agent runs")
        assert warnings[1:] == [["scale_for_tree", "2", "67%"], ["repeated_op", "1", "33%"]]

    def test_says_none_when_there_is_nothing_to_count(self):
        md = render_summary(results([job("d1", "agent", agent=agent_metrics())]))
        assert "No first-plan errors." in md
        assert "No final-plan warnings." in md

    def test_lists_the_warnings_of_reference_and_naive_plans_per_dataset(self):
        jobs = [
            job("d1", "reference", warning_codes=["repeated_op"]),
            job("d1", "naive", warning_codes=[]),
            job("d2", "naive", warning_codes=["impute_without_missing", "repeated_op"]),
        ]
        md = render_summary(results(jobs, models=(), datasets=("d1", "d2")))
        rows = table(md, "### Warnings of reference and naive plans")
        assert rows[0] == ["dataset", "reference", "naive"]
        assert row(rows, "d1") == ["d1", "repeated_op", "–"]
        assert row(rows, "d2") == ["d2", "–", "impute_without_missing, repeated_op"]


class TestFailures:
    def test_lists_every_job_that_is_not_ok_with_its_reason(self):
        jobs = [
            job(
                "d1",
                "agent",
                status="failed",
                error="out of revisions",
                error_codes=["unknown_column"],
            ),
            job("d2", "reference", status="error", error="plan file | not found\nreally"),
            job("d1", "naive"),
        ]
        rows = table(render_summary(results(jobs, datasets=("d1", "d2"))), "## Failures")
        assert rows[0] == ["dataset", "method", "run", "status", "reason"]
        assert len(rows) == 3
        assert row(rows, "d1")[1:4] == ["agent: m1", "1", "failed"]
        assert "out of revisions" in row(rows, "d1")[4]
        assert "unknown_column" in row(rows, "d1")[4]
        reason = row(rows, "d2")[4]
        assert "\n" not in reason
        assert "plan file \\| not found really" in reason

    def test_says_so_when_nothing_failed(self):
        assert "No failures." in render_summary(results([job("d1", "naive")], models=()))


class TestHeaderAndNotes:
    def test_names_the_run_and_the_ground_rules(self):
        md = render_summary(results([job("d1", "naive")], models=("m1",), repeats=3))
        assert md.startswith("# Benchmark summary")
        assert "m1" in md
        assert "3" in md
        assert "5-fold" in md
        assert "reference/README.md" in md
        assert "test set" in md

    def test_does_not_call_the_reviewed_reference_plans_drafts(self):
        md = render_summary(results([job("d1", "naive")], models=()))
        assert "draft" not in md.lower()
        assert "reference/README.md" in md

    def test_is_deterministic(self):
        data = results([job("d1", "naive", baseline=classification_baseline(0.5, 0.6))], models=())
        assert render_summary(data) == render_summary(data)
