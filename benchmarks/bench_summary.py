"""summary.md: the tables of a benchmark run, rendered from results.json alone.

Cells that compare plans show two numbers, the linear baseline model and the random forest,
because the plans are meant to be judged by what a model can do with the prepared data and no
single model is the right one. The difference to the naive plan uses the forest, the model
that is least sensitive to scaling and encoding choices.
"""

from collections import Counter
from statistics import fmean, stdev

from bench_results import BenchmarkResults, JobResult
from prep_agent.baseline import BaselineResult, BaselineScore, metric_label
from prep_agent.plan import CastToCategoryStep, DropColumnsStep, EncodeStep, Plan

METRICS = ("f1_macro", "roc_auc", "r2")
MAIN_METRIC = {"classification": "f1_macro", "regression": "r2"}
LINEAR_MODELS = ("logistic_regression", "ridge")
FOREST = "random_forest"
DASH = "–"
LEAK_MARK = "†"
NO_TREATMENT = "none"
NOISE_MARK = "≈"
TALLY_LABEL = "better / worse / within noise"
_MAX_REASON = 200


def render_summary(results: BenchmarkResults) -> str:
    agent_jobs = [job for job in results.jobs if job.method == "agent"]
    sections = [_header(results), _scores(results), _leaky_columns_kept(results)]
    if agent_jobs:
        sections.append(_against(results, "naive", "the naive plan"))
        sections.append(_against(results, "reference", "the reference plan"))
    if agent_jobs:
        sections.append(_plan_differences(results))
    if agent_jobs and results.config.repeats > 1:
        sections.append(_repeats(results))
    sections.append(_agent_runs(results, agent_jobs))
    sections.append(_issue_codes(results, agent_jobs))
    sections.append(_failures(results))
    sections.append(_notes(results, bool(agent_jobs)))
    return "\n\n".join(section for section in sections if section) + "\n"


def _header(results: BenchmarkResults) -> str:
    config = results.config
    started = results.started_at.strftime("%Y-%m-%d %H:%M UTC")
    finished = results.finished_at.strftime("%Y-%m-%d %H:%M UTC") if results.finished_at else None
    lines = [
        "# Benchmark summary",
        "",
        f"- Started: {started}" + (f", finished: {finished}" if finished else " (not finished)"),
        f"- Datasets: {', '.join(config.datasets)}",
        "- Models: " + (", ".join(config.models) if config.models else "none (no agent runs)"),
        f"- Repeats: {config.repeats} per agent run; the reference and naive plans are "
        "deterministic and run once",
    ]
    estimates = {m: c for m, c in config.estimated_max_cost.items() if c is not None}
    if estimates:
        approved = ", ".join(f"{m} {c:.2f}" for m, c in estimates.items())
        lines.append(f"- Estimated maximum cost approved ({config.currency}): {approved}")
    return "\n".join(lines)


# --- scores ---


def _scores(results: BenchmarkResults) -> str:
    labels = ["reference", "naive", *(f"agent: {model}" for model in results.config.models)]
    parts = [
        "## Scores",
        "",
        "Each cell is the linear model / the random forest, as the mean over the "
        "cross-validation folds (and over the successful repeats of an agent run).",
    ]
    intro = len(parts)
    marked = False
    for metric in METRICS:
        datasets = [
            name
            for name in results.config.datasets
            if any(_has_metric(job, metric) for job in _of(results, name))
        ]
        if not datasets:
            continue
        rows = [
            [name, *(_score_cell(_of(results, name, label), metric) for label in labels)]
            for name in datasets
        ]
        marked = marked or any(cell.endswith(LEAK_MARK) for row in rows for cell in row)
        parts += ["", f"### {metric_label(metric)}", "", _table(["dataset", *labels], rows)]
    if len(parts) == intro:
        parts += ["", "No plan has scores yet."]
    if marked:
        parts += [
            "",
            f"{LEAK_MARK} the plan kept a column that gives the target away (`leaky_columns` in "
            "datasets.yaml). Its score is inflated by leakage and cannot be compared with the "
            "others.",
        ]
    return "\n".join(parts)


def _score_cell(jobs: list[JobResult], metric: str) -> str:
    if not jobs:
        return DASH
    scored = [job for job in jobs if _has_metric(job, metric)]
    if not scored:
        if all(job.status == "error" for job in jobs):
            return "error"
        return "failed" if any(job.status != "ok" for job in jobs) else DASH
    linear = fmean(_score(job.baseline, LINEAR_MODELS, metric) for job in scored)
    forest = fmean(_score(job.baseline, (FOREST,), metric) for job in scored)
    cell = f"{linear:.3f} / {forest:.3f}"
    if len(scored) < len(jobs):
        cell += f" ({len(scored)}/{len(jobs)} ok)"
    if _leaks(scored):
        cell += f" {LEAK_MARK}"
    return cell


def _has_metric(job: JobResult, metric: str) -> bool:
    return (
        job.status == "ok"
        and job.baseline is not None
        and job.baseline.skipped_reason is None
        and metric in job.baseline.metrics
        and _score(job.baseline, (FOREST,), metric) is not None
        and _score(job.baseline, LINEAR_MODELS, metric) is not None
    )


def _score(baseline: BaselineResult | None, names: tuple[str, ...], metric: str) -> float | None:
    fold_score = _fold_score(baseline, names, metric)
    return fold_score.mean if fold_score else None


def _fold_score(
    baseline: BaselineResult | None, names: tuple[str, ...], metric: str
) -> BaselineScore | None:
    if baseline is None:
        return None
    for model in baseline.models:
        if model.name in names and metric in model.scores:
            return model.scores[metric]
    return None


# --- leaky columns kept ---


def _leaky_columns_kept(results: BenchmarkResults) -> str:
    labels = ["reference", "naive", *(f"agent: {model}" for model in results.config.models)]
    rows = []
    for name in results.config.datasets:
        leaky = next((job.leaky_columns for job in _of(results, name) if job.leaky_columns), [])
        if leaky:
            cells = [_kept_cell(_of(results, name, label), len(leaky)) for label in labels]
            rows.append([name, ", ".join(leaky), *cells])
    if not rows:
        return ""
    return "\n".join(
        [
            "## Leaky columns kept",
            "",
            "How many of the columns that give the target away (`leaky_columns` in "
            "datasets.yaml) the plan still has at its end. An agent cell adds up its runs.",
            "",
            _table(["dataset", "leaky columns", *labels], rows),
        ]
    )


def _kept_cell(jobs: list[JobResult], n_leaky: int) -> str:
    counted = [job.leaky_columns_kept for job in jobs if job.leaky_columns_kept is not None]
    if not counted:
        return DASH
    cell = f"{sum(counted)} of {n_leaky * len(counted)}"
    return cell + f" ({len(counted)} runs)" if len(counted) > 1 else cell


# --- agent against the naive and the reference plan ---


def _against(results: BenchmarkResults, other: str, name: str) -> str:
    agents = [f"agent: {model}" for model in results.config.models]
    task_types = {job.dataset: job.task_type for job in results.jobs}
    rows = []
    tally = {label: Counter() for label in agents}
    for dataset in results.config.datasets:
        if dataset not in task_types:
            continue
        metric = MAIN_METRIC[task_types[dataset]]
        other_jobs = _of(results, dataset, other)
        cells = []
        for label in agents:
            agent_jobs = _of(results, dataset, label)
            verdict, cell = _compare(agent_jobs, other_jobs, metric)
            tally[label][verdict] += 1
            cells.append(cell)
        rows.append([dataset, metric, *cells])
    rows.append([TALLY_LABEL, "", *(_tally(tally[label]) for label in agents)])
    return "\n".join(
        [
            f"## Agent against {name}",
            "",
            "Difference in the random forest's score on the main metric (macro F1 for "
            "classification, R² for regression): positive means the agent's plan scored higher. "
            f"{NOISE_MARK} marks a difference smaller than the larger of the two plans' "
            "fold standard deviations (for an agent with repeats, the mean over its runs); it is "
            "not counted as better or worse.",
            "",
            _table(["dataset", "metric", *agents], rows),
        ]
    )


def _compare(
    agent_jobs: list[JobResult], other_jobs: list[JobResult], metric: str
) -> tuple[str | None, str]:
    """The verdict (better, worse, noise, excluded; None without a score) and the cell."""
    agent, other = _forest_mean(agent_jobs, metric), _forest_mean(other_jobs, metric)
    if agent is None or other is None:
        return None, DASH
    diff = agent - other
    if _leaks(agent_jobs) or _leaks(other_jobs):
        return "excluded", f"{diff:+.3f} {LEAK_MARK}"
    spread = max(_forest_spread(agent_jobs, metric), _forest_spread(other_jobs, metric))
    if abs(diff) < spread or diff == 0:
        return "noise", f"{diff:+.3f} {NOISE_MARK}"
    return ("better" if diff > 0 else "worse"), f"{diff:+.3f}"


def _tally(counts: Counter) -> str:
    text = f"{counts['better']} / {counts['worse']} / {counts['noise']}"
    if counts["excluded"]:
        text += f" ({counts['excluded']} excluded {LEAK_MARK})"
    return text


def _leaks(jobs: list[JobResult]) -> bool:
    return any(job.kept_leaky_columns for job in jobs if job.status == "ok")


def _forest_mean(jobs: list[JobResult], metric: str) -> float | None:
    scored = [_score(job.baseline, (FOREST,), metric) for job in jobs if _has_metric(job, metric)]
    return fmean(scored) if scored else None


def _forest_spread(jobs: list[JobResult], metric: str) -> float:
    return fmean(
        _fold_score(job.baseline, (FOREST,), metric).std for job in jobs if _has_metric(job, metric)
    )


# --- plan differences ---


def _plan_differences(results: BenchmarkResults) -> str:
    rows = []
    for dataset in results.config.datasets:
        reference = next(
            (job.plan for job in _of(results, dataset, "reference") if _has_plan(job)), None
        )
        if reference is None:
            continue
        for job in _of(results, dataset):
            if job.method == "agent" and _has_plan(job):
                run = job.label.removeprefix("agent: ")
                if results.config.repeats > 1:
                    run += f" #{job.repeat}"
                rows.append([dataset, run, *_differences(reference, job.plan)])
    if not rows:
        return ""
    header = [
        "dataset",
        "agent run",
        "dropped by the reference only",
        "dropped by the agent only",
        "encoded or cast differently (reference vs agent)",
    ]
    return "\n".join(
        [
            "## Plan differences",
            "",
            "Columns the final plan of each successful agent run treats differently from the "
            f"reference plan. {DASH} means none.",
            "",
            _table(header, rows),
        ]
    )


def _has_plan(job: JobResult) -> bool:
    return job.status == "ok" and job.plan is not None


def _differences(reference: Plan, agent: Plan) -> list[str]:
    ref_dropped, agent_dropped = _dropped(reference), _dropped(agent)
    ref_treated, agent_treated = _treatments(reference), _treatments(agent)
    treated = [
        f"{column} ({ref_treated.get(column, NO_TREATMENT)} vs "
        f"{agent_treated.get(column, NO_TREATMENT)})"
        for column in sorted(ref_treated.keys() | agent_treated.keys())
        if column not in ref_dropped | agent_dropped
        and ref_treated.get(column) != agent_treated.get(column)
    ]
    return [
        ", ".join(columns) or DASH
        for columns in (
            sorted(ref_dropped - agent_dropped),
            sorted(agent_dropped - ref_dropped),
            treated,
        )
    ]


def _dropped(plan: Plan) -> set[str]:
    return {
        column
        for step in plan.steps
        if isinstance(step, DropColumnsStep)
        for column in step.columns
    }


def _treatments(plan: Plan) -> dict[str, str]:
    """How the plan encodes or casts each column it touches, e.g. `onehot(100)` for a one-hot
    encoding capped at 100 categories or `cast + onehot` for a cast followed by an encoding."""
    labels: dict[str, set[str]] = {}
    for step in plan.steps:
        if isinstance(step, EncodeStep):
            label = step.method
            if step.max_categories is not None:
                label += f"({step.max_categories})"
        elif isinstance(step, CastToCategoryStep):
            label = "cast"
        else:
            continue
        for column in step.columns:
            labels.setdefault(column, set()).add(label)
    return {column: " + ".join(sorted(names)) for column, names in labels.items()}


# --- agent repeats ---


def _repeats(results: BenchmarkResults) -> str:
    task_types = {job.dataset: job.task_type for job in results.jobs}
    rows = []
    for dataset in results.config.datasets:
        for model in results.config.models:
            jobs = _of(results, dataset, f"agent: {model}")
            if jobs:
                rows.append(
                    [dataset, model, *_repeat_cells(jobs, MAIN_METRIC[task_types[dataset]])]
                )
    return "\n".join(
        [
            "## Agent repeats",
            "",
            "Per dataset and model, over the repeats of the agent: how many runs succeeded, how "
            "many of the leaky columns the plans dropped (every run counted on its own; a run "
            "without a plan is left out) and the standard deviation of the random forest's score "
            "on the main metric from run to run (n/a below two scored runs).",
            "",
            _table(
                [
                    "dataset",
                    "model",
                    "successful runs",
                    "leaky columns caught",
                    "score std across runs",
                ],
                rows,
            ),
        ]
    )


def _repeat_cells(jobs: list[JobResult], metric: str) -> list[str]:
    ok = sum(job.status == "ok" for job in jobs)
    planned = [job for job in jobs if job.leaky_columns_kept is not None]
    leaky = sum(len(job.leaky_columns) for job in planned)
    caught = leaky - sum(job.leaky_columns_kept for job in planned)
    scores = [_score(job.baseline, (FOREST,), metric) for job in jobs if _has_metric(job, metric)]
    return [
        f"{ok} of {len(jobs)} ({round(100 * ok / len(jobs))}%)",
        f"{caught} of {leaky} ({round(100 * caught / leaky)}%)" if leaky else DASH,
        f"{stdev(scores):.3f}" if len(scores) > 1 else "n/a",
    ]


# --- agent runs ---


def _agent_runs(results: BenchmarkResults, agent_jobs: list[JobResult]) -> str:
    if not agent_jobs:
        return "## Agent runs\n\nNo agent runs."
    currency = results.config.currency
    header = [
        "model",
        "runs",
        "ok",
        "revisions",
        "questions",
        "input tokens",
        "output tokens",
        f"cost ({currency})",
        "plan seconds",
    ]
    rows = []
    for model in results.config.models:
        jobs = [job for job in agent_jobs if job.model == model]
        measured = [job for job in jobs if job.agent is not None]
        ok = sum(job.status == "ok" for job in jobs)
        with_usage = [job.agent for job in measured if job.agent.input_tokens is not None]
        costs = [job.agent.cost for job in measured if job.agent.cost is not None]
        rows.append(
            [
                model,
                str(len(jobs)),
                f"{ok} ({round(100 * ok / len(jobs))}%)" if jobs else DASH,
                _mean_text([job.agent.revisions for job in measured], "{:.1f}"),
                _mean_text([job.agent.questions for job in measured], "{:.1f}"),
                _mean_text([agent.input_tokens for agent in with_usage], "{:.0f}"),
                _mean_text([agent.output_tokens for agent in with_usage], "{:.0f}"),
                _cost_text(costs, len(measured) - len(costs)),
                _mean_text([job.plan_seconds for job in jobs if job.plan_seconds], "{:.1f}"),
            ]
        )
    return "\n".join(["## Agent runs", "", _table(header, rows)])


def _mean_text(values: list, template: str) -> str:
    return template.format(fmean(values)) if values else "n/a"


def _cost_text(costs: list[float], unknown: int) -> str:
    if not costs:
        return "n/a"
    total = f"{sum(costs):.4f}"
    return total if not unknown else f"{total}+ ({unknown} unknown)"


# --- issue codes ---


def _issue_codes(results: BenchmarkResults, agent_jobs: list[JobResult]) -> str:
    parts = ["## Most frequent issue codes"]
    measured = [job for job in agent_jobs if job.agent is not None]
    if measured:
        parts += [
            "",
            "### First-plan errors of agent runs",
            "",
            _code_table(
                [job.agent.first_plan_error_codes for job in measured],
                "No first-plan errors.",
            ),
            "",
            "### Final-plan warnings of agent runs",
            "",
            _code_table([job.warning_codes for job in measured], "No final-plan warnings."),
        ]
    fixed = [job for job in results.jobs if job.method != "agent"]
    if fixed:
        rows = []
        for name in results.config.datasets:
            cells = []
            for label in ("reference", "naive"):
                codes = {code for job in _of(results, name, label) for code in job.warning_codes}
                cells.append(", ".join(sorted(codes)) or DASH)
            rows.append([name, *cells])
        parts += [
            "",
            "### Warnings of reference and naive plans",
            "",
            _table(["dataset", "reference", "naive"], rows),
        ]
    return "\n".join(parts)


def _code_table(codes_per_run: list[list[str]], empty: str) -> str:
    counts = Counter(code for codes in codes_per_run for code in set(codes))
    if not counts:
        return empty
    total = len(codes_per_run)
    rows = [
        [code, str(count), f"{round(100 * count / total)}%"]
        for code, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    return _table(["code", "runs", "share of runs"], rows)


# --- failures and notes ---


def _failures(results: BenchmarkResults) -> str:
    failed = [job for job in results.jobs if job.status != "ok"]
    if not failed:
        return "## Failures\n\nNo failures."
    rows = [[job.dataset, job.label, str(job.repeat), job.status, _reason(job)] for job in failed]
    return "\n".join(
        ["## Failures", "", _table(["dataset", "method", "run", "status", "reason"], rows)]
    )


def _reason(job: JobResult) -> str:
    text = " ".join((job.error or "").split())
    if job.error_codes:
        text = f"{text} [{', '.join(job.error_codes)}]".strip()
    if len(text) > _MAX_REASON:
        text = text[: _MAX_REASON - 1] + "…"
    return text or DASH


def _notes(results: BenchmarkResults, has_agent_runs: bool) -> str:
    cache_notes = _cache_pricing_notes(results) if has_agent_runs else []
    return "\n".join(
        [
            "## Notes",
            "",
            "- Scores come from the baseline module: 5-fold cross-validation (fewer folds when "
            "a class is small) on each plan's own train split. The test set is not used. "
            "The spread between folds is in results.json; a difference smaller than that "
            "spread is not evidence that one plan is better.",
            "- Plans differ in the rows they keep and the columns they produce, so a score "
            "compares what a plan hands to the same models, not the models themselves. The "
            "logistic regression may stop before it converges on unscaled features; that "
            "lowers the score of a plan without scaling and is not reported separately.",
            "- The reference plans are hand-written and were reviewed by a human; see "
            "reference/README.md for the decisions behind them.",
            "- The naive plan is rule-based: drop identifier and constant columns, split, "
            "median and most-frequent imputation, one-hot encoding of every non-numeric column.",
            "- Cost is the token use the provider reported times the prices in prices.yaml; "
            "n/a means the provider reported no usage.",
            *cache_notes,
        ]
    )


def _cache_pricing_notes(results: BenchmarkResults) -> list[str]:
    notes = []
    for model in results.config.models:
        price = results.config.prices.get(model)
        if price is None:
            continue
        cached = price.cached_input_per_million is not None
        written = price.cache_write_per_million is not None
        if cached and written:
            how = "cached input and cache writes are priced separately"
        elif cached:
            how = "cached input is priced separately; cache writes are counted at the input price"
        elif written:
            how = "cache writes are priced separately; cached input is counted at the input price"
        else:
            how = (
                "no cache prices in prices.yaml, so cached and cache-write tokens are counted "
                "at the input price"
            )
        notes.append(f"- Cost of {model}: {how}.")
    return notes


# --- helpers ---


def _of(results: BenchmarkResults, dataset: str, label: str | None = None) -> list[JobResult]:
    return [
        job
        for job in results.jobs
        if job.dataset == dataset and (label is None or job.label == label)
    ]


def _table(header: list[str], rows: list[list[str]]) -> str:
    def line(cells: list[str]) -> str:
        return "| " + " | ".join(_escape(cell) for cell in cells) + " |"

    return "\n".join([line(header), line(["---"] * len(header)), *(line(row) for row in rows)])


def _escape(cell: str) -> str:
    return cell.replace("|", "\\|")
