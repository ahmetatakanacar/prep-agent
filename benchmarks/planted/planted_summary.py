"""Turns the results of a planted-fault run into summary.md.

Runs are grouped by model, variant and mode, and the scores of the `ok` runs in a group are added
up before a ratio is taken (a micro average, see `planted_scorer.combine`). A run that failed or
errored has no score: it is counted in the `runs` column and listed at the end, never counted as
a miss, because "the agent gave up" and "the agent kept the leak" are different findings.
"""

from collections import defaultdict
from dataclasses import dataclass
from statistics import mean

from planted_modes import MODES
from planted_results import PlantedResults, RunResult
from planted_scorer import Score, combine
from planted_truth import FAULT_TYPES, UNMEASURABLE_FAULTS, VARIANTS

NA = "n/a"
# The fault types with a column in the table of detection by type; the legitimate strong
# features are not faults and have a table of their own.
COLUMNS_OF_TYPES = (*FAULT_TYPES, "leak_partial")


@dataclass
class Group:
    model: str
    variant: str
    mode: str
    runs: list[RunResult]

    @property
    def ok(self) -> list[RunResult]:
        return [run for run in self.runs if run.status == "ok" and run.score is not None]

    @property
    def score(self) -> Score:
        return combine([run.score for run in self.ok if run.score is not None])

    @property
    def repeat_detection(self) -> list[float]:
        """The overall detection rate of each repeat, for the spread between repeats."""
        by_repeat: dict[int, list[Score]] = defaultdict(list)
        for run in self.ok:
            if run.score is not None:
                by_repeat[run.repeat].append(run.score)
        rates = [combine(scores).detection_rate for scores in by_repeat.values()]
        return [rate for rate in rates if rate is not None]

    def per_run(self, values: list[float | int | None]) -> float | None:
        known = [value for value in values if value is not None]
        return mean(known) if known else None

    def asked(self, tier: str | None = None, truth: str | None = None) -> tuple[int, int] | None:
        """(asked about, marked) for the columns of a tier and a truth class over the runs of the
        group; None where the mode cannot ask or no run recorded any marked column."""
        if self.mode == "no_questions":
            return None
        marked = [
            m
            for run in self.runs
            for m in run.marked
            if (tier is None or m.tier == tier) and (truth is None or m.truth == truth)
        ]
        return (sum(m.asked for m in marked), len(marked)) if marked else None

    @property
    def questions(self) -> float | None:
        return self.per_run([run.agent.questions for run in self.runs if run.agent])

    @property
    def calls(self) -> float | None:
        return self.per_run([run.agent.calls for run in self.runs if run.agent])

    @property
    def cost(self) -> float | None:
        return self.per_run([run.agent.cost for run in self.runs if run.agent])

    @property
    def cost_total(self) -> float | None:
        costs = [run.agent.cost for run in self.runs if run.agent and run.agent.cost is not None]
        return sum(costs) if costs else None


def groups(results: PlantedResults) -> list[Group]:
    """The groups present in `results`: models in the order of the config, then modes in their
    fixed order, with the variants of a mode next to each other."""
    found: dict[tuple[str, str, str], list[RunResult]] = defaultdict(list)
    for run in results.runs:
        found[(run.model, run.variant, run.mode)].append(run)
    models = {model: i for i, model in enumerate(results.config.models)}
    order = sorted(
        found,
        key=lambda k: (models.get(k[0], len(models)), MODES.index(k[2]), VARIANTS.index(k[1])),
    )
    return [Group(*key, found[key]) for key in order]


def render_summary(results: PlantedResults) -> str:
    config = results.config
    by_group = groups(results)
    lines = [
        "# Planted-fault evaluation",
        "",
        f"Started {results.started_at:%Y-%m-%d %H:%M} UTC"
        + (
            f", finished {results.finished_at:%H:%M} UTC."
            if results.finished_at
            else ", unfinished."
        ),
        f"Models: {', '.join(config.models)}. Variants: {', '.join(config.variants)}. "
        f"Modes: {', '.join(config.modes)}. Repeats: {config.repeats}. "
        f"Datasets: {len(config.datasets)}. Runs recorded: {len(results.runs)}.",
        "",
        "Rates are over the runs that ended with a valid plan; scores of several runs are added "
        "up before the ratio is taken. A plan the agent could not make valid has no score and is "
        "listed at the end.",
        "",
        "## Results by model, variant and mode",
        "",
        *_table(
            [
                "model",
                "variant",
                "mode",
                "runs (ok)",
                "detection",
                "drop precision",
                "drop recall",
                "plan accuracy",
                "must_keep dropped",
                "legit strong dropped",
                "asked: suspected",
                "asked: high",
                "avoided_op violations",
                "questions/run",
                "calls/run",
                f"cost/run ({config.currency})",
                f"cost total ({config.currency})",
            ],
            [_headline_row(group) for group in by_group],
        ),
        "",
        "## Detection by fault type",
        "",
        "Handled columns out of planted columns, over all ok runs of the group.",
        "",
        *_table(
            ["model", "variant", "mode", *COLUMNS_OF_TYPES],
            [_fault_row(group) for group in by_group],
        ),
    ]
    lines += _level_tables(by_group)
    lines += _question_table(by_group)
    comparison = _variant_comparison(by_group)
    if comparison:
        lines += [
            "",
            "## Anonymous against named",
            "",
            "The columns, values and faults are the same in both variants; only the names differ.",
            "",
            *_table(
                [
                    "model",
                    "mode",
                    "detection anonymous",
                    "detection named",
                    "detection difference",
                    "precision anonymous",
                    "precision named",
                    "precision difference",
                ],
                comparison,
            ),
        ]
    lines += ["", *_notes(by_group, results)]
    return "\n".join(lines) + "\n"


def _headline_row(group: Group) -> list[str]:
    score = group.score
    n_ok = len(group.ok)
    spread = group.repeat_detection
    detection = _rate(score.detection_rate)
    if len(spread) > 1 and score.detection_rate is not None:
        detection += f" ({min(spread):.2f}-{max(spread):.2f})"
    return [
        group.model,
        group.variant,
        group.mode,
        f"{len(group.runs)} ({n_ok})",
        detection,
        _rate(score.precision),
        _rate(score.recall),
        _rate(score.plan_accuracy),
        _count(len(score.over_dropped), n_ok),
        _legit(score),
        _ratio(group.asked("suspected")),
        _ratio(group.asked("high")),
        _count(len(score.avoided_op_violations), n_ok),
        _number(group.questions),
        _number(group.calls),
        _money(group.cost),
        _money(group.cost_total),
    ]


def _ratio(pair: tuple[int, int] | None) -> str:
    return NA if pair is None else f"{pair[0]} of {pair[1]}"


def _question_table(by_group: list[Group]) -> list[str]:
    """How often a question named a column the profile related strongly to the target, by what
    the ground truth says that column is. Empty when no run recorded such columns."""
    if not any(run.marked for g in by_group for run in g.runs):
        return []
    cells = [
        ("suspected", "leak", "leaks, 0.9 and over"),
        ("suspected", "legit", "legit, 0.9 and over"),
        ("high", "leak", "leaks, 0.6 to 0.9"),
        ("high", "legit", "legit, 0.6 to 0.9"),
    ]
    return [
        "",
        "## Questions about columns strongly related to the target",
        "",
        "Columns that a question named, out of those the profile related to the target at 0.6 or "
        "more (0.9 or more is `suspected`), by what the ground truth says the column is. The "
        "no_questions mode cannot ask.",
        "",
        *_table(
            ["model", "variant", "mode", *(label for _, _, label in cells)],
            [
                [g.model, g.variant, g.mode, *(_ratio(g.asked(t, k)) for t, k, _ in cells)]
                for g in by_group
            ],
        ),
    ]


def _legit(score: Score) -> str:
    total = sum(score.legit_planted.values())
    if not total:
        return NA
    return f"{sum(score.legit_dropped.values())} of {total}"


def _level_tables(by_group: list[Group]) -> list[str]:
    """Two tables for set D: the leaks of partial strength found, and the legitimate strong
    features dropped, by the association they were made to have. Empty for other sets."""
    scores = [(g, g.score) for g in by_group]
    leak_levels = sorted({lv for _, sc in scores for lv in sc.levels_planted})
    legit_levels = sorted({lv for _, sc in scores for lv in sc.legit_planted})
    lines: list[str] = []
    if leak_levels:
        lines += [
            "",
            "## Leaks of partial strength found, by level of association",
            "",
            "Found out of planted, over all ok runs of the group. The level is the association "
            "the column was made to have; what makes the column a leak is what it means.",
            "",
            *_table(
                ["model", "variant", "mode", *leak_levels],
                [
                    [
                        g.model,
                        g.variant,
                        g.mode,
                        *_by_level(sc.levels_found, sc.levels_planted, leak_levels),
                    ]
                    for g, sc in scores
                ],
            ),
        ]
    if legit_levels:
        lines += [
            "",
            "## Legitimate strong features dropped, by level of association",
            "",
            "Dropped out of planted, over all ok runs of the group. Every drop is a false alarm.",
            "",
            *_table(
                ["model", "variant", "mode", *legit_levels],
                [
                    [
                        g.model,
                        g.variant,
                        g.mode,
                        *_by_level(sc.legit_dropped, sc.legit_planted, legit_levels),
                    ]
                    for g, sc in scores
                ],
            ),
        ]
    return lines


def _by_level(hit, planted, levels: list[str]) -> list[str]:
    return [f"{hit[lv]}/{planted[lv]}" if planted[lv] else NA for lv in levels]


def _fault_row(group: Group) -> list[str]:
    by_type = group.score.by_type
    cells = []
    for fault in COLUMNS_OF_TYPES:
        if fault in UNMEASURABLE_FAULTS:
            cells.append("not scored")
        elif fault not in by_type:
            cells.append(NA)
        else:
            t = by_type[fault]
            cells.append(f"{t.rate:.0%} ({t.detected}/{t.n})")
    return [group.model, group.variant, group.mode, *cells]


def _variant_comparison(by_group: list[Group]) -> list[list[str]]:
    index = {(g.model, g.variant, g.mode): g.score for g in by_group if g.ok}
    rows = []
    for (model, variant, mode), anonymous in index.items():
        named = index.get((model, "named", mode))
        if variant != "anonymous" or named is None:
            continue
        rows.append(
            [
                model,
                mode,
                _rate(anonymous.detection_rate),
                _rate(named.detection_rate),
                _points(anonymous.detection_rate, named.detection_rate),
                _rate(anonymous.precision),
                _rate(named.precision),
                _points(anonymous.precision, named.precision),
            ]
        )
    return rows


def _notes(by_group: list[Group], results: PlantedResults) -> list[str]:
    lines = ["## Notes", ""]
    forced = sum(run.agent.forced_answers for run in results.runs if run.agent)
    questions = sum(run.agent.questions for run in results.runs if run.agent)
    lines.append(
        f"- Questions asked in all runs: {questions}. Answers with no other option to pick "
        f"(chosen as the recommended one, marked `forced` in results.json): {forced}."
    )
    for fault, reason in UNMEASURABLE_FAULTS.items():
        lines.append(f"- `{fault}` columns are in the data but not scored: {reason}.")
    unfinished = [run for run in results.runs if run.status != "ok"]
    if unfinished:
        lines += ["", "### Runs without a score", ""]
        lines += _table(
            ["model", "variant", "mode", "dataset", "repeat", "status", "reason"],
            [
                [
                    run.model,
                    run.variant,
                    run.mode,
                    run.dataset,
                    str(run.repeat),
                    run.status,
                    _short(run.error or ", ".join(run.error_codes)),
                ]
                for run in unfinished
            ],
        )
    return lines


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def _rate(value: float | None) -> str:
    return NA if value is None else f"{value:.2f}"


def _number(value: float | None) -> str:
    return NA if value is None else f"{value:.1f}"


def _money(value: float | None) -> str:
    return NA if value is None else f"{value:.4f}"


def _count(total: int, n_runs: int) -> str:
    return f"{total} ({total / n_runs:.2f}/run)" if n_runs else NA


def _points(before: float | None, after: float | None) -> str:
    if before is None or after is None:
        return NA
    return f"{(after - before) * 100:+.0f} pp"


def _short(text: str, limit: int = 120) -> str:
    text = " ".join(text.split()).replace("|", "/")
    return text if len(text) <= limit else text[: limit - 3] + "..."
