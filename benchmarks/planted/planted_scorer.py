"""Scores a plan against the ground truth of a planted dataset, decision by decision.

What is judged is the plan, not the data it produces: which columns `drop_columns` removes and
which columns `cast_to_category` converts. A column is looked at by its role in the truth:

- `must_drop` columns (leaks, identifiers, personal data, constants) count as found when dropped.
- A duplicate and its original are both `either`; the pair counts as found when at least one of
  the two is dropped. Dropping both is one right drop and one wrong drop, because the information
  of the column is lost; it is listed in `pairs_dropped_both`.
- `must_keep` columns that are dropped are wrong drops. A `numeric_category` column is `must_keep`
  and counts as found when `cast_to_category` converts it and it is not dropped.
- A `legit_strong` column is `must_keep`: dropping it is a wrong drop like any other, and it is
  also counted on its own (`false_alarm_rate`) with the level of association it was made to have,
  next to the leaks of partial strength (`leak_partial`) found at each level.
- `either` columns are neutral when dropped. Columns of a type that cannot be scored
  (`measurable=False`) are listed in `unmeasurable` and left out of every rate.

Rates are plain ratios of counts, so scores of several datasets are added up with `combine`
before the ratio is taken (a micro average): a dataset with more faults weighs more.
"""

from collections import Counter
from dataclasses import dataclass

from pydantic import BaseModel, Field

from planted_truth import DROP_FAULTS, DUPLICATE_FAULTS, DatasetTruth
from prep_agent.plan import CastToCategoryStep, DropColumnsStep, Plan


@dataclass(frozen=True)
class TypeScore:
    """How many columns of one fault type there were and how many the plan handled."""

    fault: str
    n: int
    detected: int

    @property
    def rate(self) -> float:
        return self.detected / self.n


class Score(BaseModel):
    """The result of scoring one plan, or the sum over several (see `combine`).

    Column lists hold `dataset/column` so that they stay readable after `combine`.
    """

    datasets: list[str]
    found: Counter[str] = Field(default_factory=Counter)
    planted: Counter[str] = Field(default_factory=Counter)
    # Drops: a correct drop is a must_drop column (or a pair) that was dropped.
    correct_drops: int = 0
    wrong_drops: int = 0
    required_drops: int = 0
    over_dropped: list[str] = Field(default_factory=list)
    pairs_dropped_both: list[str] = Field(default_factory=list)
    # Ops: columns with an expected op, and how many of them got it.
    expected_ops: int = 0
    applied_ops: int = 0
    missing_ops: list[str] = Field(default_factory=list)
    avoided_op_violations: list[str] = Field(default_factory=list)
    unmeasurable: list[str] = Field(default_factory=list)
    unknown_dropped: list[str] = Field(default_factory=list)
    # Set D. Leaks of partial strength (`leak_partial`) found, and legitimate strong features
    # (`legit_strong`) dropped, by the level of association they were made to have ("0.85").
    levels_planted: Counter[str] = Field(default_factory=Counter)
    levels_found: Counter[str] = Field(default_factory=Counter)
    legit_planted: Counter[str] = Field(default_factory=Counter)
    legit_dropped: Counter[str] = Field(default_factory=Counter)

    @property
    def false_alarm_rate(self) -> float | None:
        """The share of legitimate strong features the plan dropped; None if there were none."""
        total = sum(self.legit_planted.values())
        return sum(self.legit_dropped.values()) / total if total else None

    @property
    def by_type(self) -> dict[str, TypeScore]:
        return {
            fault: TypeScore(fault, n, self.found[fault])
            for fault, n in sorted(self.planted.items())
        }

    @property
    def detection_rate(self) -> float | None:
        """Share of all measurable faults the plan handled; None if the data had none."""
        total = sum(self.planted.values())
        return sum(self.found.values()) / total if total else None

    @property
    def precision(self) -> float | None:
        """Of the drops that matter, the share that were right; None if the plan dropped none."""
        total = self.correct_drops + self.wrong_drops
        return self.correct_drops / total if total else None

    @property
    def recall(self) -> float | None:
        """Of the drops that were needed, the share the plan made; None if none were needed."""
        return self.correct_drops / self.required_drops if self.required_drops else None

    @property
    def op_rate(self) -> float | None:
        return self.applied_ops / self.expected_ops if self.expected_ops else None


def score_plan(plan: Plan, truth: DatasetTruth) -> Score:
    dropped = {
        name for step in plan.steps if isinstance(step, DropColumnsStep) for name in step.columns
    }
    cast = {
        name for step in plan.steps if isinstance(step, CastToCategoryStep) for name in step.columns
    }
    dataset = truth.dataset
    score = Score(datasets=[dataset])
    known = {c.name for c in truth.columns}
    score.unknown_dropped = sorted(f"{dataset}/{name}" for name in dropped - known - {truth.target})

    for column in truth.columns:
        label = f"{dataset}/{column.name}"
        if not column.measurable:
            score.unmeasurable.append(f"{label} ({column.fault})")
            continue
        is_dropped = column.name in dropped
        level = f"{column.level:.2f}" if column.level is not None else None
        if column.fault == "leak_partial" and level is not None:
            score.levels_planted[level] += 1
            score.levels_found[level] += is_dropped
        if column.fault == "legit_strong" and level is not None:
            score.legit_planted[level] += 1
            score.legit_dropped[level] += is_dropped
        if column.fault in DROP_FAULTS:
            score.planted[column.fault] += 1
            score.required_drops += 1
            if is_dropped:
                score.found[column.fault] += 1
                score.correct_drops += 1
        elif column.fault in DUPLICATE_FAULTS:
            score.planted[column.fault] += 1
            score.required_drops += 1
            original = column.duplicate_of
            n_dropped = is_dropped + (original in dropped)
            if n_dropped:
                score.found[column.fault] += 1
                score.correct_drops += 1
            if n_dropped == 2:
                score.wrong_drops += 1
                score.pairs_dropped_both.append(f"{label} + {dataset}/{original}")
        elif column.role == "must_keep" and is_dropped:
            score.wrong_drops += 1
            score.over_dropped.append(label)

        if column.expected_op == "cast_to_category":
            fault = column.fault or "other"
            score.planted[fault] += 1
            score.expected_ops += 1
            if column.name in cast and not is_dropped:
                score.applied_ops += 1
                score.found[fault] += 1
            else:
                score.missing_ops.append(label)
        if column.avoid_op == "cast_to_category" and column.name in cast:
            score.avoided_op_violations.append(label)
    return score


def combine(scores: list[Score]) -> Score:
    """Add up the scores of several datasets or runs."""
    total = Score(datasets=[])
    for score in scores:
        total.datasets.extend(score.datasets)
        total.found.update(score.found)
        total.planted.update(score.planted)
        total.levels_planted.update(score.levels_planted)
        total.levels_found.update(score.levels_found)
        total.legit_planted.update(score.legit_planted)
        total.legit_dropped.update(score.legit_dropped)
        total.correct_drops += score.correct_drops
        total.wrong_drops += score.wrong_drops
        total.required_drops += score.required_drops
        total.expected_ops += score.expected_ops
        total.applied_ops += score.applied_ops
        for name in (
            "over_dropped",
            "pairs_dropped_both",
            "missing_ops",
            "avoided_op_violations",
            "unmeasurable",
            "unknown_dropped",
        ):
            getattr(total, name).extend(getattr(score, name))
    return total
