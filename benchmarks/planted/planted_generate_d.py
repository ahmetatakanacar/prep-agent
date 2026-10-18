"""Seeded generator of set D (see `planted_recipes_d`).

What is new against `planted_generate` is the columns that carry part of the target at a chosen
level, and the less obvious forms of the old faults. The ground truth is written by the code that
plants a column.

A column "at level L" is made from the target so that a model that sees only that column reaches
about L on the association score of the profile (`prep_agent.signals`), which is the share of the
possible improvement over the trivial guess:

- classification, number: a class-dependent shift. With equal spread in both classes, balanced
  accuracy at the best threshold is Phi(d/2) for a shift of d spreads, and the score is
  2 * Phi(d/2) - 1, so d = 2 * Phi^-1((1 + L) / 2).
- classification, category: four levels in two groups. The column falls in the group of the
  class with probability (1 + L) / 2, so balanced accuracy is (1 + L) / 2.
- classification, gap: the column has a value only for part of the positive rows, a share L of
  them, and never for the negative rows, so "has a value" is right for a share L of the positives
  and for all negatives.
- regression, number: the standardised target times sqrt(L) plus noise, so R-squared is L.
- regression, category: the same number cut into twelve levels with shuffled names.

The level of a leak and the level of a legitimate feature are made the same way. Nothing in the
numbers tells them apart; the ground truth says which is which by what the column means.

Run `python benchmarks/planted/planted_generate_d.py` to write the files of the dev recipes and
the ground truth of the test recipes. The data of the test recipes is written only with
`--with-test-data`, for the one run it is meant for.
"""

import argparse
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from planted_generate import (
    DATA_DIR,
    DECIMALS,
    EMAIL_DOMAINS,
    FIRST_NAMES,
    FOREIGN_NAMES,
    LAST_NAMES,
    NEWLINE,
    _ascii,
    _sigmoid,
    _standardize,
    file_stem,
)
from planted_recipes_d import ALL_D, Col, Fault, RecipeD
from planted_truth import VARIANTS, ColumnTruth, DatasetTruth, Variant

PRE_TARGET = ("numeric_category", "count_decoy", "sentinel")
LEVEL_FAULTS = ("leak_partial", "legit_strong")


def generate_d(
    recipe: RecipeD, variant: Variant = "anonymous"
) -> tuple[pd.DataFrame, DatasetTruth]:
    return _BuilderD(recipe, variant).build()


def level_column(
    rng: np.random.Generator,
    y: np.ndarray,
    task: str,
    level: float,
    form: str,
    loc: float,
    scale: float,
) -> np.ndarray:
    """A column made from the target `y` to have association `level` with it (see the module
    docstring). `form` is "numeric", "category" or, for classification only, "gap"."""
    n = len(y)
    if not 0 < level < 1:
        raise ValueError(f"level must be between 0 and 1, got {level}")
    if form not in ("numeric", "category", "gap") or (task == "regression" and form == "gap"):
        raise ValueError(f"form {form!r} is not available for a {task} target")
    if task == "classification":
        positive = y == 1
        if form == "numeric":
            shift = 2 * norm.ppf((1 + level) / 2)
            return np.round(loc + scale * (shift * positive + rng.normal(size=n)), DECIMALS)
        if form == "category":
            right = rng.random(n) < (1 + level) / 2
            group = np.where(positive == right, 0, 1)
            labels = rng.permutation([f"g{i}" for i in range(1, 5)])
            return labels[group * 2 + rng.integers(0, 2, n)].astype(object)
        if form == "gap":
            values = np.round(loc + scale * rng.normal(size=n), DECIMALS)
            return np.where(positive & (rng.random(n) < level), values, np.nan)
    elif task == "regression":
        rho = np.sqrt(level)
        raw = rho * _standardize(y) + np.sqrt(1 - rho**2) * rng.normal(size=n)
        if form == "numeric":
            return np.round(loc + scale * raw, DECIMALS)
        if form == "category":
            edges = np.quantile(raw, np.linspace(0, 1, 13)[1:-1])
            labels = rng.permutation([f"g{i:02d}" for i in range(1, 13)])
            return labels[np.searchsorted(edges, raw)].astype(object)
    raise ValueError(f"form {form!r} is not available for a {task} target")


class _BuilderD:
    def __init__(self, recipe: RecipeD, variant: Variant) -> None:
        self.recipe = recipe
        self.variant = variant
        self.rng = np.random.default_rng(recipe.seed)
        self.n = recipe.n_rows
        self.columns: dict[str, pd.Series] = {}
        self.truth: dict[str, ColumnTruth] = {}
        self.public: dict[str, str] = {}
        self.by_name: dict[str, str] = {}
        self.score = np.zeros(self.n)

    # --- bookkeeping ---

    def _put(self, name: str, values: Any, **truth: Any) -> str:
        key = f"col{len(self.columns)}"
        self.columns[key] = pd.Series(values)
        self.truth[key] = ColumnTruth(name=key, **{"role": "must_keep", **truth})
        self.public[key] = name
        self.by_name[name] = key
        return key

    def build(self) -> tuple[pd.DataFrame, DatasetTruth]:
        faults = self.recipe.faults
        self._base()
        for fault in faults:
            if fault.fault in PRE_TARGET:
                getattr(self, f"_{fault.fault}")(fault)
        y = self._target()
        self._add_missing()
        for fault in faults:
            if fault.fault in LEVEL_FAULTS:
                self._level(fault, y)
            elif fault.fault in PRE_TARGET or fault.fault.startswith("duplicate_"):
                continue
            else:
                getattr(self, f"_{fault.fault}")(fault)
        for fault in faults:
            if fault.fault.startswith("duplicate_"):
                self._duplicate(fault)
        return self._finish(y)

    # --- the base table and the target ---

    def _base(self) -> None:
        rng, n = self.rng, self.n
        self.base_keys: list[str] = []
        for col in self.recipe.base:
            if col.kind == "cat":
                levels = np.array([f"k{j}" for j in range(col.levels)], dtype=object)
                codes = rng.choice(col.levels, n, p=rng.dirichlet(np.ones(col.levels) * 2))
                self.score += rng.normal(0, 0.6, col.levels)[codes]
                values: Any = levels[codes]
            else:
                values = self._numeric(col)
                self.score += rng.normal(0, 0.8) * _standardize(np.asarray(values, dtype=float))
            self.base_keys.append(self._put(col.name, values))

    def _numeric(self, col: Col) -> np.ndarray:
        rng, n = self.rng, self.n
        if col.kind == "normal":
            values = rng.normal(col.loc if col.loc is not None else 100, col.scale or 15, n)
        elif col.kind == "lognormal":
            values = rng.lognormal(col.loc if col.loc is not None else 3, col.scale or 0.6, n)
        elif col.kind == "age":
            low = 18 if col.loc is None else int(col.loc)
            high = 79 if col.scale is None else int(col.scale)
            values = rng.integers(low, high + 1, n).astype(float)
        elif col.kind == "uniform":
            values = rng.uniform(col.loc or 0, col.scale or 1000, n)
        else:
            values = rng.poisson(col.loc if col.loc is not None else 2.5, n).astype(float)
        return np.round(values, DECIMALS)

    def _target(self) -> np.ndarray:
        rng, r = self.rng, self.recipe
        score = _standardize(self.score)
        if r.task == "classification":
            logit = np.log(r.pos_rate / (1 - r.pos_rate)) + 1.2 * score
            return (rng.random(self.n) < _sigmoid(logit)).astype(int)
        return np.round(r.mu + r.sigma * (0.85 * score + 0.53 * rng.normal(size=self.n)), DECIMALS)

    def _add_missing(self) -> None:
        rng = self.rng
        for col, key in zip(self.recipe.base, self.base_keys, strict=True):
            if col.missing <= 0:
                continue
            hit = rng.random(self.n) < col.missing
            column = self.columns[key].copy()
            column[hit] = None if col.kind == "cat" else np.nan
            self.columns[key] = column

    # --- columns that feed the target ---

    def _numeric_category(self, fault: Fault) -> None:
        rng = self.rng
        k = int(rng.integers(5, 11))
        codes = np.sort(rng.choice(np.arange(10, 130, 5), k, replace=False))
        values = rng.choice(codes, self.n, p=rng.dirichlet(np.ones(k) * 3))
        self.score += rng.normal(0, 1.0, k)[np.searchsorted(codes, values)]
        self._put(fault.name, values.astype(int), fault="numeric_category",
                  expected_op="cast_to_category")  # fmt: skip

    def _count_decoy(self, fault: Fault) -> None:
        values = self.rng.integers(1, 7, self.n)
        self.score += 0.5 * _standardize(values.astype(float))
        self._put(fault.name, values.astype(int), avoid_op="cast_to_category")

    def _sentinel(self, fault: Fault) -> None:
        rng = self.rng
        values = rng.normal(500, 120, self.n)
        self.score += rng.normal(0, 0.7) * _standardize(values)
        values = np.round(values, DECIMALS)
        hit = rng.random(self.n) < rng.uniform(0.05, 0.1)
        values[hit] = rng.choice([-999, 9999])
        self._put(fault.name, values, fault="sentinel", role="either", measurable=False)

    # --- columns made from the target ---

    def _level(self, fault: Fault, y: np.ndarray) -> None:
        values = level_column(
            self.rng, y, self.recipe.task, fault.level or 0, fault.form, fault.loc, fault.scale
        )
        role = "must_drop" if fault.fault == "leak_partial" else "must_keep"
        self._put(fault.name, values, fault=fault.fault, role=role, level=fault.level,
                  reason=fault.reason)  # fmt: skip

    # --- the old faults, less obvious ---

    def _id_sequential(self, fault: Fault) -> None:
        rng, n = self.rng, self.n
        start = int(rng.integers(1000, 200_000))
        if fault.form == "sparse":
            values: Any = start + np.cumsum(rng.integers(1, 10, n))
        elif fault.form == "prefixed":
            prefix = "".join(rng.choice(list("ABCDEFGHJKLMNPRSTUVWXYZ"), 3))
            values = np.array([f"{prefix}-{start + i:06d}" for i in range(n)], dtype=object)
        else:
            values = start + np.arange(n)
        self._put(fault.name, values, fault="id_sequential", role="must_drop")

    def _id_uuid(self, fault: Fault) -> None:
        ids = [uuid.UUID(bytes=self.rng.bytes(16), version=4) for _ in range(self.n)]
        values = [u.hex if fault.form == "hex32" else str(u) for u in ids]
        self._put(fault.name, np.array(values, dtype=object), fault="id_uuid", role="must_drop")

    def _constant(self, fault: Fault) -> None:
        value = str(self.rng.choice(["v1", "0", "1"]))
        constant = value if value == "v1" else int(value)
        self._put(fault.name, [constant] * self.n, fault="constant", role="must_drop")

    def _pii_email(self, fault: Fault) -> None:
        rng, n = self.rng, self.n
        firsts, lasts = rng.choice(FIRST_NAMES, n), rng.choice(LAST_NAMES, n)
        domains = rng.choice(EMAIL_DOMAINS, n)
        numbers = rng.integers(1, 100, n)
        if fault.form == "initial":
            values = [
                f"{_ascii(f)[0]}{_ascii(last)}{num if num % 4 == 0 else ''}@{d}"
                for f, last, d, num in zip(firsts, lasts, domains, numbers, strict=True)
            ]
        else:
            values = [
                f"{_ascii(f)}.{_ascii(last)}{num if num % 3 == 0 else ''}@{d}"
                for f, last, d, num in zip(firsts, lasts, domains, numbers, strict=True)
            ]
        self._put(fault.name, np.array(values, dtype=object), fault="pii_email", role="must_drop")

    def _pii_phone(self, fault: Fault) -> None:
        rng, n = self.rng, self.n
        digits = rng.integers(0, 10, (n, 9))
        if fault.form == "int10":
            values: Any = np.array([int("5" + "".join(map(str, d))) for d in digits])
        else:
            values = np.array(
                [f"05{''.join(map(str, d[:2]))} {''.join(map(str, d[2:5]))} "
                 f"{''.join(map(str, d[5:7]))} {''.join(map(str, d[7:]))}" for d in digits],
                dtype=object,
            )  # fmt: skip
        self._put(fault.name, values, fault="pii_phone", role="must_drop")

    def _pii_name(self, fault: Fault) -> None:
        rng, n = self.rng, self.n
        firsts, lasts = rng.choice(FIRST_NAMES, n), rng.choice(LAST_NAMES, n)
        foreign = rng.random(n) < 0.15
        values = []
        for first, last, other in zip(firsts, lasts, foreign, strict=True):
            if other:
                values.append(str(rng.choice(FOREIGN_NAMES)))
            elif fault.form == "surname_first":
                values.append(f"{last}, {first}")
            elif fault.form == "initial":
                values.append(f"{first[0]}. {last}")
            else:
                values.append(f"{first} {last}")
        self._put(fault.name, np.array(values, dtype=object), fault="pii_name", role="must_drop")

    def _duplicate(self, fault: Fault) -> None:
        rng = self.rng
        original = self.by_name[str(fault.of)]
        values = self.columns[original].copy()
        if fault.form == "relabel":
            labels = rng.permutation([f"C{i:02d}" for i in range(1, values.nunique() + 1)])
            mapping = dict(zip(sorted(values.dropna().unique()), labels, strict=True))
            values = values.map(mapping)
        elif fault.fault == "duplicate_scaled":
            factor = fault.factor or 1.0
            scaled = values * factor
            if fault.form == "coarse":
                # A unit conversion kept at a tenth of a standard deviation: a copy to a person,
                # a correlation of about 0.9996 to the profile.
                unit = 0.1 * float(scaled.std())
                values = np.round(scaled / unit) * unit
            else:
                values = np.round(scaled + rng.uniform(-3, 3), DECIMALS)
        self.truth[original] = ColumnTruth(name=original, role="either")
        self._put(fault.name, values, role="either", fault=fault.fault, duplicate_of=original)

    # --- output ---

    def _finish(self, y: np.ndarray) -> tuple[pd.DataFrame, DatasetTruth]:
        keys = list(self.columns)
        order = [keys[i] for i in self.rng.permutation(len(keys))]
        if self.variant == "anonymous":
            names = {key: f"c{i + 1:02d}" for i, key in enumerate(order)}
            target_name = "y"
        else:
            names, target_name = self.public, self.recipe.target
        frame = pd.DataFrame({names[key]: self.columns[key] for key in order})
        frame[target_name] = y
        columns = [
            self.truth[key].model_copy(
                update={
                    "name": names[key],
                    "duplicate_of": names.get(self.truth[key].duplicate_of or ""),
                }
            )
            for key in order
        ]
        truth = DatasetTruth(
            dataset=self.recipe.name,
            variant=self.variant,
            seed=self.recipe.seed,
            target=target_name,
            task_type=self.recipe.task,
            n_rows=self.n,
            columns=columns,
        )
        return frame, truth


def write_all_d(out_dir: Path = DATA_DIR, with_test_data: bool = False) -> list[str]:
    """Write the dev recipes in full and the ground truth of the test recipes. Their data goes
    out only with `with_test_data`; the frame is made but not written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for recipe in ALL_D:
        for variant in VARIANTS:
            frame, truth = generate_d(recipe, variant)
            stem = file_stem(recipe.name, variant)
            if recipe.split == "dev" or with_test_data:
                frame.to_csv(out_dir / f"{stem}.csv", index=False, lineterminator=NEWLINE)
            (out_dir / f"{stem}.truth.json").write_text(
                truth.model_dump_json(indent=2, exclude_none=True) + NEWLINE, encoding="utf-8"
            )
    return [recipe.name for recipe in ALL_D]


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--with-test-data", action="store_true")
    args = parser.parse_args()
    for name in write_all_d(args.out_dir, args.with_test_data):
        print(f"wrote {name}")


if __name__ == "__main__":
    main()
