"""Seeded generator of the planted-fault datasets.

Each dataset is a realistic base table (numeric and categorical features, some missing values,
a target that depends on them) with known faults planted on top. Column names are `c01`, `c02`,
... in shuffled order and the target is `y`, so a model cannot tell a fault from its name and the
profiler's name rules are not what is being tested. The ground truth is written by the same code
that plants the fault, so the two cannot disagree.

Run `python benchmarks/planted/planted_generate.py` to rewrite the files in `data/`. A test
regenerates them and fails if they differ from what is committed.
"""

import argparse
import dataclasses
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from planted_names import DUPLICATE_SUFFIX, NAMES
from planted_truth import VARIANTS, ColumnTruth, DatasetTruth, FaultType, Variant

DATA_DIR = Path(__file__).parent / "data"
TARGET = "y"
DECIMALS = 4
NEWLINE = chr(10)

# A plain informative integer column that really is a count (rooms, children). It is no fault:
# it is there to see whether a plan casts every small integer column to a category.
COUNT_DECOY = "count_decoy"
Item = FaultType | Literal["count_decoy"]

FIRST_NAMES = (
    "Ayşe", "Mehmet", "Fatma", "Mustafa", "Emine", "Ahmet", "Hatice", "Ali", "Zeynep", "Hüseyin",
    "Elif", "İbrahim", "Merve", "Hasan", "Özlem", "Yusuf", "Büşra", "Ömer", "Gül", "Çağlar",
    "Şule", "Burak", "Ceren", "Emre", "Gökhan", "Işıl", "Kemal", "Selin", "Tuğba", "Uğur",
)  # fmt: skip
LAST_NAMES = (
    "Yılmaz", "Kaya", "Demir", "Şahin", "Çelik", "Yıldız", "Yıldırım", "Öztürk", "Aydın", "Özdemir",
    "Arslan", "Doğan", "Kılıç", "Aslan", "Çetin", "Kara", "Koç", "Kurt", "Özkan", "Şimşek",
    "Polat", "Korkmaz", "Güneş", "Erdoğan", "Tekin", "Ünal", "Aksoy", "Güler", "Bulut", "Taş",
)  # fmt: skip
FOREIGN_NAMES = (
    "John Smith", "Maria Garcia", "Anna Müller", "David Brown", "Sofia Rossi", "Lucas Martin",
    "Emma Wilson", "Ivan Petrov", "Aisha Khan", "Chen Wei",
)  # fmt: skip
EMAIL_DOMAINS = ("gmail.com", "hotmail.com", "outlook.com", "yahoo.com", "firma.com.tr")
_ASCII = str.maketrans("ıİşŞğĞüÜöÖçÇ", "iIsSgGuUoOcC")


@dataclass(frozen=True)
class Recipe:
    name: str
    seed: int
    task: Literal["classification", "regression"]
    n_rows: int
    n_numeric: int
    n_categorical: int
    items: tuple[Item, ...]
    # The recipe whose column names and domain this one shares, if it is not its own.
    domain: str | None = None


# Every measurable fault type appears in at least two datasets.
RECIPES: tuple[Recipe, ...] = (
    Recipe("planted-01", 101, "classification", 800, 6, 2,
           ("leak_linear", "id_sequential", "pii_email", "constant", "numeric_category")),
    Recipe("planted-02", 102, "classification", 700, 5, 3,
           ("leak_categorical", "id_uuid", "pii_name", "duplicate_scaled", "sentinel")),
    Recipe("planted-03", 103, "regression", 900, 6, 2,
           ("leak_nonlinear", "id_sequential", "pii_phone", "duplicate_exact", "numeric_category",
            "count_decoy")),
    Recipe("planted-04", 104, "classification", 600, 5, 2,
           ("leak_nonlinear", "id_uuid", "pii_email", "pii_name", "constant")),
    Recipe("planted-05", 105, "regression", 750, 7, 3,
           ("leak_linear", "leak_categorical", "pii_phone", "duplicate_scaled", "sentinel",
            "numeric_category")),
    Recipe("planted-06", 106, "classification", 1000, 6, 3,
           ("leak_linear", "leak_categorical", "id_sequential", "pii_phone", "duplicate_exact",
            "numeric_category", "sentinel")),
    Recipe("planted-07", 107, "regression", 650, 5, 2,
           ("leak_categorical", "leak_nonlinear", "id_uuid", "pii_name", "pii_email", "constant",
            "duplicate_scaled")),
    Recipe("planted-08", 108, "classification", 700, 6, 2,
           ("id_sequential", "constant", "count_decoy", "pii_name")),
)  # fmt: skip

# The second set: the same recipes (task, size, faults, domain names) with other seeds. The first
# set was used to measure the agent before the profile got its leakage and duplicate signals; this
# one measures it after, and has to stay unseen by anyone tuning those signals.
RECIPES_B: tuple[Recipe, ...] = tuple(
    dataclasses.replace(
        recipe, name=recipe.name.replace("planted-", "planted-b"), seed=recipe.seed + 100,
        domain=recipe.name,
    )
    for recipe in RECIPES
)  # fmt: skip

# The third set, for the closing measurement: the same recipes again with seeds that nothing was
# tuned or looked at with.
RECIPES_C: tuple[Recipe, ...] = tuple(
    dataclasses.replace(
        recipe, name=recipe.name.replace("planted-", "planted-c"), seed=recipe.seed + 200,
        domain=recipe.name,
    )
    for recipe in RECIPES
)  # fmt: skip
ALL_RECIPES: tuple[Recipe, ...] = RECIPES + RECIPES_B + RECIPES_C


def generate(recipe: Recipe, variant: Variant = "anonymous") -> tuple[pd.DataFrame, DatasetTruth]:
    """The data and its truth. The two variants hold the same values and the same faults in the
    same order; only the column names differ."""
    return _Builder(recipe, variant).build()


def file_stem(name: str, variant: Variant) -> str:
    return name if variant == "anonymous" else f"{name}.{variant}"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def _standardize(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std()


def _ascii(text: str) -> str:
    return text.translate(_ASCII).lower()


class _Builder:
    def __init__(self, recipe: Recipe, variant: Variant) -> None:
        self.recipe = recipe
        self.variant = variant
        self.rng = np.random.default_rng(recipe.seed)
        self.n = recipe.n_rows
        # Columns are collected under internal keys; the public names come after the shuffle.
        self.columns: dict[str, pd.Series] = {}
        self.truth: dict[str, ColumnTruth] = {}
        self.score = np.zeros(self.n)
        self.numeric_keys: list[str] = []
        self.categorical_keys: list[str] = []

    def build(self) -> tuple[pd.DataFrame, DatasetTruth]:
        items = self.recipe.items
        self._base_columns()
        # Columns that feed the target come first; the faults built from the target come after.
        for item in items:
            if item == "numeric_category":
                self._numeric_category()
            elif item == COUNT_DECOY:
                self._count_decoy()
            elif item == "sentinel":
                self._sentinel()
        target, signal = self._target()
        self._add_missing()
        for item in items:
            if item.startswith("leak_"):
                self._leak(item, signal)
            elif item == "id_sequential":
                self._id_sequential()
            elif item == "id_uuid":
                self._id_uuid()
            elif item.startswith("pii_"):
                self._pii(item)
            elif item == "constant":
                self._constant()
        # Copies come last so that they also copy the missing values of their originals.
        for item in items:
            if item.startswith("duplicate_"):
                self._duplicate(item)
        return self._finish(target)

    def _put(self, values: Any, **truth: Any) -> str:
        key = f"col{len(self.columns)}"
        self.columns[key] = pd.Series(values)
        self.truth[key] = ColumnTruth(name=key, **{"role": "must_keep", **truth})
        return key

    # --- the base table ---

    def _base_columns(self) -> None:
        rng, n = self.rng, self.n
        for i in range(self.recipe.n_numeric):
            kind = ("normal", "lognormal", "age", "uniform")[i % 4]
            if kind == "normal":
                values = rng.normal(100 * rng.uniform(0.5, 2), 15, n)
            elif kind == "lognormal":
                values = rng.lognormal(rng.uniform(2, 4), 0.7, n)
            elif kind == "age":
                values = rng.integers(18, 80, n).astype(float)
            else:
                values = rng.uniform(0, 1000, n)
            self.score += rng.normal(0, 0.8) * _standardize(values)
            self.numeric_keys.append(self._put(np.round(values, DECIMALS)))
        for _ in range(self.recipe.n_categorical):
            levels = np.array([f"k{j}" for j in range(rng.integers(3, 7))], dtype=object)
            codes = rng.choice(len(levels), n, p=rng.dirichlet(np.ones(len(levels)) * 2))
            self.score += rng.normal(0, 0.6, len(levels))[codes]
            self.categorical_keys.append(self._put(levels[codes]))

    def _numeric_category(self) -> None:
        rng = self.rng
        k = int(rng.integers(5, 11))
        codes = np.sort(rng.choice(np.arange(10, 130, 5), k, replace=False))
        values = rng.choice(codes, self.n, p=rng.dirichlet(np.ones(k) * 3))
        # The effect of a code is unrelated to its size, so the numbers are labels, not quantities.
        self.score += rng.normal(0, 1.0, k)[np.searchsorted(codes, values)]
        self._put(values.astype(int), fault="numeric_category", expected_op="cast_to_category")

    def _count_decoy(self) -> None:
        values = self.rng.integers(1, 7, self.n)
        self.score += 0.5 * _standardize(values.astype(float))
        self._put(values.astype(int), avoid_op="cast_to_category")

    def _sentinel(self) -> None:
        rng = self.rng
        values = rng.normal(500, 120, self.n)
        self.score += rng.normal(0, 0.7) * _standardize(values)
        values = np.round(values, DECIMALS)
        hit = rng.random(self.n) < rng.uniform(0.05, 0.1)
        values[hit] = rng.choice([-999, 9999])
        self._put(values, fault="sentinel", role="either", measurable=False)

    def _target(self) -> tuple[np.ndarray, np.ndarray]:
        rng = self.rng
        score = _standardize(self.score)
        if self.recipe.task == "classification":
            pos_rate = rng.uniform(0.25, 0.45)
            logit = np.log(pos_rate / (1 - pos_rate)) + 1.4 * score
            y = (rng.random(self.n) < _sigmoid(logit)).astype(int)
            return y, y.astype(float)
        y = np.round(50 + 12 * score + rng.normal(0, 5, self.n), DECIMALS)
        return y, y

    def _add_missing(self) -> None:
        plain = [k for k, t in self.truth.items() if t.fault is None and t.avoid_op is None]
        numeric = [k for k in plain if self.columns[k].dtype.kind == "f"]
        text = [k for k in plain if self.columns[k].dtype.kind == "O"]
        for key in [*numeric[:2], *text[:1]]:
            hit = self.rng.random(self.n) < self.rng.uniform(0.03, 0.18)
            column = self.columns[key].copy()
            column[hit] = None if key in text else np.nan
            self.columns[key] = column

    # --- faults built on the target ---

    def _leak(self, kind: str, signal: np.ndarray) -> None:
        rng = self.rng
        z = _standardize(signal)
        if kind == "leak_linear":
            values = rng.uniform(0.5, 3) * z + rng.uniform(-5, 5) + rng.normal(0, 0.06, self.n)
            self._put(np.round(values, DECIMALS), fault=kind, role="must_drop")
        elif kind == "leak_nonlinear":
            values = np.exp(0.8 * z + rng.normal(0, 0.12, self.n))
            self._put(np.round(values, DECIMALS), fault=kind, role="must_drop")
        else:
            noisy = z + rng.normal(0, 0.25, self.n)
            if self.recipe.task == "classification":
                levels = np.where(noisy > 0.4, "m", np.where(noisy > -0.5, "n", "o"))
            else:
                labels = rng.permutation(["q1", "q2", "q3", "q4"])
                levels = labels[np.searchsorted(np.quantile(noisy, [0.25, 0.5, 0.75]), noisy)]
            self._put(levels.astype(object), fault=kind, role="must_drop")

    # --- faults that do not depend on the target ---

    def _id_sequential(self) -> None:
        start = int(self.rng.integers(1000, 200_000))
        step = int(self.rng.choice([1, 3]))
        self._put(start + step * np.arange(self.n), fault="id_sequential", role="must_drop")

    def _id_uuid(self) -> None:
        values = [str(uuid.UUID(bytes=self.rng.bytes(16), version=4)) for _ in range(self.n)]
        self._put(np.array(values, dtype=object), fault="id_uuid", role="must_drop")

    def _constant(self) -> None:
        value = str(self.rng.choice(["same", "0", "1"]))
        constant = value if value == "same" else int(value)
        self._put([constant] * self.n, fault="constant", role="must_drop")

    def _pii(self, kind: str) -> None:
        rng, n = self.rng, self.n
        firsts = rng.choice(FIRST_NAMES, n)
        lasts = rng.choice(LAST_NAMES, n)
        if kind == "pii_name":
            foreign = rng.random(n) < 0.15
            values = [
                str(rng.choice(FOREIGN_NAMES)) if hit else f"{first} {last}"
                for first, last, hit in zip(firsts, lasts, foreign, strict=True)
            ]
        elif kind == "pii_email":
            domains = rng.choice(EMAIL_DOMAINS, n)
            numbers = rng.integers(1, 100, n)
            values = [
                f"{_ascii(first)}.{_ascii(last)}{num if num % 3 == 0 else ''}@{domain}"
                for first, last, domain, num in zip(firsts, lasts, domains, numbers, strict=True)
            ]
        else:
            compact = rng.random(n) < 0.3
            prefixes = rng.choice(["+90 5", "05"], n)
            digits = ["".join(map(str, row)) for row in rng.integers(0, 10, (n, 9))]
            values = [
                f"+905{d}" if short else f"{p}{d[:2]} {d[2:5]} {d[5:7]} {d[7:]}"
                for p, d, short in zip(prefixes, digits, compact, strict=True)
            ]
        self._put(np.array(values, dtype=object), fault=kind, role="must_drop")

    def _duplicate(self, kind: str) -> None:
        rng = self.rng
        taken = {t.duplicate_of for t in self.truth.values()}
        candidates = [
            k
            for k, t in self.truth.items()
            if t.fault is None
            and t.avoid_op is None
            and k not in taken
            and self.columns[k].dtype.kind == "f"
        ]
        original = candidates[int(rng.integers(len(candidates)))]
        values = self.columns[original].copy()
        if kind == "duplicate_scaled":
            factor = float(rng.choice([2.54, 0.001, 12.0, 0.45]))
            values = np.round(values * factor + rng.uniform(-3, 3), DECIMALS)
        self.truth[original] = ColumnTruth(name=original, role="either")
        self._put(values, role="either", fault=kind, duplicate_of=original)

    # --- output ---

    def _finish(self, target: np.ndarray) -> tuple[pd.DataFrame, DatasetTruth]:
        keys = list(self.columns)
        order = [keys[i] for i in self.rng.permutation(len(keys))]
        rename = self._public_names(order)
        target_name = (
            TARGET
            if self.variant == "anonymous"
            else NAMES[self.recipe.domain or self.recipe.name].target
        )
        frame = pd.DataFrame({rename[key]: self.columns[key] for key in order})
        frame[target_name] = target
        columns = [
            self.truth[key].model_copy(
                update={
                    "name": rename[key],
                    "duplicate_of": rename.get(self.truth[key].duplicate_of or ""),
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

    def _public_names(self, order: list[str]) -> dict[str, str]:
        if self.variant == "anonymous":
            return {key: f"c{i + 1:02d}" for i, key in enumerate(order)}
        domain = NAMES[self.recipe.domain or self.recipe.name]
        names = {key: domain.numeric[i] for i, key in enumerate(self.numeric_keys)}
        names |= {key: domain.categorical[i] for i, key in enumerate(self.categorical_keys)}
        for key, truth in self.truth.items():
            if key in names or truth.fault in DUPLICATE_SUFFIX:
                continue
            names[key] = domain.faults["count_decoy" if truth.avoid_op else str(truth.fault)]
        for key, truth in self.truth.items():
            if truth.fault in DUPLICATE_SUFFIX:
                names[key] = names[str(truth.duplicate_of)] + DUPLICATE_SUFFIX[truth.fault]
        return names


def write_all(out_dir: Path = DATA_DIR) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for recipe in ALL_RECIPES:
        for variant in VARIANTS:
            frame, truth = generate(recipe, variant)
            stem = file_stem(recipe.name, variant)
            frame.to_csv(out_dir / f"{stem}.csv", index=False, lineterminator=NEWLINE)
            (out_dir / f"{stem}.truth.json").write_text(
                truth.model_dump_json(indent=2, exclude_none=True) + NEWLINE, encoding="utf-8"
            )
    return [recipe.name for recipe in ALL_RECIPES]


def load(
    name: str, variant: Variant = "anonymous", data_dir: Path = DATA_DIR
) -> tuple[pd.DataFrame, DatasetTruth]:
    """Read a committed dataset and its ground truth."""
    stem = file_stem(name, variant)
    frame = pd.read_csv(data_dir / f"{stem}.csv")
    truth = DatasetTruth.model_validate_json(
        (data_dir / f"{stem}.truth.json").read_text(encoding="utf-8")
    )
    return frame, truth


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    for name in write_all(args.out_dir):
        print(f"wrote {name}")


if __name__ == "__main__":
    main()
