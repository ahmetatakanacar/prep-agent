"""Signals the profiler computes between columns: how well one column alone predicts the target,
and which columns are copies of an earlier one.

These are plain functions of data and numbers; every threshold is passed in by the profiler,
which holds them in `ProfilerConfig` with the reasons.

Target association. The score of a column is how much of the possible improvement over the
trivial predictor (the mean, or the majority class) a model that sees only that column reaches on
rows it was not fitted on. For a regression target it is the out-of-fold R². For a
classification target it is balanced accuracy rescaled so that chance level (one over the number
of classes) is 0 and a perfect prediction is 1, which keeps it comparable across class counts and
class imbalance. Both are clipped to [0, 1]: a column that predicts worse than the trivial
predictor has no association.

The model is as small as the question allows. A numeric column gets a shallow decision tree, so a
relation of any monotone or stepped shape is found (linear, exponential, thresholded), not only a
linear one. A non-numeric column gets a lookup table from category to the mean target (or the
most common class) of the fitting rows; a category never seen in them falls back to the trivial
predictor. Out-of-fold scoring is what keeps the score honest for columns with many distinct
values: an identifier or a name that is unique per row cannot be looked up for a row it was not
fitted on, so it scores about 0 instead of looking like a perfect predictor.

Duplicates. A column is a copy of an earlier column when it has the same missing values and
carries the same information: equal values, a numeric column that is a scaled and shifted copy
(a unit conversion), or a one-to-one mapping between two sets of repeated values (a code and its
label). The earlier column is the original.
"""

import numpy as np
import pandas as pd
from pandas.api import types as pdt
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# The row order of the sample is fixed so that a profile is the same every time.
SEED = 0
_MISSING_CATEGORY = "\x00missing"


def target_association(
    df: pd.DataFrame,
    target: str,
    columns: list[str],
    *,
    task: str,
    max_rows: int,
    min_rows: int,
    folds: int,
    tree_depth: int,
) -> dict[str, float]:
    """The association score of each of `columns` with `target`, for the columns that have one.

    `task` is "classification" or "regression". Rows without a target value are left out, and
    more than `max_rows` rows are cut down to a fixed random sample. Fewer than `min_rows` rows
    give no scores at all: with so few, a score out of fold says nothing. A column that is
    constant in the sample, or all missing, has no score.
    """
    rows = np.flatnonzero(df[target].notna().to_numpy())
    if len(rows) < min_rows:
        return {}
    if len(rows) > max_rows:
        rows = np.sort(np.random.default_rng(SEED).choice(rows, max_rows, replace=False))
    sample = df.iloc[rows]

    if task == "classification":
        y = pd.factorize(sample[target])[0]
        if len(np.unique(y)) < 2:
            return {}
    else:
        y = sample[target].to_numpy(dtype=float)
        if not np.isfinite(y).all() or np.ptp(y) == 0:
            return {}

    splits = list(KFold(folds, shuffle=True, random_state=SEED).split(np.zeros(len(y))))
    leaf = max(5, len(y) // 100)
    scores = {}
    for name in columns:
        predicted = _out_of_fold(sample[name], y, task, splits, tree_depth, leaf)
        if predicted is not None:
            scores[name] = _skill(y, predicted, task)
    return scores


def _out_of_fold(
    column: pd.Series,
    y: np.ndarray,
    task: str,
    splits: list[tuple[np.ndarray, np.ndarray]],
    tree_depth: int,
    leaf: int,
) -> np.ndarray | None:
    if pdt.is_numeric_dtype(column.dtype) or pdt.is_bool_dtype(column.dtype):
        x = column.to_numpy(dtype=float)
        x = np.where(np.isfinite(x), x, np.nan)
        finite = x[~np.isnan(x)]
        if finite.size == 0 or np.ptp(finite) == 0:
            return None
        # Missing values go below every real one, so a tree can split on "is missing".
        x = np.where(np.isnan(x), finite.min() - 1, x).reshape(-1, 1)
        predictor = _tree_predictor(x, task, tree_depth, leaf)
    else:
        labels = column.astype(object).where(column.notna(), _MISSING_CATEGORY)
        codes, uniques = pd.factorize(labels)
        if len(uniques) < 2:
            return None
        n_classes = int(y.max()) + 1 if task == "classification" else 0
        predictor = _lookup_predictor(codes, len(uniques), task, n_classes)

    out = np.empty(len(y), dtype=float)
    for fit, test in splits:
        out[test] = predictor(y, fit, test)
    return out


def _tree_predictor(x: np.ndarray, task: str, depth: int, leaf: int):
    def predict(y, fit, test):
        if task == "classification":
            tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, random_state=SEED)
        else:
            tree = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=leaf, random_state=SEED)
        return tree.fit(x[fit], y[fit]).predict(x[test])

    return predict


def _lookup_predictor(codes: np.ndarray, n_codes: int, task: str, n_classes: int):
    def predict(y, fit, test):
        if task == "classification":
            counts = np.bincount(
                codes[fit] * n_classes + y[fit], minlength=n_codes * n_classes
            ).reshape(n_codes, n_classes)
            majority = np.bincount(y[fit], minlength=n_classes).argmax()
            table = np.where(counts.sum(axis=1) > 0, counts.argmax(axis=1), majority)
        else:
            sums = np.bincount(codes[fit], weights=y[fit], minlength=n_codes)
            seen = np.bincount(codes[fit], minlength=n_codes)
            table = np.where(seen > 0, sums / np.maximum(seen, 1), y[fit].mean())
        return table[codes[test]]

    return predict


def _skill(y: np.ndarray, predicted: np.ndarray, task: str) -> float:
    if task == "regression":
        total = float(((y - y.mean()) ** 2).sum())
        if total == 0:
            return 0.0
        skill = 1 - float(((y - predicted) ** 2).sum()) / total
    else:
        classes = np.unique(y)
        recalls = [float((predicted[y == c] == c).mean()) for c in classes]
        chance = 1 / len(classes)
        skill = (float(np.mean(recalls)) - chance) / (1 - chance)
    return min(max(skill, 0.0), 1.0)


def find_duplicates(
    df: pd.DataFrame, columns: list[str], *, min_abs_corr: float, corr_rows: int
) -> dict[str, str]:
    """Map each column of `columns` that copies an earlier one to that original.

    `columns` is in file order and holds the columns worth comparing (the caller leaves out the
    target and the constant ones). A column is compared only with originals that have exactly the
    same missing values, and it is a copy if one of these holds, tried in this order:

    - its values are equal to the original's;
    - both are numeric and the absolute Pearson correlation is at least `min_abs_corr`, on a
      fixed sample of at most `corr_rows` rows, which is a scaled and shifted copy up to rounding;
    - values repeat (fewer distinct values than half of the non-missing ones) and the two
      columns match one to one: every value of one goes with exactly one value of the other. The
      repeat condition matters, since two independent columns with a value for every row also
      "match one to one" without having anything to do with each other.

    A copy of a copy is mapped to the first original.
    """
    n = len(df)
    sample = np.sort(np.random.default_rng(SEED).choice(n, min(n, corr_rows), replace=False))
    originals: dict[bytes, list[str]] = {}
    found: dict[str, str] = {}
    for name in columns:
        column = df[name]
        mask = column.isna().to_numpy()
        bucket = originals.setdefault(mask.tobytes(), [])
        for original in bucket:
            if _is_copy(df[original], column, mask, sample, min_abs_corr):
                found[name] = original
                break
        else:
            bucket.append(name)
    return found


def _is_copy(
    original: pd.Series,
    column: pd.Series,
    mask: np.ndarray,
    sample: np.ndarray,
    min_abs_corr: float,
) -> bool:
    if original.equals(column):
        return True
    both_numeric = all(
        pdt.is_numeric_dtype(s.dtype) and not pdt.is_bool_dtype(s.dtype) for s in (original, column)
    )
    if both_numeric:
        a = original.to_numpy(dtype=float)[sample]
        b = column.to_numpy(dtype=float)[sample]
        usable = np.isfinite(a) & np.isfinite(b)
        if usable.sum() >= 3 and a[usable].std() > 0 and b[usable].std() > 0:
            if abs(np.corrcoef(a[usable], b[usable])[0, 1]) >= min_abs_corr:
                return True
    present = ~mask
    a_codes, a_uniques = pd.factorize(original[present])
    b_codes, b_uniques = pd.factorize(column[present])
    n_values = len(a_uniques)
    if n_values != len(b_uniques) or n_values < 2 or n_values > present.sum() / 2:
        return False
    pairs = np.unique(a_codes.astype(np.int64) * n_values + b_codes)
    return len(pairs) == n_values
