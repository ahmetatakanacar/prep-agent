"""Deterministic dataset profiler.

The profile is the only view of the data the planner (LLM) ever gets, so it has to carry
everything needed to make preparation decisions while exposing as little raw data as possible:
summary statistics, a few truncated sample values and top-value counts.

Type inference rules, applied to the non-missing values of a column in this order. Every
threshold lives in `ProfilerConfig`.

1. `constant`: at most one distinct value (this includes all-missing columns).
2. `datetime`: a datetime dtype, or strings that parse as dates (see rule 5).
3. `binary`: exactly two distinct non-missing values. A bool dtype is binary; a numeric column
   is binary only if its values are exactly {0, 1} (two-valued numerics like {1, 5} stay
   `numeric`); any other column with two distinct values is binary (yes/no, male/female, ...).
4. Numeric dtypes are `id_like` or `numeric`. Only integer dtypes can be `id_like`: every value
   is distinct, there are at least `id_min_rows` of them, and the values either form an almost
   contiguous range (span <= `id_dense_span_ratio` x row count, like 1..n) or the column name
   looks like an identifier ("id", "user_id", "orderId"). A unique integer column such as a price
   is therefore kept as `numeric`. Float columns are never `id_like`.
5. Other columns are treated as strings:
   - `datetime` if at least `datetime_min_parse_ratio` of a sample (at most
     `datetime_sample_size` values) parse as dates. A value is only tried if it has at least
     two digits and one non-digit character, so month names ("May"), zip codes ("90210") and
     other bare numbers are not mistaken for dates.
   - `id_like` if every value is distinct, there are at least `id_min_rows` of them and the mean
     length is below `text_min_mean_length` (short unique strings).
   - `text` if the mean length is at least `text_min_mean_length` and at least
     `text_min_unique_ratio` of the values are distinct (long, mostly unique strings).
   - `categorical` otherwise.

Warning `low_cardinality_numeric`: a `numeric` column (so not binary, not id_like and not
constant) whose finite values are all whole numbers and which has at most
`ProfilerConfig.low_cardinality_unique` distinct values, fewer than
`low_cardinality_max_unique_ratio` of its non-missing values. The second condition keeps out
small tables, where every whole-number column such as an age has few distinct values only
because it has few rows. It is only a hint: the numbers may be categories stored as numbers
(hour, month, a type code) or a real count (rooms, children), and the profile cannot tell which.

Privacy: a column is flagged `possible_pii` if most of its string values look like email
addresses or phone numbers, or if a part of its name (split on `_`, spaces and camelCase) is in
`ProfilerConfig.pii_name_parts` and more than `pii_name_min_unique_ratio` of its values are
distinct. The second condition keeps out columns that merely have such a name, such as a yes/no
`PhoneService` or a `department_name` with a handful of categories, which hold no identities.
For a numeric column only the parts in `ProfilerConfig.pii_numeric_name_parts` count (phone,
telefon, gsm, tckn, iban, ssn: numbers that identify a person, like a phone number stored as an
integer), under the same uniqueness condition. Short parts such as "ad" or "tel" are everyday
words in numeric names (`ad_spend`, `tel_count`) and would hide real statistics.
Such columns get no `sample_values`, no `top_values` and no
numeric statistics (min and max would expose real values); counts and ratios are kept. This
applies whether or not samples are disabled.

`unique_ratio` and the other ratios of a column use its non-missing values as denominator,
except `missing_ratio`, which uses all rows. Numeric statistics ignore missing and infinite
values. Outliers follow the IQR rule: values outside [q1 - k * IQR, q3 + k * IQR].
"""

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pandas.api import types as pdt
from pydantic import BaseModel

InferredType = Literal[
    "numeric", "categorical", "binary", "datetime", "text", "id_like", "constant"
]
TaskHint = Literal["classification", "regression"]


# The strings read as missing: pandas' default list without "None". "None" is a real category
# in many datasets (no pool, no alley, no veneer) and reading it as missing would turn "this
# feature is absent" into "unknown". The generated pipeline.py reads its input with this list
# too, so a CSV file gives the same frame in both.
NA_VALUES: tuple[str, ...] = (
    "", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN",
    "<NA>", "N/A", "NA", "NULL", "NaN", "n/a", "nan", "null",
)  # fmt: skip


class ProfilerError(ValueError):
    """Raised when a dataset cannot be loaded or profiled."""


@dataclass(frozen=True)
class ProfilerConfig:
    """Thresholds used by the profiler. See the module docstring for how they are applied."""

    # Output size limits
    max_sample_values: int = 5
    max_sample_chars: int = 50
    max_top_values: int = 10

    # Type inference
    id_min_rows: int = 10
    id_dense_span_ratio: float = 1.1
    text_min_mean_length: float = 30.0
    text_min_unique_ratio: float = 0.5
    datetime_sample_size: int = 2000
    datetime_min_parse_ratio: float = 0.9
    iqr_multiplier: float = 1.5

    # Warnings
    high_missing_ratio: float = 0.4
    high_cardinality_unique: int = 50
    low_cardinality_unique: int = 25
    low_cardinality_max_unique_ratio: float = 0.5
    high_skew_abs: float = 2.0
    many_outliers_ratio: float = 0.05

    # Privacy: a column is flagged `possible_pii` if at least this share of its string values
    # looks like an email address or phone number, or if a part of its name is in the list and
    # more than `pii_name_min_unique_ratio` of its values are distinct.
    pii_value_min_ratio: float = 0.8
    pii_name_min_unique_ratio: float = 0.5
    pii_name_parts: tuple[str, ...] = (
        "email", "mail", "phone", "telefon", "tel", "gsm", "name", "firstname", "lastname",
        "surname", "ad", "soyad", "isim", "address", "adres", "tckn", "iban", "ssn",
        "username", "fullname", "adi", "adı", "soyadi", "soyadı",
    )  # fmt: skip
    # The parts of `pii_name_parts` that can name a number. A numeric column is only flagged for
    # one of these.
    pii_numeric_name_parts: tuple[str, ...] = ("phone", "telefon", "gsm", "tckn", "iban", "ssn")

    # Task hint: a numeric target with more distinct values than this looks like regression.
    regression_min_unique: int = 20


DEFAULT_CONFIG = ProfilerConfig()

_CAMEL_ID_NAME = re.compile(r"[a-z](Id|ID)$")
_SNAKE_ID_NAME = re.compile(r"(^|[^a-z0-9])(id|uuid|guid)($|[^a-z0-9])", re.IGNORECASE)
_EMAIL = r"[^@\s]+@[^@\s]+\.[^@\s]+"
_PHONE = r"\+?[\d\s().-]{7,20}"
_PHONE_MIN_DIGITS = 7
_PHONE_MAX_DIGITS = 15


class TopValue(BaseModel):
    value: str
    count: int


class ColumnProfile(BaseModel):
    name: str
    pandas_dtype: str
    inferred_type: InferredType
    n_missing: int
    missing_ratio: float
    n_unique: int
    unique_ratio: float
    sample_values: list[str]
    warnings: list[str]

    # Numeric columns
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    std: float | None = None
    median: float | None = None
    q1: float | None = None
    q3: float | None = None
    skew: float | None = None
    n_outliers_iqr: int | None = None

    # Categorical and binary columns
    top_values: list[TopValue] | None = None

    # Text columns
    mean_length: float | None = None
    max_length: int | None = None


class DatasetProfile(BaseModel):
    n_rows: int
    n_columns: int
    n_duplicate_rows: int
    duplicate_ratio: float
    columns: list[ColumnProfile]
    target: str | None = None
    target_task_hint: TaskHint | None = None


TARGET_CONSTANT_MESSAGE = "the target has only one value, so there is nothing to predict"


def require_varying_target(profile: DatasetProfile) -> None:
    """Raise `ProfilerError` if the target the user named has a single distinct value.

    Missing values do not count as a value. Callers use this on a target that came from the user,
    so that no planner call is spent on a task that cannot work.
    """
    target = next((c for c in profile.columns if c.name == profile.target), None)
    if target is not None and target.n_unique == 1:
        raise ProfilerError(f"{TARGET_CONSTANT_MESSAGE}: column {target.name!r}")


def load_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV file, raising `ProfilerError` with a readable message on failure.

    Missing values are the strings in `NA_VALUES`; the text "None" is not one of them.
    """
    path = Path(path)
    if not path.exists():
        raise ProfilerError(f"CSV file not found: {path}")
    if not path.is_file():
        raise ProfilerError(f"not a file: {path}")
    try:
        df = pd.read_csv(path, low_memory=False, keep_default_na=False, na_values=list(NA_VALUES))
    except pd.errors.EmptyDataError:
        raise ProfilerError(f"CSV file is empty: {path}") from None
    except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
        raise ProfilerError(f"could not read CSV file {path}: {exc}") from exc
    if df.empty:
        raise ProfilerError(f"CSV file has no data rows: {path}")
    return df


def profile_dataframe(
    df: pd.DataFrame,
    target: str | None = None,
    config: ProfilerConfig = DEFAULT_CONFIG,
    include_samples: bool = True,
) -> DatasetProfile:
    """Profile every column of `df`. The input is not modified.

    With `include_samples=False`, no column gets `sample_values` or `top_values`.
    """
    if df.shape[1] == 0:
        raise ProfilerError("DataFrame is empty (no columns)")
    if df.shape[0] == 0:
        raise ProfilerError("DataFrame is empty (no rows)")
    duplicated_names = sorted({str(c) for c in df.columns[df.columns.duplicated()]})
    if duplicated_names:
        raise ProfilerError(f"duplicate column names: {duplicated_names}")
    names = [str(c) for c in df.columns]
    if target is not None and target not in names:
        raise ProfilerError(f"target column '{target}' not found; available columns: {names}")

    columns = [
        _profile_column(str(label), df[label], config, include_samples) for label in df.columns
    ]
    n_rows = int(df.shape[0])
    n_duplicate_rows = int(df.duplicated().sum())

    hint = None
    if target is not None:
        hint = _task_hint(columns[names.index(target)], config)
    return DatasetProfile(
        n_rows=n_rows,
        n_columns=int(df.shape[1]),
        n_duplicate_rows=n_duplicate_rows,
        duplicate_ratio=n_duplicate_rows / n_rows,
        columns=columns,
        target=target,
        target_task_hint=hint,
    )


def _profile_column(
    name: str, series: pd.Series, config: ProfilerConfig, include_samples: bool
) -> ColumnProfile:
    n_rows = len(series)
    non_null = series.dropna()
    n_non_null = len(non_null)
    n_unique = int(non_null.nunique())
    n_missing = n_rows - n_non_null
    missing_ratio = n_missing / n_rows
    inferred = _infer_type(name, non_null, n_unique, config)

    fields: dict = {}
    warnings: list[str] = []
    if missing_ratio >= config.high_missing_ratio:
        warnings.append("high_missing")
    if inferred == "constant":
        warnings.append("constant")
    if inferred == "categorical" and n_unique > config.high_cardinality_unique:
        warnings.append("high_cardinality")
    if inferred == "id_like":
        warnings.append("possible_id")
    possible_pii = _looks_like_pii(name, non_null, n_unique, inferred, config)
    if possible_pii:
        warnings.append("possible_pii")
    hide_values = possible_pii or not include_samples

    if inferred == "numeric" and not possible_pii:
        stats, n_inf = _numeric_stats(non_null, config)
        fields.update(stats)
        skew = fields.get("skew")
        if skew is not None and abs(skew) > config.high_skew_abs:
            warnings.append("high_skew")
        n_finite = n_non_null - n_inf
        if n_finite and fields["n_outliers_iqr"] / n_finite > config.many_outliers_ratio:
            warnings.append("many_outliers")
        if n_inf:
            warnings.append("infinite_values")
        if (
            n_unique <= config.low_cardinality_unique
            and n_unique < config.low_cardinality_max_unique_ratio * n_non_null
            and _all_whole_numbers(non_null)
        ):
            warnings.append("low_cardinality_numeric")
    elif inferred in ("categorical", "binary"):
        fields["top_values"] = [] if hide_values else _top_values(non_null, config)
    elif inferred == "text":
        lengths = non_null.astype(str).str.len()
        fields["mean_length"] = float(lengths.mean())
        fields["max_length"] = int(lengths.max())

    return ColumnProfile(
        name=name,
        pandas_dtype=str(series.dtype),
        inferred_type=inferred,
        n_missing=n_missing,
        missing_ratio=missing_ratio,
        n_unique=n_unique,
        unique_ratio=n_unique / n_non_null if n_non_null else 0.0,
        sample_values=[] if hide_values else _sample_values(non_null, config),
        warnings=warnings,
        **fields,
    )


def _infer_type(
    name: str, non_null: pd.Series, n_unique: int, config: ProfilerConfig
) -> InferredType:
    if n_unique <= 1:
        return "constant"
    dtype = non_null.dtype
    if pdt.is_datetime64_any_dtype(dtype):
        return "datetime"
    if pdt.is_bool_dtype(dtype):
        return "binary"
    n = len(non_null)
    if pdt.is_numeric_dtype(dtype):
        if n_unique == 2 and set(non_null.unique().tolist()) == {0, 1}:
            return "binary"
        if pdt.is_integer_dtype(dtype) and _is_id_like_integer(name, non_null, n_unique, config):
            return "id_like"
        return "numeric"
    if n_unique == 2:
        return "binary"

    strings = non_null.astype(str)
    if _looks_like_datetime(strings, config):
        return "datetime"
    mean_length = float(strings.str.len().mean())
    if n_unique == n and n >= config.id_min_rows and mean_length < config.text_min_mean_length:
        return "id_like"
    if mean_length >= config.text_min_mean_length and (
        n_unique / n >= config.text_min_unique_ratio
    ):
        return "text"
    return "categorical"


def _is_id_like_integer(
    name: str, non_null: pd.Series, n_unique: int, config: ProfilerConfig
) -> bool:
    n = len(non_null)
    if n < config.id_min_rows or n_unique != n:
        return False
    if _looks_like_id_name(name):
        return True
    span = int(non_null.max()) - int(non_null.min()) + 1
    return span <= config.id_dense_span_ratio * n


def _looks_like_id_name(name: str) -> bool:
    return bool(_SNAKE_ID_NAME.search(name) or _CAMEL_ID_NAME.search(name))


def _is_date_candidate(value: str) -> bool:
    n_digits = sum(ch.isdigit() for ch in value)
    return n_digits >= 2 and n_digits < len(value)


def _looks_like_datetime(strings: pd.Series, config: ProfilerConfig) -> bool:
    if len(strings) > config.datetime_sample_size:
        strings = strings.sample(config.datetime_sample_size, random_state=0)
    candidates = strings[strings.map(_is_date_candidate).astype(bool)]
    if len(candidates) < config.datetime_min_parse_ratio * len(strings):
        return False
    parsed = pd.to_datetime(candidates, format="mixed", errors="coerce", utc=True)
    return parsed.notna().sum() >= config.datetime_min_parse_ratio * len(strings)


def _looks_like_pii(
    name: str, non_null: pd.Series, n_unique: int, inferred: InferredType, config: ProfilerConfig
) -> bool:
    is_numeric = pdt.is_numeric_dtype(non_null.dtype)
    name_parts = config.pii_numeric_name_parts if is_numeric else config.pii_name_parts
    name_matches = not set(_name_parts(name)).isdisjoint(name_parts)
    if name_matches and n_unique > config.pii_name_min_unique_ratio * len(non_null):
        return True
    if inferred == "datetime" or non_null.empty or is_numeric:
        return False
    strings = non_null.astype(str).str.strip()
    is_email = strings.str.fullmatch(_EMAIL)
    n_digits = strings.str.count(r"\d")
    is_phone = strings.str.fullmatch(_PHONE) & n_digits.between(
        _PHONE_MIN_DIGITS, _PHONE_MAX_DIGITS
    )
    return bool((is_email | is_phone).mean() >= config.pii_value_min_ratio)


def _name_parts(name: str) -> list[str]:
    """Split a column name on separators, camelCase and letter/digit boundaries, lowercased."""
    parts = []
    for chunk in re.split(r"[\W_]+", name):
        start = 0
        for i in range(1, len(chunk)):
            prev, cur = chunk[i - 1], chunk[i]
            following = chunk[i + 1] if i + 1 < len(chunk) else ""
            if (
                (prev.islower() and cur.isupper())
                or (prev.isupper() and cur.isupper() and following.islower())
                or prev.isdigit() != cur.isdigit()
            ):
                parts.append(chunk[start:i])
                start = i
        parts.append(chunk[start:])
    # The dotted capital I of Turkish lowercases to two code points; map it to plain "i" first.
    return [part.replace("İ", "i").lower() for part in parts if part]


def _numeric_stats(non_null: pd.Series, config: ProfilerConfig) -> tuple[dict, int]:
    """Return the numeric statistics (finite values only) and the number of infinite values."""
    values = non_null.to_numpy(dtype="float64")
    finite = values[np.isfinite(values)]
    n_inf = len(values) - len(finite)
    if finite.size == 0:
        return {}, n_inf

    q1, median, q3 = (float(q) for q in np.percentile(finite, [25, 50, 75]))
    iqr = q3 - q1
    low = q1 - config.iqr_multiplier * iqr
    high = q3 + config.iqr_multiplier * iqr
    stats = {
        "min": _finite_or_none(finite.min()),
        "max": _finite_or_none(finite.max()),
        "mean": _finite_or_none(finite.mean()),
        "std": _finite_or_none(finite.std(ddof=1)) if finite.size > 1 else None,
        "median": _finite_or_none(median),
        "q1": _finite_or_none(q1),
        "q3": _finite_or_none(q3),
        "skew": _finite_or_none(pd.Series(finite).skew()),
        "n_outliers_iqr": int(np.count_nonzero((finite < low) | (finite > high))),
    }
    return stats, n_inf


def _all_whole_numbers(non_null: pd.Series) -> bool:
    values = non_null.to_numpy(dtype="float64")
    return bool(np.isfinite(values).all() and (values == np.floor(values)).all())


def _top_values(non_null: pd.Series, config: ProfilerConfig) -> list[TopValue]:
    counts = [(str(value), int(count)) for value, count in non_null.value_counts().items()]
    counts.sort(key=lambda item: (-item[1], item[0]))
    return [
        TopValue(value=_truncate(value, config), count=count)
        for value, count in counts[: config.max_top_values]
    ]


def _sample_values(non_null: pd.Series, config: ProfilerConfig) -> list[str]:
    unique = non_null.drop_duplicates().head(config.max_sample_values)
    return [_truncate(str(value), config) for value in unique.tolist()]


def _truncate(value: str, config: ProfilerConfig) -> str:
    return value[: config.max_sample_chars]


def _finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _task_hint(column: ColumnProfile, config: ProfilerConfig) -> TaskHint | None:
    if column.inferred_type in ("binary", "categorical"):
        return "classification"
    if column.inferred_type == "numeric":
        return "regression" if column.n_unique > config.regression_min_unique else "classification"
    return None
