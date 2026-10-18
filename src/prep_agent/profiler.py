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
   looks like an identifier ("id", "user_id", "orderId"). A column whose name has an identifier
   part (`ProfilerConfig.id_name_parts`: id, no, numara, numarasi, ...) is also `id_like` when
   more than `id_name_min_unique_ratio` of its values are distinct, because a few repeated
   records ("basvuru_no" in a file with duplicate rows) must not hide it. A unique integer column
   such as a price is therefore kept as `numeric`, and so is a code with few values
   ("sube_kodu"). Float columns are never `id_like`.
5. Other columns are treated as strings:
   - `datetime` if at least `datetime_min_parse_ratio` of a sample (at most
     `datetime_sample_size` values) parse as dates. A value is only tried if it has at least
     two digits and one non-digit character, so month names ("May"), zip codes ("90210") and
     other bare numbers are not mistaken for dates.
   - `id_like` if every value is a UUID, or a hexadecimal hash of at least `id_hash_min_length`
     characters (an md5 or sha digest), and every value is distinct. Such a value is long, so
     the rule on short strings below would call the column `text`, but it is an identifier by its
     form alone. A random hash holds a letter a-f in nearly every value, which is how a long
     number (a card number, a phone number) is kept out; see `id_hash_min_letter_ratio`.
   - `id_like` if every value is distinct (or almost, for an identifier name as in rule 4),
     there are at least `id_min_rows` of them and the mean length is below
     `text_min_mean_length` (short unique strings).
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
A non-numeric column named for a person (`ProfilerConfig.pii_person_name_parts`: soyad, ad_soyad,
firstname, lastname, surname, fullname, ...) is flagged whatever its unique ratio, since repeated
surnames are still identities.
For a numeric column only the parts in `ProfilerConfig.pii_numeric_name_parts` count (phone,
telefon, gsm, tckn, iban, ssn: numbers that identify a person, like a phone number stored as an
integer), under the same uniqueness condition. Short parts such as "ad" or "tel" are everyday
words in numeric names (`ad_spend`, `tel_count`) and would hide real statistics.
Such columns get no `sample_values`, no `top_values` and no
numeric statistics (min and max would expose real values); counts and ratios are kept. This
applies whether or not samples are disabled.

Target association: with a target, every numeric, categorical or binary column gets
`target_association`, the share of the possible improvement over the trivial predictor that a
model seeing only that column reaches on rows it was not fitted on (see `prep_agent.signals`).
A column at or above `leakage_min_score` also gets the warning `suspected_target_leakage`: it
nearly determines the target alone. A column from `high_association_min_score` up to that gets
`high_target_association`. Neither is a verdict, and neither can be: the score only says how well
one column predicts the target, and a column that is a plain fact known before the outcome (the
floor area of a house, the odor of a mushroom) can predict it as well as a column that exists
only because of the outcome. What separates them is what the column means, which the profile
cannot see; the marks tell the planner where to look and to ask.

A column can also give the target away by having a value or not, with the values themselves
useless. A column with more than `leakage_min_missing_ratio` of its rows missing therefore also
gets `missing_association`: the same score, computed the same way, for the 0/1 marker "is
missing". It is judged by the same thresholds, and `leakage_via` says what reached the lower
bound of the tier the column is in: `value`, `missingness` or `both` (a numeric or categorical
column that is missing exactly for one class
is reached by both, since the value score already sees missing values as a value of their own;
only a column of a kind the value score leaves out, such as text or dates, is reached by
missingness alone). Like the value score, the marker must nearly determine the target in both
directions: a column that is always missing for one class and missing for part of the other too
does not reach the threshold.

This is computed on all the rows before any split, with the target, so the choice of what to
drop is informed by labels of rows that will land in the test set. The risk is small and bounded,
and the design keeps it that way: the score is out of fold, so it does not reward fitting;
it is shown rounded to two decimals; it only ever causes a column to be dropped, never picks the
best columns, so no feature is selected for how well it scores; and at most `leakage_max_rows`
rows (a fixed random sample) are used. It cannot be avoided altogether: the split is made later,
by the executor, and the profile cannot know which rows it will hold out.

Duplicate columns: a column that copies an earlier one gets `duplicate_of` (the name of the
earlier column) and the warning `duplicate_column`. See `prep_agent.signals.find_duplicates` for
what counts as a copy.

`unique_ratio` and the other ratios of a column use its non-missing values as denominator,
except `missing_ratio`, which uses all rows. Numeric statistics ignore missing and infinite
values. Outliers follow the IQR rule: values outside [q1 - k * IQR, q3 + k * IQR].
"""

import codecs
import csv
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pandas.api import types as pdt
from pydantic import BaseModel, ConfigDict

from prep_agent.signals import find_duplicates, target_association

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
    # A column whose name has one of these parts (`id_name_parts`, same matching as the PII names)
    # is `id_like` if more than this share of its values are distinct, so that a few repeated
    # records do not hide an identifier.
    id_name_min_unique_ratio: float = 0.95
    id_name_parts: tuple[str, ...] = ("id", "uuid", "guid", "no", "numara", "numarasi")
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
    # more than `pii_name_min_unique_ratio` of its values are distinct. Names are compared without
    # Turkish letters (see `_name_parts`), so the list holds the ASCII spelling only. An entry with
    # an underscore is several parts in a row.
    pii_value_min_ratio: float = 0.8
    pii_name_min_unique_ratio: float = 0.5
    pii_name_parts: tuple[str, ...] = (
        "email", "mail", "phone", "telefon", "tel", "gsm", "name", "firstname", "lastname",
        "surname", "ad", "soyad", "isim", "address", "adres", "tckn", "iban", "ssn",
        "username", "fullname", "adi", "soyadi", "adsoyad", "eposta", "e_posta", "cep", "tc",
        "tc_kimlik", "kimlik_no", "dogum_tarihi",
    )  # fmt: skip
    # Names of people. A non-numeric column with one of these is flagged however few distinct
    # values it has: a surname column holds identities even when the names repeat. The general
    # parts above ("name", "ad", "isim") also name things (`department_name`, `il_adi`), so they
    # need the unique ratio.
    pii_person_name_parts: tuple[str, ...] = (
        "soyad", "soyadi", "adsoyad", "ad_soyad", "firstname", "first_name", "lastname",
        "last_name", "surname", "fullname", "full_name",
    )  # fmt: skip
    # The parts of `pii_name_parts` that can name a number. A numeric column is only flagged for
    # one of these.
    pii_numeric_name_parts: tuple[str, ...] = ("phone", "telefon", "gsm", "tckn", "iban", "ssn")

    # Task hint: a numeric target with more distinct values than this looks like regression.
    regression_min_unique: int = 20

    # Hash-shaped identifiers (UUIDs, md5/sha digests). 16 hexadecimal characters are 64 bits:
    # a shorter string can be a plain code, a longer one that is unique per row is not a word
    # or a category. A random hex string is all digits with probability (10/16)^length, under
    # 0.05% at 16, so almost every value holds a letter; requiring it of 90% of the values keeps
    # out a column of long numbers (phone or card numbers stored as text).
    id_hash_min_length: int = 16
    id_hash_min_letter_ratio: float = 0.9

    # Target association (see prep_agent.signals). The thresholds follow from what the score
    # means, not from any dataset.
    # - 0.9: the column alone reaches at least 90% of the possible improvement over the trivial
    #   predictor, so at most a tenth is left to every other column together. A column that is
    #   merely a strong predictor leaves a lot more than that to the rest; one that is a
    #   (noisy) function of the target, or of something that follows it, does not.
    # - 20,000 rows: five tree fits per column stay cheap, and the sampling error of a score on
    #   that many rows is small next to the distance between 0.9 and the scores of ordinary
    #   predictors. Fewer rows are used whole.
    # - 100 rows: under that, each of the five folds holds out about 20 rows, too few for a
    #   score out of fold to mean anything (the baseline warns under 100 train rows too).
    # - 5 folds, like the cross-validation of the baseline.
    # - depth 4: 16 leaves follow a stepped or curved relation and separate a few groups of
    #   codes along the axis, and are too few to memorise noise.
    leakage_min_score: float = 0.9
    # - 0.6 for the second tier: from here a column alone reaches more than half of the possible
    #   improvement, which an ordinary predictor in a table with several useful columns does
    #   not usually do, and which is enough to be worth a look. The tier is a call for attention
    #   and a basis for a warning, not for dropping: unlike the 0.9 tier, nothing is dropped on
    #   it alone (see the planner's rules for what happens when it may not ask).
    high_association_min_score: float = 0.6
    # - 5% missing: a column can give the target away by whether it has a value at all (a field
    #   filled in only after the outcome). The same score and the same threshold judge the 0/1
    #   marker "is missing" as a column. Under 5% of the rows the marker has so few ones that
    #   the score out of fold rests on a handful of rows; at the 100-row minimum, 5% is five.
    leakage_min_missing_ratio: float = 0.05
    leakage_max_rows: int = 20_000
    leakage_min_rows: int = 100
    leakage_folds: int = 5
    leakage_tree_depth: int = 4

    # Duplicate columns (see prep_agent.signals.find_duplicates).
    # - 0.9999: a unit conversion, even rounded to four decimals, leaves the correlation within
    #   about 1e-4 of 1, and at 0.9999 less than 0.02% of the variance is not shared. Two
    #   measurements of the same thing that merely move together stay well below it.
    # - 5,000 rows: a copy agrees on any subset of rows, so a sample decides as well as the
    #   whole column and bounds the cost of comparing many columns.
    duplicate_min_abs_corr: float = 0.9999
    duplicate_corr_rows: int = 5_000


DEFAULT_CONFIG = ProfilerConfig()

_UUID = r"[{]?[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}[}]?"
_CAMEL_ID_NAME = re.compile(r"[a-z](Id|ID)$")
_SNAKE_ID_NAME = re.compile(r"(^|[^a-z0-9])(id|uuid|guid)($|[^a-z0-9])", re.IGNORECASE)
_ASCII_TURKISH = str.maketrans("ıİşŞğĞüÜöÖçÇ", "iIsSgGuUoOcC")
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

    # Relations to other columns. `target_association` is set only when the profile has a target
    # and enough rows; `missing_association` is the same score for the marker "is missing", only
    # for a column with enough missing values; `leakage_via` says which of the two reached the
    # lower bound of the tier (suspected leakage or high association) the column is in.
    # `duplicate_of` names the earlier column this one copies.
    target_association: float | None = None
    missing_association: float | None = None
    leakage_via: Literal["value", "missingness", "both"] | None = None
    duplicate_of: str | None = None


class CsvOptions(BaseModel):
    """How a CSV file is read. `load_csv` finds them; the generated pipeline.py reads with them."""

    model_config = ConfigDict(frozen=True)

    sep: str = ","
    decimal: str = "."
    encoding: str = "utf-8"

    def read_kwargs(self) -> dict[str, str]:
        return {"sep": self.sep, "decimal": self.decimal, "encoding": self.encoding}


class DatasetProfile(BaseModel):
    n_rows: int
    n_columns: int
    n_duplicate_rows: int
    duplicate_ratio: float
    columns: list[ColumnProfile]
    target: str | None = None
    target_task_hint: TaskHint | None = None
    # How the file the profile comes from was read; None for a frame that did not come from a file.
    csv: CsvOptions | None = None


def constant_target_message(column: ColumnProfile) -> str | None:
    """Why a target column cannot be predicted, or None if it has two or more distinct values.

    Missing values do not count as a value.
    """
    if column.n_unique == 0:
        return "the target has no values, so there is nothing to predict"
    if column.n_unique == 1:
        return "the target has only one value, so there is nothing to predict"
    return None


def require_varying_target(profile: DatasetProfile) -> None:
    """Raise `ProfilerError` if the target the user named has at most one distinct value.

    Callers use this on a target that came from the user, so that no planner call is spent on a
    task that cannot work.
    """
    target = next((c for c in profile.columns if c.name == profile.target), None)
    message = constant_target_message(target) if target is not None else None
    if message is not None:
        raise ProfilerError(f"{message}: column {target.name!r}")


@dataclass(frozen=True)
class LoadedCsv:
    """A frame with the options it was read with and notes the user should see."""

    df: pd.DataFrame
    options: CsvOptions
    warnings: list[str] = field(default_factory=list)
    # The file was not UTF-8 and was read as `FALLBACK_ENCODING`; `warnings` says so in words.
    encoding_fallback: bool = False


# The encoding tried when a file is not UTF-8: the Windows code page of Turkish text.
FALLBACK_ENCODING = "cp1254"
# What is looked at to find the separator. A few KB hold enough rows without reading a large file
# twice.
_SNIFF_BYTES = 8192
_SEPARATORS = ",;\t|"
_DECODE_CHUNK = 1 << 20


def load_csv(
    path: str | Path,
    *,
    sep: str | None = None,
    decimal: str | None = None,
    encoding: str | None = None,
) -> LoadedCsv:
    """Read a CSV file, raising `ProfilerError` with a readable message on failure.

    Whatever is not given is found. Encoding: UTF-8 (a byte order mark is fine), else cp1254 with
    a warning. Separator: `csv.Sniffer` over the first few KB, among `,` `;` tab and `|`, else
    `,`. Decimal mark: `,` for a `;` file, else `.`. The options used are in the result.

    Missing values are the strings in `NA_VALUES`; the text "None" is not one of them.
    """
    path = Path(path)
    if not path.exists():
        raise ProfilerError(f"CSV file not found: {path}")
    if not path.is_file():
        raise ProfilerError(f"not a file: {path}")

    warnings: list[str] = []
    try:
        if encoding is None:
            encoding = _detect_encoding(path, warnings)
        else:
            _check_codec(encoding)
        sep = _detect_separator(path, encoding) if sep is None else _check_mark(sep, "separator")
        decimal = (
            _check_mark(decimal, "decimal mark")
            if decimal is not None
            else ("," if sep == ";" else ".")
        )
    except OSError as exc:
        raise ProfilerError(f"could not read CSV file {path}: {exc}") from exc

    options = CsvOptions(sep=sep, decimal=decimal, encoding=encoding)
    try:
        df = pd.read_csv(
            path,
            low_memory=False,
            keep_default_na=False,
            na_values=list(NA_VALUES),
            **options.read_kwargs(),
        )
    except pd.errors.EmptyDataError:
        raise ProfilerError(f"CSV file is empty: {path}") from None
    except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
        raise ProfilerError(f"could not read CSV file {path}: {exc}") from exc
    if df.empty:
        raise ProfilerError(f"CSV file has no data rows: {path}")
    return LoadedCsv(df, options, warnings, encoding_fallback=bool(warnings))


def _check_codec(encoding: str) -> None:
    try:
        codecs.lookup(encoding)
    except LookupError:
        raise ProfilerError(f"unknown encoding {encoding!r}") from None


def _check_mark(value: str, what: str) -> str:
    # `--sep "\t"` is how a tab is typed in a shell.
    if value == "\\t":
        value = "\t"
    if len(value) != 1:
        raise ProfilerError(f"the {what} must be a single character, got {value!r}")
    return value


def _detect_encoding(path: Path, warnings: list[str]) -> str:
    if _decodes(path, "utf-8"):
        return "utf-8"
    if _decodes(path, FALLBACK_ENCODING):
        warnings.append(
            f"the file is not valid UTF-8, so it was read as {FALLBACK_ENCODING} (Turkish "
            "Windows); check that the accented letters look right, or give the encoding"
        )
        return FALLBACK_ENCODING
    raise ProfilerError(
        f"could not read CSV file {path}: it is neither UTF-8 nor {FALLBACK_ENCODING}; "
        "give the encoding"
    )


def _decodes(path: Path, encoding: str) -> bool:
    decoder = codecs.getincrementaldecoder(encoding)()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(_DECODE_CHUNK):
                decoder.decode(chunk)
        decoder.decode(b"", final=True)
    except UnicodeDecodeError:
        return False
    return True


def _detect_separator(path: Path, encoding: str) -> str:
    with path.open("rb") as handle:
        head = handle.read(_SNIFF_BYTES)
    sample = head.decode(encoding, errors="ignore").lstrip("﻿")
    # A sample cut in the middle of a row would skew the count of the separators.
    if len(head) == _SNIFF_BYTES and "\n" in sample:
        sample = sample.rsplit("\n", 1)[0]
    try:
        return csv.Sniffer().sniff(sample, delimiters=_SEPARATORS).delimiter
    except csv.Error:
        return ","


def profile_dataframe(
    df: pd.DataFrame,
    target: str | None = None,
    config: ProfilerConfig = DEFAULT_CONFIG,
    include_samples: bool = True,
    csv_options: CsvOptions | None = None,
) -> DatasetProfile:
    """Profile every column of `df`. The input is not modified.

    With `include_samples=False`, no column gets `sample_values` or `top_values`. `csv_options`
    are recorded as they are.
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
    frame = df if list(df.columns) == names else df.set_axis(names, axis=1)
    _add_duplicates(columns, frame, target, config)
    if target is not None and hint is not None:
        _add_target_association(columns, frame, target, hint, config)
    return DatasetProfile(
        n_rows=n_rows,
        n_columns=int(df.shape[1]),
        n_duplicate_rows=n_duplicate_rows,
        duplicate_ratio=n_duplicate_rows / n_rows,
        columns=columns,
        target=target,
        target_task_hint=hint,
        csv=csv_options,
    )


def _add_duplicates(
    columns: list[ColumnProfile], df: pd.DataFrame, target: str | None, config: ProfilerConfig
) -> None:
    comparable = [c.name for c in columns if c.name != target and c.inferred_type != "constant"]
    copies = find_duplicates(
        df,
        comparable,
        min_abs_corr=config.duplicate_min_abs_corr,
        corr_rows=config.duplicate_corr_rows,
    )
    for column in columns:
        if column.name in copies:
            column.duplicate_of = copies[column.name]
            column.warnings.append("duplicate_column")


def _add_target_association(
    columns: list[ColumnProfile],
    df: pd.DataFrame,
    target: str,
    task: TaskHint,
    config: ProfilerConfig,
) -> None:
    candidates = [
        c.name
        for c in columns
        if c.name != target and c.inferred_type in ("numeric", "categorical", "binary")
    ]
    options = {
        "task": task,
        "max_rows": config.leakage_max_rows,
        "min_rows": config.leakage_min_rows,
        "folds": config.leakage_folds,
        "tree_depth": config.leakage_tree_depth,
    }
    scores = target_association(df, target, candidates, **options)

    # The marker of every column with enough missing values, scored like a column of its own.
    gappy = [
        c.name
        for c in columns
        if c.name != target and config.leakage_min_missing_ratio < c.missing_ratio < 1
    ]
    markers = pd.concat(
        [pd.DataFrame({name: df[name].isna().astype(float) for name in gappy}), df[[target]]],
        axis=1,
    )
    gaps = target_association(markers, target, gappy, **options) if gappy else {}

    for column in columns:
        value = round(scores[column.name], 2) if column.name in scores else None
        gap = round(gaps[column.name], 2) if column.name in gaps else None
        column.target_association = value
        column.missing_association = gap
        best = max(value or 0.0, gap or 0.0)
        if best >= config.leakage_min_score:
            floor, warning = config.leakage_min_score, "suspected_target_leakage"
        elif best >= config.high_association_min_score:
            floor, warning = config.high_association_min_score, "high_target_association"
        else:
            continue
        by_value = value is not None and value >= floor
        by_gap = gap is not None and gap >= floor
        column.leakage_via = (
            "both" if by_value and by_gap else "value" if by_value else "missingness"
        )
        column.warnings.append(warning)


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
    if n >= config.id_min_rows and n_unique == n and _looks_like_hash(strings, config):
        return "id_like"
    if _looks_like_datetime(strings, config):
        return "datetime"
    mean_length = float(strings.str.len().mean())
    if (
        n >= config.id_min_rows
        and mean_length < config.text_min_mean_length
        and (n_unique == n or _is_nearly_unique_id(name, n, n_unique, config))
    ):
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
    if n < config.id_min_rows:
        return False
    if _is_nearly_unique_id(name, n, n_unique, config):
        return True
    if n_unique != n:
        return False
    span = int(non_null.max()) - int(non_null.min()) + 1
    return span <= config.id_dense_span_ratio * n


def _is_nearly_unique_id(name: str, n: int, n_unique: int, config: ProfilerConfig) -> bool:
    return n_unique > config.id_name_min_unique_ratio * n and _looks_like_id_name(name, config)


def _looks_like_id_name(name: str, config: ProfilerConfig) -> bool:
    if _SNAKE_ID_NAME.search(name) or _CAMEL_ID_NAME.search(name):
        return True
    return not set(_name_parts(name)).isdisjoint(config.id_name_parts)


def _looks_like_hash(strings: pd.Series, config: ProfilerConfig) -> bool:
    """Every value is a UUID, or every value is hexadecimal of at least the minimum length and
    nearly all of them contain a letter."""
    if strings.str.fullmatch(_UUID).all():
        return True
    if not strings.str.fullmatch(f"[0-9a-fA-F]{{{config.id_hash_min_length},}}").all():
        return False
    return bool(strings.str.contains("[a-fA-F]").mean() >= config.id_hash_min_letter_ratio)


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
    listed = config.pii_numeric_name_parts if is_numeric else config.pii_name_parts
    joined = f"_{'_'.join(_name_parts(name))}_"
    name_matches = any(f"_{entry}_" in joined for entry in listed)
    if not is_numeric and any(f"_{entry}_" in joined for entry in config.pii_person_name_parts):
        return True
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
    """Split a column name on separators, camelCase and letter/digit boundaries, lowercased.

    Turkish letters become their ASCII look-alikes first, with the case kept, so that "Adı"
    matches "adi" and a capital "İ" is not read as a word boundary.
    """
    parts = []
    for chunk in re.split(r"[\W_]+", name.translate(_ASCII_TURKISH)):
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
    return [part.lower() for part in parts if part]


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
