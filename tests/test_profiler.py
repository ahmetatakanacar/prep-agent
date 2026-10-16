import json
import math

import numpy as np
import pandas as pd
import pytest

from prep_agent.profiler import (
    DEFAULT_CONFIG,
    NA_VALUES,
    ColumnProfile,
    DatasetProfile,
    ProfilerConfig,
    ProfilerError,
    load_csv,
    profile_dataframe,
)

N = 20


def profile_column(values, name="col", **frame_kwargs) -> ColumnProfile:
    df = pd.DataFrame({name: values}, **frame_kwargs)
    return profile_dataframe(df).columns[0]


# --- inferred_type ---------------------------------------------------------------------------


def test_numeric_column_has_statistics():
    col = profile_column([float(i) for i in range(1, N + 1)] + [1000.0])

    assert col.inferred_type == "numeric"
    assert col.min == 1.0
    assert col.max == 1000.0
    assert col.median == 11.0
    assert col.q1 is not None and col.q3 is not None
    assert col.mean is not None and col.std is not None and col.skew is not None
    assert col.n_outliers_iqr == 1
    assert col.top_values is None
    assert col.mean_length is None


def test_low_cardinality_strings_are_categorical():
    col = profile_column(["red", "green", "blue"] * 7)

    assert col.inferred_type == "categorical"
    assert [(t.value, t.count) for t in col.top_values] == [
        ("blue", 7),
        ("green", 7),
        ("red", 7),
    ]
    assert col.min is None


def test_top_values_are_ordered_by_count_and_limited():
    values = [f"v{i}" for i in range(15) for _ in range(15 - i)]
    col = profile_column(values)

    assert len(col.top_values) == DEFAULT_CONFIG.max_top_values
    assert col.top_values[0].value == "v0"
    assert col.top_values[0].count == 15
    counts = [t.count for t in col.top_values]
    assert counts == sorted(counts, reverse=True)


@pytest.mark.parametrize(
    "values",
    [
        [True, False] * 10,
        ["yes", "no"] * 10,
        ["male", "female", "female"] * 7,
        ["Y", "N", "Y", "Y"] * 5,
        [0, 1] * 10,
        [0.0, 1.0] * 10,
    ],
)
def test_two_valued_columns_are_binary(values):
    col = profile_column(values)

    assert col.inferred_type == "binary"
    assert col.n_unique == 2
    assert len(col.top_values) == 2
    assert col.min is None


@pytest.mark.parametrize("values", [[1, 5] * 10, [1.5, 2.5] * 10, [-1, 1] * 10, [2, 3] * 10])
def test_two_valued_numeric_columns_other_than_zero_one_stay_numeric(values):
    col = profile_column(values)

    assert col.inferred_type == "numeric"
    assert col.n_unique == 2
    assert col.min is not None
    assert col.top_values is None


def test_binary_ignores_missing_values():
    col = profile_column(["yes", "no", None, "yes", None] * 4)

    assert col.inferred_type == "binary"
    assert col.n_missing == 8


def test_numeric_zero_one_with_missing_values_is_binary():
    col = profile_column([0.0, 1.0, np.nan, 1.0] * 5)

    assert col.inferred_type == "binary"


def test_three_valued_zero_one_two_column_is_numeric():
    col = profile_column([0, 1, 2] * 7)

    assert col.inferred_type == "numeric"


def test_constant_column():
    col = profile_column(["x"] * N)

    assert col.inferred_type == "constant"
    assert "constant" in col.warnings


def test_all_missing_column_is_constant():
    col = profile_column([np.nan] * N)

    assert col.inferred_type == "constant"
    assert col.n_unique == 0
    assert col.unique_ratio == 0.0
    assert col.missing_ratio == 1.0
    assert "high_missing" in col.warnings


def test_datetime_dtype_column():
    col = profile_column(pd.date_range("2024-01-01", periods=N))

    assert col.inferred_type == "datetime"


def test_date_strings_are_datetime():
    col = profile_column(list(pd.date_range("2024-01-01", periods=N).strftime("%Y-%m-%d")))

    assert col.inferred_type == "datetime"


def test_mixed_date_formats_are_datetime():
    col = profile_column(["2024-01-05", "03/02/2024", "Jan 7 2024", "2024-02-11 10:30"] * 5)

    assert col.inferred_type == "datetime"


def test_month_names_are_not_datetime():
    col = profile_column(["January", "February", "March", "April"] * 5)

    assert col.inferred_type == "categorical"


def test_digit_strings_are_not_datetime():
    col = profile_column([f"{90000 + i % 5 * 111}" for i in range(N)])

    assert col.inferred_type == "categorical"


def test_sequential_integers_are_id_like():
    col = profile_column(list(range(1, N + 1)))

    assert col.inferred_type == "id_like"
    assert "possible_id" in col.warnings
    assert col.min is None


def test_unique_integers_with_id_name_are_id_like():
    col = profile_column([101, 250, 377, 481, 599, 653, 720, 835, 902, 1011], name="user_id")

    assert col.inferred_type == "id_like"


def test_unique_sparse_integers_without_id_name_are_numeric():
    col = profile_column([101, 250, 377, 481, 599, 653, 720, 835, 902, 1011], name="price")

    assert col.inferred_type == "numeric"


def test_unique_floats_are_not_id_like():
    col = profile_column([i * 1.5 for i in range(N)])

    assert col.inferred_type == "numeric"


def test_short_unique_strings_are_id_like():
    col = profile_column([f"u{i:04d}" for i in range(N)])

    assert col.inferred_type == "id_like"
    assert "possible_id" in col.warnings


def test_few_rows_are_never_id_like():
    col = profile_column([1, 2, 3, 4, 5])

    assert col.inferred_type == "numeric"


def test_long_unique_strings_are_text():
    col = profile_column([f"this is a fairly long free text comment number {i}" for i in range(N)])

    assert col.inferred_type == "text"
    assert col.mean_length == pytest.approx(
        np.mean([len(f"this is a fairly long free text comment number {i}") for i in range(N)])
    )
    assert col.max_length == len("this is a fairly long free text comment number 19")
    assert col.top_values is None


def test_long_repeated_strings_are_categorical():
    labels = [f"a rather long category label that repeats a lot {c}" for c in "ABC"]
    col = profile_column(labels * 7)

    assert col.inferred_type == "categorical"


# --- counts and ratios -----------------------------------------------------------------------


def test_missing_counts_and_ratio():
    col = profile_column([1.5, None, 3.5, None, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5])

    assert col.n_missing == 2
    assert col.missing_ratio == pytest.approx(0.2)


def test_high_missing_warning():
    col = profile_column([1.5, None, None, None, None, None, 7.5, 8.5, 9.5, 10.5])

    assert "high_missing" in col.warnings


def test_duplicate_rows_and_ratio():
    df = pd.DataFrame({"a": [1, 1, 2, 2, 2, 3], "b": ["x", "x", "y", "y", "y", "z"]})

    profile = profile_dataframe(df)

    assert profile.n_rows == 6
    assert profile.n_columns == 2
    assert profile.n_duplicate_rows == 3
    assert profile.duplicate_ratio == pytest.approx(0.5)


def test_no_duplicates():
    profile = profile_dataframe(pd.DataFrame({"a": [1, 2, 3]}))

    assert profile.n_duplicate_rows == 0
    assert profile.duplicate_ratio == 0.0


def test_unique_counts_ignore_missing_values():
    col = profile_column(["a", "b", None, "a", None, "b"] * 3 + ["c"])

    assert col.n_unique == 3
    assert col.unique_ratio == pytest.approx(3 / 13)


# --- warnings --------------------------------------------------------------------------------


def test_high_cardinality_warning():
    col = profile_column([f"city_{i % 60}" for i in range(200)])

    assert col.inferred_type == "categorical"
    assert "high_cardinality" in col.warnings


def test_high_skew_warning():
    col = profile_column([float(2**i) for i in range(N)])

    assert "high_skew" in col.warnings


def test_symmetric_numeric_has_no_warnings():
    col = profile_column([i + 0.5 for i in range(N)])

    assert col.warnings == []


def test_many_outliers_warning():
    col = profile_column([10.0 + i % 3 for i in range(N - 4)] + [500.0, 600.0, 700.0, -400.0])

    assert col.n_outliers_iqr == 4
    assert "many_outliers" in col.warnings


def test_infinite_values_are_flagged_and_excluded_from_statistics():
    col = profile_column([1.0, 2.0, 3.0, np.inf, -np.inf] * 4)

    assert "infinite_values" in col.warnings
    assert col.max == 3.0
    assert col.min == 1.0


# --- low_cardinality_numeric -----------------------------------------------------------------


def test_few_distinct_whole_numbers_get_the_low_cardinality_warning():
    hours = [i % 24 for i in range(96)]

    col = profile_column(hours)

    assert col.inferred_type == "numeric"
    assert "low_cardinality_numeric" in col.warnings


def test_whole_numbers_stored_as_floats_because_of_missing_values_get_it_too():
    col = profile_column([1.0, 2.0, None, 3.0, 4.0, 2.0, None, 1.0] * 3)

    assert col.pandas_dtype == "float64"
    assert "low_cardinality_numeric" in col.warnings


def test_a_real_count_gets_the_warning_too_because_it_is_only_a_hint():
    col = profile_column([i % 6 for i in range(60)], name="n_children")

    assert "low_cardinality_numeric" in col.warnings


def test_the_default_threshold_is_25_distinct_values_and_the_limit_itself_still_warns():
    assert DEFAULT_CONFIG.low_cardinality_unique == 25

    at_limit = profile_column([i % 25 for i in range(100)])
    above_limit = profile_column([i % 26 for i in range(104)])

    assert at_limit.n_unique == 25
    assert "low_cardinality_numeric" in at_limit.warnings
    assert above_limit.n_unique == 26
    assert "low_cardinality_numeric" not in above_limit.warnings


def test_the_threshold_comes_from_the_config():
    df = pd.DataFrame({"col": [i % 10 for i in range(100)]})

    strict = profile_dataframe(df, config=ProfilerConfig(low_cardinality_unique=9))
    lenient = profile_dataframe(df, config=ProfilerConfig(low_cardinality_unique=10))

    assert "low_cardinality_numeric" not in strict.columns[0].warnings
    assert "low_cardinality_numeric" in lenient.columns[0].warnings


@pytest.mark.parametrize(
    "values",
    [
        [0.5 * (i % 10) for i in range(100)],
        [float(i % 10) + 0.25 * (i % 2) for i in range(100)],
        [float(i % 10) for i in range(99)] + [np.inf],
    ],
    ids=["fractions", "some_fractions", "infinite"],
)
def test_values_that_are_not_all_whole_numbers_get_no_low_cardinality_warning(values):
    col = profile_column(values)

    assert col.inferred_type == "numeric"
    assert "low_cardinality_numeric" not in col.warnings


def test_binary_constant_and_id_like_columns_get_no_low_cardinality_warning():
    binary = profile_column([i % 2 for i in range(40)])
    constant = profile_column([7] * 40)
    id_like = profile_column(list(range(1, 21)))

    assert (binary.inferred_type, constant.inferred_type, id_like.inferred_type) == (
        "binary",
        "constant",
        "id_like",
    )
    for col in (binary, constant, id_like):
        assert "low_cardinality_numeric" not in col.warnings


def test_two_valued_numbers_that_are_not_01_get_the_warning():
    col = profile_column([1, 5] * 20)

    assert col.inferred_type == "numeric"
    assert "low_cardinality_numeric" in col.warnings


def test_text_columns_of_digits_get_no_low_cardinality_warning():
    col = profile_column([str(i % 7) + "x" for i in range(70)])

    assert col.inferred_type == "categorical"
    assert "low_cardinality_numeric" not in col.warnings


def test_a_small_column_of_all_distinct_whole_numbers_gets_no_low_cardinality_warning():
    col = profile_column([i * 3 for i in range(20)], name="age")

    assert col.inferred_type == "numeric"
    assert col.n_unique == 20
    assert "low_cardinality_numeric" not in col.warnings


def test_an_hour_column_in_a_large_table_still_gets_the_warning():
    col = profile_column([i % 24 for i in range(1000)], name="hour")

    assert col.n_unique == 24
    assert "low_cardinality_numeric" in col.warnings


def test_the_distinct_values_must_be_fewer_than_the_share_of_the_filled_values():
    assert DEFAULT_CONFIG.low_cardinality_max_unique_ratio == 0.5

    below = profile_column([i % 9 for i in range(20)])  # 9 of 20 distinct
    at_share = profile_column([i % 10 for i in range(20)])  # 10 of 20 distinct

    assert "low_cardinality_numeric" in below.warnings
    assert "low_cardinality_numeric" not in at_share.warnings


def test_the_share_is_taken_of_the_filled_values_not_of_all_rows():
    # 12 rows, 6 filled, 4 distinct: 4 of 6 is above half, although 4 of 12 is below it.
    col = profile_column([1.0, 2.0, 3.0, 4.0, 1.0, 2.0] + [None] * 6)

    assert col.n_unique == 4
    assert "low_cardinality_numeric" not in col.warnings


def test_the_share_comes_from_the_config():
    df = pd.DataFrame({"col": [i % 10 for i in range(20)]})

    default = profile_dataframe(df)
    lenient = profile_dataframe(df, config=ProfilerConfig(low_cardinality_max_unique_ratio=0.6))

    assert "low_cardinality_numeric" not in default.columns[0].warnings
    assert "low_cardinality_numeric" in lenient.columns[0].warnings


def test_many_distinct_whole_numbers_get_no_low_cardinality_warning():
    col = profile_column([i * 3 % 997 for i in range(300)])

    assert "low_cardinality_numeric" not in col.warnings


# --- samples ---------------------------------------------------------------------------------


def test_sample_values_are_limited_in_count_and_length():
    long_values = [f"{i}-" + "x" * 200 for i in range(N)]
    col = profile_column(long_values)

    assert len(col.sample_values) == 5
    assert all(len(v) <= 50 for v in col.sample_values)
    assert all(isinstance(v, str) for v in col.sample_values)


def test_sample_values_are_unique_strings():
    col = profile_column([3, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])

    assert col.sample_values == ["3", "4", "5", "6", "7"]


def test_sample_values_skip_missing():
    col = profile_column([None, None, "a", "b", None] * 4)

    assert col.sample_values == ["a", "b"]


# --- privacy ---------------------------------------------------------------------------------


def test_email_values_are_flagged_as_pii_and_hidden():
    col = profile_column([f"person{i % 6}@example.com" for i in range(N)])

    assert col.inferred_type == "categorical"
    assert "possible_pii" in col.warnings
    assert col.sample_values == []
    assert col.top_values == []


def test_unique_email_values_are_flagged_and_have_no_samples():
    col = profile_column([f"person{i}@example.com" for i in range(N)])

    assert "possible_pii" in col.warnings
    assert col.sample_values == []
    assert not col.top_values


@pytest.mark.parametrize(
    "phone",
    ["+90 532 123 45 67", "0532 123 45 67", "(212) 555-0147", "+1-202-555-0173", "5321234567"],
)
def test_phone_values_are_flagged_as_pii(phone):
    col = profile_column([phone, phone[:-1] + "9", phone[:-2] + "11"] * 7)

    assert "possible_pii" in col.warnings
    assert col.sample_values == []


def test_pii_value_ratio_threshold():
    mostly = [f"p{i}@example.com" for i in range(17)] + ["n/a", "unknown", "-"]
    half = [f"p{i}@example.com" for i in range(10)] + [f"note {i}" for i in range(10)]

    assert "possible_pii" in profile_column(mostly).warnings
    assert "possible_pii" not in profile_column(half).warnings


def test_pii_columns_keep_non_value_statistics():
    values = [f"p{i % 7}@example.com" for i in range(N - 4)] + [None] * 4
    col = profile_column(values)

    assert col.n_missing == 4
    assert col.missing_ratio == pytest.approx(0.2)
    assert col.n_unique == 7
    assert col.unique_ratio == pytest.approx(7 / 16)


def test_free_text_mentioning_emails_is_not_flagged_by_value_pattern():
    col = profile_column([f"contact me at someone{i}@example.com or by phone" for i in range(N)])

    assert col.inferred_type == "text"
    assert "possible_pii" not in col.warnings
    assert len(col.sample_values) == 5


@pytest.mark.parametrize(
    "name",
    [
        "email",
        "Email",
        "e_mail",
        "user_name",
        "firstName",
        "first_name",
        "firstname",
        "LastName",
        "surname",
        "Phone",
        "customer_phone2",
        "GSM",
        "telefon",
        "tel",
        "adres",
        "soyad",
        "isim",
        "address",
        "tckn",
        "IBAN",
        "ssn",
        "userEmail",
        "HTMLName",
        "username",
        "UserName",
        "fullname",
        "full_name",
        "musteri_adi",
        "adi",
        "adı",
        "soyadi",
        "soyadı",
        "musteri_soyadı",
    ],
)
def test_pii_column_names_are_flagged(name):
    col = profile_column([f"value {i % 12}" for i in range(N)], name=name)

    assert "possible_pii" in col.warnings
    assert col.sample_values == []
    assert col.top_values == []


@pytest.mark.parametrize(
    "name", ["filename", "paid", "valid", "company", "named", "telephone_x", "radio", "adidas"]
)
def test_non_pii_column_names_are_not_flagged(name):
    col = profile_column(["a", "b", "c", "d"] * 5, name=name)

    assert "possible_pii" not in col.warnings
    assert col.sample_values == ["a", "b", "c", "d"]
    assert len(col.top_values) == 4


def test_a_pii_name_with_few_distinct_values_is_not_flagged():
    yes_no = profile_column(["Yes", "No", "Yes", "Yes"] * 5, name="PhoneService")
    departments = profile_column(["Police", "Fire", "Health", "Police"] * 5, name="department_name")
    porch = profile_column([0, 0, 0, 144, 0, 0, 0, 180, 0, 0] * 2, name="3SsnPorch")

    for col in (yes_no, departments, porch):
        assert "possible_pii" not in col.warnings, col.name
    assert yes_no.sample_values == ["Yes", "No"]
    assert [v.value for v in departments.top_values] == ["Police", "Fire", "Health"]
    assert porch.max == 180.0


def test_a_pii_name_with_mostly_distinct_values_is_flagged():
    names = profile_column([f"Person {i % 12}" for i in range(N)], name="full_name")
    phones = profile_column(_sparse_phone_numbers(), name="PhoneNumber")

    assert "possible_pii" in names.warnings
    assert names.sample_values == [] and names.top_values == []
    assert phones.inferred_type == "numeric"
    assert "possible_pii" in phones.warnings
    assert phones.min is None


@pytest.mark.parametrize("name", ["ad_spend", "tel_count", "ad", "name_length", "AdClicks"])
def test_a_numeric_column_is_not_flagged_for_a_name_part_that_is_not_an_identifier(name):
    col = profile_column([i * 2.5 for i in range(N)], name=name)

    assert col.inferred_type == "numeric"
    assert "possible_pii" not in col.warnings
    assert col.min == 0.0
    assert col.max == 47.5
    assert col.sample_values != []


def test_a_text_column_named_ad_with_mostly_distinct_values_is_still_flagged():
    col = profile_column([f"Person {i}" for i in range(N)], name="ad")

    assert "possible_pii" in col.warnings
    assert col.sample_values == []


@pytest.mark.parametrize(
    "name", ["phone", "telefon", "gsm", "tckn", "iban", "ssn", "PhoneNumber", "tc_ssn"]
)
def test_a_numeric_column_with_an_identifier_name_and_distinct_values_is_flagged(name):
    col = profile_column([5321000000 + i * 7919 for i in range(N)], name=name)

    assert col.pandas_dtype == "int64"
    assert "possible_pii" in col.warnings
    assert col.sample_values == []
    assert col.min is None


def test_a_numeric_column_with_an_identifier_name_needs_mostly_distinct_values_too():
    col = profile_column([0, 0, 0, 144, 0, 0, 0, 180, 0, 0] * 2, name="phone")

    assert "possible_pii" not in col.warnings
    assert col.max == 180.0


def test_the_unique_ratio_of_a_pii_name_must_be_above_the_threshold():
    half = [f"Person {i % 10}" for i in range(N)]
    just_over = [f"Person {i % 11}" for i in range(N)]

    assert "possible_pii" not in profile_column(half, name="name").warnings
    assert "possible_pii" in profile_column(just_over, name="name").warnings


def test_the_unique_ratio_for_pii_names_is_configurable():
    df = pd.DataFrame({"name": [f"Person {i % 5}" for i in range(N)]})  # unique ratio 0.25

    def flagged(ratio):
        profile = profile_dataframe(df, config=ProfilerConfig(pii_name_min_unique_ratio=ratio))
        return "possible_pii" in profile.columns[0].warnings

    assert DEFAULT_CONFIG.pii_name_min_unique_ratio == 0.5
    assert not flagged(DEFAULT_CONFIG.pii_name_min_unique_ratio)
    assert flagged(0.2)
    assert not flagged(0.3)


def test_value_patterns_flag_a_column_whatever_its_name_and_unique_ratio():
    col = profile_column([f"person{i % 3}@example.com" for i in range(N)], name="contact")

    assert "possible_pii" in col.warnings


NUMERIC_STAT_FIELDS = ("min", "max", "mean", "std", "median", "q1", "q3", "skew", "n_outliers_iqr")


def _sparse_phone_numbers():
    return [5321000000 + i * 7919 for i in range(N - 2)] + [None, None]


def test_pii_named_numeric_column_has_no_samples_and_no_numeric_statistics():
    col = profile_column(_sparse_phone_numbers(), name="phone")

    assert col.inferred_type == "numeric"
    assert "possible_pii" in col.warnings
    assert col.sample_values == []
    for field in NUMERIC_STAT_FIELDS:
        assert getattr(col, field) is None, field


def test_pii_named_numeric_column_keeps_counts():
    col = profile_column(_sparse_phone_numbers(), name="phone")

    assert col.n_missing == 2
    assert col.missing_ratio == pytest.approx(0.1)
    assert col.n_unique == N - 2
    assert col.unique_ratio == 1.0


def test_pii_named_numeric_column_raises_no_statistics_based_warnings():
    values = [float(i) for i in range(1, N)] + [100000.0]
    col = profile_column(values, name="phone")

    assert col.warnings == ["possible_pii"]


def test_numeric_statistics_stay_for_non_pii_columns_without_samples():
    df = pd.DataFrame({"phone": _sparse_phone_numbers(), "age": [float(i) for i in range(N)]})

    profile = profile_dataframe(df, include_samples=False)

    phone, age = profile.columns
    assert phone.min is None
    assert age.min == 0.0
    assert age.max == float(N - 1)


def test_date_strings_and_zip_codes_are_not_flagged_as_phone_numbers():
    dates = profile_column(list(pd.date_range("2024-01-01", periods=N).strftime("%Y-%m-%d")))
    zips = profile_column([f"{34000 + i % 5}" for i in range(N)])

    assert "possible_pii" not in dates.warnings
    assert "possible_pii" not in zips.warnings


def test_include_samples_false_hides_samples_and_top_values_everywhere():
    df = pd.DataFrame(
        {
            "num": [float(i) for i in range(N)],
            "cat": ["red", "green", "blue", "red"] * 5,
            "flag": ["yes", "no"] * 10,
            "ident": list(range(N)),
            "text": [f"this is a fairly long free text comment number {i}" for i in range(N)],
        }
    )

    profile = profile_dataframe(df, include_samples=False)

    for col in profile.columns:
        assert col.sample_values == []
        assert not col.top_values
    by_name = {c.name: c for c in profile.columns}
    assert by_name["cat"].top_values == []
    assert by_name["flag"].top_values == []
    assert by_name["num"].top_values is None


def test_include_samples_false_keeps_other_statistics():
    df = pd.DataFrame({"num": [float(i) for i in range(N)] + [None], "cat": ["a", "b", "c"] * 7})

    with_samples = profile_dataframe(df)
    without = profile_dataframe(df, include_samples=False)

    for kept, hidden in zip(with_samples.columns, without.columns, strict=True):
        assert hidden.model_dump(exclude={"sample_values", "top_values"}) == kept.model_dump(
            exclude={"sample_values", "top_values"}
        )


def test_samples_are_included_by_default():
    col = profile_column(["a", "b", "c"] * 7)

    assert col.sample_values == ["a", "b", "c"]
    assert len(col.top_values) == 3


# --- JSON safety -----------------------------------------------------------------------------


def _reject_constant(token: str):
    raise AssertionError(f"non-standard JSON constant: {token}")


def test_profile_serializes_to_strict_json_and_round_trips():
    df = pd.DataFrame(
        {
            "x": [1.0, np.nan, np.inf, -np.inf, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] * 2,
            "all_nan": [np.nan] * N,
            "label": ["a", "b"] * 10,
        }
    )

    profile = profile_dataframe(df, target="label")
    payload = profile.model_dump_json()

    json.loads(payload, parse_constant=_reject_constant)
    assert DatasetProfile.model_validate_json(payload) == profile


def test_profile_values_are_plain_python_types():
    df = pd.DataFrame({"x": np.arange(N, dtype=np.int64) * 3, "y": np.linspace(0, 1, N)})

    profile = profile_dataframe(df)

    assert type(profile.n_rows) is int
    for col in profile.columns:
        assert type(col.n_missing) is int
        assert type(col.n_unique) is int
        assert type(col.missing_ratio) is float
        assert type(col.max) is float


def test_all_float_fields_are_finite_or_none():
    df = pd.DataFrame({"x": [np.nan, np.inf, 1.0, 2.0] * 5, "k": [7.0] * N})

    for col in profile_dataframe(df).columns:
        for field in ("min", "max", "mean", "std", "median", "q1", "q3", "skew"):
            value = getattr(col, field)
            assert value is None or math.isfinite(value)


# --- target and task hint --------------------------------------------------------------------


def test_missing_target_column_raises_clear_error():
    df = pd.DataFrame({"a": [1, 2, 3]})

    with pytest.raises(ProfilerError, match="target column 'nope' not found"):
        profile_dataframe(df, target="nope")


def test_binary_target_hints_classification():
    df = pd.DataFrame({"a": range(N), "y": ["yes", "no"] * 10})

    profile = profile_dataframe(df, target="y")

    assert profile.target == "y"
    assert profile.target_task_hint == "classification"


def test_multiclass_string_target_hints_classification():
    df = pd.DataFrame({"y": ["cat", "dog", "bird"] * 7})

    assert profile_dataframe(df, target="y").target_task_hint == "classification"


def test_low_cardinality_integer_target_hints_classification():
    df = pd.DataFrame({"y": [1, 2, 3, 4, 5] * 8})

    assert profile_dataframe(df, target="y").target_task_hint == "classification"


def test_continuous_target_hints_regression():
    df = pd.DataFrame({"y": np.linspace(0.5, 99.5, 60)})

    assert profile_dataframe(df, target="y").target_task_hint == "regression"


def test_constant_target_has_no_hint():
    df = pd.DataFrame({"y": ["a"] * N})

    assert profile_dataframe(df, target="y").target_task_hint is None


def test_no_target_means_no_hint():
    profile = profile_dataframe(pd.DataFrame({"a": [1.5, 2.5, 3.5]}))

    assert profile.target is None
    assert profile.target_task_hint is None


# --- errors and determinism ------------------------------------------------------------------


def test_empty_dataframe_raises_clear_error():
    with pytest.raises(ProfilerError, match="empty"):
        profile_dataframe(pd.DataFrame())


def test_dataframe_without_rows_raises_clear_error():
    with pytest.raises(ProfilerError, match="empty"):
        profile_dataframe(pd.DataFrame({"a": []}))


def test_duplicate_column_names_raise_clear_error():
    df = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "a"])

    with pytest.raises(ProfilerError, match="duplicate column names"):
        profile_dataframe(df)


def test_profiling_is_deterministic():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "num": rng.normal(size=3000),
            "when": pd.date_range("2020-01-01", periods=3000).strftime("%Y-%m-%d"),
            "cat": rng.choice(["a", "b", "c", "d"], size=3000),
        }
    )

    first = profile_dataframe(df, target="cat")
    second = profile_dataframe(df.copy(), target="cat")

    assert first == second
    assert first.model_dump_json() == second.model_dump_json()


def test_profiling_does_not_modify_input():
    df = pd.DataFrame({"a": [1.5, np.nan, 3.5], "b": ["x", "y", "x"]})
    expected = df.copy()

    profile_dataframe(df)

    pd.testing.assert_frame_equal(df, expected)


# --- load_csv --------------------------------------------------------------------------------


def test_load_csv_reads_file(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("a,b\n1,x\n2,y\n", encoding="utf-8")

    df = load_csv(path).df

    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2


def test_load_csv_missing_file(tmp_path):
    with pytest.raises(ProfilerError, match="not found"):
        load_csv(tmp_path / "missing.csv")


def test_load_csv_directory(tmp_path):
    with pytest.raises(ProfilerError, match="not a file"):
        load_csv(tmp_path)


def test_load_csv_empty_file(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ProfilerError, match="empty"):
        load_csv(path)


def test_load_csv_header_only(tmp_path):
    path = tmp_path / "header.csv"
    path.write_text("a,b\n", encoding="utf-8")

    with pytest.raises(ProfilerError, match="no data rows"):
        load_csv(path)


def test_load_csv_unreadable_encoding(tmp_path):
    path = tmp_path / "bad.csv"
    # Neither UTF-8 nor cp1254 defines the byte 0x81.
    path.write_bytes(b"a,b\n\x81\xff,1\n")

    with pytest.raises(ProfilerError, match="could not read"):
        load_csv(path)


# --- missing value spellings -----------------------------------------------------------------


def write_csv(tmp_path, *lines):
    path = tmp_path / "data.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_none_text_stays_a_category_and_is_not_missing(tmp_path):
    rows = ["BrkFace", "None", "Stone", "None", "None", "BrkFace"]
    path = write_csv(tmp_path, "veneer,price", *[f"{v},{i}" for i, v in enumerate(rows)])

    df = load_csv(path).df

    assert df["veneer"].tolist() == rows
    column = profile_dataframe(df).columns[0]
    assert column.n_missing == 0
    assert column.inferred_type == "categorical"
    assert {v.value: v.count for v in column.top_values}["None"] == 3


@pytest.mark.parametrize(
    "spelling", ["", "NA", "NaN", "nan", "null", "NULL", "N/A", "n/a", "#N/A", "<NA>", "#NA"]
)
def test_the_other_missing_value_spellings_are_still_missing(tmp_path, spelling):
    path = write_csv(tmp_path, "kind,price", "a,1", f"{spelling},2", "b,3", "None,4")

    df = load_csv(path).df

    assert df["kind"].isna().tolist() == [False, True, False, False]
    assert df["kind"].tolist()[3] == "None"


def test_missing_cells_keep_a_numeric_column_numeric(tmp_path):
    path = write_csv(tmp_path, "a,b", "1,x", ",y", "NA,z", "4,w")

    df = load_csv(path).df

    assert pd.api.types.is_float_dtype(df["a"])
    assert df["a"].isna().sum() == 2


def test_none_is_not_in_the_missing_value_list():
    assert "None" not in NA_VALUES
    assert {"", "NA", "NaN", "null", "N/A", "n/a", "nan", "NULL"} <= set(NA_VALUES)
