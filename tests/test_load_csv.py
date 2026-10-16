"""Reading a CSV file: the encoding, separator and decimal mark are found, not assumed."""

import pandas as pd
import pytest

from prep_agent.profiler import CsvOptions, ProfilerError, load_csv

TURKISH = ["çalışan", "öğretmen", "şoför", "ığdır", "ÜSKÜDAR", "İzmir"]


def write(tmp_path, text: str, encoding: str = "utf-8", name: str = "data.csv"):
    path = tmp_path / name
    path.write_bytes(text.encode(encoding))
    return path


def rows(n: int, sep: str, decimal: str) -> str:
    lines = [f"id{sep}price{sep}city"]
    for i in range(n):
        price = f"{10 + i}{decimal}{(i % 4) * 25 + 5:02d}"
        lines.append(f"{i}{sep}{price}{sep}{TURKISH[i % len(TURKISH)]}")
    return "\n".join(lines) + "\n"


def test_a_plain_utf8_comma_file_gets_the_default_options(tmp_path):
    loaded = load_csv(write(tmp_path, rows(12, ",", ".")))

    assert loaded.options == CsvOptions(sep=",", decimal=".", encoding="utf-8")
    assert loaded.encoding_fallback is False
    assert loaded.warnings == []
    assert loaded.df["price"].tolist()[:2] == [10.05, 11.30]
    assert loaded.df["city"].tolist()[:2] == ["çalışan", "öğretmen"]


def test_the_options_default_to_comma_dot_and_utf8():
    assert CsvOptions() == CsvOptions(sep=",", decimal=".", encoding="utf-8")


def test_a_semicolon_file_is_read_with_a_decimal_comma(tmp_path):
    loaded = load_csv(write(tmp_path, rows(12, ";", ",")))

    assert loaded.options == CsvOptions(sep=";", decimal=",", encoding="utf-8")
    assert list(loaded.df.columns) == ["id", "price", "city"]
    assert loaded.df["price"].tolist()[:2] == [10.05, 11.30]


@pytest.mark.parametrize("sep", ["\t", "|"])
def test_tab_and_pipe_files_keep_the_decimal_point(tmp_path, sep):
    loaded = load_csv(write(tmp_path, rows(12, sep, ".")))

    assert loaded.options == CsvOptions(sep=sep, decimal=".", encoding="utf-8")
    assert loaded.df["price"].tolist()[:2] == [10.05, 11.30]


def test_commas_inside_quoted_fields_do_not_decide_the_separator(tmp_path):
    text = 'id;note\n1;"a, b, c"\n2;"d, e, f"\n3;"g, h, i"\n4;"j, k, l"\n'

    loaded = load_csv(write(tmp_path, text))

    assert loaded.options.sep == ";"
    assert loaded.df["note"].tolist() == ["a, b, c", "d, e, f", "g, h, i", "j, k, l"]


def test_a_file_with_one_column_falls_back_to_the_comma(tmp_path):
    loaded = load_csv(write(tmp_path, "a\n1\n2\n3\n"))

    assert loaded.options.sep == ","
    assert loaded.df["a"].tolist() == [1, 2, 3]


def test_a_utf8_file_with_a_byte_order_mark_is_utf8_and_keeps_its_first_column_name(tmp_path):
    path = write(tmp_path, rows(12, ";", ","), encoding="utf-8-sig")

    loaded = load_csv(path)

    assert loaded.options.encoding == "utf-8"
    assert loaded.warnings == []
    assert list(loaded.df.columns) == ["id", "price", "city"]


def test_a_file_that_is_not_utf8_falls_back_to_cp1254_with_a_warning(tmp_path):
    loaded = load_csv(write(tmp_path, rows(12, ";", ","), encoding="cp1254"))

    assert loaded.options == CsvOptions(sep=";", decimal=",", encoding="cp1254")
    assert loaded.encoding_fallback is True
    [warning] = loaded.warnings
    assert "cp1254" in warning
    assert "UTF-8" in warning
    assert loaded.df["city"].tolist()[:6] == TURKISH


def test_a_file_that_neither_utf8_nor_cp1254_can_read_is_an_error(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_bytes(b"a,b\n\x81\xff,1\n")

    with pytest.raises(ProfilerError, match="could not read"):
        load_csv(path)


def test_an_encoding_that_cannot_decode_the_file_is_an_error(tmp_path):
    path = write(tmp_path, rows(12, ",", "."), encoding="cp1254")

    with pytest.raises(ProfilerError, match="could not read"):
        load_csv(path, encoding="utf-8")


def test_the_given_options_replace_what_would_have_been_found(tmp_path):
    path = write(tmp_path, rows(12, ";", ","))

    loaded = load_csv(path, sep=";", decimal=".", encoding="latin-1")

    assert loaded.options == CsvOptions(sep=";", decimal=".", encoding="latin-1")
    assert loaded.encoding_fallback is False
    assert loaded.warnings == []
    # The comma is not a decimal mark any more, so the prices stay text.
    assert not pd.api.types.is_numeric_dtype(loaded.df["price"])


def test_the_decimal_mark_follows_a_given_separator(tmp_path):
    # Found alone this file is a comma file; with the semicolon given, its decimal is a comma.
    text = "id;price\n1;1,5\n2;2,5\n3;3,5\n"

    loaded = load_csv(write(tmp_path, text), sep=";")

    assert loaded.options == CsvOptions(sep=";", decimal=",", encoding="utf-8")
    assert loaded.df["price"].tolist() == [1.5, 2.5, 3.5]


def test_only_the_options_that_are_given_replace_the_found_ones(tmp_path):
    loaded = load_csv(write(tmp_path, rows(12, ";", ",")), decimal=",")

    assert loaded.options.sep == ";"


def test_a_tab_can_be_given_as_a_backslash_t(tmp_path):
    loaded = load_csv(write(tmp_path, rows(12, "\t", ".")), sep="\\t")

    assert loaded.options.sep == "\t"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sep": ";;"}, "separator"),
        ({"sep": ""}, "separator"),
        ({"decimal": ",,"}, "decimal"),
        ({"decimal": ""}, "decimal"),
        ({"encoding": "not-a-codec"}, "unknown encoding"),
    ],
)
def test_options_that_cannot_work_are_refused(tmp_path, kwargs, message):
    path = write(tmp_path, rows(12, ",", "."))

    with pytest.raises(ProfilerError, match=message):
        load_csv(path, **kwargs)
