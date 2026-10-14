import ast

import pytest

from prep_agent.ops.code import (
    MAX_LINE_WIDTH,
    Assign,
    CodeError,
    Import,
    call,
    comment_lines,
    literal,
    ref,
    render_imports,
    render_statements,
)

HOSTILE_STRINGS = [
    "",
    "plain",
    "with space",
    "it's",
    'say "hi"',
    "both ' and \"",
    "back\\slash",
    "\\'",
    "ends with backslash\\",
    'x"); import os  #',
    "line\nbreak",
    "cr\rlf\r\n",
    "tab\there",
    "müşteri yaşı İ ğ ş",
    "日本語の列",
    "emoji \U0001f600",
    "nul\x00char",
    "bell\x07",
    "bidi ‮ override",
    "nbsp here",
    "line sep   and  ",
    "'''triple'''",
    '"""triple"""',
    "{braces} and %s and \\n",
]


@pytest.mark.parametrize("text", HOSTILE_STRINGS)
def test_string_literals_round_trip_through_python(text):
    source = literal(text).flat()

    assert ast.literal_eval(source) == text
    assert "\n" not in source
    assert "\r" not in source


@pytest.mark.parametrize("text", HOSTILE_STRINGS)
def test_string_literals_hide_nothing_from_a_reader(text):
    # Control and format characters (bidi overrides, nul, line separators) must be escaped so
    # that the source file shows what it contains.
    source = literal(text).flat()

    assert all(ch.isprintable() for ch in source)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("plain", '"plain"'),
        ("it's", '"it\'s"'),
        ('say "hi"', "'say \"hi\"'"),
        ("both ' and \"", '"both \' and \\""'),
        ("müşteri", '"müşteri"'),
        ("a\nb", '"a\\nb"'),
        ("a\\b", '"a\\\\b"'),
        ("‮", '"\\u202e"'),
        ("\x00", '"\\x00"'),
    ],
)
def test_string_literals_use_the_quotes_ruff_format_keeps(text, expected):
    assert literal(text).flat() == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, "True"),
        (False, "False"),
        (None, "None"),
        (0, "0"),
        (-3, "-3"),
        (0.2, "0.2"),
        (-1.5, "-1.5"),
        (1e-05, "1e-05"),
        (1e16, "1e16"),
        (1.5e300, "1.5e300"),
        (float("nan"), 'float("nan")'),
        (float("inf"), 'float("inf")'),
        (float("-inf"), 'float("-inf")'),
        (["a", 1, None], '["a", 1, None]'),
        ([], "[]"),
    ],
)
def test_other_literals(value, expected):
    assert literal(value).flat() == expected


@pytest.mark.parametrize("value", [{"a": 1}, {1}, b"bytes", object(), 1 + 2j, ["ok", {"no": 1}]])
def test_unsupported_literals_are_rejected(value):
    with pytest.raises(CodeError):
        literal(value)


def test_float_literals_round_trip():
    for value in (0.1, 1 / 3, 2.5e-10, 123456.789, 1e22, 5e-324):
        assert ast.literal_eval(literal(value).flat()) == value


@pytest.mark.parametrize("name", ["os.system('x')", "a b", "1abc", "", "a-b", "x; y", "a.", "ü"])
def test_names_must_be_plain_identifiers(name):
    with pytest.raises(CodeError):
        ref(name)
    with pytest.raises(CodeError):
        call(name)


def test_keyword_names_must_be_plain_identifiers():
    with pytest.raises(CodeError):
        call("f", **{"a=1, b": 1})


def test_a_bare_string_argument_is_data_never_code():
    source = call("f", "os.system('x')", key="import os").flat()

    tree = ast.parse(source, mode="eval")
    [arg] = tree.body.args
    assert isinstance(arg, ast.Constant)
    assert isinstance(tree.body.keywords[0].value, ast.Constant)


def test_short_statements_stay_on_one_line():
    stmt = Assign("df", call("df.drop", columns=["a", "b"]))

    assert render_statements([stmt], indent=4) == ['    df = df.drop(columns=["a", "b"])']


def test_the_assignment_target_must_be_names():
    with pytest.raises(CodeError):
        Assign("df[0]", literal(1))
    with pytest.raises(CodeError):
        Assign("df = x; y", literal(1))
    Assign("train, test", literal(1))


def test_long_statements_explode_with_trailing_commas():
    columns = [f"a_rather_long_column_name_{i}" for i in range(6)]
    stmt = Assign("df", call("df.drop", columns=columns))

    lines = render_statements([stmt], indent=4)

    assert lines[0] == "    df = df.drop("
    assert lines[1] == "        columns=["
    assert lines[2] == '            "a_rather_long_column_name_0",'
    assert lines[-2] == "        ],"
    assert lines[-1] == "    )"
    assert len(lines) == 10
    assert all(len(line) <= MAX_LINE_WIDTH for line in lines)


def test_exploded_and_flat_layouts_mean_the_same_code():
    columns = [f"a_rather_long_column_name_{i}" for i in range(6)]
    stmt = Assign("df", call("df.drop", columns=columns, errors="raise"))

    exploded = "\n".join(line[4:] for line in render_statements([stmt], indent=4))
    flat = f"df = {stmt.value.flat()}"

    assert ast.dump(ast.parse(exploded)) == ast.dump(ast.parse(flat))


def test_nested_calls_and_lists_explode_level_by_level():
    inner = call("SimpleImputer", strategy="constant", fill_value="a" * 30, add_indicator=False)
    outer = call("Wrap", [("name", inner, ["x", "y"])], remainder="pass")
    stmt = Assign("steps", outer)

    lines = render_statements([stmt], indent=0)

    assert all(len(line) <= MAX_LINE_WIDTH for line in lines)
    assert lines[0] == "steps = Wrap("
    assert lines[-1] == ")"
    # Every exploded bracket ends its items with a comma, so ruff keeps the layout.
    assert all(line.endswith((",", "(", "[")) or line.strip() in {")", "]"} for line in lines)
    assert ast.dump(ast.parse("\n".join(lines))) == ast.dump(ast.parse(f"steps = {outer.flat()}"))


def test_a_method_call_stays_attached_to_the_closing_bracket():
    base = call(
        "ColumnTransformer", [call("A", x="y" * 40)], remainder="passthrough", other="z" * 30
    )
    stmt = Assign("ct", base.then("set_output", transform="pandas"))

    lines = render_statements([stmt], indent=0)

    assert lines[-1] == ').set_output(transform="pandas")'
    assert all(len(line) <= MAX_LINE_WIDTH for line in lines)


def test_then_only_accepts_plain_values():
    with pytest.raises(CodeError):
        call("f").then("g", x=call("h"))


def test_width_counts_wide_characters_twice():
    wide = "列" * 40
    stmt = Assign("x", call("f", wide))

    # 40 wide characters are 80 columns, which together with the rest overflows 88.
    assert len(render_statements([stmt], indent=0)) > 1


def test_single_item_tuples_keep_their_comma():
    assert literal(("a",)).flat() == '("a",)'
    assert literal(("a", 1)).flat() == '("a", 1)'


def test_comment_text_cannot_leave_the_comment():
    reason = "first\nimport os\r\nos.system('x') and\x0bmore\x00‮ end"

    lines = comment_lines(reason, indent=0)

    assert len(lines) >= 1
    assert all(line.startswith("# ") for line in lines)
    assert all(ch.isprintable() for line in lines for ch in line)
    assert ast.parse("\n".join(lines)).body == []


def test_comments_are_wrapped_to_the_line_width():
    lines = comment_lines("word " * 80, indent=8)

    assert len(lines) > 1
    assert all(len(line) <= MAX_LINE_WIDTH for line in lines)
    assert all(line.startswith("        # ") for line in lines)


def test_a_word_longer_than_the_line_is_broken_instead_of_overflowing():
    lines = comment_lines("x" * 300, indent=4)

    assert all(len(line) <= MAX_LINE_WIDTH for line in lines)
    assert "".join(line.removeprefix("    # ") for line in lines) == "x" * 300


def test_blank_comment_text_gives_no_lines():
    assert comment_lines(" \n\t ", indent=0) == []


def test_imports_are_sorted_and_grouped_like_isort():
    imports = [
        Import("sklearn.preprocessing", "StandardScaler"),
        Import("pandas", alias="pd"),
        Import("pathlib", "Path"),
        Import("argparse"),
        Import("sklearn.compose", "ColumnTransformer"),
        Import("sklearn.preprocessing", "MinMaxScaler"),
        Import("sklearn.preprocessing", "StandardScaler"),
    ]

    assert render_imports(imports) == [
        "import argparse",
        "from pathlib import Path",
        "",
        "import pandas as pd",
        "from sklearn.compose import ColumnTransformer",
        "from sklearn.preprocessing import MinMaxScaler, StandardScaler",
    ]


def test_long_import_lists_are_wrapped_with_a_trailing_comma():
    names = [
        "FunctionTransformer",
        "MinMaxScaler",
        "OneHotEncoder",
        "OrdinalEncoder",
        "RobustScaler",
    ]
    lines = render_imports([Import("sklearn.preprocessing", name) for name in names])

    assert lines == [
        "from sklearn.preprocessing import (",
        *[f"    {name}," for name in names],
        ")",
    ]


def test_import_names_must_be_plain_identifiers():
    with pytest.raises(CodeError):
        Import("os; import sys")
    with pytest.raises(CodeError):
        Import("sklearn", "a, b")
