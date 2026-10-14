"""Building blocks for the code that ops generate for the standalone `pipeline.py`.

Ops describe their code as small expression and statement objects instead of strings, so that
nothing from a plan can end up in the script as code:

- A plain Python value (string, number, bool, None, list) is always rendered as a literal. A bare
  string handed to `call` is data, never a name; names go through `ref` and must be identifiers.
- Strings are escaped by `literal`, which keeps everything printable and escapes the rest, so
  a column name such as `x"); import os  #` is just a string in the script.
- Comments come from `comment_lines`, which turns every line break and control character into
  a space, so a comment can never end and start a code line.

The layout is chosen to be stable under `ruff format`: a node is written on one line when it fits
in `MAX_LINE_WIDTH` and is otherwise exploded one item per line, every item followed by a comma
(ruff keeps such a layout). Strings use the quotes ruff prefers.
"""

import re
import sys
import textwrap
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

MAX_LINE_WIDTH = 88
INDENT = 4

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*")
_SIMPLE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_TARGETS = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(, [A-Za-z_][A-Za-z0-9_]*)*")
_SHORT_ESCAPES = {"\\": "\\\\", "\n": "\\n", "\r": "\\r", "\t": "\\t"}


class CodeError(ValueError):
    """Raised when something that is not safe or not supported would end up in generated code."""


def display_width(text: str) -> int:
    """Width of `text` in terminal columns, the way ruff measures a line."""
    width = 0
    for char in text:
        if unicodedata.combining(char):
            continue
        width += 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
    return width


def _check(pattern: re.Pattern[str], text: str, what: str) -> str:
    if not pattern.fullmatch(text):
        raise CodeError(f"not a valid {what}: {text!r}")
    return text


@dataclass(frozen=True)
class Expr:
    """An expression. `flat()` is its single-line source."""

    def flat(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class Atom(Expr):
    """A leaf whose source is already safe: a literal or a validated name."""

    text: str

    def flat(self) -> str:
        return self.text


@dataclass(frozen=True)
class Seq(Expr):
    """A list or a tuple."""

    items: tuple[Expr, ...]
    is_tuple: bool = False

    def flat(self) -> str:
        inner = ", ".join(item.flat() for item in self.items)
        if not self.is_tuple:
            return f"[{inner}]"
        return f"({inner},)" if len(self.items) == 1 else f"({inner})"


@dataclass(frozen=True)
class Call(Expr):
    """A call, optionally followed by method calls on its result (`tail`)."""

    func: str
    args: tuple[Expr, ...]
    kwargs: tuple[tuple[str, Expr], ...]
    tail: tuple[tuple[str, tuple[tuple[str, Atom], ...]], ...] = ()

    def flat(self) -> str:
        parts = [arg.flat() for arg in self.args]
        parts += [f"{key}={value.flat()}" for key, value in self.kwargs]
        text = f"{self.func}({', '.join(parts)})"
        for name, kwargs in self.tail:
            text += f".{name}({', '.join(f'{k}={v.flat()}' for k, v in kwargs)})"
        return text

    def then(self, method: str, **kwargs: Any) -> "Call":
        """Append `.method(key=value)`. Values must be plain literals or names."""
        _check(_SIMPLE_NAME, method, "method name")
        pairs = []
        for key, value in kwargs.items():
            atom = literal(value)
            if not isinstance(atom, Atom):
                raise CodeError("then() only takes plain values")
            pairs.append((_check(_SIMPLE_NAME, key, "keyword name"), atom))
        return Call(self.func, self.args, self.kwargs, (*self.tail, (method, tuple(pairs))))


def ref(name: str) -> Atom:
    """A reference to a name, such as `df` or `pd.concat`."""
    return Atom(_check(_NAME, name, "name"))


def call(func: str, /, *args: Any, **kwargs: Any) -> Call:
    """A call of `func`. Arguments that are not expressions are rendered as literals."""
    return Call(
        _check(_NAME, func, "function name"),
        tuple(literal(arg) for arg in args),
        tuple((_check(_SIMPLE_NAME, key, "keyword name"), literal(v)) for key, v in kwargs.items()),
    )


def literal(value: Any) -> Expr:
    """The literal for a Python value; expressions pass through unchanged."""
    if isinstance(value, Expr):
        return value
    # bool is an int, so it has to be looked at first.
    if value is None or isinstance(value, bool):
        return Atom(repr(value))
    if isinstance(value, int):
        return Atom(str(value))
    if isinstance(value, float):
        return Atom(_float_literal(value))
    if isinstance(value, str):
        return Atom(_string_literal(value))
    if isinstance(value, list):
        return Seq(tuple(literal(item) for item in value))
    if isinstance(value, tuple):
        return Seq(tuple(literal(item) for item in value), is_tuple=True)
    raise CodeError(f"cannot write a value of type {type(value).__name__} as a literal")


def _float_literal(value: float) -> str:
    if value != value:
        return 'float("nan")'
    if value in (float("inf"), float("-inf")):
        return f'float("{value}")'
    # ruff format writes 1e16, not 1e+16.
    return repr(value).replace("e+", "e")


def _string_literal(text: str) -> str:
    # Same choice as ruff: double quotes, unless that needs more escapes than single quotes.
    quote = '"' if text.count('"') <= text.count("'") else "'"
    out = []
    for char in text:
        if char == quote:
            out.append("\\" + char)
        elif char in _SHORT_ESCAPES:
            out.append(_SHORT_ESCAPES[char])
        elif char.isprintable():
            out.append(char)
        elif ord(char) <= 0xFF:
            out.append(f"\\x{ord(char):02x}")
        elif ord(char) <= 0xFFFF:
            out.append(f"\\u{ord(char):04x}")
        else:
            out.append(f"\\U{ord(char):08x}")
    return quote + "".join(out) + quote


@dataclass(frozen=True)
class Assign:
    """`targets = value`, where `targets` is one name or names separated by ", "."""

    targets: str
    value: Expr

    def __post_init__(self) -> None:
        _check(_TARGETS, self.targets, "assignment target")


def render_statements(statements: Sequence[Assign], indent: int = 0) -> list[str]:
    """Source lines for `statements`, each starting at column `indent`."""
    lines: list[str] = []
    for statement in statements:
        lead = " " * indent + f"{statement.targets} = "
        lines += _render(statement.value, indent, lead, "")
    return lines


def _render(expr: Expr, indent: int, lead: str, trail: str) -> list[str]:
    flat = expr.flat()
    explodable = (isinstance(expr, Seq) and expr.items) or (
        isinstance(expr, Call) and (expr.args or expr.kwargs)
    )
    if not explodable or display_width(lead + flat + trail) <= MAX_LINE_WIDTH:
        return [lead + flat + trail]

    inner = indent + INDENT
    pad = " " * inner
    if isinstance(expr, Seq):
        opening, closing = ("(", ")") if expr.is_tuple else ("[", "]")
        tail = ""
        lines = [lead + opening]
        for item in expr.items:
            lines += _render(item, inner, pad, ",")
    else:
        opening, closing = f"{expr.func}(", ")"
        tail = "".join(
            f".{name}({', '.join(f'{k}={v.flat()}' for k, v in kwargs)})"
            for name, kwargs in expr.tail
        )
        lines = [lead + opening]
        for arg in expr.args:
            lines += _render(arg, inner, pad, ",")
        for key, value in expr.kwargs:
            lines += _render(value, inner, f"{pad}{key}=", ",")
    lines.append(" " * indent + closing + tail + trail)
    return lines


def comment_lines(text: str, indent: int = 0) -> list[str]:
    """`text` as `# ` comment lines wrapped to the line width, never more than comments.

    Anything that is not a printable character, line breaks included, becomes a space.
    """
    cleaned = "".join(c if c.isprintable() else " " for c in text)
    words = " ".join(cleaned.split())
    if not words:
        return []
    prefix = " " * indent + "# "
    width = max(MAX_LINE_WIDTH - len(prefix), 20)
    return [prefix + piece for piece in textwrap.wrap(words, width=width, break_on_hyphens=False)]


@dataclass(frozen=True)
class Import:
    """`import module [as alias]`, or `from module import name` when `name` is given."""

    module: str
    name: str | None = None
    alias: str | None = None

    def __post_init__(self) -> None:
        _check(_NAME, self.module, "module name")
        if self.name is not None:
            _check(_SIMPLE_NAME, self.name, "imported name")
            if self.alias is not None:
                raise CodeError("an alias is only supported for plain imports")
        if self.alias is not None:
            _check(_SIMPLE_NAME, self.alias, "alias")


def render_imports(imports: Iterable[Import]) -> list[str]:
    """Import lines in the order `ruff check` (isort) expects, with a blank line between the
    standard library and third-party blocks."""
    stdlib: list[Import] = []
    third_party: list[Import] = []
    for item in sorted(set(imports), key=lambda i: (i.module, i.name or "", i.alias or "")):
        section = stdlib if item.module.split(".")[0] in sys.stdlib_module_names else third_party
        section.append(item)
    blocks = [_render_block(block) for block in (stdlib, third_party) if block]
    lines: list[str] = []
    for block in blocks:
        lines += ([""] if lines else []) + block
    return lines


def _render_block(imports: list[Import]) -> list[str]:
    lines = []
    for item in imports:
        if item.name is None:
            lines.append(f"import {item.module}" + (f" as {item.alias}" if item.alias else ""))
    names_by_module: dict[str, list[str]] = {}
    for item in imports:
        if item.name is not None:
            names_by_module.setdefault(item.module, []).append(item.name)
    for module, names in names_by_module.items():
        names = sorted(names, key=_member_order)
        flat = f"from {module} import {', '.join(names)}"
        if display_width(flat) <= MAX_LINE_WIDTH:
            lines.append(flat)
        else:
            lines += [f"from {module} import (", *[f"    {n}," for n in names], ")"]
    return lines


def _member_order(name: str) -> tuple[int, str]:
    # isort's default: CONSTANTS, then Classes, then functions and variables.
    if name.isupper() and len(name) > 1:
        rank = 0
    elif name[0].isupper():
        rank = 1
    else:
        rank = 2
    return rank, name.lower()
