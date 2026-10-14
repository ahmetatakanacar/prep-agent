"""Codegen: writes the standalone `pipeline.py` of a run.

The script repeats what the executor does, on a raw CSV file, with nothing but pandas and
scikit-learn: the stateless steps, the split, and an sklearn `Pipeline` of the stateful steps
that is fit on train only. Each op writes its own part (`codegen` of its spec, next to the op
itself); this module only puts the parts in order around a fixed frame of functions.

The frame has no data in it. Everything that comes from the plan or the run (column names,
parameters, the target) is rendered by `prep_agent.ops.code` as a literal, and the reasons and
run facts only ever appear in comments, so nothing from a plan is written as code.

The script is laid out the way `ruff format` would write it, so it passes `ruff check` and
`ruff format --check` as it is.
"""

import pandas as pd
import sklearn

from prep_agent.ops import get_op
from prep_agent.ops.code import (
    Assign,
    Import,
    call,
    comment_lines,
    literal,
    ref,
    render_imports,
    render_statements,
)
from prep_agent.plan import Plan, Step

_BODY = 4

_DOCSTRING = [
    '"""Prepare a raw CSV file for modelling with the steps prep-agent chose for this run.',
    "",
    "Usage: python pipeline.py <input.csv> [--out-dir <folder>]",
    "",
    "train.csv and test.csv are written to the folder (default: the current directory).",
    "Existing files are only replaced with --force.",
    "",
    "The pieces can also be imported and called one by one: apply_stateless_steps,",
    "split_train_test, build_pipeline and prepare. Only pandas and scikit-learn are needed.",
    '"""',
]

_READ_INPUT = [
    "def read_input(path: Path) -> pd.DataFrame:",
    '    """Read the raw CSV file the way prep-agent did."""',
    "    return pd.read_csv(path, low_memory=False)",
]

_PREPARE = [
    "def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:",
    '    """Run every step on a raw frame and return the prepared train and test frames."""',
    "    train, test = split_train_test(apply_stateless_steps(df))",
    "    x_train = train.drop(columns=TARGET)",
    "    x_test = test.drop(columns=TARGET)",
    "    y_train = train[TARGET].copy()",
    "    y_test = test[TARGET].copy()",
    "    pipeline = build_pipeline()",
    "    x_train_out = pipeline.fit_transform(x_train)",
    "    x_test_out = pipeline.transform(x_test)",
    "    train_out = pd.concat([x_train_out, y_train], axis=1)",
    "    test_out = pd.concat([x_test_out, y_test], axis=1)",
    "    return train_out, test_out",
]

_MAIN = [
    "def main(argv: list[str] | None = None) -> None:",
    '    parser = argparse.ArgumentParser(description="Prepare a raw CSV file.")',
    '    parser.add_argument("input", type=Path, help="the raw CSV file")',
    "    parser.add_argument(",
    '        "--out-dir",',
    "        type=Path,",
    '        default=Path("."),',
    '        help="folder for train.csv and test.csv (default: the current directory)",',
    "    )",
    "    parser.add_argument(",
    '        "--force", action="store_true", help="overwrite existing train.csv and test.csv"',
    "    )",
    "    args = parser.parse_args(argv)",
    '    existing = [n for n in ("train.csv", "test.csv") if (args.out_dir / n).exists()]',
    "    if existing and not args.force:",
    '        names = " and ".join(existing)',
    '        raise SystemExit(f"error: {names} already exists; use --force to overwrite")',
    "    train, test = prepare(read_input(args.input))",
    "    args.out_dir.mkdir(parents=True, exist_ok=True)",
    '    train.to_csv(args.out_dir / "train.csv", index=False, lineterminator="\\n")',
    '    test.to_csv(args.out_dir / "test.csv", index=False, lineterminator="\\n")',
    '    print(f"Wrote train.csv ({len(train)} rows) and test.csv ({len(test)} rows).")',
]

_ENTRY_POINT = ['if __name__ == "__main__":', "    main()"]


def generate_pipeline_script(
    plan: Plan, *, run_id: str, prep_agent_version: str, input_sha256: str
) -> str:
    """The source of the standalone pipeline.py for `plan`.

    The plan must have passed the plan validator: stateless steps first, then exactly one split,
    then the stateful steps. Steps are taken by kind in plan order, as the executor does.
    """
    steps = [(index, step, get_op(step.op)) for index, step in enumerate(plan.steps)]
    stateless = [(i, s, op) for i, s, op in steps if op.kind == "stateless"]
    splits = [(i, s, op) for i, s, op in steps if op.kind == "split"]
    stateful = [(i, s, op) for i, s, op in steps if op.kind == "stateful"]
    if len(splits) != 1:
        raise ValueError(f"the plan needs exactly one split step, it has {len(splits)}")

    imports = {
        Import("argparse"),
        Import("pathlib", "Path"),
        Import("pandas", alias="pd"),
        Import("sklearn.pipeline", "Pipeline"),
    }
    stateless_body, split_body = [], []
    for index, step, op in [*stateless, *splits]:
        code = op.codegen(step)
        imports.update(code.imports)
        body = _step_comment(index, step) + render_statements(code.statements, _BODY)
        (split_body if op.kind == "split" else stateless_body).extend(body)

    pipeline_body = []
    step_names = []
    for index, step, op in stateful:
        code = op.codegen(step)
        imports.update(code.imports)
        variable = f"step_{index}"
        step_names.append(variable)
        pipeline_body += _step_comment(index, step)
        pipeline_body += render_statements(
            [Assign(variable, literal((f"{index}_{step.op}", code.expression)))], _BODY
        )
    if stateful:
        steps_value = literal([ref(name) for name in step_names])
    else:
        imports.add(Import("sklearn.preprocessing", "FunctionTransformer"))
        steps_value = literal([("identity", call("FunctionTransformer"))])
    pipeline_body += render_statements([Assign("steps", steps_value)], _BODY)

    sections = [
        _header(plan, run_id, prep_agent_version, input_sha256) + _DOCSTRING,
        render_imports(imports),
        render_statements([Assign("TARGET", literal(plan.target))]),
        _READ_INPUT,
        [
            "def apply_stateless_steps(df: pd.DataFrame) -> pd.DataFrame:",
            '    """Apply the steps that learn nothing from the data, before the split."""',
            *stateless_body,
            "    return df",
        ],
        [
            "def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:",
            '    """Split the frame into train and test sets."""',
            *split_body,
            "    return train, test",
        ],
        [
            "def build_pipeline() -> Pipeline:",
            '    """Build the unfitted pipeline of the steps that learn from data."""',
            *pipeline_body,
            '    return Pipeline(steps).set_output(transform="pandas")',
        ],
        _PREPARE,
        _MAIN,
        _ENTRY_POINT,
    ]
    # One blank line between the docstring, the imports and the constant, two before the code.
    text = (
        "\n".join(sections[0]) + "\n\n" + "\n".join(sections[1]) + "\n\n" + "\n".join(sections[2])
    )
    for section in sections[3:]:
        text += "\n\n\n" + "\n".join(section)
    return text + "\n"


def _header(plan: Plan, run_id: str, prep_agent_version: str, input_sha256: str) -> list[str]:
    facts = [
        ("prep-agent version", prep_agent_version),
        ("run_id", run_id),
        ("input sha256", input_sha256),
        ("target", plan.target),
        ("task_type", plan.task_type),
        ("model_family", plan.model_family),
        ("pandas version", pd.__version__),
        ("scikit-learn version", sklearn.__version__),
    ]
    lines = comment_lines("Generated by prep-agent. Needs only pandas and scikit-learn.")
    for name, value in facts:
        lines += comment_lines(f"{name}: {value}")
    return lines


def _step_comment(index: int, step: Step) -> list[str]:
    return comment_lines(f"Step {index} ({step.op}): {step.reason}", _BODY)
