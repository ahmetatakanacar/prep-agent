"""The generated pipeline.py must give exactly the executor's output, with nothing but pandas
and scikit-learn installed."""

import ast
import hashlib
import importlib.util
import io
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sklearn

from prep_agent.codegen import generate_pipeline_script
from prep_agent.executor import execute
from prep_agent.plan import Plan
from prep_agent.profiler import NA_VALUES, load_csv, profile_dataframe

FIXTURES = Path(__file__).parent / "fixtures"
CUSTOMERS_CSV = FIXTURES / "customers.csv"
CUSTOMERS_PLAN = FIXTURES / "customers_plan.json"

RUN_ID = "test-run"
ALLOWED_IMPORTS = {"argparse", "hashlib", "pandas", "pathlib", "sklearn", "sys"}


def raw_frame(n: int = 150, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    num1 = rng.normal(50, 10, n).round(3)
    num1[rng.random(n) < 0.12] = np.nan
    num2 = rng.exponential(5, n).round(3)
    num2[rng.random(n) < 0.15] = np.nan
    cat_low = rng.choice(["basic", "pro", "enterprise"], n).astype(object)
    cat_low[rng.random(n) < 0.1] = None
    label = rng.choice(["yes", "no"], n, p=[0.35, 0.65]).astype(object)
    label[rng.random(n) < 0.05] = None
    price = (num1 * 2).round(2)
    price[rng.random(n) < 0.05] = np.nan
    # Categories that occur once: one of them is likely to end up in test only, where the
    # encoders have to deal with a category they never saw in train.
    cat_high = rng.choice([f"city_{i}" for i in range(12)], n).astype(object)
    cat_high[20:30] = [f"rare_{i}" for i in range(10)]
    df = pd.DataFrame(
        {
            "id": np.arange(n),
            "num1": num1,
            "num2": num2,
            "num3": rng.integers(0, 40, n),
            "n_items": rng.integers(1, 9, n),
            "flag": rng.random(n) < 0.5,
            "cat_low": cat_low,
            "cat_high": cat_high,
            "cat_ord": rng.choice(["S", "M", "L", "XL"], n),
            "const": "x",
            "label": label,
            "price": price,
        }
    )
    return pd.concat([df, df.iloc[:10]], ignore_index=True)


def split(**extra):
    return {"op": "split", "stratify": False, **extra}


PLANS = {
    "everything_classification": {
        "target": "label",
        "task_type": "classification",
        "model_family": "any",
        "steps": [
            {"op": "drop_columns", "columns": ["id", "const"]},
            {"op": "dedupe", "subset": None, "keep": "first"},
            {"op": "drop_missing_target"},
            split(stratify=True, test_size=0.25, random_state=3),
            {"op": "impute", "columns": ["num1"], "strategy": "mean", "add_indicator": True},
            {"op": "impute", "columns": ["num2"], "strategy": "median"},
            {"op": "impute", "columns": ["cat_low"], "strategy": "most_frequent"},
            {"op": "encode", "columns": ["cat_low"], "method": "onehot"},
            {"op": "encode", "columns": ["cat_high"], "method": "onehot", "max_categories": 5},
            {"op": "encode", "columns": ["cat_ord"], "method": "ordinal"},
            {"op": "scale", "columns": ["num1", "num2", "num3"], "method": "standard"},
        ],
    },
    "dedupe_subset_constants_minmax": {
        "target": "label",
        "task_type": "classification",
        "model_family": "any",
        "steps": [
            {"op": "drop_columns", "columns": ["id"]},
            {"op": "dedupe", "subset": ["num3", "cat_ord"], "keep": "last"},
            {"op": "drop_missing_target"},
            split(test_size=0.3, random_state=7),
            {"op": "impute", "columns": ["num1"], "strategy": "constant", "fill_value": -1},
            {"op": "impute", "columns": ["num2"], "strategy": "constant", "fill_value": 2.5},
            {
                "op": "impute",
                "columns": ["cat_low"],
                "strategy": "constant",
                "fill_value": "unknown",
            },
            {"op": "encode", "columns": ["cat_low", "cat_high"], "method": "onehot"},
            {"op": "scale", "columns": ["num3"], "method": "minmax"},
        ],
    },
    "regression_robust": {
        "target": "price",
        "task_type": "regression",
        "model_family": "linear",
        "steps": [
            {"op": "drop_columns", "columns": ["id", "const", "label"]},
            {"op": "drop_missing_target"},
            split(random_state=11),
            {"op": "impute", "columns": ["num1"], "strategy": "mean"},
            {"op": "impute", "columns": ["num2"], "strategy": "median"},
            {"op": "impute", "columns": ["cat_low"], "strategy": "most_frequent"},
            {"op": "encode", "columns": ["cat_ord"], "method": "ordinal"},
            {
                "op": "encode",
                "columns": ["cat_low", "cat_high"],
                "method": "onehot",
                "max_categories": 4,
            },
            {"op": "scale", "columns": ["num1", "num2", "num3"], "method": "robust"},
        ],
    },
    "no_stateful_steps": {
        "target": "label",
        "task_type": "classification",
        "model_family": "tree",
        "steps": [
            {"op": "drop_columns", "columns": ["id", "const"]},
            {"op": "drop_missing_target"},
            split(stratify=True),
        ],
    },
    "no_stateless_steps": {
        "target": "price",
        "task_type": "regression",
        "model_family": "any",
        "steps": [split(test_size=0.1), {"op": "scale", "columns": ["num3"], "method": "standard"}],
    },
    "indicator_then_ordinal": {
        "target": "price",
        "task_type": "regression",
        "model_family": "tree",
        "steps": [
            {"op": "drop_columns", "columns": ["id", "const", "label"]},
            {"op": "drop_missing_target"},
            split(),
            {
                "op": "impute",
                "columns": ["cat_low"],
                "strategy": "most_frequent",
                "add_indicator": True,
            },
            {"op": "impute", "columns": ["num1", "num2"], "strategy": "median"},
            {"op": "encode", "columns": ["cat_low", "cat_ord", "cat_high"], "method": "ordinal"},
        ],
    },
}

HARD_NAMES = {
    "age": "müşteri yaşı",
    "apostrophe": "it's",
    "quoted": 'say "hi"',
    "backslash": "back\\slash",
    "injection": 'x"); import os  #',
    "tab": "tab\there",
    "newline": "new\nline",
    "cjk": "日本語",
    "payload": 'a"); open("pwned.txt", "w").close(); ("',
    "target": 'hedef (y); import sys  #"',
}


def hard_frame(n: int = 120, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.normal(40, 8, n).round(2)
    age[rng.random(n) < 0.1] = np.nan
    injection = rng.choice(["p", "q", "r"], n).astype(object)
    injection[rng.random(n) < 0.1] = None
    return pd.DataFrame(
        {
            HARD_NAMES["age"]: age,
            HARD_NAMES["apostrophe"]: rng.choice(["a", "b", "c"], n),
            HARD_NAMES["quoted"]: rng.normal(0, 1, n).round(3),
            HARD_NAMES["backslash"]: rng.choice(["u", "v"], n),
            HARD_NAMES["injection"]: injection,
            HARD_NAMES["tab"]: rng.integers(0, 5, n),
            HARD_NAMES["newline"]: rng.normal(10, 3, n).round(3),
            HARD_NAMES["cjk"]: rng.normal(5, 2, n).round(3),
            HARD_NAMES["payload"]: rng.normal(1, 1, n).round(3),
            HARD_NAMES["target"]: rng.choice([0, 1], n),
        }
    )


def hard_plan() -> dict:
    n = HARD_NAMES
    return {
        "target": n["target"],
        "task_type": "classification",
        "model_family": "any",
        "steps": [
            {"op": "drop_columns", "columns": [n["tab"]]},
            {"op": "dedupe", "subset": [n["backslash"], n["age"]], "keep": "first"},
            {"op": "drop_missing_target"},
            split(stratify=True),
            {"op": "impute", "columns": [n["age"]], "strategy": "median"},
            {"op": "impute", "columns": [n["injection"]], "strategy": "most_frequent"},
            {
                "op": "encode",
                "columns": [n["apostrophe"], n["backslash"], n["injection"]],
                "method": "onehot",
            },
            {
                "op": "scale",
                "columns": [n["age"], n["quoted"], n["newline"], n["cjk"], n["payload"]],
                "method": "standard",
            },
        ],
    }


def build_plan(raw: dict) -> Plan:
    return Plan.model_validate(
        {
            **raw,
            "steps": [{"reason": f"reason of step {i}", **s} for i, s in enumerate(raw["steps"])],
        }
    )


def reference(csv_path: Path, plan: Plan):
    df = load_csv(csv_path)
    return execute(df, plan, profile_dataframe(df, target=plan.target))


def make_script(tmp_path: Path, csv_path: Path, plan: Plan) -> Path:
    sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    script = tmp_path / "pipeline.py"
    script.write_text(
        generate_pipeline_script(plan, run_id=RUN_ID, prep_agent_version="9.9.9", input_sha256=sha),
        encoding="utf-8",
        newline="\n",
    )
    return script


def run_python(args: list, cwd: Path) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return subprocess.run(
        [sys.executable, "-W", "error", *map(str, args)],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
        check=False,
    )


def csv_text(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


@pytest.fixture
def workdirs(tmp_path):
    """A folder for the script and a different one to run it in, away from the repository."""
    script_dir, run_dir = tmp_path / "script", tmp_path / "elsewhere"
    script_dir.mkdir()
    run_dir.mkdir()
    return script_dir, run_dir


def check_subprocess_run(workdirs, csv_path: Path, plan: Plan) -> None:
    script_dir, run_dir = workdirs
    expected = reference(csv_path, plan)
    script = make_script(script_dir, csv_path, plan)
    out_dir = run_dir / "out"

    done = run_python([script, csv_path, "--out-dir", out_dir], cwd=run_dir)

    assert done.returncode == 0, done.stderr
    for name, frame in (("train.csv", expected.train), ("test.csv", expected.test)):
        produced = out_dir / name
        assert produced.read_bytes() == csv_text(frame), name
        pd.testing.assert_frame_equal(
            pd.read_csv(produced), pd.read_csv(io.BytesIO(csv_text(frame)))
        )


def load_module(script: Path):
    spec = importlib.util.spec_from_file_location("generated_pipeline", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- equality with the executor ---


def test_fixture_plan_matches_the_executor(workdirs):
    plan = Plan.model_validate_json(CUSTOMERS_PLAN.read_text(encoding="utf-8"))

    check_subprocess_run(workdirs, CUSTOMERS_CSV, plan)


@pytest.mark.parametrize("name", PLANS)
def test_generated_script_matches_the_executor(workdirs, tmp_path, name):
    csv_path = tmp_path / "raw.csv"
    raw_frame().to_csv(csv_path, index=False)

    check_subprocess_run(workdirs, csv_path, build_plan(PLANS[name]))


@pytest.mark.parametrize("name", PLANS)
def test_generated_functions_match_the_executor_exactly(tmp_path, name):
    csv_path = tmp_path / "raw.csv"
    raw_frame().to_csv(csv_path, index=False)
    plan = build_plan(PLANS[name])
    expected = reference(csv_path, plan)

    module = load_module(make_script(tmp_path, csv_path, plan))
    train, test = module.prepare(module.read_input(csv_path))

    pd.testing.assert_frame_equal(train, expected.train)
    pd.testing.assert_frame_equal(test, expected.test)


@pytest.mark.parametrize(
    "name", ["everything_classification", "regression_robust", "indicator_then_ordinal"]
)
def test_the_test_split_has_categories_unseen_in_train(tmp_path, name):
    # Guards the equality tests: they only exercise handle_unknown if this holds.
    csv_path = tmp_path / "raw.csv"
    raw_frame().to_csv(csv_path, index=False)
    expected = reference(csv_path, build_plan(PLANS[name]))

    unseen = set(expected.split_test["cat_high"]) - set(expected.split_train["cat_high"])

    assert unseen


def test_the_steps_can_be_called_one_by_one(tmp_path):
    plan = Plan.model_validate_json(CUSTOMERS_PLAN.read_text(encoding="utf-8"))
    expected = reference(CUSTOMERS_CSV, plan)
    module = load_module(make_script(tmp_path, CUSTOMERS_CSV, plan))

    stateless = module.apply_stateless_steps(module.read_input(CUSTOMERS_CSV))
    train, test = module.split_train_test(stateless)
    pipeline = module.build_pipeline()

    pd.testing.assert_frame_equal(train, expected.split_train)
    pd.testing.assert_frame_equal(test, expected.split_test)
    pd.testing.assert_frame_equal(
        pipeline.fit_transform(train.drop(columns=module.TARGET)),
        expected.train.drop(columns=plan.target),
    )
    assert [name for name, _ in pipeline.steps] == [name for name, _ in expected.pipeline.steps]


def test_the_default_output_folder_is_the_working_directory(workdirs):
    script_dir, run_dir = workdirs
    plan = Plan.model_validate_json(CUSTOMERS_PLAN.read_text(encoding="utf-8"))
    script = make_script(script_dir, CUSTOMERS_CSV, plan)

    done = run_python([script, CUSTOMERS_CSV], cwd=run_dir)

    assert done.returncode == 0, done.stderr
    assert (run_dir / "train.csv").is_file()
    assert (run_dir / "test.csv").is_file()
    assert "train.csv" in done.stdout


def run_script_in(workdirs, *args):
    script_dir, run_dir = workdirs
    plan = Plan.model_validate_json(CUSTOMERS_PLAN.read_text(encoding="utf-8"))
    script = script_dir / "pipeline.py"
    if not script.exists():
        make_script(script_dir, CUSTOMERS_CSV, plan)
    return run_python([script, CUSTOMERS_CSV, *args], cwd=run_dir)


def test_a_second_run_into_the_same_folder_is_refused(workdirs):
    _, run_dir = workdirs
    assert run_script_in(workdirs).returncode == 0
    before = {n: (run_dir / n).read_bytes() for n in ("train.csv", "test.csv")}

    again = run_script_in(workdirs)

    assert again.returncode != 0
    assert "train.csv and test.csv already exist;" in again.stderr
    assert "--force" in again.stderr
    assert "Traceback" not in again.stderr
    assert {n: (run_dir / n).read_bytes() for n in before} == before


def test_force_overwrites_the_existing_files(workdirs):
    _, run_dir = workdirs
    assert run_script_in(workdirs).returncode == 0
    (run_dir / "train.csv").write_text("stale", encoding="utf-8")

    forced = run_script_in(workdirs, "--force")

    assert forced.returncode == 0, forced.stderr
    assert (run_dir / "train.csv").read_text(encoding="utf-8") != "stale"


@pytest.mark.parametrize("existing", ["train.csv", "test.csv"])
def test_one_existing_file_is_named_and_nothing_is_written(workdirs, existing):
    _, run_dir = workdirs
    other = "test.csv" if existing == "train.csv" else "train.csv"
    (run_dir / existing).write_text("mine", encoding="utf-8")

    done = run_script_in(workdirs)

    assert done.returncode != 0
    assert f"{existing} already exists;" in done.stderr
    assert other not in done.stderr
    assert (run_dir / existing).read_text(encoding="utf-8") == "mine"
    assert not (run_dir / other).exists()


def test_the_input_hash_is_a_constant_equal_to_the_header(tmp_path):
    csv_path = tmp_path / "raw.csv"
    raw_frame().to_csv(csv_path, index=False)
    sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()

    source = make_script(tmp_path, csv_path, build_plan(PLANS["no_stateful_steps"])).read_text(
        encoding="utf-8"
    )

    assert f"# input sha256: {sha}" in source.splitlines()
    assert f'INPUT_SHA256 = "{sha}"' in source.splitlines()
    assert load_module(tmp_path / "pipeline.py").INPUT_SHA256 == sha


def test_the_same_data_gives_no_warning(workdirs):
    done = run_script_in(workdirs)

    assert done.returncode == 0, done.stderr
    assert done.stderr == ""


def test_other_data_warns_on_stderr_and_still_runs(workdirs, tmp_path):
    script_dir, run_dir = workdirs
    other = tmp_path / "other.csv"
    other.write_text(CUSTOMERS_CSV.read_text(encoding="utf-8"), encoding="utf-8")
    with other.open("a", encoding="utf-8", newline="") as handle:
        handle.write("\n")
    plan = Plan.model_validate_json(CUSTOMERS_PLAN.read_text(encoding="utf-8"))
    script = make_script(script_dir, CUSTOMERS_CSV, plan)

    done = run_python([script, other, "--out-dir", run_dir / "out"], cwd=run_dir)

    assert done.returncode == 0, done.stderr
    assert "differs" in done.stderr
    assert "Traceback" not in done.stderr
    assert (run_dir / "out" / "train.csv").is_file()
    assert (run_dir / "out" / "test.csv").is_file()
    assert "differs" not in done.stdout


# --- independence from prep-agent ---


def imported_modules(source: str) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            found.add(node.module.split(".")[0])
    return found


@pytest.mark.parametrize("name", PLANS)
def test_the_script_imports_only_pandas_scikit_learn_and_the_standard_library(tmp_path, name):
    csv_path = tmp_path / "raw.csv"
    raw_frame().to_csv(csv_path, index=False)

    source = make_script(tmp_path, csv_path, build_plan(PLANS[name])).read_text(encoding="utf-8")

    assert imported_modules(source) <= ALLOWED_IMPORTS
    assert "prep_agent" not in source


def test_the_script_runs_where_prep_agent_cannot_be_imported(workdirs):
    script_dir, run_dir = workdirs
    plan = Plan.model_validate_json(CUSTOMERS_PLAN.read_text(encoding="utf-8"))
    expected = reference(CUSTOMERS_CSV, plan)
    script = make_script(script_dir, CUSTOMERS_CSV, plan)
    # prep_agent is installed in this environment, so block it: a None entry in sys.modules
    # makes every import of it fail.
    runner = (
        "import runpy, sys\n"
        "sys.modules['prep_agent'] = None\n"
        "sys.argv = sys.argv[1:]\n"
        "runpy.run_path(sys.argv[0], run_name='__main__')\n"
    )

    blocked = run_python(["-c", "import prep_agent"], cwd=run_dir)
    assert blocked.returncode == 0, "sanity check: prep_agent is importable here"
    done = run_python(
        ["-c", runner, script, CUSTOMERS_CSV, "--out-dir", run_dir / "out"], cwd=run_dir
    )

    assert done.returncode == 0, done.stderr
    assert (run_dir / "out" / "train.csv").read_bytes() == csv_text(expected.train)


# --- hostile column names and reasons ---


def test_hard_column_names_work_and_are_never_executed(workdirs, tmp_path):
    csv_path = tmp_path / "hard.csv"
    hard_frame().to_csv(csv_path, index=False)
    plan = build_plan(hard_plan())

    check_subprocess_run(workdirs, csv_path, plan)

    script_dir, run_dir = workdirs
    source = (script_dir / "pipeline.py").read_text(encoding="utf-8")
    assert not (run_dir / "pwned.txt").exists()
    assert not (script_dir / "pwned.txt").exists()
    # The names are in the file only as string literals.
    tree = ast.parse(source)
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert "os" not in imported
    assert imported_modules(source) <= ALLOWED_IMPORTS
    literals = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)}
    assert set(HARD_NAMES.values()) <= literals


def test_hard_column_names_work_when_imported(tmp_path):
    csv_path = tmp_path / "hard.csv"
    hard_frame().to_csv(csv_path, index=False)
    plan = build_plan(hard_plan())
    expected = reference(csv_path, plan)

    module = load_module(make_script(tmp_path, csv_path, plan))
    train, test = module.prepare(module.read_input(csv_path))

    pd.testing.assert_frame_equal(train, expected.train)
    pd.testing.assert_frame_equal(test, expected.test)
    assert module.TARGET == HARD_NAMES["target"]


EVIL_REASONS = [
    "fine\nimport os\nos.system('echo pwned')",
    "fine\r\nimport os\r\n",
    "fine import os import sys",
    "fine\x0bimport os\x0cimport sys\x1cx\x85y",
    "fine \\\nimport os",
    "a" * 300 + "\nimport os",
    "bidi ‮ import os ⁦",
    "\n\n\n",
]


@pytest.mark.parametrize("reason", EVIL_REASONS)
def test_a_reason_cannot_leave_its_comment(tmp_path, reason):
    raw = {**PLANS["no_stateful_steps"]}
    plain = build_plan(raw)
    evil = plain.model_copy(deep=True)
    evil.steps[0] = evil.steps[0].model_copy(update={"reason": reason})

    plain_source = generate_pipeline_script(
        plain, run_id=RUN_ID, prep_agent_version="9.9.9", input_sha256="0" * 64
    )
    evil_source = generate_pipeline_script(
        evil, run_id=RUN_ID, prep_agent_version="9.9.9", input_sha256="0" * 64
    )

    assert ast.dump(ast.parse(evil_source)) == ast.dump(ast.parse(plain_source))
    assert "import os" not in [line.strip() for line in evil_source.splitlines()]
    for line in evil_source.splitlines():
        if "import os" in line or "pwned" in line:
            assert line.lstrip().startswith("#"), line
    assert all(ch.isprintable() or ch == "\n" for ch in evil_source)


def test_hostile_text_in_the_header_cannot_leave_its_comment():
    plan = Plan.model_validate(
        {
            **PLANS["no_stateful_steps"],
            "target": "label",
            "steps": [{"reason": "r", **s} for s in PLANS["no_stateful_steps"]["steps"]],
        }
    )

    source = generate_pipeline_script(
        plan, run_id="r1", prep_agent_version="1\nimport os", input_sha256="abc\nimport sys"
    )

    assert imported_modules(source) <= ALLOWED_IMPORTS


# --- content ---


def test_header_and_reasons(tmp_path):
    csv_path = tmp_path / "raw.csv"
    raw_frame().to_csv(csv_path, index=False)
    plan = build_plan(PLANS["everything_classification"])

    source = make_script(tmp_path, csv_path, plan).read_text(encoding="utf-8")
    lines = source.splitlines()
    header = "\n".join(lines[:10])

    assert lines[0].startswith("#")
    assert "9.9.9" in header
    assert RUN_ID in header
    assert hashlib.sha256(csv_path.read_bytes()).hexdigest() in header
    assert "label" in header
    assert "classification" in header
    assert "any" in header
    assert pd.__version__ in header
    assert sklearn.__version__ in header
    # Each step's reason is a comment, in plan order, right above the step's code.
    positions = []
    for index in range(len(plan.steps)):
        [position] = [
            i for i, line in enumerate(lines) if line.endswith(f": reason of step {index}")
        ]
        assert lines[position].lstrip().startswith("#")
        positions.append(position)
    assert positions == sorted(positions)


def cast_plan(*cast_steps) -> Plan:
    return build_plan(
        {
            "target": "label",
            "task_type": "classification",
            "model_family": "tree",
            "steps": [*cast_steps, split(stratify=True)],
        }
    )


def test_a_cast_step_makes_the_script_define_its_helpers_once():
    plan = cast_plan(
        {"op": "cast_to_category", "columns": ["num3"]},
        {"op": "cast_to_category", "columns": ["n_items"]},
    )

    source = generate_pipeline_script(
        plan, run_id=RUN_ID, prep_agent_version="1", input_sha256="0" * 64
    )

    assert source.count("def cast_to_category(") == 1
    assert source.count("def category_text(") == 1
    assert source.count("cast_to_category(df, ") == 2
    assert source.index("def cast_to_category(") < source.index("def apply_stateless_steps(")


def test_a_plan_without_a_cast_step_has_no_helpers():
    source = generate_pipeline_script(
        build_plan(PLANS["no_stateful_steps"]),
        run_id=RUN_ID,
        prep_agent_version="1",
        input_sha256="0" * 64,
    )

    assert "cast_to_category" not in source
    assert "category_text" not in source


def test_a_plan_without_a_split_cannot_be_generated():
    plan = build_plan(
        {**PLANS["no_stateful_steps"], "steps": PLANS["no_stateful_steps"]["steps"][:2]}
    )

    with pytest.raises(ValueError, match="split"):
        generate_pipeline_script(plan, run_id=RUN_ID, prep_agent_version="1", input_sha256="0")


def test_generation_is_deterministic():
    plan = build_plan(PLANS["everything_classification"])
    args = {"run_id": RUN_ID, "prep_agent_version": "1", "input_sha256": "0" * 64}

    assert generate_pipeline_script(plan, **args) == generate_pipeline_script(plan, **args)


# --- style ---

# Run with ruff's own defaults and with the project's rules at its line length and at 88, the
# width ruff formats to when a script is checked outside this repository.
RUFF_CONFIGS = {
    "defaults": ([], []),
    "project": (["--select", "E,F,I,B,UP", "--line-length", "100"], ["--line-length", "100"]),
    "width-88": (["--select", "E,F,I,B,UP", "--line-length", "88"], ["--line-length", "88"]),
}


@pytest.mark.parametrize("config", RUFF_CONFIGS)
@pytest.mark.parametrize("name", [*PLANS, "hard_names"])
def test_generated_script_passes_ruff(tmp_path, name, config):
    csv_path = tmp_path / "raw.csv"
    if name == "hard_names":
        hard_frame().to_csv(csv_path, index=False)
        plan = build_plan(hard_plan())
    else:
        raw_frame().to_csv(csv_path, index=False)
        plan = build_plan(PLANS[name])
    script = make_script(tmp_path, csv_path, plan)
    check_flags, format_flags = RUFF_CONFIGS[config]

    check = run_python(
        ["-m", "ruff", "check", "--isolated", "--no-cache", *check_flags, script], tmp_path
    )
    fmt = run_python(
        [
            "-m",
            "ruff",
            "format",
            "--isolated",
            "--no-cache",
            "--check",
            "--diff",
            *format_flags,
            script,
        ],
        tmp_path,
    )

    assert check.returncode == 0, check.stdout + check.stderr
    assert fmt.returncode == 0, fmt.stdout + fmt.stderr


# --- the text "None" ---

NONE_PLAN = {
    "target": "t",
    "task_type": "classification",
    "model_family": "any",
    "steps": [
        split(stratify=True, test_size=0.25),
        {"op": "impute", "columns": ["veneer"], "strategy": "most_frequent"},
        {"op": "encode", "columns": ["veneer"], "method": "onehot"},
    ],
}


def write_none_csv(tmp_path: Path) -> Path:
    rows = ["id,veneer,t"] + [
        f"{i},{['None', 'Stone', '', 'NA', 'BrkFace'][i % 5]},{i % 2}" for i in range(60)
    ]
    csv_path = tmp_path / "none.csv"
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return csv_path


def test_the_script_reads_none_and_missing_values_as_the_loader_does(tmp_path):
    csv_path = write_none_csv(tmp_path)
    module = load_module(make_script(tmp_path, csv_path, build_plan(NONE_PLAN)))

    produced = module.read_input(csv_path)

    pd.testing.assert_frame_equal(produced, load_csv(csv_path))
    assert (produced["veneer"] == "None").sum() == 12
    assert produced["veneer"].isna().sum() == 24
    assert module.NA_VALUES == list(NA_VALUES)


def test_a_none_category_gives_the_same_result_in_the_script_and_the_executor(workdirs, tmp_path):
    csv_path = write_none_csv(tmp_path)
    plan = build_plan(NONE_PLAN)

    check_subprocess_run(workdirs, csv_path, plan)

    assert "veneer_None" in reference(csv_path, plan).train.columns
