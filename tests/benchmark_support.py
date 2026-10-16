"""Two tiny datasets, their plans and a scripted planner, shared by the benchmark tests."""

from pathlib import Path

import pandas as pd
import yaml

from bench_datasets import DatasetSpec
from prep_agent.baseline import BaselineResult, BaselineScore, ModelBaseline
from prep_agent.plan import Plan
from prep_agent.planner import PlannerResponse, Question


def classification_frame(n=80) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "uid": range(n),
            "age": [float(20 + i % 30) if i % 7 else float("nan") for i in range(n)],
            "city": [["a", "b", "c", "d"][i % 4] for i in range(n)],
            "income": [float((i * 37) % 101) for i in range(n)],
            "y": ["yes" if i % 2 else "no" for i in range(n)],
        }
    )


def regression_frame(n=60) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rid": range(n),
            "size": [i * 1.5 for i in range(n)],
            "kind": [["x", "y", "z"][i % 3] for i in range(n)],
            "price": [3 * i * 1.5 + i % 5 for i in range(n)],
        }
    )


def classification_plan(drop="uid") -> dict:
    return {
        "target": "y",
        "task_type": "classification",
        "model_family": "any",
        "steps": [
            {"op": "drop_columns", "columns": [drop], "reason": "The id carries no signal."},
            {"op": "split", "test_size": 0.2, "stratify": True, "random_state": 42, "reason": "s"},
            {"op": "impute", "columns": ["age"], "strategy": "median", "reason": "Age has gaps."},
            {"op": "encode", "columns": ["city"], "method": "onehot", "reason": "Text column."},
        ],
    }


def regression_plan() -> dict:
    return {
        "target": "price",
        "task_type": "regression",
        "model_family": "any",
        "steps": [
            {"op": "drop_columns", "columns": ["rid"], "reason": "The id carries no signal."},
            {"op": "split", "test_size": 0.2, "stratify": False, "random_state": 42, "reason": "s"},
            {"op": "encode", "columns": ["kind"], "method": "onehot", "reason": "Text column."},
        ],
    }


def answer(plan: dict) -> PlannerResponse:
    return PlannerResponse(language="English", kind="plan", plan=Plan.model_validate(plan))


def questions_answer() -> PlannerResponse:
    question = Question(
        id="q1", question="Keep outliers?", options=["yes", "no"], recommended="no", reason="r"
    )
    return PlannerResponse(language="English", kind="questions", questions=[question])


def spec(name, target, task_type, n_rows, n_columns) -> DatasetSpec:
    return DatasetSpec(
        name=name,
        openml_id={"cls": 1, "reg": 2}.get(name, 3),
        openml_name=name,
        openml_version=1,
        target=target,
        task_type=task_type,
        goal=f"Prepare {name} for modelling.",
        license="Public",
        challenge="Tiny.",
        n_rows=n_rows,
        n_columns=n_columns,
    )


def write_workspace(root: Path) -> list[DatasetSpec]:
    """Write the two datasets, their reference plans and a registry under `root`."""
    (root / "datasets").mkdir()
    (root / "reference").mkdir()
    classification_frame().to_csv(root / "datasets" / "cls.csv", index=False)
    regression_frame().to_csv(root / "datasets" / "reg.csv", index=False)
    for name, plan in (("cls", classification_plan()), ("reg", regression_plan())):
        (root / "reference" / f"{name}.plan.json").write_text(
            Plan.model_validate(plan).model_dump_json(), encoding="utf-8"
        )
    specs = [
        spec("cls", "y", "classification", 80, 5),
        spec("reg", "price", "regression", 60, 4),
    ]
    (root / "datasets.yaml").write_text(
        yaml.safe_dump({"datasets": [s.model_dump() for s in specs]}), encoding="utf-8"
    )
    return specs


def stub_baseline(train, pipeline, target, task_type, **kwargs) -> BaselineResult:
    """A canned baseline: the real cross-validation takes seconds per plan."""
    names = (
        ("dummy_most_frequent", "logistic_regression", "random_forest")
        if task_type == "classification"
        else ("dummy_mean", "ridge", "random_forest")
    )
    metric = "f1_macro" if task_type == "classification" else "r2"
    return BaselineResult(
        task_type=task_type,
        n_rows=len(train),
        random_state=42,
        n_folds=5,
        metrics=[metric],
        models=[
            ModelBaseline(name=name, scores={metric: BaselineScore(mean=0.1 * i, std=0.0)})
            for i, name in enumerate(names)
        ],
    )
