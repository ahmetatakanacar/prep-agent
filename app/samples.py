"""Datasets the UI offers instead of an upload: the test fixture and the downloaded benchmark
datasets that are present on disk."""

from dataclasses import dataclass
from pathlib import Path

from app.config import REPO_ROOT

CUSTOMERS_ID = "customers"
CUSTOMERS_PATH = REPO_ROOT / "tests" / "fixtures" / "customers.csv"
BENCHMARK_DATASETS = REPO_ROOT / "benchmarks" / "datasets"
BENCHMARK_REGISTRY = REPO_ROOT / "benchmarks" / "datasets.yaml"


@dataclass(frozen=True)
class Sample:
    id: str
    label: str
    path: Path
    goal: str
    goal_tr: str | None
    target: str | None
    task_type: str | None
    note: str
    n_rows: int | None = None
    n_columns: int | None = None

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "goal": self.goal,
            "goal_tr": self.goal_tr,
            "target": self.target,
            "task_type": self.task_type,
            "note": self.note,
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
        }


def list_samples(fake: bool) -> list[Sample]:
    """The samples that exist on disk. The demo script only fits the customers file, so the
    demo mode offers nothing else."""
    samples = []
    if CUSTOMERS_PATH.is_file():
        samples.append(
            Sample(
                id=CUSTOMERS_ID,
                label="customers",
                path=CUSTOMERS_PATH,
                goal="I want to predict which customers will churn.",
                goal_tr="Hangi müşterilerin ayrılacağını (churn) tahmin etmek istiyorum.",
                target="churned",
                task_type="classification",
                note="",
                n_rows=20,
                n_columns=7,
            )
        )
    if not fake:
        samples.extend(_benchmark_samples())
    return samples


def find_sample(sample_id: str, fake: bool) -> Sample | None:
    return next((s for s in list_samples(fake) if s.id == sample_id), None)


def _benchmark_samples() -> list[Sample]:
    if not BENCHMARK_DATASETS.is_dir():
        return []
    entries = _read_registry()
    samples = []
    for path in sorted(BENCHMARK_DATASETS.glob("*.csv")):
        entry = entries.get(path.stem, {})
        samples.append(
            Sample(
                id=f"benchmark:{path.stem}",
                label=path.stem,
                path=path,
                goal=entry.get("goal", ""),
                goal_tr=None,
                target=entry.get("target"),
                task_type=entry.get("task_type"),
                note=entry.get("challenge", ""),
                n_rows=entry.get("n_rows"),
                n_columns=entry.get("n_columns"),
            )
        )
    return samples


def _read_registry() -> dict[str, dict]:
    # The registry only adds goals and notes; a missing or unreadable one must not hide the files.
    try:
        import yaml
    except ImportError:
        return {}
    try:
        raw = yaml.safe_load(BENCHMARK_REGISTRY.read_text(encoding="utf-8"))
        return {entry["name"]: entry for entry in raw["datasets"]}
    except (OSError, yaml.YAMLError, KeyError, TypeError, ValueError):
        return {}
