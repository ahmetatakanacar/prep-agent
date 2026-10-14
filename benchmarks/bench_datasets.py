"""The benchmark's dataset registry: which OpenML datasets it uses and how they are described.

`datasets.yaml` is the single source. Each entry names one OpenML dataset by id (names and
versions are only unique together, the id is exact), the target column, the task type, a goal
sentence written the way a user would write it, the license as OpenML reports it and one
sentence on what makes the dataset a useful test. `leaky_columns` names the columns that give
the target away, if any. `n_rows` and `n_columns` (the target included) are what the dataset
had when it was verified; the downloader refuses a file that does not match, so a changed
dataset cannot slip in unnoticed.
"""

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

BENCHMARKS_DIR = Path(__file__).resolve().parent
DEFAULT_REGISTRY = BENCHMARKS_DIR / "datasets.yaml"
DEFAULT_DATASETS_DIR = BENCHMARKS_DIR / "datasets"
DEFAULT_REFERENCE_DIR = BENCHMARKS_DIR / "reference"

# The name becomes a file name, so it is restricted to what is safe on every platform.
Name = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_-]*$")]
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


def _distinct(names: list[str]) -> list[str]:
    if len(set(names)) != len(names):
        raise ValueError("column names must be distinct")
    return names


class RegistryError(ValueError):
    """The registry file is missing, unreadable or wrong, or a dataset name is unknown."""


class DatasetSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Name
    openml_id: int = Field(gt=0)
    openml_name: Text
    openml_version: int = Field(gt=0)
    target: Text
    task_type: Literal["classification", "regression"]
    goal: Text
    license: Text
    challenge: Text
    n_rows: int = Field(gt=0)
    n_columns: int = Field(gt=0)
    # Columns that give the target away (recorded after the outcome, or derived from it). A
    # plan that keeps one scores higher for a reason that has nothing to do with preparation,
    # so the results flag such plans.
    leaky_columns: Annotated[list[Text], AfterValidator(_distinct)] = Field(default_factory=list)

    @model_validator(mode="after")
    def _target_does_not_leak(self) -> "DatasetSpec":
        if self.target in self.leaky_columns:
            raise ValueError("leaky_columns must not contain the target")
        return self


def load_registry(path: Path = DEFAULT_REGISTRY) -> list[DatasetSpec]:
    """Read and validate the registry; the order of the file is kept."""
    path = Path(path)
    if not path.exists():
        raise RegistryError(f"dataset registry not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise RegistryError(f"could not read the dataset registry {path}: {exc}") from exc
    entries = raw.get("datasets") if isinstance(raw, dict) else None
    if not entries:
        raise RegistryError(f"the dataset registry {path} lists no datasets")
    try:
        specs = [DatasetSpec.model_validate(entry) for entry in entries]
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        raise RegistryError(f"invalid dataset registry {path}: {problems}") from None

    _reject_duplicates([spec.name for spec in specs], "dataset name")
    _reject_duplicates([spec.openml_id for spec in specs], "OpenML id")
    return specs


def select_datasets(specs: list[DatasetSpec], names: list[str] | None) -> list[DatasetSpec]:
    """The named subset in registry order; all of them when `names` is None."""
    if names is None:
        return list(specs)
    known = {spec.name for spec in specs}
    unknown = [name for name in names if name not in known]
    if unknown:
        raise RegistryError(
            f"unknown dataset(s): {', '.join(unknown)}; known: {', '.join(sorted(known))}"
        )
    return [spec for spec in specs if spec.name in set(names)]


def csv_path(spec: DatasetSpec, datasets_dir: Path) -> Path:
    return Path(datasets_dir) / f"{spec.name}.csv"


def reference_plan_path(spec: DatasetSpec, reference_dir: Path) -> Path:
    return Path(reference_dir) / f"{spec.name}.plan.json"


def _reject_duplicates(values: list, what: str) -> None:
    seen = set()
    for value in values:
        if value in seen:
            raise RegistryError(f"duplicate {what}: {value!r}")
        seen.add(value)
