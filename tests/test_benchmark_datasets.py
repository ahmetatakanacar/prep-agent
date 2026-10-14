from pathlib import Path

import pytest
import yaml

from bench_datasets import (
    DEFAULT_REGISTRY,
    DatasetSpec,
    RegistryError,
    csv_path,
    load_registry,
    reference_plan_path,
    select_datasets,
)


def _entry(**overrides) -> dict:
    entry = {
        "name": "toy",
        "openml_id": 1,
        "openml_name": "toy",
        "openml_version": 1,
        "target": "y",
        "task_type": "classification",
        "goal": "Predict y.",
        "license": "Public",
        "challenge": "Nothing in particular.",
        "n_rows": 10,
        "n_columns": 3,
    }
    return entry | overrides


def _write(tmp_path: Path, entries: list[dict]) -> Path:
    path = tmp_path / "datasets.yaml"
    path.write_text(yaml.safe_dump({"datasets": entries}), encoding="utf-8")
    return path


def test_load_registry_reads_every_field(tmp_path):
    [spec] = load_registry(_write(tmp_path, [_entry()]))
    assert spec == DatasetSpec(**_entry())


def test_load_registry_keeps_file_order(tmp_path):
    names = ["b", "a", "c"]
    specs = load_registry(
        _write(tmp_path, [_entry(name=n, openml_id=i) for i, n in enumerate(names, start=1)])
    )
    assert [spec.name for spec in specs] == names


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        ([_entry(), _entry(openml_id=2)], "duplicate dataset name"),
        ([_entry(), _entry(name="other")], "duplicate OpenML id"),
        ([_entry(name="Has Space")], "name"),
        ([_entry(name="../escape")], "name"),
        ([_entry(task_type="clustering")], "task_type"),
        ([_entry(n_rows=0)], "n_rows"),
        ([_entry(goal="  ")], "goal"),
        ([{**_entry(), "surprise": 1}], "surprise"),
    ],
)
def test_load_registry_rejects_a_bad_entry(tmp_path, entries, message):
    with pytest.raises(RegistryError, match=message):
        load_registry(_write(tmp_path, entries))


def test_load_registry_rejects_an_empty_list(tmp_path):
    with pytest.raises(RegistryError, match="no datasets"):
        load_registry(_write(tmp_path, []))


def test_load_registry_reports_a_missing_file(tmp_path):
    with pytest.raises(RegistryError, match="not found"):
        load_registry(tmp_path / "nope.yaml")


def test_load_registry_reports_invalid_yaml(tmp_path):
    path = tmp_path / "datasets.yaml"
    path.write_text("datasets: [unclosed", encoding="utf-8")
    with pytest.raises(RegistryError, match="could not read"):
        load_registry(path)


def test_select_datasets_keeps_registry_order(tmp_path):
    specs = load_registry(
        _write(tmp_path, [_entry(name=n, openml_id=i) for i, n in enumerate("abc", start=1)])
    )
    assert [s.name for s in select_datasets(specs, ["c", "a"])] == ["a", "c"]


def test_select_datasets_without_names_returns_all(tmp_path):
    specs = load_registry(_write(tmp_path, [_entry()]))
    assert select_datasets(specs, None) == specs


def test_select_datasets_rejects_unknown_names(tmp_path):
    specs = load_registry(_write(tmp_path, [_entry()]))
    with pytest.raises(RegistryError, match="unknown dataset.*nope"):
        select_datasets(specs, ["toy", "nope"])


def test_paths_follow_the_dataset_name(tmp_path):
    spec = DatasetSpec(**_entry(name="toy"))
    assert csv_path(spec, tmp_path) == tmp_path / "toy.csv"
    assert reference_plan_path(spec, tmp_path) == tmp_path / "toy.plan.json"


class TestShippedRegistry:
    def test_has_between_eight_and_ten_datasets(self):
        assert 8 <= len(load_registry(DEFAULT_REGISTRY)) <= 10

    def test_covers_both_task_types_with_at_least_two_regressions(self):
        specs = load_registry(DEFAULT_REGISTRY)
        assert sum(spec.task_type == "regression" for spec in specs) >= 2
        assert sum(spec.task_type == "classification" for spec in specs) >= 2

    def test_spans_small_and_medium_sizes(self):
        rows = [spec.n_rows for spec in load_registry(DEFAULT_REGISTRY)]
        assert min(rows) < 1000
        assert max(rows) >= 10_000

    def test_every_entry_says_why_it_was_chosen(self):
        for spec in load_registry(DEFAULT_REGISTRY):
            assert spec.challenge.strip().endswith("."), spec.name
