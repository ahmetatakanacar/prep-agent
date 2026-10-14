import hashlib
import json

import pandas as pd
import pytest
from typer.testing import CliRunner

from bench_datasets import DatasetSpec
from download import FetchedDataset, download_datasets
from download import app as download_app


def _spec(name="toy", openml_id=1, version=1, rows=4, columns=3, **overrides) -> DatasetSpec:
    return DatasetSpec(
        **{
            "name": name,
            "openml_id": openml_id,
            "openml_name": name,
            "openml_version": version,
            "target": "y",
            "task_type": "classification",
            "goal": "Predict y.",
            "license": "Public",
            "challenge": "Nothing.",
            "n_rows": rows,
            "n_columns": columns,
        }
        | overrides
    )


def _frame(rows=4) -> pd.DataFrame:
    return pd.DataFrame(
        {"a": range(rows), "b": ["x", "y"] * (rows // 2), "y": [0, 1] * (rows // 2)}
    )


class StubFetcher:
    """Stands in for OpenML: serves a frame per dataset id and counts the calls."""

    def __init__(self, frames=None):
        self.frames = frames or {}
        self.calls = []

    def __call__(self, spec):
        self.calls.append(spec.name)
        frame = self.frames.get(spec.name, _frame())
        return FetchedDataset(frame=frame, name=spec.openml_name, version=spec.openml_version)


def _manifest(directory) -> dict:
    return json.loads((directory / "manifest.json").read_text(encoding="utf-8"))["datasets"]


def test_downloads_a_dataset_as_csv_without_an_index(tmp_path):
    report = download_datasets([_spec()], tmp_path, fetch=StubFetcher())
    assert [o.status for o in report.outcomes] == ["downloaded"]
    written = pd.read_csv(tmp_path / "toy.csv")
    assert list(written.columns) == ["a", "b", "y"]
    assert len(written) == 4


def test_manifest_records_hash_shape_and_origin(tmp_path):
    download_datasets([_spec()], tmp_path, fetch=StubFetcher())
    entry = _manifest(tmp_path)["toy"]
    assert entry["sha256"] == hashlib.sha256((tmp_path / "toy.csv").read_bytes()).hexdigest()
    assert (entry["n_rows"], entry["n_columns"]) == (4, 3)
    assert entry["file"] == "toy.csv"
    assert (entry["openml_id"], entry["openml_name"], entry["openml_version"]) == (1, "toy", 1)
    assert entry["license"] == "Public"


def test_existing_files_are_not_downloaded_again(tmp_path):
    download_datasets([_spec()], tmp_path, fetch=StubFetcher())
    fetcher = StubFetcher()
    report = download_datasets([_spec()], tmp_path, fetch=fetcher)
    assert fetcher.calls == []
    assert [o.status for o in report.outcomes] == ["exists"]


def test_an_existing_file_is_added_to_a_missing_manifest(tmp_path):
    download_datasets([_spec()], tmp_path, fetch=StubFetcher())
    (tmp_path / "manifest.json").unlink()
    download_datasets([_spec()], tmp_path, fetch=StubFetcher())
    assert _manifest(tmp_path)["toy"]["sha256"]


def test_force_downloads_again(tmp_path):
    download_datasets([_spec()], tmp_path, fetch=StubFetcher())
    fetcher = StubFetcher()
    report = download_datasets([_spec()], tmp_path, fetch=fetcher, force=True)
    assert fetcher.calls == ["toy"]
    assert [o.status for o in report.outcomes] == ["downloaded"]


def test_a_file_that_changed_since_the_manifest_is_reported(tmp_path):
    download_datasets([_spec()], tmp_path, fetch=StubFetcher())
    with (tmp_path / "toy.csv").open("a", encoding="utf-8") as handle:
        handle.write("9,x,1\n")
    report = download_datasets([_spec()], tmp_path, fetch=StubFetcher())
    [outcome] = report.outcomes
    assert outcome.status == "failed"
    assert "changed" in outcome.message


def test_an_existing_file_with_the_wrong_shape_is_reported(tmp_path):
    (tmp_path / "toy.csv").write_text("a,y\n1,0\n", encoding="utf-8")
    [outcome] = download_datasets([_spec()], tmp_path, fetch=StubFetcher()).outcomes
    assert outcome.status == "failed"
    assert "1 rows x 2 columns" in outcome.message


def test_a_different_dataset_version_is_refused_and_nothing_is_written(tmp_path):
    class Wrong(StubFetcher):
        def __call__(self, spec):
            return FetchedDataset(frame=_frame(), name="toy", version=2)

    [outcome] = download_datasets([_spec()], tmp_path, fetch=Wrong()).outcomes
    assert outcome.status == "failed"
    assert "version" in outcome.message
    assert not (tmp_path / "toy.csv").exists()


def test_a_different_dataset_name_is_refused(tmp_path):
    class Wrong(StubFetcher):
        def __call__(self, spec):
            return FetchedDataset(frame=_frame(), name="other", version=1)

    [outcome] = download_datasets([_spec()], tmp_path, fetch=Wrong()).outcomes
    assert outcome.status == "failed"
    assert "other" in outcome.message


def test_the_name_check_ignores_case(tmp_path):
    class Shouting(StubFetcher):
        def __call__(self, spec):
            return FetchedDataset(frame=_frame(), name="TOY", version=1)

    [outcome] = download_datasets([_spec()], tmp_path, fetch=Shouting()).outcomes
    assert outcome.status == "downloaded"


def test_a_download_with_the_wrong_shape_is_refused(tmp_path):
    fetcher = StubFetcher({"toy": _frame(rows=6)})
    [outcome] = download_datasets([_spec()], tmp_path, fetch=fetcher).outcomes
    assert outcome.status == "failed"
    assert "6 rows x 3 columns" in outcome.message
    assert not (tmp_path / "toy.csv").exists()


def test_a_failing_dataset_does_not_stop_the_others(tmp_path):
    class Flaky(StubFetcher):
        def __call__(self, spec):
            if spec.name == "bad":
                raise OSError("network down")
            return super().__call__(spec)

    specs = [_spec("bad", 1), _spec("good", 2)]
    report = download_datasets(specs, tmp_path, fetch=Flaky())
    assert [(o.name, o.status) for o in report.outcomes] == [
        ("bad", "failed"),
        ("good", "downloaded"),
    ]
    assert "network down" in report.outcomes[0].message
    assert set(_manifest(tmp_path)) == {"good"}
    assert not report.ok


def test_the_manifest_keeps_entries_of_datasets_not_requested_now(tmp_path):
    download_datasets([_spec("first", 1)], tmp_path, fetch=StubFetcher())
    download_datasets([_spec("second", 2)], tmp_path, fetch=StubFetcher())
    assert set(_manifest(tmp_path)) == {"first", "second"}


def test_an_interrupted_write_leaves_no_file_behind(tmp_path, monkeypatch):
    def write_half_then_fail(self, path, *args, **kwargs):
        path.write_text("a,b\n1,", encoding="utf-8")
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_csv", write_half_then_fail)
    [outcome] = download_datasets([_spec()], tmp_path, fetch=StubFetcher()).outcomes
    assert outcome.status == "failed"
    assert [p.name for p in tmp_path.iterdir() if p.name != "manifest.json"] == []


class TestCommandLine:
    @pytest.fixture
    def registry(self, tmp_path):
        import yaml

        entries = [_spec("one", 1).model_dump(), _spec("two", 2).model_dump()]
        path = tmp_path / "datasets.yaml"
        path.write_text(yaml.safe_dump({"datasets": entries}), encoding="utf-8")
        return path

    def _invoke(self, monkeypatch, registry, tmp_path, *args, fetch=None):
        monkeypatch.setattr("download.fetch_from_openml", fetch or StubFetcher())
        return CliRunner().invoke(
            download_app,
            ["--registry", str(registry), "--datasets-dir", str(tmp_path / "out"), *args],
        )

    def test_downloads_everything_and_exits_zero(self, monkeypatch, registry, tmp_path):
        result = self._invoke(monkeypatch, registry, tmp_path)
        assert result.exit_code == 0, result.output
        assert (tmp_path / "out" / "one.csv").exists()
        assert (tmp_path / "out" / "two.csv").exists()

    def test_only_downloads_the_named_datasets(self, monkeypatch, registry, tmp_path):
        result = self._invoke(monkeypatch, registry, tmp_path, "--only", "two")
        assert result.exit_code == 0, result.output
        assert not (tmp_path / "out" / "one.csv").exists()

    def test_exits_one_when_a_dataset_fails(self, monkeypatch, registry, tmp_path):
        result = self._invoke(
            monkeypatch, registry, tmp_path, fetch=StubFetcher({"one": _frame(rows=6)})
        )
        assert result.exit_code == 1
        assert (tmp_path / "out" / "two.csv").exists()

    def test_exits_one_for_an_unknown_dataset(self, monkeypatch, registry, tmp_path):
        result = self._invoke(monkeypatch, registry, tmp_path, "--only", "nope")
        assert result.exit_code == 1
        assert "unknown dataset" in result.output
