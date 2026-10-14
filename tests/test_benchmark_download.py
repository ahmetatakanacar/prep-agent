import hashlib
import json

import pandas as pd
import pytest
from typer.testing import CliRunner

from bench_datasets import DEFAULT_MANIFEST, DEFAULT_REGISTRY, DatasetSpec, load_registry
from download import FetchedDataset
from download import app as download_app
from download import download_datasets as _download_datasets


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


@pytest.fixture
def data(tmp_path):
    """The datasets folder; the manifest sits beside it, as in the repository."""
    path = tmp_path / "datasets"
    path.mkdir()
    return path


def download_datasets(specs, directory, **kwargs):
    return _download_datasets(specs, directory, directory.parent / "manifest.json", **kwargs)


def _manifest(directory) -> dict:
    path = directory.parent / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))["datasets"]


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tamper(directory, name="toy"):
    """Change a value but not the shape, so the file still passes the registry check."""
    path = directory / f"{name}.csv"
    path.write_text(path.read_text(encoding="utf-8").replace("0,x,0", "5,x,0", 1), encoding="utf-8")


def test_downloads_a_dataset_as_csv_without_an_index(data):
    report = download_datasets([_spec()], data, fetch=StubFetcher())
    assert [o.status for o in report.outcomes] == ["downloaded"]
    written = pd.read_csv(data / "toy.csv")
    assert list(written.columns) == ["a", "b", "y"]
    assert len(written) == 4


def test_manifest_records_hash_shape_and_origin(data):
    download_datasets([_spec()], data, fetch=StubFetcher())
    entry = _manifest(data)["toy"]
    assert entry["sha256"] == hashlib.sha256((data / "toy.csv").read_bytes()).hexdigest()
    assert (entry["n_rows"], entry["n_columns"]) == (4, 3)
    assert entry["file"] == "toy.csv"
    assert (entry["openml_id"], entry["openml_name"], entry["openml_version"]) == (1, "toy", 1)
    assert entry["license"] == "Public"


def test_existing_files_are_not_downloaded_again(data):
    download_datasets([_spec()], data, fetch=StubFetcher())
    fetcher = StubFetcher()
    report = download_datasets([_spec()], data, fetch=fetcher)
    assert fetcher.calls == []
    assert [o.status for o in report.outcomes] == ["exists"]


def test_an_existing_file_is_added_to_a_missing_manifest(data):
    download_datasets([_spec()], data, fetch=StubFetcher())
    (data.parent / "manifest.json").unlink()
    download_datasets([_spec()], data, fetch=StubFetcher())
    assert _manifest(data)["toy"]["sha256"]


def test_force_downloads_again(data):
    download_datasets([_spec()], data, fetch=StubFetcher())
    fetcher = StubFetcher()
    report = download_datasets([_spec()], data, fetch=fetcher, force=True)
    assert fetcher.calls == ["toy"]
    assert [o.status for o in report.outcomes] == ["downloaded"]


def test_the_manifest_is_not_written_inside_the_datasets_folder(data):
    download_datasets([_spec()], data, fetch=StubFetcher())

    assert (data.parent / "manifest.json").exists()
    assert not (data / "manifest.json").exists()


def test_a_file_that_differs_from_the_manifest_is_a_warning_and_the_manifest_stays(data):
    download_datasets([_spec()], data, fetch=StubFetcher())
    recorded = _manifest(data)["toy"]["sha256"]
    _tamper(data)

    [outcome] = download_datasets([_spec()], data, fetch=StubFetcher()).outcomes

    assert outcome.status == "exists"
    assert "differs from the manifest" in outcome.warning
    assert recorded[:12] in outcome.warning
    assert _manifest(data)["toy"]["sha256"] == recorded


def test_a_download_that_differs_from_the_manifest_is_a_warning_too(data):
    download_datasets([_spec()], data, fetch=StubFetcher())
    recorded = _manifest(data)["toy"]["sha256"]
    changed = _frame().assign(a=lambda f: f["a"] + 100)

    [outcome] = download_datasets(
        [_spec()], data, fetch=StubFetcher({"toy": changed}), force=True
    ).outcomes

    assert outcome.status == "downloaded"
    assert "differs from the manifest" in outcome.warning
    assert _manifest(data)["toy"]["sha256"] == recorded


def test_a_file_that_matches_the_manifest_has_no_warning(data):
    download_datasets([_spec()], data, fetch=StubFetcher())

    [again] = download_datasets([_spec()], data, fetch=StubFetcher(), force=True).outcomes

    assert again.warning == ""


def test_a_dataset_the_manifest_does_not_know_yet_is_added_without_a_warning(data):
    download_datasets([_spec("first", 1)], data, fetch=StubFetcher())

    [outcome] = download_datasets([_spec("second", 2)], data, fetch=StubFetcher()).outcomes

    assert outcome.warning == ""
    assert set(_manifest(data)) == {"first", "second"}


def test_update_manifest_accepts_the_new_hash(data):
    download_datasets([_spec()], data, fetch=StubFetcher())
    _tamper(data)

    [outcome] = download_datasets(
        [_spec()], data, fetch=StubFetcher(), update_manifest=True
    ).outcomes

    assert outcome.warning == ""
    assert _manifest(data)["toy"]["sha256"] == _sha256(data / "toy.csv")


def test_a_warning_does_not_make_the_report_fail(data):
    download_datasets([_spec()], data, fetch=StubFetcher())
    _tamper(data)

    assert download_datasets([_spec()], data, fetch=StubFetcher()).ok


def test_an_existing_file_with_the_wrong_shape_is_reported(data):
    (data / "toy.csv").write_text("a,y\n1,0\n", encoding="utf-8")
    [outcome] = download_datasets([_spec()], data, fetch=StubFetcher()).outcomes
    assert outcome.status == "failed"
    assert "1 rows x 2 columns" in outcome.message


def test_a_different_dataset_version_is_refused_and_nothing_is_written(data):
    class Wrong(StubFetcher):
        def __call__(self, spec):
            return FetchedDataset(frame=_frame(), name="toy", version=2)

    [outcome] = download_datasets([_spec()], data, fetch=Wrong()).outcomes
    assert outcome.status == "failed"
    assert "version" in outcome.message
    assert not (data / "toy.csv").exists()


def test_a_different_dataset_name_is_refused(data):
    class Wrong(StubFetcher):
        def __call__(self, spec):
            return FetchedDataset(frame=_frame(), name="other", version=1)

    [outcome] = download_datasets([_spec()], data, fetch=Wrong()).outcomes
    assert outcome.status == "failed"
    assert "other" in outcome.message


def test_the_name_check_ignores_case(data):
    class Shouting(StubFetcher):
        def __call__(self, spec):
            return FetchedDataset(frame=_frame(), name="TOY", version=1)

    [outcome] = download_datasets([_spec()], data, fetch=Shouting()).outcomes
    assert outcome.status == "downloaded"


def test_a_download_with_the_wrong_shape_is_refused(data):
    fetcher = StubFetcher({"toy": _frame(rows=6)})
    [outcome] = download_datasets([_spec()], data, fetch=fetcher).outcomes
    assert outcome.status == "failed"
    assert "6 rows x 3 columns" in outcome.message
    assert not (data / "toy.csv").exists()


def test_a_failing_dataset_does_not_stop_the_others(data):
    class Flaky(StubFetcher):
        def __call__(self, spec):
            if spec.name == "bad":
                raise OSError("network down")
            return super().__call__(spec)

    specs = [_spec("bad", 1), _spec("good", 2)]
    report = download_datasets(specs, data, fetch=Flaky())
    assert [(o.name, o.status) for o in report.outcomes] == [
        ("bad", "failed"),
        ("good", "downloaded"),
    ]
    assert "network down" in report.outcomes[0].message
    assert set(_manifest(data)) == {"good"}
    assert not report.ok


def test_the_manifest_keeps_entries_of_datasets_not_requested_now(data):
    download_datasets([_spec("first", 1)], data, fetch=StubFetcher())
    download_datasets([_spec("second", 2)], data, fetch=StubFetcher())
    assert set(_manifest(data)) == {"first", "second"}


def test_an_interrupted_write_leaves_no_file_behind(data, monkeypatch):
    def write_half_then_fail(self, path, *args, **kwargs):
        path.write_text("a,b\n1,", encoding="utf-8")
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_csv", write_half_then_fail)
    [outcome] = download_datasets([_spec()], data, fetch=StubFetcher()).outcomes
    assert outcome.status == "failed"
    assert list(data.iterdir()) == []


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
            [
                "--registry",
                str(registry),
                "--datasets-dir",
                str(tmp_path / "out"),
                "--manifest",
                str(tmp_path / "manifest.json"),
                *args,
            ],
        )

    def test_downloads_everything_and_exits_zero(self, monkeypatch, registry, tmp_path):
        result = self._invoke(monkeypatch, registry, tmp_path)
        assert result.exit_code == 0, result.output
        assert (tmp_path / "out" / "one.csv").exists()
        assert (tmp_path / "out" / "two.csv").exists()
        assert set(_manifest(tmp_path / "out")) == {"one", "two"}

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

    def test_warns_about_a_changed_file_and_exits_zero(self, monkeypatch, registry, tmp_path):
        self._invoke(monkeypatch, registry, tmp_path)
        _tamper(tmp_path / "out", "one")

        result = self._invoke(monkeypatch, registry, tmp_path)

        assert result.exit_code == 0, result.output
        assert "warning" in result.output.lower()
        assert "differs from the manifest" in result.output
        assert "--update-manifest" in result.output

    def test_update_manifest_takes_the_new_hash(self, monkeypatch, registry, tmp_path):
        self._invoke(monkeypatch, registry, tmp_path)
        _tamper(tmp_path / "out", "one")

        result = self._invoke(monkeypatch, registry, tmp_path, "--update-manifest")

        assert result.exit_code == 0, result.output
        assert "differs" not in result.output
        assert _manifest(tmp_path / "out")["one"]["sha256"] == _sha256(tmp_path / "out" / "one.csv")


def test_the_manifest_is_committed_beside_the_datasets_folder():
    assert DEFAULT_MANIFEST.parent == DEFAULT_REGISTRY.parent
    assert DEFAULT_MANIFEST.name == "manifest.json"
    assert DEFAULT_MANIFEST.exists()


def test_the_committed_manifest_describes_every_dataset_of_the_registry():
    entries = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))["datasets"]

    for spec in load_registry():
        entry = entries[spec.name]
        assert len(entry["sha256"]) == 64 and set(entry["sha256"]) <= set("0123456789abcdef")
        assert entry["file"] == f"{spec.name}.csv"
        assert (entry["n_rows"], entry["n_columns"]) == (spec.n_rows, spec.n_columns)
        assert (entry["openml_id"], entry["openml_name"], entry["openml_version"]) == (
            spec.openml_id,
            spec.openml_name,
            spec.openml_version,
        )
        assert entry["license"] == spec.license
