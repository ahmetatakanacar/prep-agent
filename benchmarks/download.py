"""Download the benchmark datasets from OpenML as CSV files.

    uv run python benchmarks/download.py

Files go to `benchmarks/datasets/<name>.csv` (git-ignored). A dataset whose file exists is not
downloaded again, but it is still checked against the registry and the manifest. The manifest,
`benchmarks/manifest.json`, is committed: it records the sha256 of every file next to the OpenML
id, name and version it came from, so a later download can be compared with the data the
reference scores were made on. A file whose hash differs from the manifest is reported as a
warning and the manifest keeps the recorded hash; `--update-manifest` accepts the new one.

OpenML leaves out the columns a dataset marks as row id or ignored, so the CSV files hold what
`fetch_openml` returns and not necessarily every column of the original source.
"""

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

import pandas as pd
import typer
from rich.console import Console
from rich.text import Text

from bench_datasets import (
    DEFAULT_DATASETS_DIR,
    DEFAULT_MANIFEST,
    DEFAULT_REGISTRY,
    DatasetSpec,
    RegistryError,
    csv_path,
    load_registry,
    select_datasets,
)

_HASH_CHUNK = 1 << 20


@dataclass(frozen=True)
class FetchedDataset:
    frame: pd.DataFrame
    name: str
    version: int


Fetcher = Callable[[DatasetSpec], FetchedDataset]


@dataclass(frozen=True)
class Outcome:
    name: str
    status: Literal["downloaded", "exists", "failed"]
    message: str = ""
    # The file is usable but differs from the manifest.
    warning: str = ""


@dataclass
class DownloadReport:
    outcomes: list[Outcome] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(outcome.status != "failed" for outcome in self.outcomes)


class _Refused(Exception):
    """A dataset or file does not match the registry; the message says how."""


def fetch_from_openml(spec: DatasetSpec) -> FetchedDataset:
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml(data_id=spec.openml_id, as_frame=True, parser="auto")
    return FetchedDataset(
        frame=bunch.frame,
        name=str(bunch.details["name"]),
        version=int(bunch.details["version"]),
    )


def download_datasets(
    specs: list[DatasetSpec],
    datasets_dir: Path,
    manifest_path: Path,
    *,
    fetch: Fetcher | None = None,
    force: bool = False,
    update_manifest: bool = False,
) -> DownloadReport:
    """Fetch every dataset that has no file yet (or all of them with `force`) and update the
    manifest. A dataset that fails is reported and does not stop the others.

    A dataset the manifest does not list yet is added to it. A file whose hash differs from the
    recorded one gets a warning and the recorded entry stays, unless `update_manifest` is set.
    """
    fetch = fetch or fetch_from_openml
    datasets_dir = Path(datasets_dir)
    datasets_dir.mkdir(parents=True, exist_ok=True)
    manifest = _read_manifest(manifest_path)
    report = DownloadReport()
    for spec in specs:
        try:
            outcome = _ensure(spec, datasets_dir, manifest, fetch, force, update_manifest)
        except _Refused as exc:
            outcome = Outcome(spec.name, "failed", str(exc))
        # Whatever the network or OpenML raises must end up in the report, not stop the loop.
        except Exception as exc:
            outcome = Outcome(spec.name, "failed", f"{type(exc).__name__}: {exc}")
        report.outcomes.append(outcome)
    _write_json(manifest_path, {"datasets": manifest})
    return report


def _ensure(
    spec: DatasetSpec,
    datasets_dir: Path,
    manifest: dict,
    fetch: Fetcher,
    force: bool,
    update_manifest: bool,
) -> Outcome:
    path = csv_path(spec, datasets_dir)
    if path.exists() and not force:
        frame = pd.read_csv(path, low_memory=False)
        _check_shape(spec, frame, path.name)
        status = "exists"
    else:
        frame = _download(spec, path, fetch)
        status = "downloaded"
    warning = _record(spec, path, frame, manifest, update_manifest)
    return Outcome(spec.name, status, warning=warning)


def _download(spec: DatasetSpec, path: Path, fetch: Fetcher) -> pd.DataFrame:
    fetched = fetch(spec)
    _check_origin(spec, fetched)
    _check_shape(spec, fetched.frame, f"OpenML dataset {spec.openml_id}")
    if spec.target not in fetched.frame.columns:
        raise _Refused(f"the target {spec.target!r} is not a column of the downloaded data")
    # Written under another name first, so an interrupted download never leaves a file that
    # looks complete.
    partial = path.with_name(path.name + ".part")
    try:
        fetched.frame.to_csv(partial, index=False, lineterminator="\n")
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)
    return fetched.frame


def _record(
    spec: DatasetSpec, path: Path, frame: pd.DataFrame, manifest: dict, update_manifest: bool
) -> str:
    """Enter the file in the manifest; the return value is the warning, if there is one."""
    digest = _sha256(path)
    known = manifest.get(spec.name)
    if known is not None and known["sha256"] != digest and not update_manifest:
        return (
            f"{path.name} differs from the manifest (sha256 {digest[:12]}, recorded "
            f"{known['sha256'][:12]}); scores may not be comparable with earlier ones. The "
            "manifest was not changed; use --update-manifest to accept this file."
        )
    manifest[spec.name] = _entry(spec, path, digest, frame)
    return ""


def _check_origin(spec: DatasetSpec, fetched: FetchedDataset) -> None:
    if fetched.name.lower() != spec.openml_name.lower() or fetched.version != spec.openml_version:
        raise _Refused(
            f"OpenML dataset {spec.openml_id} is {fetched.name!r} version {fetched.version}, "
            f"the registry expects {spec.openml_name!r} version {spec.openml_version}"
        )


def _check_shape(spec: DatasetSpec, frame: pd.DataFrame, where: str) -> None:
    rows, columns = frame.shape
    if (rows, columns) != (spec.n_rows, spec.n_columns):
        raise _Refused(
            f"{where} has {rows} rows x {columns} columns, the registry expects "
            f"{spec.n_rows} x {spec.n_columns}"
        )


def _entry(spec: DatasetSpec, path: Path, digest: str, frame: pd.DataFrame) -> dict:
    return {
        "file": path.name,
        "sha256": digest,
        "n_rows": int(frame.shape[0]),
        "n_columns": int(frame.shape[1]),
        "openml_id": spec.openml_id,
        "openml_name": spec.openml_name,
        "openml_version": spec.openml_version,
        "license": spec.license,
    }


def _read_manifest(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))["datasets"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RegistryError(f"could not read {path}: {exc}; delete it to start a new one") from exc


def _write_json(path: Path, data: dict) -> None:
    Path(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


app = typer.Typer(add_completion=False, help="Download the benchmark datasets from OpenML.")


@app.command()
def main(
    only: Annotated[
        str | None, typer.Option(help="Comma-separated dataset names; default: all.")
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Download again even if the file exists.")
    ] = False,
    update_manifest: Annotated[
        bool,
        typer.Option("--update-manifest", help="Record the hash of files that differ from it."),
    ] = False,
    registry: Annotated[Path, typer.Option(help="Dataset registry file.")] = DEFAULT_REGISTRY,
    datasets_dir: Annotated[
        Path, typer.Option(help="Directory that receives the CSV files.")
    ] = DEFAULT_DATASETS_DIR,
    manifest: Annotated[
        Path, typer.Option(help="Manifest with the sha256 of every file.")
    ] = DEFAULT_MANIFEST,
) -> None:
    console = Console()
    try:
        names = [n.strip() for n in only.split(",") if n.strip()] if only else None
        specs = select_datasets(load_registry(registry), names)
        report = download_datasets(
            specs, datasets_dir, manifest, force=force, update_manifest=update_manifest
        )
    except RegistryError as exc:
        Console(stderr=True).print(Text.assemble(("Error: ", "red"), str(exc)))
        raise typer.Exit(1) from None

    for outcome in report.outcomes:
        detail = f": {outcome.message}" if outcome.message else ""
        style = "red" if outcome.status == "failed" else None
        console.print(Text(f"{outcome.status:<10} {outcome.name}{detail}", style=style))
        if outcome.warning:
            console.print(Text(f"Warning: {outcome.warning}", style="yellow"), soft_wrap=True)
    console.print(Text(f"Manifest: {manifest}"), soft_wrap=True)
    if not report.ok:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
