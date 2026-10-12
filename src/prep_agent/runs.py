"""Run artifacts: everything one execution leaves on disk under `<runs_dir>/<run_id>/`.

Files: profile.json, plan.json, train.csv, test.csv, pipeline.joblib, issues.json (plan and
result issues) and run.json (see `RunInfo`).

`pipeline.joblib` holds only the fitted stateful steps (impute, encode, scale). It does not
include the stateless steps or the split, so it can only be applied to a frame that has gone
through those already. The full, standalone `pipeline.py` is not generated yet.

A run directory is created exclusively: an existing directory is never written into. If writing
fails halfway, the directory this call created is removed again.
"""

import hashlib
import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import joblib
from pydantic import BaseModel

from prep_agent import __version__
from prep_agent.executor import ExecutionResult
from prep_agent.plan import Plan
from prep_agent.profiler import DatasetProfile
from prep_agent.validators import has_errors

PIPELINE_JOBLIB_NOTE = (
    "Fitted stateful steps only (impute, encode, scale), fit on train. Stateless steps and "
    "the split are not included; the full pipeline.py is not generated yet."
)
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_HASH_CHUNK = 1 << 20


class RunExistsError(FileExistsError):
    """Raised when the run directory already exists."""


class Counts(BaseModel):
    before: int
    train: int
    test: int


class RunInfo(BaseModel):
    """Contents of run.json."""

    run_id: str
    created_at: datetime
    prep_agent_version: str
    input_path: str
    input_sha256: str
    target: str
    rows: Counts
    columns: Counts
    status: Literal["ok", "failed"]
    pipeline_joblib: str


def save_run(
    runs_dir: Path,
    *,
    input_path: Path,
    profile: DatasetProfile,
    plan: Plan,
    result: ExecutionResult,
    run_id: str | None = None,
) -> Path:
    """Write the artifacts of one execution and return the run directory.

    `status` is "failed" if the result has validation errors; the artifacts are still written
    so the run can be inspected. `run_id` defaults to the UTC timestamp.
    """
    created_at = datetime.now(UTC)
    run_id = run_id if run_id is not None else created_at.strftime("%Y%m%d-%H%M%S")
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError(
            f"invalid run_id {run_id!r}: use letters, digits, '.', '_' and '-', "
            "starting with a letter or digit"
        )

    info = RunInfo(
        run_id=run_id,
        created_at=created_at,
        prep_agent_version=__version__,
        input_path=str(input_path),
        input_sha256=_sha256(Path(input_path)),
        target=plan.target,
        rows=Counts(before=profile.n_rows, train=len(result.train), test=len(result.test)),
        columns=Counts(
            before=profile.n_columns,
            train=result.train.shape[1],
            test=result.test.shape[1],
        ),
        status="failed" if has_errors(result.result_issues) else "ok",
        pipeline_joblib=PIPELINE_JOBLIB_NOTE,
    )

    run_dir = Path(runs_dir) / run_id
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_dir.mkdir()
    except FileExistsError:
        raise RunExistsError(f"run directory already exists: {run_dir}") from None

    try:
        _write_artifacts(run_dir, info, profile, plan, result)
    except BaseException:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise
    return run_dir


def _write_artifacts(
    run_dir: Path, info: RunInfo, profile: DatasetProfile, plan: Plan, result: ExecutionResult
) -> None:
    _write_text(run_dir / "profile.json", profile.model_dump_json(indent=2))
    _write_text(run_dir / "plan.json", plan.model_dump_json(indent=2))
    result.train.to_csv(run_dir / "train.csv", index=False, lineterminator="\n")
    result.test.to_csv(run_dir / "test.csv", index=False, lineterminator="\n")
    joblib.dump(result.pipeline, run_dir / "pipeline.joblib")
    issues = {
        "plan_issues": [issue.model_dump() for issue in result.plan_issues],
        "result_issues": [issue.model_dump() for issue in result.result_issues],
    }
    _write_text(run_dir / "issues.json", json.dumps(issues, indent=2))
    _write_text(run_dir / "run.json", info.model_dump_json(indent=2))


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()
