"""The HTTP side of the local web UI: FastAPI routes over `SessionStore`.

Nothing here waits for the agent. A run starts in a thread of its own (see `sessions`) and the
browser polls the session state.
"""

import argparse
import os
import tempfile
import zipfile
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.config import DEFAULT_MAX_UPLOAD_MB, DEFAULT_PORT, HOST, Settings
from app.demo import DemoError, DemoLLM
from app.results import DOWNLOADABLE_SUFFIXES, build_result
from app.samples import find_sample, list_samples
from app.sessions import (
    AnswerError,
    RunRequest,
    Session,
    SessionBusy,
    SessionStore,
)
from prep_agent.env import load_dotenv_file
from prep_agent.llm.base import LLMClient, LLMError
from prep_agent.llm.factory import create_llm_client
from prep_agent.profiler import ProfilerError, load_csv

STATIC_DIR = Path(__file__).resolve().parent / "static"
UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
_READ_CHUNK = 1 << 20

LLMFactory = Callable[[Session, RunRequest], LLMClient]


class ApiError(HTTPException):
    """An error the UI can translate: `code` and `params` name it, `detail` is the English text."""

    def __init__(self, status: int, code: str, detail: str, **params: str) -> None:
        super().__init__(status, detail)
        self.code = code
        self.params = params


class PlanRequest(BaseModel):
    goal: str
    target: str | None = None
    model_family: Literal["tree", "linear", "any"] | None = None
    # The language of the interface. Only the demo script uses it; a real planner answers in the
    # language of the goal.
    lang: Literal["en", "tr"] = "en"


class AnswersRequest(BaseModel):
    answers: dict[str, str]


class SampleRequest(BaseModel):
    id: str


def default_llm_factory(settings: Settings) -> LLMFactory:
    if settings.fake:

        def demo(session: Session, request: RunRequest) -> LLMClient:
            return DemoLLM(
                session.profile,
                request.target,
                request.model_family,
                answers=lambda: {record.question.id: record.answer for record in session.qa},
                lang=request.lang,
            )

        return demo

    def real(session: Session, request: RunRequest) -> LLMClient:
        return create_llm_client()

    return real


def create_app(settings: Settings | None = None, llm_factory: LLMFactory | None = None) -> FastAPI:
    settings = settings or Settings()
    make_llm = llm_factory or default_llm_factory(settings)
    store = SessionStore()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        store.close_all()

    app = FastAPI(title="prep-agent", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(settings.allowed_hosts))

    @app.middleware("http")
    async def guard(request: Request, call_next):
        # A page on another origin can send a form to this server; refuse anything that says
        # it comes from somewhere else.
        origin = request.headers.get("origin")
        if request.method in UNSAFE_METHODS and origin is not None:
            host = origin.split("://", 1)[-1].split("/", 1)[0].rsplit(":", 1)[0]
            if host not in settings.allowed_hosts:
                return JSONResponse(
                    {"detail": "cross-origin request refused", "code": "cross_origin"},
                    status_code=403,
                )
        response = await call_next(request)
        # The page is being designed: a reload must always show the files on disk.
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_: Request, exc: RequestValidationError) -> JSONResponse:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in err['loc'][1:]) or 'request'}: {err['msg']}"
            for err in exc.errors()
        )
        return JSONResponse({"detail": problems}, status_code=422)

    @app.exception_handler(ApiError)
    async def api_error(_: Request, exc: ApiError) -> JSONResponse:
        body = {"detail": exc.detail, "code": exc.code, "params": exc.params}
        return JSONResponse(body, status_code=exc.status_code)

    def session_or_404(session_id: str) -> Session:
        session = store.get(session_id)
        if session is None:
            raise ApiError(404, "session_not_found", "session not found; it may have been deleted")
        return session

    def open_session(path: Path, filename: str) -> dict:
        try:
            df = load_csv(path)
            session = store.create(filename, df, path)
        except ProfilerError as exc:
            raise ApiError(422, "unreadable_file", str(exc)) from None
        return session.snapshot()

    # --- pages and configuration ---

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/config")
    def config() -> dict:
        return {
            "fake": settings.fake,
            "max_upload_bytes": settings.max_upload_bytes,
            "samples": [sample.to_json() for sample in list_samples(settings.fake)],
        }

    # --- sessions ---

    @app.post("/api/sessions", status_code=201)
    def upload(file: Annotated[UploadFile, File()]) -> dict:
        data = file.file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            limit = _megabytes(settings.max_upload_bytes)
            raise ApiError(
                413, "file_too_large", f"the file is larger than the limit of {limit}", limit=limit
            )
        if not data:
            raise ApiError(422, "file_empty", "the file is empty")
        handle, name = tempfile.mkstemp(dir=store.root, suffix=".csv")
        staged = Path(name)
        try:
            with os.fdopen(handle, "wb") as out:
                out.write(data)
            return open_session(staged, Path(file.filename or "upload.csv").name)
        finally:
            staged.unlink(missing_ok=True)

    @app.post("/api/sessions/sample", status_code=201)
    def use_sample(body: SampleRequest) -> dict:
        sample = find_sample(body.id, settings.fake)
        if sample is None:
            raise ApiError(404, "unknown_sample", "unknown sample")
        if sample.path.stat().st_size > settings.max_upload_bytes:
            limit = _megabytes(settings.max_upload_bytes)
            raise ApiError(
                413,
                "file_too_large",
                f"the sample is larger than the limit of {limit}",
                limit=limit,
            )
        return open_session(sample.path, sample.path.name)

    @app.get("/api/sessions/{session_id}")
    def state(session_id: str) -> dict:
        return session_or_404(session_id).snapshot()

    @app.delete("/api/sessions/{session_id}", status_code=204)
    def delete(session_id: str) -> None:
        if not store.delete(session_id):
            raise ApiError(404, "session_not_found", "session not found")

    # --- a run ---

    @app.post("/api/sessions/{session_id}/plan", status_code=202)
    def plan(session_id: str, body: PlanRequest) -> dict:
        session = session_or_404(session_id)
        goal = body.goal.strip()
        if not goal:
            raise ApiError(422, "goal_empty", "the goal must not be empty")
        columns = [column.name for column in session.profile.columns]
        if body.target is not None and body.target not in columns:
            raise ApiError(
                422,
                "unknown_target",
                f"target {body.target!r} is not a column of this file",
                target=body.target,
            )
        request = RunRequest(goal, body.target, body.model_family, body.lang)
        try:
            llm = make_llm(session, request)
        except DemoError as exc:
            raise ApiError(422, "demo_unsupported", str(exc)) from None
        except LLMError as exc:
            raise ApiError(400, "llm_unavailable", str(exc)) from None
        try:
            session.start(request, llm)
        except SessionBusy as exc:
            raise ApiError(409, "session_busy", str(exc)) from None
        return session.snapshot()

    @app.post("/api/sessions/{session_id}/reset")
    def reset(session_id: str) -> dict:
        """Drop the run (cancelling it if it is still going) and open the brief again. The file
        and its profile stay."""
        session = session_or_404(session_id)
        session.reset()
        return session.snapshot()

    @app.post("/api/sessions/{session_id}/answers")
    def answer(session_id: str, body: AnswersRequest) -> dict:
        session = session_or_404(session_id)
        try:
            session.answer(body.answers)
        except AnswerError as exc:
            raise ApiError(422, "answers_invalid", str(exc)) from None
        return session.snapshot()

    @app.get("/api/sessions/{session_id}/result")
    def result(session_id: str) -> dict:
        session = session_or_404(session_id)
        if session.outcome is None or session.run_dir is None:
            raise ApiError(409, "no_run", "this session has no finished run")
        return build_result(session)

    @app.get("/api/sessions/{session_id}/files/{name}")
    def one_file(session_id: str, name: str) -> FileResponse:
        session = session_or_404(session_id)
        run_dir = _finished_run_dir(session)
        path = run_dir / name
        # The name is checked against the folder listing, not joined blindly into a path.
        if name not in {p.name for p in run_dir.iterdir() if p.is_file()} or (
            path.suffix not in DOWNLOADABLE_SUFFIXES
        ):
            raise ApiError(404, "no_file", "no such file in the run folder")
        return FileResponse(path, filename=name)

    @app.get("/api/sessions/{session_id}/download")
    def download(session_id: str) -> FileResponse:
        session = session_or_404(session_id)
        run_dir = _finished_run_dir(session)
        # Outside the run folder, so the archive never contains itself.
        archive = session.directory / f"{run_dir.name}.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(run_dir.rglob("*")):
                if path.is_file():
                    zf.write(path, Path(run_dir.name) / path.relative_to(run_dir))
        stem = Path(session.filename).stem or "data"
        return FileResponse(
            archive, filename=f"prep-agent-{stem}-{run_dir.name}.zip", media_type="application/zip"
        )

    def _finished_run_dir(session: Session) -> Path:
        if session.run_dir is None or not session.run_dir.is_dir():
            raise ApiError(409, "no_run", "this session has no finished run")
        return session.run_dir

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def _megabytes(size: int) -> str:
    return f"{size / (1024 * 1024):g} MB"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m app", description="Local web UI for prep-agent (127.0.0.1 only)."
    )
    parser.add_argument(
        "--fake",
        action="store_true",
        help="demo mode: a scripted planner on tests/fixtures/customers.csv, no API calls",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--max-upload-mb",
        type=float,
        default=DEFAULT_MAX_UPLOAD_MB,
        help=f"upload size limit in MB (default {DEFAULT_MAX_UPLOAD_MB})",
    )
    args = parser.parse_args(argv)

    if not args.fake:
        load_dotenv_file()
    settings = Settings(fake=args.fake, max_upload_bytes=int(args.max_upload_mb * 1024 * 1024))
    mode = "demo mode (scripted planner, no API calls)" if args.fake else "real LLM"
    print(f"prep-agent web UI: http://{HOST}:{args.port}  [{mode}]", flush=True)
    uvicorn.run(create_app(settings), host=HOST, port=args.port, log_level="warning")
