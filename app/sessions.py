"""Sessions: one uploaded CSV file and the agent runs made on it, kept in memory.

`run_agent` blocks while it waits for answers, so each run gets a thread of its own. The agent's
`ask_fn` parks that thread: it publishes the questions in the session state and waits until the
answer endpoint hands the answers over. The HTTP requests never wait for the agent; the browser
polls `Session.snapshot`.

The files of a session live in a temporary directory that is removed with the session.
"""

import logging
import re
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel

from prep_agent.agent import AgentError, AgentResult, AgentTrace, run_agent
from prep_agent.baseline import BaselineResult, evaluate_baselines
from prep_agent.llm.base import LLMClient, LLMError, LLMResult
from prep_agent.planner import QARecord, Question
from prep_agent.profiler import CsvOptions, DatasetProfile, ProfilerError, profile_dataframe
from prep_agent.runs import RunExistsError, save_run

log = logging.getLogger("app.sessions")

Status = Literal["ready", "running", "waiting", "done", "failed", "error"]
Phase = Literal["planning", "revising", "validating", "baseline", "saving"]
ModelFamily = Literal["tree", "linear", "any"]


JOIN_TIMEOUT_SECONDS = 30.0


class SessionClosed(Exception):
    """The run was cancelled (reset or deleted session) while the agent was waiting."""


class SessionBusy(Exception):
    """The session already has a run in progress."""


class AnswerError(ValueError):
    """The submitted answers do not fit the pending questions."""


@dataclass(frozen=True)
class RunRequest:
    goal: str
    target: str | None
    model_family: ModelFamily | None
    lang: str = "en"


class Session:
    def __init__(
        self,
        root: Path,
        filename: str,
        df: pd.DataFrame,
        source: Path,
        csv_options: CsvOptions | None = None,
        csv_warnings: list[str] | None = None,
    ) -> None:
        # Profiling comes first: it can refuse the frame, and then no directory may be left behind.
        self.profile = profile_dataframe(df, csv_options=csv_options)
        # Codes the page translates, one per thing that went wrong while reading the file.
        self.csv_warnings = csv_warnings or []
        self.id = uuid.uuid4().hex
        self.filename = filename
        self.directory = root / self.id
        self.directory.mkdir(parents=True)
        # The run records the name of its input file, so the copy keeps the name it was given.
        self.csv_path = self.directory / _safe_name(filename)
        shutil.copyfile(source, self.csv_path)
        self.runs_dir = self.directory / "runs"
        self.df = df

        self.lock = threading.RLock()
        self.status: Status = "ready"
        self.phase: Phase | None = None
        self.request: RunRequest | None = None
        self.questions: list[Question] | None = None
        self.qa: list[QARecord] = []
        self.llm_calls = 0
        self.error: str | None = None
        self.baseline_error: str | None = None
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.run_number = 0
        self.run_dir: Path | None = None
        self.outcome: Outcome | None = None

        self._threads: list[threading.Thread] = []
        self._last_kind: str | None = None
        self._answers: dict[str, str] | None = None
        self._answered = threading.Event()
        self._closed = False
        # Every run has a token. A reset or a new run makes the token of the old one stale, and a
        # stale run's thread may not change the session any more.
        self._token = 0

    # --- state ---

    def snapshot(self) -> dict:
        with self.lock:
            now = time.time()
            end = self.finished_at if self.finished_at is not None else now
            return {
                "id": self.id,
                "filename": self.filename,
                "status": self.status,
                "phase": self.phase,
                "profile": self.profile.model_dump(mode="json", exclude_none=True),
                "csv_warnings": self.csv_warnings,
                "request": None
                if self.request is None
                else {
                    "goal": self.request.goal,
                    "target": self.request.target,
                    "model_family": self.request.model_family,
                },
                "questions": None
                if self.questions is None
                else [q.model_dump(mode="json") for q in self.questions],
                "qa": [record.model_dump(mode="json") for record in self.qa],
                "llm_calls": self.llm_calls,
                "elapsed_seconds": None
                if self.started_at is None
                else round(end - self.started_at, 1),
                "error": self.error,
                "run_number": self.run_number,
                "has_result": self.outcome is not None,
            }

    def _set(self, token: int, **changes) -> None:
        with self.lock:
            if token != self._token:
                return
            for name, value in changes.items():
                setattr(self, name, value)

    # --- run ---

    def start(self, request: RunRequest, llm: LLMClient) -> None:
        with self.lock:
            if self.status in ("running", "waiting"):
                raise SessionBusy("this session already has a run in progress")
            self.run_number += 1
            self.status = "running"
            self.phase = "planning"
            self.request = request
            self.questions = None
            self.qa = []
            self.llm_calls = 0
            self.error = None
            self.baseline_error = None
            self.outcome = None
            self.run_dir = None
            self.started_at = time.time()
            self.finished_at = None
            self._last_kind = None
            self._answers = None
            self._token += 1
            token = self._token
        thread = threading.Thread(
            target=self._run,
            args=(request, llm, self.run_number, token),
            name=f"run-{self.id[:8]}",
        )
        thread.daemon = True
        with self.lock:
            self._threads = [t for t in self._threads if t.is_alive()] + [thread]
        thread.start()

    def answer(self, answers: dict[str, str]) -> None:
        with self.lock:
            if self.status != "waiting" or self.questions is None:
                raise AnswerError("the agent is not waiting for answers")
            _check_answers(self.questions, answers)
            self._answers = dict(answers)
            self.status = "running"
            self.phase = "planning"
            event = self._answered
        event.set()

    def reset(self) -> None:
        """Forget the run, stop it from changing the session and make the session ready again."""
        with self.lock:
            self._token += 1
            self.status = "ready"
            self.phase = None
            self.request = None
            self.questions = None
            self.qa = []
            self.llm_calls = 0
            self.error = None
            self.baseline_error = None
            self.outcome = None
            self.run_dir = None
            self.started_at = None
            self.finished_at = None
            self._answers = None
            event = self._answered
        event.set()

    def close(self) -> None:
        with self.lock:
            self._closed = True
            event = self._answered
        event.set()
        shutil.rmtree(self.directory, ignore_errors=True)

    def join(self, timeout: float) -> None:
        """Wait for the run threads to notice that the session is closed and return."""
        deadline = time.monotonic() + timeout
        with self.lock:
            threads = list(self._threads)
        for thread in threads:
            thread.join(max(0.0, deadline - time.monotonic()))

    def _current(self, token: int) -> bool:
        with self.lock:
            return token == self._token and not self._closed

    # --- the thread ---

    def _ask(self, questions: list[Question], token: int) -> dict[str, str]:
        event = threading.Event()
        with self.lock:
            if token != self._token or self._closed:
                raise SessionClosed
            self.questions = list(questions)
            self.status = "waiting"
            self.phase = None
            self._answers = None
            self._answered = event
        event.wait()
        with self.lock:
            if token != self._token or self._closed:
                raise SessionClosed
            answers = self._answers or {}
            self.qa.extend(QARecord(question=q, answer=answers[q.id]) for q in questions)
            self.questions = None
        return answers

    def _run(self, request: RunRequest, llm: LLMClient, run_number: int, token: int) -> None:
        run_id = f"run-{run_number}"
        try:
            result = run_agent(
                self.df,
                request.goal,
                _ObservedLLM(llm, self, token),
                lambda questions: self._ask(questions, token),
                target=request.target,
                model_family=request.model_family,
                csv_options=self.profile.csv,
            )
            self._finish_ok(result, run_id, token)
        except SessionClosed:
            return
        except AgentError as exc:
            self._finish_failed(exc, run_id, token)
        except (ProfilerError, LLMError) as exc:
            self._finish_error(str(exc), token)
        except Exception as exc:
            # A thread has nobody to raise to; the UI shows the message and the log keeps the trace.
            log.exception("run failed unexpectedly")
            self._finish_error(f"unexpected error: {type(exc).__name__}: {exc}", token)

    def _finish_ok(self, result: AgentResult, run_id: str, token: int) -> None:
        if not self._current(token):
            return
        self._set(token, phase="baseline")
        baseline: BaselineResult | None = None
        try:
            baseline = evaluate_baselines(
                result.execution.split_train,
                result.execution.pipeline,
                result.plan.target,
                result.plan.task_type,
            )
        except Exception as exc:
            # The baseline is a bonus on top of a valid run; its failure is shown, not fatal.
            log.exception("baseline failed")
            self._set(token, baseline_error=f"{type(exc).__name__}: {exc}")
        if not self._current(token):
            return
        self._set(token, phase="saving")
        try:
            run_dir = save_run(
                self.runs_dir,
                input_path=self.csv_path,
                profile=result.profile,
                plan=result.plan,
                result=result.execution,
                trace=result.trace,
                baseline=baseline,
                run_id=run_id,
            )
        except (OSError, RunExistsError) as exc:
            self._finish_error(f"could not save the run: {exc}", token)
            return
        self._complete(
            token,
            "done",
            run_dir,
            Outcome(
                profile=result.profile,
                trace=result.trace,
                plan=result.plan.model_dump(mode="json"),
                issues=[i.model_dump(mode="json") for i in result.issues],
                baseline=baseline,
            ),
        )

    def _finish_failed(self, error: AgentError, run_id: str, token: int) -> None:
        if not self._current(token):
            return
        self._set(token, phase="saving")
        try:
            run_dir = save_run(
                self.runs_dir,
                input_path=self.csv_path,
                profile=error.profile,
                plan=error.plan,
                result=error.execution,
                plan_issues=error.issues,
                trace=error.trace,
                error=str(error),
                run_id=run_id,
            )
        except (OSError, RunExistsError) as exc:
            self._finish_error(f"{error}; the run could not be saved: {exc}", token)
            return
        issues = (
            error.execution.plan_issues + error.execution.result_issues
            if error.execution is not None
            else error.issues
        )
        self._complete(
            token,
            "failed",
            run_dir,
            Outcome(
                profile=error.profile,
                trace=error.trace,
                plan=None if error.plan is None else error.plan.model_dump(mode="json"),
                issues=[i.model_dump(mode="json") for i in issues],
                baseline=None,
            ),
            error=str(error),
        )

    def _finish_error(self, message: str, token: int) -> None:
        with self.lock:
            if token != self._token or self._closed:
                return
            self.status = "error"
            self.phase = None
            self.error = message
            self.finished_at = time.time()

    def _complete(
        self,
        token: int,
        status: Status,
        run_dir: Path,
        outcome: "Outcome",
        error: str | None = None,
    ) -> None:
        with self.lock:
            if token != self._token or self._closed:
                return
            self.status = status
            self.phase = None
            self.run_dir = run_dir
            self.outcome = outcome
            self.error = error
            self.finished_at = time.time()

    # --- observed LLM ---

    def _llm_started(self, token: int) -> None:
        with self.lock:
            if token != self._token:
                return
            self.llm_calls += 1
            # A plan is only followed by another call when the validators sent it back.
            self.phase = "revising" if self._last_kind == "plan" else "planning"

    def _llm_finished(self, parsed: BaseModel, token: int) -> None:
        with self.lock:
            if token != self._token:
                return
            self._last_kind = getattr(parsed, "kind", None)
            self.phase = "validating"


@dataclass(frozen=True)
class Outcome:
    """What a finished run, successful or not, leaves in memory next to its files."""

    profile: DatasetProfile
    trace: AgentTrace
    plan: dict | None
    issues: list[dict]
    baseline: BaselineResult | None


class _ObservedLLM:
    """Reports to the session when a planner call starts and ends. Changes nothing else."""

    def __init__(self, inner: LLMClient, session: Session, token: int) -> None:
        self._inner = inner
        self._session = session
        self._token = token

    def generate[T: BaseModel](self, system: str, user: str, schema: type[T]) -> LLMResult[T]:
        self._session._llm_started(self._token)
        result = self._inner.generate(system, user, schema)
        self._session._llm_finished(result.parsed, self._token)
        return result


def _safe_name(filename: str) -> str:
    name = re.sub(r"[^\w.\- ]", "_", Path(filename).name).strip(" .") or "data"
    return name if name.lower().endswith(".csv") else f"{name}.csv"


def _check_answers(questions: list[Question], answers: dict[str, str]) -> None:
    known = {q.id for q in questions}
    unknown = sorted(set(answers) - known)
    if unknown:
        raise AnswerError(f"unknown question id(s): {', '.join(unknown)}")
    for question in questions:
        if question.id not in answers:
            raise AnswerError(f"no answer for question {question.id!r}")
        answer = answers[question.id]
        if question.allow_custom and not answer.strip():
            raise AnswerError(f"the answer to {question.id!r} is empty")
        if not question.allow_custom and answer not in question.options:
            raise AnswerError(f"the answer to {question.id!r} must be one of its options")


class SessionStore:
    def __init__(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="prep-agent-web-"))
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(
        self,
        filename: str,
        df: pd.DataFrame,
        source: Path,
        csv_options: CsvOptions | None = None,
        csv_warnings: list[str] | None = None,
    ) -> Session:
        session = Session(self.root, filename, df, source, csv_options, csv_warnings)
        with self._lock:
            self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            return self._sessions.get(session_id)

    def delete(self, session_id: str) -> bool:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        session.close()
        return True

    def close_all(self) -> None:
        with self._lock:
            sessions, self._sessions = list(self._sessions.values()), {}
        for session in sessions:
            session.close()
        # A cross-validation cannot be interrupted, so wait for it only up to a limit.
        for session in sessions:
            session.join(JOIN_TIMEOUT_SECONDS)
        shutil.rmtree(self.root, ignore_errors=True)
