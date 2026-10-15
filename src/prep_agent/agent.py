"""The agent loop: from a goal and a dataset to executed, validated data preparation.

Flow: profile -> planner (LLM) -> [questions -> user -> planner]* -> plan validation ->
execution -> result validation. A plan or result with errors goes back to the planner together
with the validator's errors, and its warnings ride along. Warnings alone never trigger a
revision.

Two limits keep the loop finite. `max_question_rounds` caps how often the user is asked; once it
is reached the planner is told to answer with a plan. If it asks anyway, it gets one more call
with a plan-only instruction, and that call is charged to the revision budget. `max_revisions`
caps how often a plan is sent back, and plan errors, result errors and that retry draw on the
same budget. When the budget is used up
and errors remain, `RevisionBudgetError` carries the last plan and its issues.

An answer that does not match the response schema is not a plan to revise: the call is repeated,
with the validation error added to the prompt, up to `max_schema_retries` times (a limit of its
own, `MAX_SCHEMA_RETRIES` by default, per planner call and apart from the revision budget). If
the last attempt fails too, `AgentError` is raised.

The planner only ever sees the profile (see `prep_agent.planner`), never the data frame.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal

import pandas as pd
from pydantic import BaseModel

from prep_agent.executor import ExecutionError, ExecutionResult, PlanValidationError, execute
from prep_agent.llm.base import LLMClient, LLMError, LLMSchemaError, TokenUsage
from prep_agent.plan import Plan
from prep_agent.planner import (
    PlannerResponse,
    QARecord,
    Question,
    Revision,
    build_system_prompt,
    build_user_prompt,
)
from prep_agent.profiler import DatasetProfile, profile_dataframe
from prep_agent.validators import ValidationIssue, partition_issues

MAX_SCHEMA_RETRIES = 1
"""How often one planner call is repeated after an answer that does not match the schema."""

# The validation error goes back to the model; this keeps a long list of errors from bloating
# the prompt.
SCHEMA_ERROR_LIMIT = 600

AskFn = Callable[[list[Question]], dict[str, str]]
"""Takes the planner's questions and returns `{question id: chosen option}` for every one."""


class LLMCallRecord(BaseModel):
    """What one planner call left behind. The prompts are not kept: they carry profile values."""

    index: int
    purpose: Literal["plan", "revise", "schema_retry"]
    model: str | None
    usage: TokenUsage | None
    # Codes of the validator errors this call was asked to fix; empty for a first plan.
    issue_codes: list[str]
    # Codes of the warnings it was asked to fix too if it could.
    warning_codes: list[str]
    # What the planner answered with; `purpose` only says why it was called. "invalid" is an
    # answer that did not match the response schema, so there is no response to keep.
    response_kind: Literal["plan", "questions", "invalid"]
    response: PlannerResponse | None


@dataclass(frozen=True)
class AgentTrace:
    """Everything the agent asked, answered and was told, in order."""

    goal: str
    calls: list[LLMCallRecord]
    qa_history: list[QARecord]

    @property
    def model(self) -> str | None:
        return next((call.model for call in self.calls if call.model is not None), None)

    @property
    def total_usage(self) -> TokenUsage | None:
        """Sum over the calls that reported usage; None if no call did. A cache count is summed
        over the calls that reported it and is None if none did."""
        reported = [call.usage for call in self.calls if call.usage is not None]
        if not reported:
            return None
        return TokenUsage(
            input_tokens=sum(usage.input_tokens for usage in reported),
            output_tokens=sum(usage.output_tokens for usage in reported),
            cached_input_tokens=_sum_known(usage.cached_input_tokens for usage in reported),
            cache_write_tokens=_sum_known(usage.cache_write_tokens for usage in reported),
        )


def _sum_known(counts: Iterable[int | None]) -> int | None:
    known = [count for count in counts if count is not None]
    return sum(known) if known else None


@dataclass(frozen=True)
class AgentResult:
    """A plan that passed validation, its execution and the record of how it came about.

    `issues` are the final plan's issues (plan and result), warnings only: any error would have
    sent the plan back. Issues of earlier attempts are in `calls` as the codes they were sent as.
    """

    plan: Plan
    profile: DatasetProfile
    execution: ExecutionResult
    issues: list[ValidationIssue]
    trace: AgentTrace

    @property
    def calls(self) -> list[LLMCallRecord]:
        return self.trace.calls

    @property
    def qa_history(self) -> list[QARecord]:
        return self.trace.qa_history

    @property
    def model(self) -> str | None:
        return self.trace.model

    @property
    def total_usage(self) -> TokenUsage | None:
        return self.trace.total_usage


class AgentError(RuntimeError):
    """The agent could not deliver a valid result.

    Carries the state at the point of failure so the run can still be inspected: the last plan
    (None if the planner never produced one), the profile it was checked against, its issues,
    the execution if the plan got that far, and the trace.
    """

    def __init__(
        self,
        message: str,
        *,
        trace: AgentTrace,
        profile: DatasetProfile,
        plan: Plan | None = None,
        issues: list[ValidationIssue] | None = None,
        execution: ExecutionResult | None = None,
    ) -> None:
        super().__init__(message)
        self.trace = trace
        self.profile = profile
        self.plan = plan
        self.issues = issues or []
        self.execution = execution


class RevisionBudgetError(AgentError):
    """The plan still had errors after the last allowed revision."""


@dataclass(frozen=True)
class _Attempt:
    plan: Plan
    profile: DatasetProfile
    issues: list[ValidationIssue]
    execution: ExecutionResult | None


def run_agent(
    df: pd.DataFrame,
    goal: str,
    llm: LLMClient,
    ask_fn: AskFn,
    target: str | None = None,
    model_family: str | None = None,
    max_question_rounds: int = 2,
    max_revisions: int = 2,
    max_schema_retries: int = MAX_SCHEMA_RETRIES,
) -> AgentResult:
    """Prepare `df` for `goal`. Raises `ProfilerError` for an unusable frame or target, and
    `AgentError` if no valid result could be produced. `df` is not modified."""
    if max_question_rounds < 0 or max_revisions < 0 or max_schema_retries < 0:
        raise ValueError(
            "max_question_rounds, max_revisions and max_schema_retries must not be negative"
        )
    return _Agent(
        df,
        goal,
        llm,
        ask_fn,
        target,
        model_family,
        max_question_rounds,
        max_revisions,
        max_schema_retries,
    ).run()


class _Agent:
    def __init__(
        self,
        df: pd.DataFrame,
        goal: str,
        llm: LLMClient,
        ask_fn: AskFn,
        target: str | None,
        model_family: str | None,
        max_question_rounds: int,
        max_revisions: int,
        max_schema_retries: int,
    ) -> None:
        self.df = df
        self.goal = goal
        self.llm = llm
        self.ask_fn = ask_fn
        self.target = target
        self.model_family = model_family
        self.max_question_rounds = max_question_rounds
        self.max_revisions = max_revisions
        self.max_schema_retries = max_schema_retries

        self.profile = profile_dataframe(df, target=target)
        self.system_prompt = build_system_prompt()
        self.calls: list[LLMCallRecord] = []
        self.qa_history: list[QARecord] = []
        self.question_rounds = 0
        self.revisions = 0
        self.plan_only_retried = False
        self.attempt: _Attempt | None = None
        self._profiles: dict[str, DatasetProfile] = {}

    def run(self) -> AgentResult:
        revision: Revision | None = None
        plan_only = False
        while True:
            response = self._call_planner(revision, plan_only)
            plan_only = False
            if response.kind == "questions":
                if self.question_rounds < self.max_question_rounds:
                    self._ask_user(response.questions or [])
                else:
                    self._charge_plan_only_retry()
                    plan_only = True
                continue

            plan = response.plan
            assert plan is not None  # guaranteed by PlannerResponse for kind "plan"
            attempt = self._try(plan)
            errors, warnings = partition_issues(attempt.issues)
            if not errors:
                assert attempt.execution is not None
                return AgentResult(
                    plan, attempt.profile, attempt.execution, attempt.issues, self._trace()
                )
            if self.revisions >= self.max_revisions:
                raise RevisionBudgetError(
                    f"the plan still has {len(errors)} error(s) after {self.revisions} "
                    f"revision(s): {_describe(errors)}",
                    **self._failure(attempt),
                )
            self.revisions += 1
            revision = Revision(plan=plan, issues=errors, warnings=warnings)

    def _call_planner(self, revision: Revision | None, plan_only: bool) -> PlannerResponse:
        purpose: Literal["plan", "revise", "schema_retry"] = (
            "revise" if revision or plan_only else "plan"
        )
        schema_error: str | None = None
        attempts = 0
        while True:
            attempts += 1
            user = build_user_prompt(
                goal=self.goal,
                profile=self.profile,
                target=self.target,
                model_family=self.model_family,
                qa_history=self.qa_history,
                revision=revision,
                question_rounds_left=self.max_question_rounds - self.question_rounds,
                plan_only=plan_only,
                schema_error=schema_error,
            )
            try:
                result = self.llm.generate(self.system_prompt, user, PlannerResponse)
            except LLMSchemaError as exc:
                self._record(purpose, revision, exc.model, exc.usage, response=None)
                if attempts > self.max_schema_retries:
                    raise AgentError(
                        "the planner's answer did not match the response schema in "
                        f"{attempts} attempt(s); last error: {exc}",
                        **self._failure(),
                    ) from exc
                schema_error = _shorten(str(exc))
                purpose = "schema_retry"
                continue
            except LLMError as exc:
                raise AgentError(f"the LLM call failed: {exc}", **self._failure()) from exc
            self._record(purpose, revision, result.model, result.usage, result.parsed)
            return result.parsed

    def _record(
        self,
        purpose: Literal["plan", "revise", "schema_retry"],
        revision: Revision | None,
        model: str | None,
        usage: TokenUsage | None,
        response: PlannerResponse | None,
    ) -> None:
        self.calls.append(
            LLMCallRecord(
                index=len(self.calls),
                purpose=purpose,
                model=model,
                usage=usage,
                issue_codes=[issue.code for issue in revision.issues] if revision else [],
                warning_codes=[issue.code for issue in revision.warnings] if revision else [],
                response_kind="invalid" if response is None else response.kind,
                response=response,
            )
        )

    def _charge_plan_only_retry(self) -> None:
        """Allow one more call after questions past the limit, or give up."""
        if self.plan_only_retried or self.revisions >= self.max_revisions:
            raise AgentError(
                "the planner kept asking questions although the limit of "
                f"{self.max_question_rounds} question round(s) was reached",
                **self._failure(),
            )
        self.plan_only_retried = True
        self.revisions += 1

    def _ask_user(self, questions: list[Question]) -> None:
        answers = self.ask_fn(list(questions))
        for question in questions:
            if question.id not in answers:
                raise AgentError(
                    f"no answer was given for question {question.id!r}", **self._failure()
                )
            answer = answers[question.id]
            if question.allow_custom and not answer.strip():
                raise AgentError(
                    f"the answer to question {question.id!r} is empty", **self._failure()
                )
            if not question.allow_custom and answer not in question.options:
                raise AgentError(
                    f"the answer {answer!r} to question {question.id!r} is not "
                    f"one of the options: {question.options}",
                    **self._failure(),
                )
        self.qa_history.extend(QARecord(question=q, answer=answers[q.id]) for q in questions)
        self.question_rounds += 1

    def _try(self, plan: Plan) -> _Attempt:
        profile = self._profile_for(plan)
        # `execute` validates the plan itself and refuses to run one with errors.
        try:
            execution = execute(self.df, plan, profile, self.model_family)
        except PlanValidationError as exc:
            self.attempt = _Attempt(plan, profile, exc.issues, None)
            return self.attempt
        except ExecutionError as exc:
            self.attempt = _Attempt(plan, profile, [], None)
            raise AgentError(f"executing the plan failed: {exc}", **self._failure()) from exc
        issues = execution.plan_issues + execution.result_issues
        self.attempt = _Attempt(plan, profile, issues, execution)
        return self.attempt

    def _profile_for(self, plan: Plan) -> DatasetProfile:
        """The profile to check `plan` against.

        With a target from the user, the profile was built with it. Without one, the planner
        picked the target, so the profile is rebuilt around that choice to carry the target and
        its task hint. A target that is not a column keeps the plain profile; the validator
        reports it.
        """
        if self.profile.target is not None or plan.target not in {
            c.name for c in self.profile.columns
        }:
            return self.profile
        if plan.target not in self._profiles:
            self._profiles[plan.target] = profile_dataframe(self.df, target=plan.target)
        return self._profiles[plan.target]

    def _trace(self) -> AgentTrace:
        return AgentTrace(self.goal, list(self.calls), list(self.qa_history))

    def _failure(self, attempt: _Attempt | None = None) -> dict:
        attempt = attempt or self.attempt
        if attempt is None:
            return {"trace": self._trace(), "profile": self.profile}
        return {
            "trace": self._trace(),
            "profile": attempt.profile,
            "plan": attempt.plan,
            "issues": attempt.issues,
            "execution": attempt.execution,
        }


def _shorten(text: str) -> str:
    if len(text) <= SCHEMA_ERROR_LIMIT:
        return text
    return text[: SCHEMA_ERROR_LIMIT - 3] + "..."


def _describe(issues: list[ValidationIssue]) -> str:
    parts = []
    for issue in issues:
        where = []
        if issue.step_index is not None:
            where.append(f"step {issue.step_index}")
        if issue.column is not None:
            where.append(f"column {issue.column!r}")
        location = f" ({', '.join(where)})" if where else ""
        parts.append(f"{issue.code}{location}: {issue.message}")
    return "; ".join(parts)
