"""The agent loop: from a goal and a dataset to executed, validated data preparation.

Flow: profile -> planner (LLM) -> [questions -> user -> planner]* -> plan validation ->
execution -> result validation. A plan or result with errors goes back to the planner together
with the validator's issues. Warnings never do.

Two limits keep the loop finite. `max_question_rounds` caps how often the user is asked; once it
is reached the planner is told to answer with a plan. If it asks anyway, it gets one more call
with a plan-only instruction, and that call is charged to the revision budget. `max_revisions`
caps how often a plan is sent back, and plan errors, result errors and that retry draw on the
same budget. When the budget is used up
and errors remain, `RevisionBudgetError` carries the last plan and its issues.

The planner only ever sees the profile (see `prep_agent.planner`), never the data frame.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import pandas as pd
from pydantic import BaseModel

from prep_agent.executor import ExecutionError, ExecutionResult, PlanValidationError, execute
from prep_agent.llm.base import LLMClient, LLMError, TokenUsage
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

AskFn = Callable[[list[Question]], dict[str, str]]
"""Takes the planner's questions and returns `{question id: chosen option}` for every one."""


class LLMCallRecord(BaseModel):
    """What one planner call left behind. The prompts are not kept: they carry profile values."""

    index: int
    purpose: Literal["plan", "revise"]
    model: str
    usage: TokenUsage | None
    # Codes of the validator errors this call was asked to fix; empty for a first plan.
    issue_codes: list[str]
    response: PlannerResponse


@dataclass(frozen=True)
class AgentTrace:
    """Everything the agent asked, answered and was told, in order."""

    goal: str
    calls: list[LLMCallRecord]
    qa_history: list[QARecord]

    @property
    def model(self) -> str | None:
        return self.calls[0].model if self.calls else None

    @property
    def total_usage(self) -> TokenUsage | None:
        """Sum over the calls that reported usage; None if no call did."""
        reported = [call.usage for call in self.calls if call.usage is not None]
        if not reported:
            return None
        return TokenUsage(
            input_tokens=sum(usage.input_tokens for usage in reported),
            output_tokens=sum(usage.output_tokens for usage in reported),
        )


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
) -> AgentResult:
    """Prepare `df` for `goal`. Raises `ProfilerError` for an unusable frame or target, and
    `AgentError` if no valid result could be produced. `df` is not modified."""
    if max_question_rounds < 0 or max_revisions < 0:
        raise ValueError("max_question_rounds and max_revisions must not be negative")
    return _Agent(
        df, goal, llm, ask_fn, target, model_family, max_question_rounds, max_revisions
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
    ) -> None:
        self.df = df
        self.goal = goal
        self.llm = llm
        self.ask_fn = ask_fn
        self.target = target
        self.model_family = model_family
        self.max_question_rounds = max_question_rounds
        self.max_revisions = max_revisions

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
            errors, _ = partition_issues(attempt.issues)
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
            revision = Revision(plan=plan, issues=errors)

    def _call_planner(self, revision: Revision | None, plan_only: bool) -> PlannerResponse:
        user = build_user_prompt(
            goal=self.goal,
            profile=self.profile,
            target=self.target,
            model_family=self.model_family,
            qa_history=self.qa_history,
            revision=revision,
            question_rounds_left=self.max_question_rounds - self.question_rounds,
            plan_only=plan_only,
        )
        try:
            result = self.llm.generate(self.system_prompt, user, PlannerResponse)
        except LLMError as exc:
            raise AgentError(f"the LLM call failed: {exc}", **self._failure()) from exc
        self.calls.append(
            LLMCallRecord(
                index=len(self.calls),
                purpose="revise" if revision or plan_only else "plan",
                model=result.model,
                usage=result.usage,
                issue_codes=[issue.code for issue in revision.issues] if revision else [],
                response=result.parsed,
            )
        )
        return result.parsed

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
