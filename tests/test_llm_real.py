"""Tests that call the real OpenAI API. They cost a few cents and only run on request:

$env:PREP_AGENT_RUN_LLM_TESTS="1"; uv run pytest -m llm -v -s
"""

import json
from pathlib import Path

import pytest

from prep_agent.llm.factory import create_llm_client
from prep_agent.llm.ping import PING_SYSTEM, PING_USER, PingReply
from prep_agent.plan import Plan
from prep_agent.profiler import load_csv, profile_dataframe
from prep_agent.validators import validate_plan

pytestmark = pytest.mark.llm

CUSTOMERS = Path(__file__).parent / "fixtures" / "customers.csv"


def test_ping_schema_round_trip():
    result = create_llm_client().generate(PING_SYSTEM, PING_USER, PingReply)

    assert isinstance(result.parsed, PingReply)
    assert result.model
    assert result.usage is not None and result.usage.input_tokens > 0


def test_plan_schema_is_accepted_by_strict_structured_outputs():
    # Plan quality is not asserted here (that is the planner's job in a later stage). The issue
    # codes are printed so a run shows how far a bare, unprompted answer is from a valid plan.
    profile = profile_dataframe(load_csv(CUSTOMERS).df, target="churned")
    system = (
        "You prepare tabular data for machine learning. Return a preparation plan as JSON "
        "that matches the schema. Use only the columns listed in the profile."
    )
    user = (
        f"Target: churned, binary classification, model family: any.\n{profile.model_dump_json()}"
    )

    result = create_llm_client().generate(system, user, Plan)

    assert isinstance(result.parsed, Plan)
    issues = validate_plan(result.parsed, profile)
    print(f"\nmodel: {result.model} | usage: {result.usage}")
    print(f"steps: {[step.op for step in result.parsed.steps]}")
    print(f"validate_plan issue codes: {[f'{i.severity}:{i.code}' for i in issues] or 'none'}")


def test_prepare_flow_runs_end_to_end_with_the_real_planner(invoke_cli, tmp_path):
    # Everything the model decided is printed first, so a failing run can be inspected with -s.
    result = invoke_cli(
        [
            "prepare",
            str(CUSTOMERS),
            "--goal",
            "Prepare this customer data to predict whether a customer churns.",
            "--target",
            "churned",
            "--non-interactive",
            "--out-dir",
            str(tmp_path / "runs"),
        ]
    )
    runs = list((tmp_path / "runs").iterdir())
    run_dir = runs[0] if runs else None
    print(f"\nexit code: {result.exit_code}")
    if run_dir is not None:
        info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        plan = json.loads((run_dir / "plan.json").read_text(encoding="utf-8"))
        trace = json.loads((run_dir / "llm_trace.json").read_text(encoding="utf-8"))
        qa = json.loads((run_dir / "qa.json").read_text(encoding="utf-8"))
        issues = json.loads((run_dir / "issues.json").read_text(encoding="utf-8"))
        print(f"status: {info['status']} | model: {info['model']} | usage: {info['token_usage']}")
        print(f"model_family: {plan['model_family']} | task_type: {plan['task_type']}")
        for index, step in enumerate(plan["steps"]):
            print(f"step {index}: {step['op']} | {step['reason']}")
        for q in qa:
            print(f"question: {q['question']['question']} -> {q['answer']}")
        print(f"questions asked: {len(qa)}")
        for call in trace["calls"]:
            print(f"call {call['index']} ({call['purpose']}): sent issues {call['issue_codes']}")
        final = issues["plan_issues"] + issues["result_issues"]
        print(f"final issue codes: {[f'{i["severity"]}:{i["code"]}' for i in final] or 'none'}")
    else:
        print(result.output)

    assert result.exit_code == 0, result.output
    assert info["status"] == "ok"
