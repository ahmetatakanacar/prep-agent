"""Tests that call the real OpenAI API. They cost a few cents and only run on request:

$env:PREP_AGENT_RUN_LLM_TESTS="1"; uv run pytest -m llm -v
"""

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
    profile = profile_dataframe(load_csv(CUSTOMERS), target="churned")
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
