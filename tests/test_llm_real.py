"""Tests that call the real OpenAI API. They cost a few cents and only run on request:

$env:PREP_AGENT_RUN_LLM_TESTS="1"; uv run pytest -m llm -v
"""

from pathlib import Path

import pytest

from prep_agent.llm.factory import create_llm_client
from prep_agent.llm.ping import PING_SYSTEM, PING_USER, PingReply
from prep_agent.plan import Plan
from prep_agent.profiler import load_csv, profile_dataframe

pytestmark = pytest.mark.llm

CUSTOMERS = Path(__file__).parent / "fixtures" / "customers.csv"


def test_ping_schema_round_trip():
    result = create_llm_client().generate(PING_SYSTEM, PING_USER, PingReply)

    assert isinstance(result.parsed, PingReply)
    assert result.model
    assert result.usage is not None and result.usage.input_tokens > 0


def test_plan_schema_is_accepted_by_strict_structured_outputs():
    # This is the empirical check of the Plan schema against OpenAI's strict mode: the schema
    # has oneOf/discriminator and defaults, which the docs do not list as supported.
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
    assert result.parsed.target == "churned"
