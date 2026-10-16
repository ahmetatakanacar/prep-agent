"""Endpoint tests for the local web UI in `app/`. They run the real agent with `FakeLLM`; no
network is touched."""

import io
import json
import re
import time
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from app.config import Settings  # noqa: E402
from app.results import _baseline_view  # noqa: E402
from app.server import create_app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from prep_agent.baseline import (  # noqa: E402
    NOT_CONVERGED_NOTE,
    BaselineResult,
    BaselineScore,
    ModelBaseline,
)
from prep_agent.llm.base import LLMConfigError, TokenUsage  # noqa: E402
from prep_agent.llm.fake import FakeLLM  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
CUSTOMERS = FIXTURES / "customers.csv"
TEST_HOSTS = ("testserver",)
FINISHED = {"done", "failed", "error"}


def make_client(llm_factory=None, **settings) -> TestClient:
    app = create_app(Settings(allowed_hosts=TEST_HOSTS, **settings), llm_factory=llm_factory)
    return TestClient(app)


@pytest.fixture
def demo():
    with make_client(fake=True) as client:
        yield client


def open_sample(client: TestClient) -> str:
    response = client.post("/api/sessions/sample", json={"id": "customers"})
    assert response.status_code == 201
    return response.json()["id"]


def wait_for(client: TestClient, session_id: str, statuses: set[str], timeout: float = 60.0):
    deadline = time.monotonic() + timeout
    snap = {}
    while time.monotonic() < deadline:
        snap = client.get(f"/api/sessions/{session_id}").json()
        if snap["status"] in statuses:
            return snap
        time.sleep(0.05)
    raise AssertionError(f"session did not reach {statuses}; last state: {snap}")


def plan_response(path: Path) -> dict:
    return {"kind": "plan", "plan": json.loads(path.read_text(encoding="utf-8")), "questions": None}


def scripted(*responses):
    return lambda session, request: FakeLLM(responses)


def start(client: TestClient, session_id: str, **body):
    return client.post(f"/api/sessions/{session_id}/plan", json={"goal": "predict churn", **body})


def answer(client: TestClient, session_id: str, answers: dict):
    return client.post(f"/api/sessions/{session_id}/answers", json={"answers": answers})


def test_upload_returns_the_profile(demo):
    response = demo.post("/api/sessions", files={"file": ("my data.csv", CUSTOMERS.read_bytes())})
    assert response.status_code == 201
    snap = response.json()
    assert snap["status"] == "ready"
    assert snap["filename"] == "my data.csv"
    assert snap["profile"]["n_rows"] == 20
    assert [c["name"] for c in snap["profile"]["columns"]][:2] == ["customer_id", "age"]


@pytest.mark.parametrize("content", [b"", b"a,b\n"])
def test_upload_rejects_unusable_files(demo, content):
    response = demo.post("/api/sessions", files={"file": ("x.csv", content)})
    assert response.status_code == 422
    assert response.json()["detail"]


def test_upload_over_the_limit_is_refused():
    with make_client(max_upload_bytes=100) as client:
        response = client.post("/api/sessions", files={"file": ("big.csv", CUSTOMERS.read_bytes())})
    assert response.status_code == 413
    assert "limit" in response.json()["detail"]


def test_config_lists_only_the_customers_sample_in_demo_mode(demo):
    config = demo.get("/api/config").json()
    assert config["fake"] is True
    assert [s["id"] for s in config["samples"]] == ["customers"]


def test_unknown_session_and_sample_are_404(demo):
    assert demo.get("/api/sessions/nope").status_code == 404
    assert demo.post("/api/sessions/sample", json={"id": "nope"}).status_code == 404


def test_plan_request_is_validated(demo):
    session_id = open_sample(demo)
    assert start(demo, session_id, goal="  ").status_code == 422
    assert start(demo, session_id, target="nope").status_code == 422
    assert start(demo, session_id, model_family="forest").status_code == 422


def test_demo_mode_refuses_a_target_its_script_does_not_cover(demo):
    session_id = open_sample(demo)

    response = start(demo, session_id, target="plan")

    assert response.status_code == 422
    assert response.json()["code"] == "demo_unsupported"
    assert response.json()["params"] == {"target": "plan", "demo_target": "churned"}
    assert "churned" in response.json()["detail"]


def test_demo_mode_refuses_a_file_its_script_does_not_cover(demo):
    csv = "a,b\n" + "".join(f"{i},{i % 2}\n" for i in range(30))
    session_id = demo.post("/api/sessions", files={"file": ("other.csv", csv.encode())}).json()[
        "id"
    ]

    response = start(demo, session_id)

    assert response.status_code == 422
    assert response.json()["code"] == "demo_unsupported_file"
    assert "customer_id" in response.json()["params"]["columns"]


def test_config_names_the_only_target_of_the_demo_script(demo):
    assert demo.get("/api/config").json()["demo_target"] == "churned"
    with make_client() as live:
        assert live.get("/api/config").json()["demo_target"] is None


@pytest.mark.parametrize(
    "key", ["api.demo_unsupported", "api.demo_unsupported_file", "brief.demo_target_note"]
)
def test_the_demo_texts_are_translated_and_used_by_the_page(key):
    texts = (STATIC / "i18n.js").read_text(encoding="utf-8")
    page = (STATIC / "app.js").read_text(encoding="utf-8")

    assert texts.count(f'"{key}":') == 2, "one entry for English and one for Turkish"
    if key.startswith("brief"):
        assert f't("{key}"' in page
        assert "demo_target" in page


def test_question_answer_flow_ends_in_a_corrected_plan(demo):
    session_id = open_sample(demo)
    assert start(demo, session_id).status_code == 202

    waiting = wait_for(demo, session_id, {"waiting"})
    assert [q["id"] for q in waiting["questions"]] == ["target", "signup_date"]
    target_question = waiting["questions"][0]
    assert target_question["recommended"] == "churned"
    assert target_question["allow_custom"] is True
    assert not waiting["questions"][1]["allow_custom"]

    # A second run, or a wrong answer, must not disturb the waiting agent.
    assert start(demo, session_id).status_code == 409
    assert answer(demo, session_id, {"signup_date": "Maybe"}).status_code == 422
    assert demo.get(f"/api/sessions/{session_id}").json()["status"] == "waiting"

    answers = {"target": "churned", "signup_date": "Drop it"}
    assert answer(demo, session_id, answers).status_code == 200
    done = wait_for(demo, session_id, FINISHED)
    assert done["status"] == "done", done["error"]
    assert [r["answer"] for r in done["qa"]] == ["churned", "Drop it"]

    result = demo.get(f"/api/sessions/{session_id}/result").json()
    assert result["status"] == "ok"
    assert len(result["revisions"]) == 1
    revision = result["revisions"][0]
    codes = [i["code"] for i in revision["rejected_issues"]]
    assert codes == ["unknown_column", "stateful_before_split", "scale_non_numeric"]
    fates = {s["op"]: s["fate"] for s in revision["rejected_steps"]}
    assert fates["impute"] == "moved"
    assert result["plan"]["target"] == "churned"
    assert result["run"]["rows"] == {"before": 20, "train": 16, "test": 4}
    assert result["baseline"]["models"][0]["is_reference"] is True
    # A script used no tokens, so the demo reports no usage at all, per run and per call.
    assert result["llm"]["usage"] is None
    assert all(call["usage"] is None for call in result["llm"]["calls"])
    assert {f["name"] for f in result["files"]} >= {"plan.json", "train.csv", "report.md"}
    assert result["report_md"].startswith("# Preparation report")
    dropped = {c["column"]: c for group in result["dropped"] for c in group["columns"]}
    assert sorted(dropped) == ["country", "customer_id", "signup_date"]
    assert dropped["customer_id"]["kinds"] == ["identifier"]
    assert dropped["country"]["kinds"] == ["constant"]
    # The date cannot be used by any op, and the user was asked about it.
    assert dropped["signup_date"]["kinds"] == ["unsupported_type", "asked"]
    assert dropped["signup_date"]["asked"][0]["answer"] == "Drop it"


def test_a_chosen_target_means_no_target_question(demo):
    session_id = open_sample(demo)
    start(demo, session_id, target="churned")
    waiting = wait_for(demo, session_id, {"waiting"})
    assert [q["id"] for q in waiting["questions"]] == ["signup_date"]


def test_the_demo_asks_in_the_language_of_the_interface(demo):
    session_id = open_sample(demo)
    start(demo, session_id, lang="tr")
    waiting = wait_for(demo, session_id, {"waiting"})
    date_question = waiting["questions"][1]
    assert date_question["options"] == ["Düşür", "Tut"]
    assert answer(demo, session_id, {"target": "churned", "signup_date": "Tut"}).status_code == 200
    assert wait_for(demo, session_id, FINISHED)["status"] == "done"
    result = demo.get(f"/api/sessions/{session_id}/result").json()
    drop = next(s for s in result["plan"]["steps"] if s["op"] == "drop_columns")
    assert "signup_date" not in drop["columns"]
    assert "düşürmeyi sen seçtin" not in drop["reason"]
    # Keeping a date the ops cannot use is the user's call; the validators only warn.
    assert any(i["code"] == "unencoded_column_remains" for i in result["issues"])


def test_a_free_text_answer_is_accepted_where_the_question_allows_it(demo):
    session_id = open_sample(demo)
    start(demo, session_id)
    wait_for(demo, session_id, {"waiting"})
    empty = {"target": "  ", "signup_date": "Drop it"}
    assert answer(demo, session_id, empty).status_code == 422
    # The date question is limited to its options.
    free = {"target": "age", "signup_date": "whenever"}
    assert answer(demo, session_id, free).status_code == 422
    free = {"target": "age", "signup_date": "Drop it"}
    assert answer(demo, session_id, free).status_code == 200
    snap = wait_for(demo, session_id, FINISHED)
    # The script has no plan for another target and says so instead of pretending.
    assert snap["status"] == "failed"
    assert "churned" in snap["error"]


def test_reset_cancels_a_waiting_run_and_the_session_can_plan_again(demo):
    session_id = open_sample(demo)
    start(demo, session_id)
    wait_for(demo, session_id, {"waiting"})

    snap = demo.post(f"/api/sessions/{session_id}/reset").json()
    assert snap["status"] == "ready"
    assert snap["request"] is None and snap["questions"] is None and snap["qa"] == []
    assert snap["profile"]["n_rows"] == 20

    start(demo, session_id, target="churned")
    wait_for(demo, session_id, {"waiting"})
    assert answer(demo, session_id, {"signup_date": "Drop it"}).status_code == 200
    done = wait_for(demo, session_id, FINISHED)
    # The first, cancelled run must not have written over the second one.
    assert done["status"] == "done"
    assert done["run_number"] == 2
    assert demo.get(f"/api/sessions/{session_id}/result").json()["run"]["run_id"] == "run-2"


def test_reset_after_a_finished_run_clears_the_result(demo):
    session_id = open_sample(demo)
    start(demo, session_id, target="churned")
    wait_for(demo, session_id, {"waiting"})
    answer(demo, session_id, {"signup_date": "Drop it"})
    wait_for(demo, session_id, {"done"})
    assert demo.post(f"/api/sessions/{session_id}/reset").json()["status"] == "ready"
    assert demo.get(f"/api/sessions/{session_id}/result").status_code == 409


def test_errors_carry_a_code_for_the_interface_to_translate():
    with make_client(max_upload_bytes=100) as client:
        too_big = client.post("/api/sessions", files={"file": ("b.csv", CUSTOMERS.read_bytes())})
        missing = client.get("/api/sessions/nope")
    assert too_big.json()["code"] == "file_too_large"
    assert "limit" in too_big.json()["params"]
    assert missing.json()["code"] == "session_not_found"


def test_result_before_a_run_finishes_is_409(demo):
    session_id = open_sample(demo)
    assert demo.get(f"/api/sessions/{session_id}/result").status_code == 409
    assert demo.get(f"/api/sessions/{session_id}/download").status_code == 409


def test_download_is_the_whole_run_folder(demo):
    session_id = open_sample(demo)
    start(demo, session_id)
    wait_for(demo, session_id, {"waiting"})
    answer(demo, session_id, {"target": "churned", "signup_date": "Drop it"})
    wait_for(demo, session_id, FINISHED)

    response = demo.get(f"/api/sessions/{session_id}/download")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        names = set(archive.namelist())
    expected = {"profile.json", "plan.json", "train.csv", "test.csv", "pipeline.joblib"}
    expected |= {"pipeline.py", "report.md", "llm_trace.json", "baseline.json"}
    assert {f"run-1/{name}" for name in expected} <= names

    single = demo.get(f"/api/sessions/{session_id}/files/plan.json")
    assert single.status_code == 200
    assert json.loads(single.content)["target"] == "churned"
    assert demo.get(f"/api/sessions/{session_id}/files/..%2Finput.csv").status_code == 404
    assert demo.get(f"/api/sessions/{session_id}/files/missing.csv").status_code == 404


def test_a_valid_first_plan_needs_no_questions_or_revisions():
    plan = plan_response(FIXTURES / "customers_plan.json")
    with make_client(llm_factory=scripted(plan)) as client:
        session_id = open_sample(client)
        start(client, session_id)
        assert wait_for(client, session_id, FINISHED)["status"] == "done"
        result = client.get(f"/api/sessions/{session_id}/result").json()
    assert result["revisions"] == []
    assert result["qa"] == []
    # FakeLLM reports no usage unless it is given some.
    assert result["llm"]["usage"] is None


def test_the_result_has_token_counts_and_no_cost():
    plan = plan_response(FIXTURES / "customers_plan.json")
    usage = TokenUsage(input_tokens=1200, output_tokens=300, cached_input_tokens=400)

    def factory(session, request):
        return FakeLLM([plan], usage=usage)

    with make_client(llm_factory=factory) as client:
        session_id = open_sample(client)
        start(client, session_id)
        wait_for(client, session_id, FINISHED)
        llm = client.get(f"/api/sessions/{session_id}/result").json()["llm"]
    # Prices change, so the interface shows what a run used and never what it would cost.
    assert "cost" not in llm
    assert llm["usage"] == {
        "input_tokens": 1200,
        "output_tokens": 300,
        "cached_input_tokens": 400,
        "cache_write_tokens": None,
    }
    assert len(llm["calls"]) == 1
    assert llm["calls"][0]["usage"]["cached_input_tokens"] == 400


@pytest.mark.parametrize("lang", ["en", "tr"])
def test_the_demo_ends_with_a_plan_that_has_no_issues_at_all(demo, lang):
    session_id = open_sample(demo)
    assert start(demo, session_id, lang=lang).status_code == 202
    wait_for(demo, session_id, {"waiting"})
    answers = {"target": "churned", "signup_date": "Drop it" if lang == "en" else "Düşür"}
    assert answer(demo, session_id, answers).status_code == 200
    assert wait_for(demo, session_id, FINISHED)["status"] == "done"

    result = demo.get(f"/api/sessions/{session_id}/result").json()

    assert result["issues"] == []
    # The corrected plan follows the rule the agent follows: no step the data does not need.
    assert "drop_missing_target" not in [step["op"] for step in result["plan"]["steps"]]


STATIC = Path(__file__).parent.parent / "app" / "static"


def test_every_percentage_of_the_page_comes_from_one_locale_aware_formatter():
    # The page is plain JavaScript and no JS runner is available to the tests, so this checks the
    # source: the formatter is built on Intl.NumberFormat with the locale of the chosen language,
    # numbers and percentages share the locale table, and no text puts a % sign around a value
    # by hand (Turkish writes the sign first, English last, and the formatter knows which).
    page = (STATIC / "app.js").read_text(encoding="utf-8")
    texts = (STATIC / "i18n.js").read_text(encoding="utf-8")

    assert 'style: "percent"' in page
    assert "Intl.NumberFormat(LOCALES[lang]" in page
    assert "toLocaleString(LOCALES[lang])" in page
    assert not re.search(r"%\{|\}%", texts)
    assert "toFixed(1)}%" not in page
    assert "Math.round(p.test_size * 100)" not in page


def baseline_with_a_note() -> BaselineResult:
    score = {"accuracy": BaselineScore(mean=0.8, std=0.1)}
    return BaselineResult(
        task_type="classification",
        n_rows=200,
        random_state=42,
        n_folds=5,
        metrics=["accuracy"],
        models=[
            ModelBaseline(name="dummy_most_frequent", scores=score),
            ModelBaseline(name="logistic_regression", scores=score, note=NOT_CONVERGED_NOTE),
        ],
    )


def test_the_result_carries_the_note_of_a_model_that_did_not_converge():
    models = _baseline_view(baseline_with_a_note())["models"]

    assert [m["note"] for m in models] == [None, NOT_CONVERGED_NOTE]
    assert [m["not_converged"] for m in models] == [False, True]


@pytest.mark.parametrize("key", ["model.not_converged", "base.not_converged"])
def test_the_convergence_note_is_translated_and_used_by_the_page(key):
    static = Path(__file__).parent.parent / "app" / "static"
    texts = (static / "i18n.js").read_text(encoding="utf-8")
    page = (static / "app.js").read_text(encoding="utf-8")

    assert texts.count(f'"{key}":') == 2, "one entry for English and one for Turkish"
    assert f't("{key}")' in page


def test_a_plan_that_stays_invalid_ends_as_a_failed_run_that_can_still_be_inspected():
    bad = plan_response(FIXTURES / "customers_plan_invalid.json")
    with make_client(llm_factory=scripted(bad, bad, bad)) as client:
        session_id = open_sample(client)
        start(client, session_id)
        snap = wait_for(client, session_id, FINISHED)
        assert snap["status"] == "failed"
        assert "revision" in snap["error"]
        result = client.get(f"/api/sessions/{session_id}/result").json()
        assert client.get(f"/api/sessions/{session_id}/download").status_code == 200
    assert result["status"] == "failed"
    assert result["baseline"] is None
    assert result["plan"] is not None
    assert {i["code"] for i in result["issues"]} >= {"stateful_before_split", "unknown_column"}
    assert len(result["revisions"]) == 2


def test_an_llm_that_cannot_be_built_is_reported_before_a_run_starts():
    def broken(session, request):
        raise LLMConfigError("OPENAI_API_KEY is not set")

    with make_client(llm_factory=broken) as client:
        session_id = open_sample(client)
        response = start(client, session_id)
        assert response.status_code == 400
        assert "OPENAI_API_KEY" in response.json()["detail"]
        assert client.get(f"/api/sessions/{session_id}").json()["status"] == "ready"


def test_deleting_a_session_removes_it(demo):
    session_id = open_sample(demo)
    assert demo.delete(f"/api/sessions/{session_id}").status_code == 204
    assert demo.get(f"/api/sessions/{session_id}").status_code == 404
    assert demo.delete(f"/api/sessions/{session_id}").status_code == 404


def test_deleting_a_waiting_session_releases_the_agent_thread(demo):
    session_id = open_sample(demo)
    start(demo, session_id)
    wait_for(demo, session_id, {"waiting"})
    assert demo.delete(f"/api/sessions/{session_id}").status_code == 204


def test_session_files_are_removed_when_the_server_stops(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    with make_client(fake=True) as client:
        open_sample(client)
        assert any(tmp_path.rglob("customers.csv"))
    assert not any(tmp_path.iterdir())


def test_cross_origin_posts_and_foreign_hosts_are_refused(demo):
    session_id = open_sample(demo)
    refused = demo.post(
        f"/api/sessions/{session_id}/plan",
        json={"goal": "x"},
        headers={"Origin": "http://evil.example"},
    )
    assert refused.status_code == 403
    assert demo.get("/api/config", headers={"Host": "evil.example"}).status_code == 400


THREE_COLUMN_REASON = (
    "id identifies a row, email is personal data, and leaky is recorded after the outcome, "
    "which is target leakage."
)


@pytest.mark.parametrize(
    ("column", "columns", "reason", "expected"),
    [
        # The leakage word belongs to the column it follows, not to its neighbours.
        ("leaky", ["id", "email", "leaky"], THREE_COLUMN_REASON, True),
        ("id", ["id", "email", "leaky"], THREE_COLUMN_REASON, False),
        ("email", ["id", "email", "leaky"], THREE_COLUMN_REASON, False),
        # A sentence boundary ends the reading too.
        ("a", ["a", "b"], "a is an identifier. b leaks the target.", False),
        ("b", ["a", "b"], "a is an identifier. b leaks the target.", True),
        # The only column of a step is charged even when the reason does not name it.
        ("boat", ["boat"], "It gives the target away.", True),
        ("boat", ["boat", "body"], "It gives the target away.", False),
    ],
)
def test_a_leakage_claim_is_read_per_column(column, columns, reason, expected):
    from app.results import claims_leakage

    assert claims_leakage(column, columns, reason) is expected
