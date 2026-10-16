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
    return {
        "language": "English",
        "kind": "plan",
        "plan": json.loads(path.read_text(encoding="utf-8")),
        "questions": None,
    }


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


def test_the_profile_names_the_csv_settings_that_were_found(demo):
    csv = FIXTURES / "customers_tr.csv"

    snap = demo.post("/api/sessions", files={"file": ("tr.csv", csv.read_bytes())}).json()

    assert snap["profile"]["csv"] == {"sep": ";", "decimal": ",", "encoding": "cp1254"}
    assert snap["profile"]["n_rows"] == 20
    monthly = next(c for c in snap["profile"]["columns"] if c["name"] == "monthly_spend")
    assert monthly["inferred_type"] == "numeric"
    plain = demo.get(f"/api/sessions/{open_sample(demo)}").json()
    assert plain["profile"]["csv"] == {"sep": ",", "decimal": ".", "encoding": "utf-8"}


def test_a_file_that_is_not_utf8_comes_with_a_warning_code_and_others_with_none(demo):
    csv = FIXTURES / "customers_tr.csv"

    snap = demo.post("/api/sessions", files={"file": ("tr.csv", csv.read_bytes())}).json()
    plain = demo.get(f"/api/sessions/{open_sample(demo)}").json()

    assert snap["csv_warnings"] == ["encoding_fallback"]
    assert plain["csv_warnings"] == []


def test_the_warning_about_the_encoding_is_translated_and_shown_above_the_profile():
    texts = (STATIC / "i18n.js").read_text(encoding="utf-8")
    page = (STATIC / "app.js").read_text(encoding="utf-8")

    assert texts.count('"profile.encoding_fallback":') == 2
    assert (
        "This file is not UTF-8, so it was read as Turkish Windows-1254. If letters look wrong, "
        "save the file as UTF-8 and upload it again."
    ) in texts
    assert (
        "Bu dosya UTF-8 değil, bu yüzden Türkçe Windows-1254 olarak okundu. Harfler bozuk "
        "görünürse dosyayı UTF-8 olarak kaydedip yeniden yükle."
    ) in texts
    assert 't("profile.encoding_fallback"' in page
    assert "csv_warnings" in page


def test_a_run_on_a_semicolon_cp1254_file_keeps_the_settings_in_its_run_folder():
    response = plan_response(FIXTURES / "customers_plan.json")
    csv = FIXTURES / "customers_tr.csv"
    with make_client(llm_factory=scripted(response)) as client:
        session_id = client.post(
            "/api/sessions", files={"file": ("tr.csv", csv.read_bytes())}
        ).json()["id"]
        assert start(client, session_id, target="churned", model_family="any").status_code == 202
        snap = wait_for(client, session_id, FINISHED)
        assert snap["status"] == "done", snap["error"]
        info = client.get(f"/api/sessions/{session_id}/files/run.json").json()
        script = client.get(f"/api/sessions/{session_id}/files/pipeline.py").text

    assert info["csv"] == {"sep": ";", "decimal": ",", "encoding": "cp1254"}
    assert 'CSV_OPTIONS = {"sep": ";", "decimal": ",", "encoding": "cp1254"}' in script


@pytest.mark.parametrize("key", ["profile.sep", "profile.encoding"])
def test_the_csv_settings_of_the_profile_are_translated_and_shown_by_the_page(key):
    texts = (STATIC / "i18n.js").read_text(encoding="utf-8")
    page = (STATIC / "app.js").read_text(encoding="utf-8")

    assert texts.count(f'"{key}":') == 2, "one entry for English and one for Turkish"
    assert f't("{key}"' in page


def test_the_separator_of_the_profile_is_shown_by_name_in_both_languages():
    texts = (STATIC / "i18n.js").read_text(encoding="utf-8")
    page = (STATIC / "app.js").read_text(encoding="utf-8")

    names = {
        "profile.sep_comma": ("comma", "virgül"),
        "profile.sep_semicolon": ("semicolon", "noktalı virgül"),
        "profile.sep_tab": ("tab", "sekme"),
        "profile.sep_pipe": ("pipe", "dikey çizgi"),
    }
    for key, (english, turkish) in names.items():
        assert texts.count(f'"{key}":') == 2
        assert f'"{key}": "{english}"' in texts
        assert f'"{key}": "{turkish}"' in texts
        assert f'"{key}"' in page
    assert '"profile.sep": "separator: {sep}"' in texts
    assert '"profile.sep": "ayraç: {sep}"' in texts


def test_the_demo_mode_text_of_the_top_bar_is_short_enough_for_its_cell():
    texts = (STATIC / "i18n.js").read_text(encoding="utf-8")

    assert '"tb.mode_demo": "demo, scripted"' in texts
    assert '"tb.mode_demo": "demo (betikli)"' in texts


def test_the_language_notes_of_the_brief_and_plan_exist_in_both_languages_and_are_shown_in_both():
    texts = (STATIC / "i18n.js").read_text(encoding="utf-8")
    page = (STATIC / "app.js").read_text(encoding="utf-8")

    for key in ("brief.langnote", "plan.en_note"):
        assert texts.count(f'"{key}":') == 2, "one entry for English and one for Turkish"
        assert f't("{key}")' in page
        assert f'lang === "tr" && hasText("{key}")' not in page
    assert "Questions and reasons come in the language you write the goal in." in texts
    assert "The agent's reasons are in the language of the goal, not of the interface." in texts


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


def test_a_target_with_one_value_ends_the_run_before_any_llm_call():
    llms = []

    def factory(session, request):
        llms.append(FakeLLM([]))
        return llms[-1]

    csv = "a,t\n" + "".join(f"{i},yes\n" for i in range(30))
    with make_client(llm_factory=factory) as client:
        session_id = client.post("/api/sessions", files={"file": ("one.csv", csv.encode())}).json()[
            "id"
        ]

        assert start(client, session_id, target="t").status_code == 202
        snap = wait_for(client, session_id, FINISHED)

    assert snap["status"] == "error"
    assert "only one value" in snap["error"]
    assert llms[0].calls == []


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
    assert [q["id"] for q in waiting["questions"]] == ["target", "model_family"]
    target_question, family_question = waiting["questions"]
    assert target_question["recommended"] == "churned"
    assert target_question["allow_custom"] is True
    assert family_question["options"] == ["tree", "linear", "any"]
    assert family_question["recommended"] == "any"
    assert family_question["allow_custom"] is False

    # A second run, or a wrong answer, must not disturb the waiting agent.
    assert start(demo, session_id).status_code == 409
    assert answer(demo, session_id, {"model_family": "forest"}).status_code == 422
    assert demo.get(f"/api/sessions/{session_id}").json()["status"] == "waiting"

    answers = {"target": "churned", "model_family": "any"}
    assert answer(demo, session_id, answers).status_code == 200
    done = wait_for(demo, session_id, FINISHED)
    assert done["status"] == "done", done["error"]
    assert [r["answer"] for r in done["qa"]] == ["churned", "any"]

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
    # No op can use a date, so it goes without a question to the user.
    assert dropped["signup_date"]["kinds"] == ["unsupported_type"]
    assert dropped["signup_date"]["asked"] == []


def test_a_chosen_target_means_no_target_question(demo):
    session_id = open_sample(demo)
    start(demo, session_id, target="churned")
    waiting = wait_for(demo, session_id, {"waiting"})
    assert [q["id"] for q in waiting["questions"]] == ["model_family"]


def test_a_brief_with_a_target_and_a_family_gets_a_plan_without_questions(demo):
    session_id = open_sample(demo)
    start(demo, session_id, target="churned", model_family="tree")

    done = wait_for(demo, session_id, FINISHED)

    assert done["status"] == "done", done["error"]
    assert done["qa"] == []
    result = demo.get(f"/api/sessions/{session_id}/result").json()
    # The story of a rejected and a corrected plan does not depend on the questions.
    assert len(result["revisions"]) == 1
    assert result["issues"] == []


FAMILY_LABELS = {
    "en": {"tree": "tree", "linear": "linear", "any": "any"},
    "tr": {"tree": "ağaç", "linear": "doğrusal", "any": "fark etmez"},
}


def test_the_demo_asks_in_the_language_of_the_interface(demo):
    session_id = open_sample(demo)
    start(demo, session_id, lang="tr")
    waiting = wait_for(demo, session_id, {"waiting"})
    target_question, family_question = waiting["questions"]
    assert "hangi sütunu" in target_question["question"]
    assert "tür modeli" in family_question["question"]
    # The options carry the labels of the family buttons of the form.
    assert family_question["options"] == ["ağaç", "doğrusal", "fark etmez"]
    assert family_question["recommended"] == "fark etmez"
    assert "`any`" not in family_question["reason"]
    assert "fark etmez" in family_question["reason"]
    assert "any" not in family_question["options"]
    answers = {"target": "churned", "model_family": "fark etmez"}
    assert answer(demo, session_id, answers).status_code == 200
    assert wait_for(demo, session_id, FINISHED)["status"] == "done"
    result = demo.get(f"/api/sessions/{session_id}/result").json()
    drop = next(s for s in result["plan"]["steps"] if s["op"] == "drop_columns")
    assert "signup_date" in drop["columns"]
    assert "ham bir tarih metni" in drop["reason"]
    assert "düşürmeyi sen seçtin" not in drop["reason"]


def test_the_english_family_question_keeps_its_plain_options(demo):
    session_id = open_sample(demo)
    start(demo, session_id, target="churned")
    [question] = wait_for(demo, session_id, {"waiting"})["questions"]

    assert question["options"] == ["tree", "linear", "any"]
    assert question["recommended"] == "any"
    assert "`any`" in question["reason"]


@pytest.mark.parametrize("lang", ["en", "tr"])
@pytest.mark.parametrize("family", ["tree", "linear", "any"])
def test_the_answer_about_the_family_shapes_the_plan(demo, lang, family):
    session_id = open_sample(demo)
    start(demo, session_id, target="churned", lang=lang)
    wait_for(demo, session_id, {"waiting"})
    label = FAMILY_LABELS[lang][family]
    assert answer(demo, session_id, {"model_family": label}).status_code == 200
    assert wait_for(demo, session_id, FINISHED)["status"] == "done"

    plan = demo.get(f"/api/sessions/{session_id}/result").json()["plan"]

    assert plan["model_family"] == family
    scales = [s for s in plan["steps"] if s["op"] == "scale"]
    if family == "tree":
        assert scales == []
    else:
        assert [s["parameters"]["method"] for s in scales] == ["robust"]


def test_a_free_text_answer_is_accepted_where_the_question_allows_it(demo):
    session_id = open_sample(demo)
    start(demo, session_id)
    wait_for(demo, session_id, {"waiting"})
    empty = {"target": "  ", "model_family": "any"}
    assert answer(demo, session_id, empty).status_code == 422
    # The family question is limited to its options.
    free = {"target": "age", "model_family": "whenever"}
    assert answer(demo, session_id, free).status_code == 422
    free = {"target": "age", "model_family": "any"}
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
    assert answer(demo, session_id, {"model_family": "any"}).status_code == 200
    done = wait_for(demo, session_id, FINISHED)
    # The first, cancelled run must not have written over the second one.
    assert done["status"] == "done"
    assert done["run_number"] == 2
    assert demo.get(f"/api/sessions/{session_id}/result").json()["run"]["run_id"] == "run-2"


def test_reset_after_a_finished_run_clears_the_result(demo):
    session_id = open_sample(demo)
    start(demo, session_id, target="churned")
    wait_for(demo, session_id, {"waiting"})
    answer(demo, session_id, {"model_family": "any"})
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
    answer(demo, session_id, {"target": "churned", "model_family": "any"})
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
@pytest.mark.parametrize(
    ("brief", "asked"),
    [
        ({}, ["target", "model_family"]),
        ({"target": "churned"}, ["model_family"]),
        ({"target": "churned", "model_family": "linear"}, []),
    ],
    ids=["nothing given", "target given", "target and family given"],
)
def test_the_demo_ends_with_a_plan_that_has_no_issues_at_all(demo, lang, brief, asked):
    session_id = open_sample(demo)
    assert start(demo, session_id, lang=lang, **brief).status_code == 202
    if asked:
        waiting = wait_for(demo, session_id, {"waiting"})
        assert [q["id"] for q in waiting["questions"]] == asked
        answers = {"target": "churned", "model_family": FAMILY_LABELS[lang]["any"]}
        assert answer(demo, session_id, {k: answers[k] for k in asked}).status_code == 200
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
