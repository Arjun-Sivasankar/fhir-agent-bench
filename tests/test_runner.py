"""The benchmark runner end to end on the fixture with the gold-replay stub (the CI smoke run)."""

import json
from pathlib import Path

import pytest

from fhir_agent_bench.questions import read_jsonl
from fhir_agent_bench.runner import main, open_rows, run_benchmark
from fhir_agent_bench.store import BundleStore
from fhir_agent_bench.stub import ScriptedModel, final_message, gold_replay_model

QUESTIONS_PATH = Path(__file__).parents[1] / "questions" / "fixture.jsonl"


def test_cli_smoke_run_scores_perfectly(tmp_path, capsys):
    main(
        [
            "--name",
            "smoke",
            "--model-kind",
            "stub-gold",
            "--questions",
            str(QUESTIONS_PATH),
            "--results-dir",
            str(tmp_path),
        ]
    )
    out = json.loads((tmp_path / "smoke.json").read_text(encoding="utf-8"))
    s, meta = out["summary"], out["meta"]
    n = len(read_jsonl(QUESTIONS_PATH))
    assert s["n"] == n
    assert (s["tool_selection"], s["args"], s["answer"], s["refusal_correct"]) == (1, 1, 1, 1)
    assert s["format_failures"] == 0 and s["missed_refusals"] == 0
    assert meta["model"] == "gold-replay stub" and meta["data"] == "fixture"
    rows = (tmp_path / "smoke.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == n and json.loads(rows[0])["run"]["trace"]
    audit = (tmp_path / "smoke.audit.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(audit) == sum(len(q.gold_calls) for q in read_jsonl(QUESTIONS_PATH))


async def test_resume_skips_done_questions(tmp_path):
    qs = read_jsonl(QUESTIONS_PATH)[:4]
    rows = tmp_path / "r.jsonl"
    cfg = {"model": "x"}
    calls = []

    def model_for(q):
        calls.append(q.id)
        return gold_replay_model(q)

    await run_benchmark(qs[:2], BundleStore.fixture(), model_for, rows, cfg)
    summary = await run_benchmark(qs, BundleStore.fixture(), model_for, rows, cfg)
    assert calls == [q.id for q in qs]  # first two not re-run
    assert summary["n"] == 4


def test_resume_with_other_settings_is_refused(tmp_path):
    rows = tmp_path / "r.jsonl"
    open_rows(rows, {"model": "a"})
    with pytest.raises(SystemExit, match="other settings"):
        open_rows(rows, {"model": "b"})


def test_partial_last_line_is_dropped(tmp_path):
    rows = tmp_path / "r.jsonl"
    open_rows(rows, {"model": "a"})
    rows.write_text('{"id": "q001", "answer": 1}\n{"id": "q00', encoding="utf-8")
    assert list(open_rows(rows, {"model": "a"})) == ["q001"]
    assert rows.read_text(encoding="utf-8") == '{"id": "q001", "answer": 1}\n'


async def test_a_wrong_model_scores_badly(tmp_path):
    qs = [q for q in read_jsonl(QUESTIONS_PATH) if q.category != "refusal"][:3]

    def model_for(q):  # answers without calling any tool
        return ScriptedModel(script=[final_message(["42"])])

    s = await run_benchmark(qs, BundleStore.fixture(), model_for, tmp_path / "w.jsonl", {})
    assert s["answer"] == 0 and s["tool_selection"] == 0 and s["args"] == 0


def test_openai_kind_requires_endpoint_details(tmp_path):
    with pytest.raises(SystemExit):
        main(["--name", "x", "--model-kind", "openai", "--results-dir", str(tmp_path)])


async def test_a_failing_question_is_recorded_and_the_run_goes_on(tmp_path):
    qs = read_jsonl(QUESTIONS_PATH)[:3]

    def model_for(q):
        if q.id == qs[1].id:
            raise RuntimeError("model server returned 500")
        return gold_replay_model(q)

    s = await run_benchmark(qs, BundleStore.fixture(), model_for, tmp_path / "e.jsonl", {})
    assert s["n"] == 3 and s["run_errors"] == 1 and s["format_failures"] == 1
    assert s["answer"] == pytest.approx(2 / 3)


async def test_duplicate_ids_are_rejected(tmp_path):
    q = read_jsonl(QUESTIONS_PATH)[0]
    with pytest.raises(ValueError, match="unique"):
        await run_benchmark(
            [q, q], BundleStore.fixture(), gold_replay_model, tmp_path / "d.jsonl", {}
        )


def test_resume_with_a_different_limit_is_refused(tmp_path):
    base = [
        "--name",
        "lim",
        "--model-kind",
        "stub-gold",
        "--questions",
        str(QUESTIONS_PATH),
        "--results-dir",
        str(tmp_path),
    ]
    main([*base, "--limit", "2"])
    with pytest.raises(SystemExit, match="other settings"):
        main([*base, "--limit", "3"])


def test_opencode_go_preset_headers_per_question():
    from fhir_agent_bench.runner import PROVIDERS, USER_AGENT, chat_model_factory

    model_for = chat_model_factory("https://x.test/v1", "m", "k", PROVIDERS["opencode-go"])
    qs = read_jsonl(QUESTIONS_PATH)[:2]
    a, b = model_for(qs[0]), model_for(qs[1])
    assert a.default_headers["User-Agent"] == b.default_headers["User-Agent"] == USER_AGENT
    assert a.default_headers["x-opencode-session"] != b.default_headers["x-opencode-session"]
    # the headers reach the underlying OpenAI HTTP client
    assert (
        a.root_client.default_headers["x-opencode-session"]
        == (a.default_headers["x-opencode-session"])
    )
    assert a.root_client.default_headers["User-Agent"] == USER_AGENT
    shared = chat_model_factory("https://x.test/v1", "m", "k")
    assert shared(qs[0]) is shared(qs[1])  # no session header: one stateless client


def test_provider_run_reads_key_from_env_and_never_stores_it(tmp_path, monkeypatch):
    import fhir_agent_bench.runner as runner

    secret = "sk-test-SECRET-0123456789"
    monkeypatch.setenv("OPENCODE_API_KEY", secret)
    seen = {}

    def fake_make_chat_model(base_url, model, api_key="EMPTY", temperature=0.0, headers=None):
        seen.update(base_url=base_url, model=model, api_key=api_key, headers=headers)
        return ScriptedModel(script=[final_message([])])

    monkeypatch.setattr(runner, "make_chat_model", fake_make_chat_model)
    main(
        [
            "--name",
            "og",
            "--provider",
            "opencode-go",
            "--model",
            "deepseek-v4.1-flash",
            "--questions",
            str(QUESTIONS_PATH),
            "--limit",
            "2",
            "--results-dir",
            str(tmp_path),
        ]
    )
    assert seen["api_key"] == secret and seen["model"] == "deepseek-v4.1-flash"
    assert seen["base_url"] == "https://opencode.ai/inference/go/openai/v1"
    assert "x-opencode-session" in seen["headers"]
    meta = json.loads((tmp_path / "og.json").read_text(encoding="utf-8"))["meta"]
    assert meta["api_key_env"] == "OPENCODE_API_KEY" and meta["backend"] == "OpenCode Go API"
    for f in tmp_path.iterdir():
        assert secret not in f.read_text(encoding="utf-8"), f.name


def test_missing_key_variable_is_an_error(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        main(
            [
                "--name",
                "x",
                "--provider",
                "opencode-go",
                "--model",
                "m",
                "--results-dir",
                str(tmp_path),
            ]
        )


def test_public_url_drops_credentials_and_query():
    from fhir_agent_bench.runner import _public_url

    assert _public_url("https://user:pw@host.test:8443/v1?key=abc") == "https://host.test:8443/v1"
    assert _public_url("") == ""


def test_provider_with_stub_kind_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        main(
            [
                "--name",
                "x",
                "--provider",
                "opencode-go",
                "--model-kind",
                "stub-gold",
                "--model",
                "m",
                "--results-dir",
                str(tmp_path),
            ]
        )
