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
