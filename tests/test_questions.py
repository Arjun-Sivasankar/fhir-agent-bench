"""The fixture question set: gold answers reproduce from gold calls, and gold calls are valid
calls on the real MCP server."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest
from mcp import Client

from fhir_agent_bench.questions import (
    Question,
    _since_year,
    answer_from_results,
    clinical_name,
    fmt_value,
    generate,
    ingredient,
    read_jsonl,
    run_call,
    stats,
)
from fhir_agent_bench.server import create_server
from fhir_agent_bench.store import BundleStore

QUESTIONS_PATH = Path(__file__).parents[1] / "questions" / "fixture.jsonl"


@pytest.fixture(scope="module")
def store():
    return BundleStore.fixture()


@pytest.fixture(scope="module")
def questions():
    return read_jsonl(QUESTIONS_PATH)


def test_committed_file_matches_generator(store, questions):
    assert [asdict(q) for q in questions] == [asdict(q) for q in generate(store)]


def test_ids_unique_and_categories(questions):
    assert len({q.id for q in questions}) == len(questions)
    s = stats(questions)
    assert set(s["by_category"]) == {"lookup", "filtered", "multi_step", "refusal"}
    assert s["refusal_share"] >= 0.15


def test_every_gold_answer_reproduces_from_its_gold_calls(store, questions):
    for q in (q for q in questions if q.answer_kind != "refuse"):
        results = [run_call(store, c) for c in q.gold_calls]
        assert [p.id for p in results[0]] == [q.patient_id], q.id  # name finds exactly one
        assert answer_from_results(q, results) == (q.answer_kind, q.answer), q.id


def test_refusals_have_no_calls_and_no_answer(questions):
    for q in (q for q in questions if q.category == "refusal"):
        assert q.answer_kind == "refuse" and q.gold_calls == [] and q.answer == []


async def test_gold_calls_are_valid_calls_on_the_mcp_server(store, questions):
    server = create_server(store)
    async with Client(server) as c:
        for q in questions:
            for call in q.gold_calls:
                res = await c.call_tool(call["tool"], call["args"])
                assert not res.is_error, (q.id, call, res.content[0].text)
                assert json.loads(res.content[0].text)["count"] == len(run_call(store, call))


def test_hand_checked_examples(questions):
    by_text = {q.question: q for q in questions}
    q = by_text["What are Maria Gonzalez's last 3 HbA1c results?"]
    assert (q.answer_kind, q.answer) == ("ordered", ["6.7", "6.8", "7.1"])
    assert q.gold_calls[1] == {
        "tool": "get_observations",
        "args": {"patient_id": "p001", "loinc_code": "4548-4", "limit": 3},
    }
    q = by_text["Which medications is Mei Chen currently taking?"]
    assert (q.answer_kind, q.answer) == ("set", ["cholecalciferol", "levothyroxine"])
    q = by_text["Does Mei Chen have any recorded allergies?"]
    assert (q.answer_kind, q.answer) == ("none", [])
    q = by_text[
        "What was James Okafor's most recent systolic blood pressure, and when was it measured?"
    ]
    assert q.answer == ["132", "2024-03-11"]  # from the blood-pressure panel component


def test_more_literal_gold_answers(questions):
    gold = {q.id: (q.answer_kind, q.answer) for q in questions}
    assert gold["q002"] == ("ordered", ["6.7", "6.8"])  # HbA1c on or after 2024-01-01
    assert gold["q011"] == ("set", ["Diabetes mellitus type 2", "Hypertension"])
    assert gold["q012"] == ("set", ["Acute bronchitis"])  # onset on or after 2019-01-01
    assert gold["q014"] == (
        "set",
        ["Diabetes mellitus type 2", "Hypertension", "lisinopril", "metformin"],
    )
    assert gold["q016"] == ("ordered", ["5.9", "5.8"])  # includes the 23:30-05:00 timestamp
    assert gold["q024"] == ("set", ["Allergy to peanut", "Latex allergy"])
    assert gold["q031"] == ("set", ["Hypothyroidism", "Migraine"])


def test_since_questions_filter_something(store, questions):
    for q in (q for q in questions if q.note == "since_values"):
        call = q.gold_calls[1]
        unfiltered = run_call(
            store, {**call, "args": {**call["args"], "since": None, "limit": 10_000}}
        )
        assert 2 <= len(q.answer) < len(unfiltered), q.id


def test_ambiguous_names_are_skipped():
    from fhir_agent_bench.fixture import build_bundle

    bundle = build_bundle()
    twin = {
        "resourceType": "Patient",
        "id": "p009",
        "name": [{"family": "Gonzalez", "given": ["Maria"]}],
        "gender": "female",
        "birthDate": "1980-01-01",
    }
    bundle["entry"].append({"resource": twin})
    qs = generate(BundleStore(bundle))
    assert not any(q.patient_id in ("p001", "p009") for q in qs)  # both Marias skipped
    assert any(q.patient_id == "p002" for q in qs)


def test_helpers():
    assert fmt_value(142.0) == "142" and fmt_value(6.7) == "6.7" and fmt_value(None) == ""
    assert fmt_value(5.123456789) == "5.123456789"  # no rounding
    assert ingredient("Metformin hydrochloride 500 MG Oral Tablet") == "metformin"
    assert ingredient("24 HR Metformin hydrochloride 500 MG Extended Release Oral Tablet") == (
        "metformin"
    )
    assert ingredient("Insulin Isophane Human 70 UNT/ML Injectable Suspension [Humulin]") == (
        "insulin"
    )
    with pytest.raises(ValueError):
        ingredient("500 MG")
    assert clinical_name("Diabetes mellitus type 2 (disorder)") == "Diabetes mellitus type 2"
    assert clinical_name("Hypertension") == "Hypertension"
    assert _since_year(["2024-05-01", "2023-01-01", "2022-01-01"]) == "2023"
    assert _since_year(["2024-05-01", "2024-01-01"]) is None  # no cut drops a value
    q = Question("x", "lookup", "?", [], "", note="nonsense")
    with pytest.raises(ValueError):
        answer_from_results(q, [[], []])
