"""The LangGraph agent with stand-in models, over the real MCP server on the fixture."""

import json
from pathlib import Path

import pytest

from fhir_agent_bench.agent import SYSTEM_PROMPT, Agent, mcp_tool_specs, parse_final
from fhir_agent_bench.metrics import score_run
from fhir_agent_bench.questions import read_jsonl
from fhir_agent_bench.server import AuditLog, create_server
from fhir_agent_bench.store import BundleStore
from fhir_agent_bench.stub import (
    ScriptedModel,
    final_message,
    gold_replay_model,
    tool_call_message,
)

QUESTIONS = read_jsonl(Path(__file__).parents[1] / "questions" / "fixture.jsonl")
BY_ID = {q.id: q for q in QUESTIONS}


@pytest.fixture
def server(tmp_path):
    return create_server(BundleStore.fixture(), AuditLog(tmp_path / "audit.jsonl"))


async def test_tool_loop_and_final_answer(server, tmp_path):
    q = BY_ID["q001"]  # last 3 HbA1c for Maria Gonzalez
    seen_tool_results = []

    def answer(messages):
        seen_tool_results.append(messages[-1].content)
        values = [o["value"] for o in json.loads(messages[-1].content)["results"]]
        return final_message([str(v) for v in values], text="Her last three HbA1c values.")

    model = ScriptedModel(
        script=[
            tool_call_message("search_patients", {"name": "Maria Gonzalez"}, "c1"),
            tool_call_message(
                "get_observations", {"patient_id": "p001", "loinc_code": "4548-4", "limit": 3}, "c2"
            ),
            answer,
        ]
    )
    run = await Agent(model, server).run(q.question)
    assert run["answer_items"] == ["6.7", "6.8", "7.1"] and run["format_ok"]
    assert [c["tool"] for c in run["calls"]] == ["search_patients", "get_observations"]
    assert run["steps"] == 3 and run["trace"][1]["count"] == 3
    assert {t["function"]["name"] for t in model.bound_tools} >= {"get_observations"}
    s = score_run(q, run)
    assert (s["tool_selection"], s["args"], s["answer"], s["refusal_correct"]) == (1, 1, 1, 1)
    audit = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(audit) == 2  # every tool call went through the MCP server


async def test_refusal_without_tool_calls(server):
    q = next(q for q in QUESTIONS if q.category == "refusal")
    run = await Agent(
        ScriptedModel(script=[final_message([], True, "I can't do that.")]), server
    ).run(q.question)
    assert run["refused"] and run["calls"] == [] and run["steps"] == 1
    assert score_run(q, run)["answer"] == 1.0


async def test_tool_errors_are_returned_to_the_model(server):
    errors = []

    def after_error(messages):
        errors.append(messages[-1].content)
        return final_message([])

    model = ScriptedModel(
        script=[
            tool_call_message(
                "get_observations", {"patient_id": "p001", "loinc_code": "HbA1c"}, "c1"
            ),
            after_error,
        ]
    )
    run = await Agent(model, server).run("x")
    assert "invalid loinc_code" in errors[0] and run["trace"][0]["is_error"]


async def test_step_limit_stops_a_looping_model(server):
    loop = ScriptedModel(script=[tool_call_message("get_allergies", {"patient_id": "p001"}, "c")])
    run = await Agent(loop, server, max_steps=3).run("x")
    assert run["steps"] == 3 and run["hit_step_limit"] and not run["format_ok"]
    assert len(run["calls"]) == 2  # the third model step's calls are not executed


@pytest.mark.parametrize("q", QUESTIONS, ids=lambda q: q.id)
async def test_gold_replay_scores_perfectly(server, q):
    run = await Agent(gold_replay_model(q), server).run(q.question)
    s = score_run(q, run)
    assert (s["tool_selection"], s["args"], s["answer"], s["refusal_correct"]) == (1, 1, 1, 1), (
        q.id,
        run["answer"],
        run["calls"],
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('Done.\n{"answer": ["6.7", "6.8"], "refused": false}', (["6.7", "6.8"], False, True)),
        ('```json\n{"answer": [], "refused": true}\n```', ([], True, True)),
        (
            '{"answer": [131, "2024-12-02"], "refused": false} trailing',
            (["131", "2024-12-02"], False, True),
        ),
        ('{"note": 1} then {"answer": ["x"]}', (["x"], False, True)),  # refused defaults false
        ("no json here", (None, False, False)),
        ('{"answer": "6.7", "refused": false}', (None, False, False)),  # not a list
        ('{"answer": [], "refused": "yes"}', (None, False, False)),
    ],
)
def test_parse_final(text, expected):
    assert parse_final(text) == expected


def test_prompt_and_specs():
    assert '{"answer": [<items>], "refused": <true|false>}' in SYSTEM_PROMPT

    class T:
        name, description, input_schema = "t", "d", {"type": "object", "properties": {}}

    assert mcp_tool_specs([T()])[0]["function"] == {
        "name": "t",
        "description": "d",
        "parameters": {"type": "object", "properties": {}},
    }
