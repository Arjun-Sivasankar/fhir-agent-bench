"""The MCP server end to end through an in-process MCP client, on the fixture store."""

import json

import pytest
from mcp import Client

from fhir_agent_bench.server import AuditLog, create_server
from fhir_agent_bench.store import BundleStore

TOOLS = {
    "search_patients",
    "get_patient",
    "get_observations",
    "get_medications",
    "get_conditions",
    "get_allergies",
}


@pytest.fixture
def audit_path(tmp_path):
    return tmp_path / "audit.jsonl"


@pytest.fixture
def server(audit_path):
    return create_server(BundleStore.fixture(), AuditLog(audit_path))


def _audit(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _payload(res):
    """The tool's returned dict (structured content, or the JSON text block)."""
    if res.structured_content:
        sc = res.structured_content
        return sc.get("result", sc)
    return json.loads(res.content[0].text)


async def test_tools_are_listed_read_only(server):
    async with Client(server) as c:
        tools = (await c.list_tools()).tools
    assert {t.name for t in tools} == TOOLS
    for t in tools:
        assert t.annotations.read_only_hint is True and t.annotations.destructive_hint is False
        assert t.description


async def test_search_then_query(server, audit_path):
    async with Client(server) as c:
        found = _payload(await c.call_tool("search_patients", {"name": "gonzalez"}))
        assert found["count"] == 1 and found["results"][0]["id"] == "p001"
        obs = _payload(
            await c.call_tool(
                "get_observations", {"patient_id": "p001", "loinc_code": "4548-4", "limit": 3}
            )
        )
    assert [o["value"] for o in obs["results"]] == [6.7, 6.8, 7.1]
    rows = _audit(audit_path)
    assert [r["tool"] for r in rows] == ["search_patients", "get_observations"]
    assert rows[1]["args"] == {"patient_id": "p001", "loinc_code": "4548-4", "limit": 3}
    assert rows[1]["count"] == 3 and rows[1]["error"] is None and rows[1]["duration_ms"] >= 0


@pytest.mark.parametrize(
    ("tool", "args", "message"),
    [
        ("get_observations", {"patient_id": "p001", "loinc_code": "HbA1c"}, "invalid loinc_code"),
        (
            "get_observations",
            {"patient_id": "p001", "loinc_code": "4548-4", "limit": 0},
            "invalid limit",
        ),
        (
            "get_observations",
            {"patient_id": "p001", "loinc_code": "4548-4", "since": "2024-13-01"},
            "invalid since",
        ),
        ("get_patient", {"patient_id": "p001; DROP TABLE"}, "invalid patient_id"),
        ("get_medications", {"patient_id": "p001", "status": "all"}, "invalid status"),
        (
            "get_conditions",
            {"patient_id": "p001", "clinical_status": "open"},
            "invalid clinical_status",
        ),
        ("search_patients", {}, "give a name"),
        ("search_patients", {"name": "<script>"}, "invalid name"),
    ],
)
async def test_invalid_arguments_are_refused_and_audited(server, audit_path, tool, args, message):
    async with Client(server) as c:
        res = await c.call_tool(tool, args)
    assert res.is_error and message in res.content[0].text
    [row] = _audit(audit_path)
    assert row["tool"] == tool and row["count"] is None and message in row["error"]


async def test_filters_and_empty_results(server, audit_path):
    async with Client(server) as c:
        meds = _payload(
            await c.call_tool("get_medications", {"patient_id": "p001", "status": "active"})
        )
        conds = _payload(
            await c.call_tool("get_conditions", {"patient_id": "p001", "since": "2020-01-01"})
        )
        none = _payload(await c.call_tool("get_allergies", {"patient_id": "p003"}))
        missing = _payload(await c.call_tool("get_patient", {"patient_id": "p999"}))
    assert meds["count"] == 2 and conds["results"][0]["name"] == "Acute bronchitis"
    assert none == {"count": 0, "results": []} and missing["count"] == 0
    assert [r["count"] for r in _audit(audit_path)] == [2, 1, 0, 0]


async def test_backend_failure_is_reported_and_audited(audit_path):
    class Broken(BundleStore):
        def allergies(self, patient_id):
            raise ConnectionError("FHIR server down")

    server = create_server(Broken({"entry": []}), AuditLog(audit_path))
    async with Client(server) as c:
        res = await c.call_tool("get_allergies", {"patient_id": "p001"})
    assert res.is_error and "FHIR backend error" in res.content[0].text
    assert "ConnectionError" in _audit(audit_path)[0]["error"]


async def test_tool_schemas_show_the_real_parameters(server):
    async with Client(server) as c:
        tools = {t.name: t for t in (await c.list_tools()).tools}
    obs = tools["get_observations"].input_schema
    assert set(obs["properties"]) == {"patient_id", "loinc_code", "since", "limit"}
    assert set(obs["required"]) == {"patient_id", "loinc_code"}
    assert obs["properties"]["limit"]["type"] == "integer"
    assert set(tools["search_patients"].input_schema["properties"]) == {"name", "birth_date"}


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("get_observations", {"loinc_code": "4548-4"}),  # missing patient_id
        ("get_observations", {"patient_id": "p001", "loinc_code": "4548-4", "limit": True}),
        ("get_observations", {"patient_id": "p001", "loinc_code": "4548-4", "limit": "5"}),
        ("delete_patient", {"patient_id": "p001"}),  # no such tool
    ],
)
async def test_calls_rejected_by_the_sdk_are_audited_too(server, audit_path, tool, args):
    async with Client(server) as c:
        res = await c.call_tool(tool, args)
    assert res.is_error
    [row] = _audit(audit_path)
    assert row["tool"] == tool and row["count"] is None and row["error"]


async def test_audit_truncates_long_values(server, audit_path):
    async with Client(server) as c:
        await c.call_tool("get_patient", {"patient_id": "x" * 5000})
    [row] = _audit(audit_path)
    assert len(row["args"]["patient_id"]) <= 203 and len(row["error"]) <= 203


def test_patient_id_with_trailing_newline_is_rejected():
    from mcp.server.mcpserver.exceptions import ToolError

    from fhir_agent_bench.server import _date, _patient_id

    with pytest.raises(ToolError):
        _patient_id("p001\n")
    with pytest.raises(ToolError):
        _date("２０２４-01-01", "since")  # full-width digits


async def test_synthea_style_names_are_accepted(server):
    async with Client(server) as c:
        res = await c.call_tool("search_patients", {"name": "Abe604 Ziemann110"})
    assert not res.is_error and _payload(res)["count"] == 0
