"""Read-only MCP server over a FHIR store.

Six tools, all read-only (annotated `readOnlyHint`); there are no write tools at all. Every
argument is validated inside the tool, so rejected calls are refused with a clear message and
still recorded. Every call - successful, empty or rejected - is appended to a JSONL audit log
(time, tool, arguments, result count, error, duration).

Run over stdio: `fhir-mcp` (fixture bundle) or `fhir-mcp --fhir-url http://host:8080/fhir`.
"""

from __future__ import annotations

import argparse
import functools
import json
import re
import threading
import time
from datetime import UTC, date, datetime
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import StrictInt

from .store import BundleStore, FhirStore, HapiStore, as_dicts

ID = re.compile(r"[A-Za-z0-9\-.]{1,64}")  # FHIR id rule (used with fullmatch)
LOINC = re.compile(r"[0-9]{1,7}-[0-9]")
ISO_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")  # ASCII digits only
# letters and digits (Synthea appends digits to names: "Abe604 Ziemann110"), spaces, ' . -
NAME = re.compile(r"[^\W_](?:[^\W_]|[ '.\-])*")
MED_STATUS = {
    "active",
    "on-hold",
    "cancelled",
    "completed",
    "entered-in-error",
    "stopped",
    "draft",
    "unknown",
}
CLINICAL_STATUS = {"active", "recurrence", "relapse", "inactive", "remission", "resolved"}
MAX_LIMIT = 100
READ_ONLY = ToolAnnotations(
    read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False
)


MAX_LOGGED_CHARS = 200


def _clip(value):
    """Values as logged: long strings truncated, anything non-JSON stringified."""
    if isinstance(value, str):
        return value if len(value) <= MAX_LOGGED_CHARS else value[:MAX_LOGGED_CHARS] + "..."
    if isinstance(value, bool | int | float) or value is None:
        return value
    return _clip(repr(value))


class AuditLog:
    """Append-only JSONL audit trail of tool calls (thread-safe).

    Fail-closed: if the audit line cannot be written, the call fails instead of returning data
    that was never recorded.
    """

    def __init__(self, path: Path | None):
        self.path = path
        self._lock = threading.Lock()
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, tool: str, args: dict, count: int | None, error: str | None, ms: float):
        if not self.path:
            return
        line = json.dumps(
            {
                "time": datetime.now(UTC).isoformat(timespec="milliseconds"),
                "tool": _clip(tool),
                "args": {_clip(k): _clip(v) for k, v in (args or {}).items()},
                "count": count,
                "error": _clip(error) if error else None,
                "duration_ms": round(ms, 2),
            }
        )
        with self._lock, open(self.path, "a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")


def _count(result) -> int | None:
    """Result count from a tool result's structured content ({"count": ...} or wrapped)."""
    sc = getattr(result, "structured_content", None)
    if not sc:  # plain dict results arrive as one JSON text block
        try:
            sc = json.loads(result.content[0].text)
        except (AttributeError, IndexError, TypeError, ValueError):
            return None
    sc = sc.get("result", sc) if isinstance(sc, dict) else {}
    return sc.get("count") if isinstance(sc, dict) else None


class AuditedServer(MCPServer):
    """MCPServer that audits at the call_tool entry point, so every call is recorded - including
    ones the SDK rejects before the tool runs (unknown tool, missing or mistyped arguments)."""

    def __init__(self, *args, audit: AuditLog, **kwargs):
        super().__init__(*args, **kwargs)
        self.audit = audit

    async def call_tool(self, name, arguments, context=None):
        t0 = time.perf_counter()
        try:
            result = await super().call_tool(name, arguments, context)
        except Exception as e:  # noqa: BLE001  - record every failure, then let the SDK report it
            self.audit.record(
                name, arguments, None, f"{type(e).__name__}: {e}", (time.perf_counter() - t0) * 1e3
            )
            raise
        error = None
        if getattr(result, "is_error", False):
            error = " ".join(getattr(c, "text", "") for c in result.content) or "tool error"
        self.audit.record(
            name,
            arguments,
            None if error else _count(result),
            error,
            (time.perf_counter() - t0) * 1e3,
        )
        return result


# ---------------------------------------------------------------- validation


def _patient_id(value: str) -> str:
    if not ID.fullmatch(value or ""):
        raise ToolError(f"invalid patient_id {_clip(value)!r}: use the id from search_patients")
    return value


def _date(value: str, field: str) -> str | None:
    if not value:
        return None
    if not ISO_DATE.fullmatch(value):
        raise ToolError(f"invalid {field} {_clip(value)!r}: use an ISO date YYYY-MM-DD")
    try:
        date.fromisoformat(value)
    except ValueError:
        raise ToolError(f"invalid {field} {value!r}: not a real date") from None
    return value


def _choice(value: str, allowed: set[str], field: str) -> str | None:
    if not value:
        return None
    if value not in allowed:
        raise ToolError(f"invalid {field} {value!r}: one of {sorted(allowed)}")
    return value


# ---------------------------------------------------------------- server


def create_server(store: FhirStore, audit: AuditLog | None = None) -> MCPServer:
    server = AuditedServer(
        "fhir-agent-bench",
        audit=audit or AuditLog(None),
        instructions=(
            "Read-only access to synthetic FHIR patient records. Find the patient first with "
            "search_patients, then query by patient_id. Dates are ISO YYYY-MM-DD; lab codes are "
            "LOINC (e.g. 4548-4 for HbA1c). There are no tools to create, change or delete data."
        ),
    )

    def audited(fn):
        """Turn backend failures (FHIR server down, bad response) into clear tool errors.
        Auditing itself happens in AuditedServer.call_tool."""

        @functools.wraps(fn)
        def wrapper(**kwargs):
            try:
                return fn(**kwargs)
            except ToolError:
                raise
            except Exception as e:  # noqa: BLE001
                raise ToolError(f"FHIR backend error: {type(e).__name__}: {e}") from e

        return wrapper

    def result(records: list) -> dict:
        rows = as_dicts(records)
        return {"count": len(rows), "results": rows}

    @server.tool(annotations=READ_ONLY)
    @audited
    def search_patients(name: str = "", birth_date: str = "") -> dict:
        """Find patients by (part of) their name and/or exact birth date (YYYY-MM-DD).
        Returns id, name, gender and birth date for each match."""
        name = name.strip()
        if not name and not birth_date:
            raise ToolError("give a name, a birth_date or both")
        if name and (len(name) > 100 or not NAME.fullmatch(name)):
            raise ToolError(
                f"invalid name {name!r}: letters, digits, spaces, apostrophes and hyphens only"
            )
        return result(store.search_patients(name or None, _date(birth_date, "birth_date")))

    @server.tool(annotations=READ_ONLY)
    @audited
    def get_patient(patient_id: str) -> dict:
        """Demographics of one patient by id."""
        p = store.get_patient(_patient_id(patient_id))
        return result([p] if p else [])

    @server.tool(annotations=READ_ONLY)
    @audited
    def get_observations(
        patient_id: str, loinc_code: str, since: str = "", limit: StrictInt = 10
    ) -> dict:
        """Lab results and vital signs for one LOINC code, newest first. `since` (YYYY-MM-DD) is
        inclusive; `limit` is 1-100. Examples: 4548-4 HbA1c, 13457-7 LDL cholesterol,
        3016-3 TSH, 8480-6 systolic and 8462-4 diastolic blood pressure."""
        if not LOINC.fullmatch(loinc_code or ""):
            raise ToolError(f"invalid loinc_code {loinc_code!r}: a LOINC code such as 4548-4")
        if not 1 <= limit <= MAX_LIMIT:
            raise ToolError(f"invalid limit {limit}: 1 to {MAX_LIMIT}")
        return result(
            store.observations(_patient_id(patient_id), loinc_code, _date(since, "since"), limit)
        )

    @server.tool(annotations=READ_ONLY)
    @audited
    def get_medications(patient_id: str, status: str = "") -> dict:
        """Medication requests for one patient, optionally filtered by status
        (active, stopped, completed, ...)."""
        return result(
            store.medications(_patient_id(patient_id), _choice(status, MED_STATUS, "status"))
        )

    @server.tool(annotations=READ_ONLY)
    @audited
    def get_conditions(patient_id: str, clinical_status: str = "", since: str = "") -> dict:
        """Conditions (diagnoses) for one patient, optionally by clinical status (active,
        resolved, ...) and onset on or after `since` (YYYY-MM-DD)."""
        return result(
            store.conditions(
                _patient_id(patient_id),
                _choice(clinical_status, CLINICAL_STATUS, "clinical_status"),
                _date(since, "since"),
            )
        )

    @server.tool(annotations=READ_ONLY)
    @audited
    def get_allergies(patient_id: str) -> dict:
        """Recorded allergies and intolerances for one patient."""
        return result(store.allergies(_patient_id(patient_id)))

    return server


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="fhir-mcp", description=__doc__)
    ap.add_argument("--fhir-url", default="", help="HAPI FHIR R4 base URL (default: fixture)")
    ap.add_argument("--audit-log", default="logs/audit.jsonl", type=Path)
    args = ap.parse_args(argv)
    store = HapiStore(args.fhir_url) if args.fhir_url else BundleStore.fixture()
    try:
        create_server(store, AuditLog(args.audit_log)).run("stdio")
    finally:
        if isinstance(store, HapiStore):
            store.close()


if __name__ == "__main__":
    main()
