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

from .store import BundleStore, FhirStore, HapiStore, as_dicts

ID = re.compile(r"^[A-Za-z0-9\-.]{1,64}$")  # FHIR id rule
LOINC = re.compile(r"^\d{1,7}-\d$")
NAME = re.compile(r"^[^\W\d_](?:[^\W\d_]|[ '.\-])*$")  # letters, spaces, ' . -
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


class AuditLog:
    """Append-only JSONL audit trail of tool calls (thread-safe)."""

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
                "tool": tool,
                "args": args,
                "count": count,
                "error": error,
                "duration_ms": round(ms, 2),
            }
        )
        with self._lock, open(self.path, "a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")


# ---------------------------------------------------------------- validation


def _patient_id(value: str) -> str:
    if not ID.match(value or ""):
        raise ToolError(f"invalid patient_id {value!r}: use the id returned by search_patients")
    return value


def _date(value: str, field: str) -> str | None:
    if not value:
        return None
    try:
        date.fromisoformat(value)
    except ValueError:
        raise ToolError(f"invalid {field} {value!r}: use an ISO date YYYY-MM-DD") from None
    if len(value) != 10:
        raise ToolError(f"invalid {field} {value!r}: use an ISO date YYYY-MM-DD")
    return value


def _choice(value: str, allowed: set[str], field: str) -> str | None:
    if not value:
        return None
    if value not in allowed:
        raise ToolError(f"invalid {field} {value!r}: one of {sorted(allowed)}")
    return value


# ---------------------------------------------------------------- server


def create_server(store: FhirStore, audit: AuditLog | None = None) -> MCPServer:
    audit = audit or AuditLog(None)
    server = MCPServer(
        "fhir-agent-bench",
        instructions=(
            "Read-only access to synthetic FHIR patient records. Find the patient first with "
            "search_patients, then query by patient_id. Dates are ISO YYYY-MM-DD; lab codes are "
            "LOINC (e.g. 4548-4 for HbA1c). There are no tools to create, change or delete data."
        ),
    )

    def audited(fn):
        """Record every call with its arguments, result count or error, and duration."""

        @functools.wraps(fn)
        def wrapper(**kwargs):
            t0 = time.perf_counter()
            try:
                out = fn(**kwargs)
            except ToolError as e:
                audit.record(fn.__name__, kwargs, None, str(e), (time.perf_counter() - t0) * 1e3)
                raise
            except Exception as e:  # noqa: BLE001  - backend failure: audit, then report
                msg = f"FHIR backend error: {type(e).__name__}: {e}"
                audit.record(fn.__name__, kwargs, None, msg, (time.perf_counter() - t0) * 1e3)
                raise ToolError(msg) from e
            audit.record(fn.__name__, kwargs, out["count"], None, (time.perf_counter() - t0) * 1e3)
            return out

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
        if name and (len(name) > 100 or not NAME.match(name)):
            raise ToolError(f"invalid name {name!r}: letters, spaces, apostrophes and hyphens only")
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
        patient_id: str, loinc_code: str, since: str = "", limit: int = 10
    ) -> dict:
        """Lab results and vital signs for one LOINC code, newest first. `since` (YYYY-MM-DD) is
        inclusive; `limit` is 1-100. Examples: 4548-4 HbA1c, 13457-7 LDL cholesterol,
        3016-3 TSH, 8480-6 systolic and 8462-4 diastolic blood pressure."""
        if not LOINC.match(loinc_code or ""):
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
    create_server(store, AuditLog(args.audit_log)).run("stdio")


if __name__ == "__main__":
    main()
