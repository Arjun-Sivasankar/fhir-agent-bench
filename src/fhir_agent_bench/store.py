"""FHIR access layer: one read-only interface, two backends.

- `BundleStore`: in memory, from a FHIR R4 Bundle (the fixture; tests and CI).
- `HapiStore`: FHIR REST search against a HAPI FHIR R4 server (Synthea data; benchmark runs).

Both parse resources with the same functions into small normalised records, so the MCP tools,
gold answers and metrics see identical shapes whichever backend is used.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Protocol

import httpx

LOINC_SYSTEM = "http://loinc.org"


@dataclass(frozen=True)
class Patient:
    id: str
    name: str
    gender: str
    birth_date: str


@dataclass(frozen=True)
class Observation:
    id: str
    patient_id: str
    code: str
    display: str
    value: float | None
    unit: str
    date: str  # ISO date (YYYY-MM-DD)


@dataclass(frozen=True)
class Medication:
    id: str
    patient_id: str
    code: str
    name: str
    status: str
    authored_on: str


@dataclass(frozen=True)
class Condition:
    id: str
    patient_id: str
    code: str
    name: str
    clinical_status: str
    onset: str


@dataclass(frozen=True)
class Allergy:
    id: str
    patient_id: str
    code: str
    substance: str
    criticality: str


def as_dicts(records: Iterable) -> list[dict]:
    return [asdict(r) for r in records]


# ---------------------------------------------------------------- parsing (shared)


def _ref_id(ref: dict | None) -> str:
    return (ref or {}).get("reference", "").rsplit("/", 1)[-1]


def _coding(cc: dict | None) -> tuple[str, str]:
    cc = cc or {}
    coding = (cc.get("coding") or [{}])[0]
    return coding.get("code", ""), coding.get("display") or cc.get("text", "")


def _date(value: str | None) -> str:
    return (value or "")[:10]


def parse_patient(r: dict) -> Patient:
    n = (r.get("name") or [{}])[0]
    name = " ".join([*n.get("given", []), n.get("family", "")]).strip()
    return Patient(r["id"], name, r.get("gender", ""), r.get("birthDate", ""))


def parse_observation(r: dict) -> Observation:
    code, display = _coding(r.get("code"))
    q = r.get("valueQuantity") or {}
    return Observation(
        r["id"],
        _ref_id(r.get("subject")),
        code,
        display,
        q.get("value"),
        q.get("unit", ""),
        _date(r.get("effectiveDateTime") or (r.get("effectivePeriod") or {}).get("start")),
    )


def parse_medication(r: dict) -> Medication:
    code, name = _coding(r.get("medicationCodeableConcept"))
    return Medication(
        r["id"],
        _ref_id(r.get("subject")),
        code,
        name,
        r.get("status", ""),
        _date(r.get("authoredOn")),
    )


def parse_condition(r: dict) -> Condition:
    code, name = _coding(r.get("code"))
    status, _ = _coding(r.get("clinicalStatus"))
    return Condition(
        r["id"], _ref_id(r.get("subject")), code, name, status, _date(r.get("onsetDateTime"))
    )


def parse_allergy(r: dict) -> Allergy:
    code, substance = _coding(r.get("code"))
    return Allergy(r["id"], _ref_id(r.get("patient")), code, substance, r.get("criticality", ""))


# ---------------------------------------------------------------- interface


class FhirStore(Protocol):
    def search_patients(
        self, name: str | None = None, birth_date: str | None = None
    ) -> list[Patient]: ...
    def get_patient(self, patient_id: str) -> Patient | None: ...
    def observations(
        self, patient_id: str, code: str, since: str | None = None, limit: int = 10
    ) -> list[Observation]: ...
    def medications(self, patient_id: str, status: str | None = None) -> list[Medication]: ...
    def conditions(
        self, patient_id: str, clinical_status: str | None = None, since: str | None = None
    ) -> list[Condition]: ...
    def allergies(self, patient_id: str) -> list[Allergy]: ...


# ---------------------------------------------------------------- in-memory backend


class BundleStore:
    """Answers the interface's queries from the resources of one Bundle.

    Semantics (shared with HapiStore): name matching is a case-insensitive substring of the full
    name; observations are newest first and cut at `limit`; `since` dates are inclusive.
    """

    def __init__(self, bundle: dict):
        by_type: dict[str, list[dict]] = {}
        for e in bundle.get("entry", []):
            r = e["resource"]
            by_type.setdefault(r["resourceType"], []).append(r)
        self._patients = [parse_patient(r) for r in by_type.get("Patient", [])]
        self._obs = [parse_observation(r) for r in by_type.get("Observation", [])]
        self._meds = [parse_medication(r) for r in by_type.get("MedicationRequest", [])]
        self._conds = [parse_condition(r) for r in by_type.get("Condition", [])]
        self._allergies = [parse_allergy(r) for r in by_type.get("AllergyIntolerance", [])]

    @classmethod
    def fixture(cls) -> BundleStore:
        import json

        from .fixture import FIXTURE_PATH

        return cls(json.loads(FIXTURE_PATH.read_text(encoding="utf-8")))

    def search_patients(self, name=None, birth_date=None):
        out = self._patients
        if name:
            out = [p for p in out if name.lower() in p.name.lower()]
        if birth_date:
            out = [p for p in out if p.birth_date == birth_date]
        return sorted(out, key=lambda p: p.id)

    def get_patient(self, patient_id):
        return next((p for p in self._patients if p.id == patient_id), None)

    def observations(self, patient_id, code, since=None, limit=10):
        out = [o for o in self._obs if o.patient_id == patient_id and o.code == code]
        if since:
            out = [o for o in out if o.date >= since]
        return sorted(out, key=lambda o: o.date, reverse=True)[:limit]

    def medications(self, patient_id, status=None):
        out = [m for m in self._meds if m.patient_id == patient_id]
        if status:
            out = [m for m in out if m.status == status]
        return sorted(out, key=lambda m: (m.authored_on, m.id))

    def conditions(self, patient_id, clinical_status=None, since=None):
        out = [c for c in self._conds if c.patient_id == patient_id]
        if clinical_status:
            out = [c for c in out if c.clinical_status == clinical_status]
        if since:
            out = [c for c in out if c.onset >= since]
        return sorted(out, key=lambda c: (c.onset, c.id))

    def allergies(self, patient_id):
        return sorted(
            (a for a in self._allergies if a.patient_id == patient_id), key=lambda a: a.id
        )


# ---------------------------------------------------------------- HAPI backend


class HapiStore:
    """FHIR REST search against a HAPI FHIR R4 server, following `next` links for paging.

    Server-side filters are used where FHIR search supports them; the final ordering and the
    name match are applied client-side so results match `BundleStore` exactly.
    """

    def __init__(self, base_url: str, timeout: float = 30.0, transport=None, page_size: int = 100):
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            transport=transport,
            headers={"Accept": "application/fhir+json"},
        )
        self.page_size = page_size

    def close(self) -> None:
        self._http.close()

    def _search(self, resource: str, params: dict) -> list[dict]:
        params = {k: v for k, v in params.items() if v is not None}
        params["_count"] = self.page_size
        r = self._http.get(f"/{resource}", params=params)
        out = []
        while True:
            r.raise_for_status()
            bundle = r.json()
            out += [
                e["resource"]
                for e in bundle.get("entry", [])
                if e.get("resource", {}).get("resourceType") == resource
            ]
            nxt = next(
                (link["url"] for link in bundle.get("link", []) if link.get("relation") == "next"),
                None,
            )
            if not nxt:
                return out
            r = self._http.get(nxt)

    def search_patients(self, name=None, birth_date=None):
        found = [
            parse_patient(r)
            for r in self._search("Patient", {"name": name, "birthdate": birth_date})
        ]
        if name:  # FHIR `name` matches word prefixes; apply the shared substring rule
            found = [p for p in found if name.lower() in p.name.lower()]
        return sorted(found, key=lambda p: p.id)

    def get_patient(self, patient_id):
        r = self._http.get(f"/Patient/{patient_id}")
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return parse_patient(r.json())

    def observations(self, patient_id, code, since=None, limit=10):
        rows = [
            parse_observation(r)
            for r in self._search(
                "Observation",
                {
                    "patient": patient_id,
                    "code": f"{LOINC_SYSTEM}|{code}",
                    "date": f"ge{since}" if since else None,
                    "_sort": "-date",
                },
            )
        ]
        return sorted(rows, key=lambda o: o.date, reverse=True)[:limit]

    def medications(self, patient_id, status=None):
        rows = [
            parse_medication(r)
            for r in self._search("MedicationRequest", {"patient": patient_id, "status": status})
        ]
        return sorted(rows, key=lambda m: (m.authored_on, m.id))

    def conditions(self, patient_id, clinical_status=None, since=None):
        rows = [
            parse_condition(r)
            for r in self._search(
                "Condition",
                {
                    "patient": patient_id,
                    "clinical-status": clinical_status,
                    "onset-date": f"ge{since}" if since else None,
                },
            )
        ]
        return sorted(rows, key=lambda c: (c.onset, c.id))

    def allergies(self, patient_id):
        rows = [
            parse_allergy(r) for r in self._search("AllergyIntolerance", {"patient": patient_id})
        ]
        return sorted(rows, key=lambda a: a.id)
