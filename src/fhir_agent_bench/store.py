"""FHIR access layer: one read-only interface, two backends.

- `BundleStore`: in memory, from a FHIR R4 Bundle (the fixture; tests and CI).
- `HapiStore`: FHIR REST search against a HAPI FHIR R4 server (Synthea data; benchmark runs).

Both parse resources with the same functions into small normalised records and apply the same
rules, so the MCP tools, gold answers and metrics see identical results whichever backend runs:

- dates are the date part of the recorded value (the patient-local day, before any timezone
  offset), compared as ISO strings; `since` is inclusive and applied client-side by both
  backends, so server timezone handling cannot move a boundary
- an Observation with `component[]` (e.g. a blood-pressure panel) yields one record per
  component quantity as well, so "systolic blood pressure" finds values inside panels
- ties are broken by id, so ordering is deterministic
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Protocol
from urllib.parse import urljoin, urlparse

import httpx

LOINC_SYSTEM = "http://loinc.org"
CLINICAL_STATUS_SYSTEM = "http://terminology.hl7.org/CodeSystem/condition-clinical"


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
    value: float | None  # numeric value; None for coded or text results
    unit: str
    date: str  # ISO date (YYYY-MM-DD)
    text: str = ""  # coded/text value (valueCodeableConcept / valueString), if any


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
    """Date part of a FHIR dateTime as recorded: '2024-12-02T23:30:00-05:00' -> '2024-12-02'."""
    return (value or "")[:10]


def parse_patient(r: dict) -> Patient:
    n = (r.get("name") or [{}])[0]
    name = " ".join([*n.get("given", []), n.get("family", "")]).strip()
    return Patient(r["id"], name, r.get("gender", ""), r.get("birthDate", ""))


def _value(node: dict) -> tuple[float | None, str, str]:
    if "valueQuantity" in node:
        q = node["valueQuantity"]
        return q.get("value"), q.get("unit", ""), ""
    if "valueCodeableConcept" in node:
        return None, "", _coding(node["valueCodeableConcept"])[1]
    return None, "", str(node.get("valueString", ""))


def parse_observations(r: dict) -> list[Observation]:
    """The observation itself (if it has a value) plus one record per component with a value."""
    patient = _ref_id(r.get("subject"))
    date = _date(r.get("effectiveDateTime") or (r.get("effectivePeriod") or {}).get("start"))
    out = []
    code, display = _coding(r.get("code"))
    value, unit, text = _value(r)
    if value is not None or text:
        out.append(Observation(r["id"], patient, code, display, value, unit, date, text))
    for i, comp in enumerate(r.get("component") or []):
        c_code, c_display = _coding(comp.get("code"))
        value, unit, text = _value(comp)
        if value is not None or text:
            out.append(
                Observation(f"{r['id']}#c{i}", patient, c_code, c_display, value, unit, date, text)
            )
    return out


def parse_medication(r: dict) -> Medication:
    if "medicationCodeableConcept" in r:
        code, name = _coding(r["medicationCodeableConcept"])
    else:  # medicationReference: use its display (resolving would need another request)
        ref = r.get("medicationReference") or {}
        code, name = "", ref.get("display", "") or _ref_id(ref)
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


# ---------------------------------------------------------------- shared query rules


def _match_name(patients: Iterable[Patient], name: str | None) -> list[Patient]:
    return [p for p in patients if not name or name.lower() in p.name.lower()]


def _obs_result(obs: Iterable[Observation], code: str, since: str | None, limit: int):
    rows = [o for o in obs if o.code == code and (not since or o.date >= since)]
    return sorted(rows, key=lambda o: (o.date, o.id), reverse=True)[:limit]


def _cond_result(conds: Iterable[Condition], since: str | None):
    rows = [c for c in conds if not since or c.onset >= since]
    return sorted(rows, key=lambda c: (c.onset, c.id))


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
    """Answers the interface's queries from the resources of one Bundle."""

    def __init__(self, bundle: dict):
        by_type: dict[str, list[dict]] = {}
        for e in bundle.get("entry", []):
            r = e["resource"]
            by_type.setdefault(r["resourceType"], []).append(r)
        self._patients = [parse_patient(r) for r in by_type.get("Patient", [])]
        self._obs = [o for r in by_type.get("Observation", []) for o in parse_observations(r)]
        self._meds = [parse_medication(r) for r in by_type.get("MedicationRequest", [])]
        self._conds = [parse_condition(r) for r in by_type.get("Condition", [])]
        self._allergies = [parse_allergy(r) for r in by_type.get("AllergyIntolerance", [])]

    @classmethod
    def fixture(cls) -> BundleStore:
        import json

        from .fixture import FIXTURE_PATH

        return cls(json.loads(FIXTURE_PATH.read_text(encoding="utf-8")))

    def search_patients(self, name=None, birth_date=None):
        out = _match_name(self._patients, name)
        if birth_date:
            out = [p for p in out if p.birth_date == birth_date]
        return sorted(out, key=lambda p: p.id)

    def get_patient(self, patient_id):
        return next((p for p in self._patients if p.id == patient_id), None)

    def observations(self, patient_id, code, since=None, limit=10):
        return _obs_result((o for o in self._obs if o.patient_id == patient_id), code, since, limit)

    def medications(self, patient_id, status=None):
        out = [m for m in self._meds if m.patient_id == patient_id]
        if status:
            out = [m for m in out if m.status == status]
        return sorted(out, key=lambda m: (m.authored_on, m.id))

    def conditions(self, patient_id, clinical_status=None, since=None):
        out = [c for c in self._conds if c.patient_id == patient_id]
        if clinical_status:
            out = [c for c in out if c.clinical_status == clinical_status]
        return _cond_result(out, since)

    def allergies(self, patient_id):
        return sorted(
            (a for a in self._allergies if a.patient_id == patient_id), key=lambda a: a.id
        )


# ---------------------------------------------------------------- HAPI backend


class HapiStore:
    """FHIR REST search against a HAPI FHIR R4 server, following `next` links for paging.

    Server-side search narrows by patient, code and status; date limits, name matching and
    ordering are applied client-side with the same rules as `BundleStore`.
    """

    def __init__(self, base_url: str, timeout: float = 30.0, transport=None, page_size: int = 100):
        self.base_url = base_url.rstrip("/") + "/"
        self._http = httpx.Client(
            timeout=timeout, transport=transport, headers={"Accept": "application/fhir+json"}
        )
        self.page_size = page_size

    def close(self) -> None:
        self._http.close()

    def _url(self, path_or_url: str) -> str:
        """Resolve relative or absolute links against the base; refuse other hosts."""
        url = urljoin(self.base_url, path_or_url)
        if urlparse(url).netloc != urlparse(self.base_url).netloc:
            raise ValueError(f"refusing to follow a link to another host: {url}")
        return url

    def _search(self, resource: str, params: dict) -> list[dict]:
        params = {k: v for k, v in params.items() if v is not None}
        params["_count"] = self.page_size
        r = self._http.get(self._url(resource), params=params)
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
            r = self._http.get(self._url(nxt))

    def search_patients(self, name=None, birth_date=None):
        # FHIR `name` matches word prefixes; ask for the last word, then apply the shared rule
        server_name = name.split()[-1] if name and name.split() else None
        found = [
            parse_patient(r)
            for r in self._search("Patient", {"name": server_name, "birthdate": birth_date})
        ]
        return sorted(_match_name(found, name), key=lambda p: p.id)

    def get_patient(self, patient_id):
        r = self._http.get(self._url(f"Patient/{patient_id}"))
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return parse_patient(r.json())

    def observations(self, patient_id, code, since=None, limit=10):
        # combo-code matches Observation.code or a component code (blood-pressure panels)
        resources = self._search(
            "Observation", {"patient": patient_id, "combo-code": f"{LOINC_SYSTEM}|{code}"}
        )
        return _obs_result(
            (o for r in resources for o in parse_observations(r)), code, since, limit
        )

    def medications(self, patient_id, status=None):
        rows = [
            parse_medication(r)
            for r in self._search("MedicationRequest", {"patient": patient_id, "status": status})
        ]
        return sorted(rows, key=lambda m: (m.authored_on, m.id))

    def conditions(self, patient_id, clinical_status=None, since=None):
        token = f"{CLINICAL_STATUS_SYSTEM}|{clinical_status}" if clinical_status else None
        rows = [
            parse_condition(r)
            for r in self._search("Condition", {"patient": patient_id, "clinical-status": token})
        ]
        if clinical_status:
            rows = [c for c in rows if c.clinical_status == clinical_status]
        return _cond_result(rows, since)

    def allergies(self, patient_id):
        rows = [
            parse_allergy(r) for r in self._search("AllergyIntolerance", {"patient": patient_id})
        ]
        return sorted(rows, key=lambda a: a.id)
