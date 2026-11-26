"""The fixture and both store backends. HapiStore runs against a mock FHIR server (httpx
MockTransport) that answers searches over the fixture's resources, with paging."""

import json
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from fhir_agent_bench.fixture import FIXTURE_PATH, build_bundle
from fhir_agent_bench.store import BundleStore, HapiStore

HBA1C, LDL, TSH = "4548-4", "13457-7", "3016-3"


@pytest.fixture(scope="module")
def store():
    return BundleStore.fixture()


def test_committed_fixture_matches_code():
    assert json.loads(FIXTURE_PATH.read_text(encoding="utf-8")) == build_bundle()


def test_search_patients(store):
    assert [p.id for p in store.search_patients(name="gonz")] == ["p001"]
    assert store.search_patients(name="Maria Gonzalez")[0].birth_date == "1958-04-12"
    assert [p.id for p in store.search_patients(birth_date="1972-11-03")] == ["p002"]
    assert store.search_patients(name="nobody") == []
    assert len(store.search_patients()) == 3


def test_observations_newest_first_with_limit_and_since(store):
    last3 = store.observations("p001", HBA1C, limit=3)
    assert [(o.date, o.value) for o in last3] == [
        ("2024-12-02", 6.7),
        ("2024-05-20", 6.8),
        ("2023-11-06", 7.1),
    ]
    assert last3[0].unit == "%" and last3[0].display.startswith("Hemoglobin A1c")
    since = store.observations("p001", HBA1C, since="2024-01-01", limit=50)
    assert [o.date for o in since] == ["2024-12-02", "2024-05-20"]
    assert len(store.observations("p001", HBA1C, limit=50)) == 12
    assert store.observations("p003", HBA1C) == []


def test_medications_conditions_allergies(store):
    active = store.medications("p001", status="active")
    assert [m.name.split()[0] for m in active] == ["Metformin", "Lisinopril"]
    assert len(store.medications("p001")) == 3
    conds = store.conditions("p001", clinical_status="active")
    assert [c.name for c in conds] == ["Diabetes mellitus type 2", "Hypertensive disorder"]
    assert [c.name for c in store.conditions("p001", since="2020-01-01")] == ["Acute bronchitis"]
    assert [a.substance for a in store.allergies("p002")] == ["Allergy to peanut", "Latex allergy"]
    assert store.allergies("p003") == []
    assert store.get_patient("p404") is None


# ------------------------------------------------------------ HAPI backend vs a mock server


def _mock_fhir_server(bundle: dict, page_size: int = 2):
    """A tiny FHIR search server over a Bundle's resources: patient, code, status,
    clinical-status, date ge / onset-date ge, name (word prefix), birthdate, paging."""
    resources = [e["resource"] for e in bundle["entry"]]
    requests = []

    def subject(r):
        ref = (r.get("subject") or r.get("patient") or {}).get("reference", "")
        return ref.rsplit("/", 1)[-1]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        url = urlparse(str(request.url))
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        path = url.path.removeprefix("/fhir").strip("/").split("/")
        if len(path) == 2:  # read: /Patient/{id}
            hit = [r for r in resources if r["resourceType"] == path[0] and r["id"] == path[1]]
            return httpx.Response(200, json=hit[0]) if hit else httpx.Response(404)
        rtype = path[0]
        rows = [r for r in resources if r["resourceType"] == rtype]
        if "patient" in q:
            rows = [r for r in rows if subject(r) == q["patient"]]
        if "code" in q:
            system, code = q["code"].split("|")
            rows = [
                r
                for r in rows
                if r["code"]["coding"][0]
                == {**r["code"]["coding"][0], "system": system, "code": code}
            ]
        if "status" in q:
            rows = [r for r in rows if r.get("status") == q["status"]]
        if "clinical-status" in q:
            rows = [
                r for r in rows if r["clinicalStatus"]["coding"][0]["code"] == q["clinical-status"]
            ]
        if "date" in q:
            rows = [r for r in rows if r["effectiveDateTime"] >= q["date"][2:]]
        if "onset-date" in q:
            rows = [r for r in rows if r["onsetDateTime"] >= q["onset-date"][2:]]
        if "name" in q:
            words = lambda r: [w.lower() for n in r["name"] for w in [*n["given"], n["family"]]]  # noqa: E731
            rows = [r for r in rows if any(w.startswith(q["name"].lower()) for w in words(r))]
        if "birthdate" in q:
            rows = [r for r in rows if r["birthDate"] == q["birthdate"]]
        offset = int(q.get("_offset", 0))
        page = rows[offset : offset + page_size]
        links = []
        if offset + page_size < len(rows):
            nxt = dict(q, _offset=str(offset + page_size))
            links.append(
                {
                    "relation": "next",
                    "url": f"http://hapi.test/fhir/{rtype}?"
                    + "&".join(f"{k}={v}" for k, v in nxt.items()),
                }
            )
        return httpx.Response(
            200,
            json={
                "resourceType": "Bundle",
                "type": "searchset",
                "link": links,
                "entry": [{"resource": r} for r in page],
            },
        )

    return httpx.MockTransport(handler), requests


@pytest.fixture
def hapi():
    transport, requests = _mock_fhir_server(build_bundle(), page_size=2)
    return HapiStore("http://hapi.test/fhir", transport=transport, page_size=2), requests


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("search_patients", {"name": "gonzalez"}),
        ("search_patients", {"birth_date": "1990-07-21"}),
        ("get_patient", {"patient_id": "p002"}),
        ("get_patient", {"patient_id": "p404"}),
        ("observations", {"patient_id": "p001", "code": HBA1C, "limit": 5}),
        ("observations", {"patient_id": "p001", "code": HBA1C, "since": "2022-01-01", "limit": 50}),
        ("observations", {"patient_id": "p003", "code": TSH}),
        ("medications", {"patient_id": "p001", "status": "active"}),
        ("medications", {"patient_id": "p003"}),
        ("conditions", {"patient_id": "p001", "clinical_status": "resolved"}),
        ("conditions", {"patient_id": "p002", "since": "2015-01-01"}),
        ("allergies", {"patient_id": "p002"}),
    ],
)
def test_hapi_store_matches_bundle_store(store, hapi, method, kwargs):
    hstore, _ = hapi
    assert getattr(hstore, method)(**kwargs) == getattr(store, method)(**kwargs)


def test_hapi_requests_use_fhir_search_parameters(hapi):
    hstore, requests = hapi
    hstore.observations("p001", HBA1C, since="2022-01-01", limit=3)
    q = parse_qs(urlparse(str(requests[0].url)).query)
    assert q["code"] == [f"http://loinc.org|{HBA1C}"] and q["date"] == ["ge2022-01-01"]
    assert q["patient"] == ["p001"] and q["_sort"] == ["-date"]
    assert len(requests) > 1  # 12 matches at page size 2 -> followed `next` links
