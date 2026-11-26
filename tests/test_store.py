"""The fixture and both store backends. HapiStore runs against a mock FHIR server (httpx
MockTransport) that answers searches over the fixture's resources, with paging."""

import json
from urllib.parse import parse_qs, urlencode, urlparse

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
        if "combo-code" in q:  # Observation.code or any component code
            wanted = tuple(q["combo-code"].split("|"))

            def codes(r):
                nodes = [r["code"], *[c["code"] for c in r.get("component", [])]]
                return {(n["coding"][0]["system"], n["coding"][0]["code"]) for n in nodes}

            rows = [r for r in rows if wanted in codes(r)]
        if "status" in q:
            rows = [r for r in rows if r.get("status") == q["status"]]
        if "clinical-status" in q:
            system, code = q["clinical-status"].split("|")
            rows = [
                r
                for r in rows
                if (
                    r["clinicalStatus"]["coding"][0]["system"],
                    r["clinicalStatus"]["coding"][0]["code"],
                )
                == (system, code)
            ]
        assert "date" not in q and "onset-date" not in q  # dates are filtered client-side
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
                    # relative, as some servers return; the store resolves it against its base
                    "url": f"{rtype}?" + urlencode(nxt),
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
    assert q["combo-code"] == [f"http://loinc.org|{HBA1C}"] and q["patient"] == ["p001"]
    assert "date" not in q  # date limits are applied client-side by both backends
    assert len(requests) > 1  # 12 matches at page size 2 -> followed relative `next` links
    assert all(str(r.url).startswith("http://hapi.test/fhir/") for r in requests)
    hstore.conditions("p001", clinical_status="active")
    q = parse_qs(urlparse(str(requests[-1].url)).query)
    assert q["clinical-status"] == [
        "http://terminology.hl7.org/CodeSystem/condition-clinical|active"
    ]


def test_timestamps_use_the_recorded_date(store):
    # recorded 2023-06-19T23:30:00-05:00 (2023-06-20 in UTC): counts as 2023-06-19
    got = store.observations("p002", HBA1C, since="2023-06-19")
    assert [(o.date, o.value) for o in got] == [("2023-06-19", 5.9)]
    assert store.observations("p002", HBA1C, since="2023-06-20") == []


def test_panel_components_are_searchable(store):
    sbp = store.observations("p002", "8480-6", limit=10)
    assert [(o.date, o.value) for o in sbp] == [
        ("2024-03-11", 132.0),
        ("2023-06-19", 128.0),
        ("2022-06-13", 124.0),
    ]
    assert sbp[0].id == "p002-bp-0#c0"
    assert [o.value for o in store.observations("p002", "8462-4")] == [84.0]


def test_medication_reference_uses_its_display(store):
    meds = store.medications("p003", status="active")
    assert "Cholecalciferol 1000 UNT Oral Tablet" in [m.name for m in meds]


def test_multi_word_name_search_matches_on_hapi(store, hapi):
    hstore, _ = hapi
    assert [p.id for p in hstore.search_patients(name="Maria Gonzalez")] == ["p001"]


def test_next_link_to_another_host_is_refused():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "resourceType": "Bundle",
                "entry": [],
                "link": [{"relation": "next", "url": "http://evil.test/fhir/Patient"}],
            },
        )

    hstore = HapiStore("http://hapi.test/fhir", transport=httpx.MockTransport(handler))
    with pytest.raises(ValueError, match="another host"):
        hstore.search_patients(name="x")
