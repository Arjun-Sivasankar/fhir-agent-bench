"""A small, hand-written, fully synthetic FHIR R4 Bundle for tests and the fixture benchmark.

The three patients and every value below are invented. `python -m fhir_agent_bench.fixture`
regenerates `data/fixture_bundle.json`; a test checks the committed file matches this code.
"""

from __future__ import annotations

import json
from pathlib import Path

FIXTURE_PATH = Path(__file__).parent / "data" / "fixture_bundle.json"

LOINC = "http://loinc.org"
SNOMED = "http://snomed.info/sct"
RXNORM = "http://www.nlm.nih.gov/research/umls/rxnorm"

HBA1C = ("4548-4", "Hemoglobin A1c/Hemoglobin.total in Blood", "%")
LDL = ("13457-7", "Cholesterol in LDL [Mass/volume] in Serum or Plasma by calculation", "mg/dL")
TSH = ("3016-3", "Thyrotropin [Units/volume] in Serum or Plasma", "m[IU]/L")
SBP = ("8480-6", "Systolic blood pressure", "mm[Hg]")

PATIENTS = [
    ("p001", "Gonzalez", "Maria", "female", "1958-04-12"),
    ("p002", "Okafor", "James", "male", "1972-11-03"),
    ("p003", "Chen", "Mei", "female", "1990-07-21"),
]

OBSERVATIONS = {
    "p001": [
        (
            HBA1C,
            [
                ("2019-02-11", 8.4),
                ("2019-08-20", 8.1),
                ("2020-02-03", 7.9),
                ("2020-09-14", 7.6),
                ("2021-03-01", 7.4),
                ("2021-09-27", 7.5),
                ("2022-03-14", 7.2),
                ("2022-10-03", 7.0),
                ("2023-04-17", 6.9),
                ("2023-11-06", 7.1),
                ("2024-05-20", 6.8),
                ("2024-12-02", 6.7),
            ],
        ),
        (
            LDL,
            [
                ("2020-02-03", 142.0),
                ("2021-03-01", 118.0),
                ("2022-03-14", 96.0),
                ("2023-04-17", 101.0),
            ],
        ),
        (SBP, [("2023-04-17", 148.0), ("2024-05-20", 136.0), ("2024-12-02", 131.0)]),
    ],
    "p002": [
        (HBA1C, [("2021-06-07", 5.6), ("2022-06-13", 5.8), ("2023-06-19", 5.9)]),
        (SBP, [("2022-06-13", 124.0), ("2023-06-19", 128.0)]),
    ],
    "p003": [
        (
            TSH,
            [
                ("2020-01-20", 9.8),
                ("2020-05-04", 5.1),
                ("2021-01-18", 2.4),
                ("2022-01-24", 2.1),
                ("2023-02-06", 1.9),
            ],
        ),
    ],
}

CONDITIONS = {  # (snomed, name, clinical status, onset date)
    "p001": [
        ("44054006", "Diabetes mellitus type 2", "active", "2015-03-02"),
        ("38341003", "Hypertensive disorder", "active", "2018-06-10"),
        ("10509002", "Acute bronchitis", "resolved", "2021-01-15"),
    ],
    "p002": [
        ("195967001", "Asthma", "active", "2010-09-01"),
        ("444814009", "Viral sinusitis", "resolved", "2022-12-05"),
    ],
    "p003": [
        ("40930008", "Hypothyroidism", "active", "2020-02-01"),
        ("37796009", "Migraine", "resolved", "2019-05-10"),
    ],
}

MEDICATIONS = {  # (rxnorm, name, status, authored on)
    "p001": [
        ("860975", "Metformin hydrochloride 500 MG Oral Tablet", "active", "2015-03-02"),
        ("314076", "Lisinopril 10 MG Oral Tablet", "active", "2018-06-10"),
        ("617312", "Atorvastatin 20 MG Oral Tablet", "stopped", "2020-02-03"),
    ],
    "p002": [
        ("745679", "Albuterol 0.09 MG/ACTUAT Metered Dose Inhaler", "active", "2010-09-01"),
        (
            "896209",
            "Fluticasone propionate 0.11 MG/ACTUAT Metered Dose Inhaler",
            "active",
            "2016-04-11",
        ),
    ],
    "p003": [
        ("966247", "Levothyroxine Sodium 0.075 MG Oral Tablet", "active", "2020-02-01"),
        ("310965", "Ibuprofen 200 MG Oral Tablet", "completed", "2019-05-10"),
    ],
}

ALLERGIES = {  # (snomed, substance, criticality)
    "p001": [("91936005", "Allergy to penicillin", "high")],
    "p002": [("91935009", "Allergy to peanut", "high"), ("300916003", "Latex allergy", "low")],
    "p003": [],
}


def _cc(system: str, code: str, display: str) -> dict:
    return {"coding": [{"system": system, "code": code, "display": display}], "text": display}


def build_bundle() -> dict:
    """The fixture as a FHIR R4 collection Bundle."""
    entries: list[dict] = []

    def add(resource: dict) -> None:
        entries.append(
            {
                "fullUrl": f"urn:uuid:{resource['resourceType']}-{resource['id']}",
                "resource": resource,
            }
        )

    for pid, family, given, gender, birth in PATIENTS:
        add(
            {
                "resourceType": "Patient",
                "id": pid,
                "name": [{"family": family, "given": [given]}],
                "gender": gender,
                "birthDate": birth,
            }
        )
    for pid, series in OBSERVATIONS.items():
        for (code, display, unit), points in series:
            for i, (date, value) in enumerate(points):
                add(
                    {
                        "resourceType": "Observation",
                        "id": f"{pid}-obs-{code}-{i}",
                        "status": "final",
                        "code": _cc(LOINC, code, display),
                        "subject": {"reference": f"Patient/{pid}"},
                        "effectiveDateTime": date,
                        "valueQuantity": {
                            "value": value,
                            "unit": unit,
                            "system": "http://unitsofmeasure.org",
                            "code": unit,
                        },
                    }
                )
    for pid, conds in CONDITIONS.items():
        for i, (code, name, status, onset) in enumerate(conds):
            add(
                {
                    "resourceType": "Condition",
                    "id": f"{pid}-cond-{i}",
                    "clinicalStatus": _cc(
                        "http://terminology.hl7.org/CodeSystem/condition-clinical", status, status
                    ),
                    "code": _cc(SNOMED, code, name),
                    "subject": {"reference": f"Patient/{pid}"},
                    "onsetDateTime": onset,
                }
            )
    for pid, meds in MEDICATIONS.items():
        for i, (code, name, status, authored) in enumerate(meds):
            add(
                {
                    "resourceType": "MedicationRequest",
                    "id": f"{pid}-med-{i}",
                    "status": status,
                    "intent": "order",
                    "medicationCodeableConcept": _cc(RXNORM, code, name),
                    "subject": {"reference": f"Patient/{pid}"},
                    "authoredOn": authored,
                }
            )
    for pid, alls in ALLERGIES.items():
        for i, (code, substance, criticality) in enumerate(alls):
            add(
                {
                    "resourceType": "AllergyIntolerance",
                    "id": f"{pid}-allergy-{i}",
                    "code": _cc(SNOMED, code, substance),
                    "criticality": criticality,
                    "patient": {"reference": f"Patient/{pid}"},
                }
            )
    return {"resourceType": "Bundle", "type": "collection", "entry": entries}


def write_fixture(path: Path = FIXTURE_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_bundle(), indent=1) + "\n", encoding="utf-8", newline="\n")
    return path


if __name__ == "__main__":
    print("wrote", write_fixture())
