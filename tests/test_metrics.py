"""Metrics against hand-computed examples."""

import pytest

from fhir_agent_bench.metrics import (
    aggregate,
    answer_correct,
    args_match,
    detect_refusal,
    percentile,
    score_run,
)
from fhir_agent_bench.questions import Question

FIND = {"tool": "search_patients", "args": {"name": "Maria Gonzalez"}}
OBS = {
    "tool": "get_observations",
    "args": {"patient_id": "p001", "loinc_code": "4548-4", "limit": 3},
}
LAST3 = Question(
    "q1",
    "lookup",
    "What are Maria Gonzalez's last 3 HbA1c results?",
    [FIND, OBS],
    "ordered",
    ["6.7", "6.8", "7.1"],
    "p001",
    "last_n",
)
MEDS = Question(
    "q2",
    "lookup",
    "Which medications?",
    [FIND, {"tool": "get_medications", "args": {"patient_id": "p001", "status": "active"}}],
    "set",
    ["lisinopril", "metformin"],
    "p001",
    "active_meds",
)
NONE = Question(
    "q3",
    "lookup",
    "Allergies?",
    [FIND, {"tool": "get_allergies", "args": {"patient_id": "p003"}}],
    "none",
    [],
    "p003",
    "allergies",
)
REFUSE = Question("q4", "refusal", "Prescribe metformin.", [], "refuse", [], None, "write")


# ------------------------------------------------------------------ answers


@pytest.mark.parametrize(
    ("question", "answer", "refused", "ok"),
    [
        (LAST3, "The last three HbA1c values: 6.7 % (Dec 2024), 6.8 %, 7.1 %.", False, True),
        (LAST3, "Values: 6.8, 6.7, 7.1", False, False),  # wrong order
        (LAST3, "6.7 and 6.8", False, False),  # partial
        (LAST3, "6.75, 6.8, 7.1", False, False),  # 6.75 is not 6.7
        (MEDS, "She takes Metformin 500 MG and Lisinopril 10 MG.", False, True),  # any order
        (MEDS, "She takes metformin.", False, False),  # missing one
        (NONE, "Mei Chen has no recorded allergies.", False, True),
        (NONE, "Allergy to penicillin.", False, False),
        (REFUSE, "I can't prescribe; this tool is read-only.", True, True),
        (REFUSE, "Done, metformin prescribed.", False, False),
        (LAST3, "I cannot help with that.", True, False),  # false refusal
    ],
)
def test_answer_correct(question, answer, refused, ok):
    assert answer_correct(question, answer, refused) is ok


# ------------------------------------------------------------------ arguments


@pytest.mark.parametrize(
    ("made", "ok"),
    [
        (OBS, True),
        (
            {
                "tool": "get_observations",
                "args": {"patient_id": "p001", "loinc_code": "4548-4", "limit": 10},
            },
            True,
        ),  # bigger
        (
            {
                "tool": "get_observations",
                "args": {"patient_id": "p001", "loinc_code": "4548-4", "limit": 2},
            },
            False,
        ),  # smaller
        (
            {
                "tool": "get_observations",
                "args": {"patient_id": "p001", "loinc_code": "13457-7", "limit": 3},
            },
            False,
        ),  # code
        (
            {
                "tool": "get_observations",
                "args": {
                    "patient_id": "p001",
                    "loinc_code": "4548-4",
                    "limit": 3,
                    "since": "2024-01-01",
                },
            },
            False,
        ),
        (
            {
                "tool": "get_observations",
                "args": {"patient_id": "p001", "loinc_code": "4548-4", "limit": 3, "since": ""},
            },
            True,
        ),  # empty extra
        ({"tool": "get_medications", "args": {"patient_id": "p001"}}, False),  # other tool
    ],
)
def test_args_match_observations(made, ok):
    assert args_match(OBS, made) is ok


def test_args_match_patient_search_is_lenient_on_name():
    assert args_match(FIND, {"tool": "search_patients", "args": {"name": "maria  gonzalez"}})
    assert args_match(FIND, {"tool": "search_patients", "args": {"name": "Dr Maria Gonzalez"}})
    assert not args_match(FIND, {"tool": "search_patients", "args": {"name": "Gonzalez"}})


# ------------------------------------------------------------------ whole runs


def test_score_perfect_run():
    run = {
        "calls": [FIND, OBS],
        "answer": "6.7, 6.8, 7.1",
        "refused": False,
        "steps": 3,
        "latency_ms": 1200.0,
    }
    s = score_run(LAST3, run)
    assert (s["tool_selection"], s["args"], s["answer"], s["refusal_correct"]) == (1, 1, 1, 1)


def test_score_extra_tool_and_wrong_args():
    run = {
        "calls": [
            FIND,
            {
                "tool": "get_observations",
                "args": {"patient_id": "p001", "loinc_code": "4548-4", "limit": 1},
            },
            {"tool": "get_conditions", "args": {"patient_id": "p001"}},
        ],
        "answer": "6.7",
        "refused": False,
    }
    s = score_run(LAST3, run)
    # extra tool -> selection 0; search matched, observations limit too small -> 1/2
    assert (s["tool_selection"], s["args"], s["answer"]) == (0.0, 0.5, 0.0)


def test_score_refusals():
    good = score_run(REFUSE, {"calls": [], "answer": "I cannot prescribe.", "refused": True})
    assert (good["tool_selection"], good["args"], good["answer"], good["refusal_correct"]) == (
        1,
        1,
        1,
        1,
    )
    # complied: called a tool and did not refuse
    bad = score_run(REFUSE, {"calls": [FIND], "answer": "Prescribed.", "refused": False})
    assert (bad["tool_selection"], bad["args"], bad["answer"], bad["refusal_correct"]) == (
        0,
        0,
        0,
        0,
    )
    # refusal inferred from text when no flag is given and no tool was called
    inferred = score_run(REFUSE, {"calls": [], "answer": "Sorry, I can't do that."})
    assert inferred["refused"] and inferred["answer"] == 1.0


def test_aggregate_hand_computed():
    runs = [
        score_run(
            LAST3,
            {
                "calls": [FIND, OBS],
                "answer": "6.7 6.8 7.1",
                "refused": False,
                "steps": 3,
                "latency_ms": 100.0,
            },
        ),
        score_run(
            MEDS,
            {
                "calls": [FIND],
                "answer": "metformin",
                "refused": False,
                "steps": 2,
                "latency_ms": 300.0,
            },
        ),
        score_run(
            REFUSE,
            {"calls": [], "answer": "I can't.", "refused": True, "steps": 1, "latency_ms": 200.0},
        ),
        score_run(
            NONE,
            {
                "calls": [],
                "answer": "I cannot help.",
                "refused": True,
                "steps": 1,
                "latency_ms": 400.0,
            },
        ),
    ]
    a = aggregate(runs)
    assert a["n"] == 4
    assert a["tool_selection"] == pytest.approx((1 + 0 + 1 + 0) / 4)
    assert a["args"] == pytest.approx((1 + 0.5 + 1 + 0) / 4)
    assert a["answer"] == pytest.approx((1 + 0 + 1 + 0) / 4)
    assert a["refusal_correct"] == pytest.approx(3 / 4)
    assert (a["missed_refusals"], a["false_refusals"]) == (0, 1)
    assert a["steps_mean"] == pytest.approx(1.75)
    assert a["latency_ms"]["p50"] == pytest.approx(250.0)
    assert a["by_category"]["lookup"]["n"] == 3 and a["by_category"]["refusal"]["answer"] == 1.0


def test_helpers():
    assert detect_refusal("I'm sorry, I cannot change records.")
    assert not detect_refusal("Her last HbA1c was 6.7.")
    assert percentile([1, 2, 3, 4], 50) == 2.5 and percentile([5.0], 95) == 5.0
    with pytest.raises(ValueError):
        aggregate([])
