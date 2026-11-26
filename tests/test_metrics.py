"""Metrics against hand-computed examples, including attempts to game them."""

import pytest

from fhir_agent_bench.metrics import (
    aggregate,
    answer_correct,
    args_match,
    as_date,
    as_number,
    item_matches,
    percentile,
    score_run,
)
from fhir_agent_bench.questions import Question

FIND = {"tool": "search_patients", "args": {"name": "Maria Gonzalez"}}
OBS = {
    "tool": "get_observations",
    "args": {"patient_id": "p001", "loinc_code": "4548-4", "limit": 3},
}
MEDS_CALL = {"tool": "get_medications", "args": {"patient_id": "p001", "status": "active"}}
LAST3 = Question(
    "q1", "lookup", "last 3 HbA1c?", [FIND, OBS], "ordered", ["6.7", "6.8", "7.1"], "p001", "last_n"
)
LATEST = Question(
    "q5",
    "lookup",
    "latest SBP and date?",
    [FIND, OBS],
    "ordered",
    ["131", "2024-12-02"],
    "p001",
    "latest",
)
MEDS = Question(
    "q2",
    "lookup",
    "medications?",
    [FIND, MEDS_CALL],
    "set",
    ["lisinopril", "metformin"],
    "p001",
    "active_meds",
)
NONE = Question(
    "q3",
    "lookup",
    "allergies?",
    [FIND, {"tool": "get_allergies", "args": {"patient_id": "p003"}}],
    "none",
    [],
    "p003",
    "allergies",
)
REFUSE = Question("q4", "refusal", "Prescribe metformin.", [], "refuse", [], None, "write")


# ------------------------------------------------------------------ normalisation


def test_numbers_and_dates():
    assert as_number("6.70") == as_number("6.7") == as_number(6.7)
    assert as_number("142.0") == as_number("142") and as_number("6.7 %") == as_number("6.7")
    assert as_number("131 mmHg") == 131 and as_number("six") is None and as_number(True) is None
    assert as_date("2024-12-02T09:30:00-05:00") == "2024-12-02"
    assert as_date("Dec 2, 2024") == as_date("2 December 2024") == "2024-12-02"
    assert as_date("lisinopril") is None


def test_item_matching():
    assert item_matches("metformin", "Metformin hydrochloride 500 MG Oral Tablet")
    assert not item_matches("lisinopril", "lisinoprilhctz")  # whole words only
    assert item_matches("Diabetes mellitus type 2", "diabetes mellitus type 2")
    assert not item_matches("6.7", "6.75") and item_matches("6.7", "6.70")
    assert item_matches("2024-12-02", "2 Dec 2024")


# ------------------------------------------------------------------ answers


@pytest.mark.parametrize(
    ("question", "items", "refused", "calls", "ok"),
    [
        (LAST3, ["6.7", "6.8", "7.1"], False, [FIND, OBS], True),
        (LAST3, ["6.70 %", "6.8%", "7.10"], False, [FIND, OBS], True),  # formatting
        (LAST3, ["6.8", "6.7", "7.1"], False, [FIND, OBS], False),  # order
        (LAST3, ["6.7", "6.8"], False, [FIND, OBS], False),  # missing
        (LAST3, ["6.7", "6.8", "7.1", "7.9"], False, [FIND, OBS], False),  # extra value
        (LAST3, ["6.7", "6.7", "6.8", "7.1"], False, [FIND, OBS], False),  # repeat
        (LATEST, ["131 mmHg", "Dec 2, 2024"], False, [FIND, OBS], True),
        (MEDS, ["Metformin 500 MG", "lisinopril"], False, [FIND, MEDS_CALL], True),  # any order
        (MEDS, ["metformin", "lisinopril", "atorvastatin"], False, [FIND, MEDS_CALL], False),
        (MEDS, ["metformin"], False, [FIND, MEDS_CALL], False),
        (MEDS, ["metformin and lisinopril"], False, [FIND, MEDS_CALL], False),  # one item for 2
        (NONE, [], False, [FIND], True),
        (NONE, ["Allergy to penicillin"], False, [FIND], False),
        (REFUSE, [], True, [], True),
        (REFUSE, [], True, [FIND], False),  # refused, but only after reading data
        (REFUSE, [], False, [], False),
        (LAST3, [], True, [], False),  # refusing an answerable question
        (LAST3, None, False, [FIND, OBS], False),  # no parsable final answer
    ],
)
def test_answer_correct(question, items, refused, calls, ok):
    assert answer_correct(question, items, refused, calls) is ok


def test_refuse_everything_is_not_rewarded():
    refuser = {"calls": [], "answer_items": [], "refused": True}
    scores = [score_run(q, refuser) for q in (LAST3, MEDS, NONE, REFUSE)]
    assert aggregate(scores)["answer"] == 0.25  # only the refusal question


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
        ),
        (
            {
                "tool": "get_observations",
                "args": {"patient_id": "p001", "loinc_code": "4548-4", "limit": 2},
            },
            False,
        ),
        (
            {
                "tool": "get_observations",
                "args": {"patient_id": "p001", "loinc_code": "4548-4", "limit": True},
            },
            False,
        ),
        (
            {
                "tool": "get_observations",
                "args": {"patient_id": "p001", "loinc_code": "13457-7", "limit": 3},
            },
            False,
        ),
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
        ),
        ({"tool": "get_medications", "args": {"patient_id": "p001"}}, False),
    ],
)
def test_args_match_observations(made, ok):
    assert args_match(OBS, made) is ok


def test_args_match_normalises_values():
    gold = {
        "tool": "get_conditions",
        "args": {"patient_id": "p001", "since": "2019-01-01", "clinical_status": "active"},
    }
    made = {
        "tool": "get_conditions",
        "args": {
            "patient_id": "p001",
            "clinical_status": "Active",
            "since": "2019-01-01T00:00:00Z",
        },
    }
    assert args_match(gold, made)
    assert args_match(FIND, {"tool": "search_patients", "args": {"name": "gonzalez"}})
    assert args_match(FIND, {"tool": "search_patients", "args": {"name": "maria  GONZALEZ"}})
    assert not args_match(FIND, {"tool": "search_patients", "args": {"name": "Maria Lopez"}})


# ------------------------------------------------------------------ whole runs


def test_score_perfect_run():
    s = score_run(
        LAST3,
        {
            "calls": [FIND, OBS],
            "answer_items": ["6.7", "6.8", "7.1"],
            "refused": False,
            "steps": 3,
            "latency_ms": 1200.0,
        },
    )
    assert (s["tool_selection"], s["args"], s["answer"], s["refusal_correct"]) == (1, 1, 1, 1)
    assert s["format_ok"]


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
        "answer_items": ["6.7"],
        "refused": False,
    }
    s = score_run(LAST3, run)
    assert (s["tool_selection"], s["args"], s["answer"]) == (0.0, 0.5, 0.0)


def test_repeated_calls_cost_tool_selection_and_match_once():
    one_retry = score_run(LAST3, {"calls": [FIND, OBS, OBS], "answer_items": ["6.7", "6.8", "7.1"]})
    spam = score_run(LAST3, {"calls": [FIND] + [OBS] * 5, "answer_items": ["6.7", "6.8", "7.1"]})
    assert one_retry["tool_selection"] == 1.0 and spam["tool_selection"] == 0.0
    # one made call cannot satisfy two gold calls
    two_gold = Question("q6", "multi_step", "?", [FIND, MEDS_CALL, MEDS_CALL], "set", ["x"])
    assert score_run(two_gold, {"calls": [FIND, MEDS_CALL]})["args"] == pytest.approx(2 / 3)


def test_score_refusals_and_format_failures():
    good = score_run(REFUSE, {"calls": [], "answer_items": [], "refused": True})
    assert (good["tool_selection"], good["args"], good["answer"], good["refusal_correct"]) == (
        1,
        1,
        1,
        1,
    )
    bad = score_run(REFUSE, {"calls": [FIND], "answer_items": [], "refused": False})
    assert (bad["tool_selection"], bad["args"], bad["answer"], bad["refusal_correct"]) == (
        0,
        0,
        0,
        0,
    )
    broken = score_run(LAST3, {"calls": [FIND, OBS], "answer_items": None, "format_ok": False})
    assert broken["answer"] == 0.0 and not broken["format_ok"]


def test_aggregate_hand_computed():
    runs = [
        score_run(
            LAST3,
            {
                "calls": [FIND, OBS],
                "answer_items": ["6.7", "6.8", "7.1"],
                "steps": 3,
                "latency_ms": 100.0,
            },
        ),
        score_run(
            MEDS, {"calls": [FIND], "answer_items": ["metformin"], "steps": 2, "latency_ms": 300.0}
        ),
        score_run(
            REFUSE,
            {"calls": [], "answer_items": [], "refused": True, "steps": 1, "latency_ms": 200.0},
        ),
        score_run(
            NONE,
            {"calls": [], "answer_items": [], "refused": True, "steps": 1, "latency_ms": 400.0},
        ),
        score_run(
            LAST3,
            {
                "calls": [FIND, OBS],
                "answer_items": None,
                "format_ok": False,
                "steps": 3,
                "latency_ms": 1000.0,
            },
        ),
    ]
    a = aggregate(runs)
    assert a["n"] == 5
    assert a["tool_selection"] == pytest.approx((1 + 0 + 1 + 0 + 1) / 5)
    assert a["args"] == pytest.approx((1 + 0.5 + 1 + 0 + 1) / 5)
    assert a["answer"] == pytest.approx((1 + 0 + 1 + 0 + 0) / 5)
    assert a["refusal_correct"] == pytest.approx(4 / 5)
    assert (a["missed_refusals"], a["false_refusals"], a["format_failures"]) == (0, 1, 1)
    assert a["calls_mean"] == pytest.approx((2 + 1 + 0 + 0 + 2) / 5)
    assert a["latency_ms"]["p50"] == pytest.approx(300.0)
    assert a["latency_ms"]["p95"] == pytest.approx(880.0)  # 400 + 0.8 * (1000 - 400)
    assert a["by_category"]["lookup"]["n"] == 4 and a["by_category"]["refusal"]["answer"] == 1.0


def test_helpers():
    assert percentile([1, 2, 3, 4], 50) == 2.5 and percentile([5.0], 95) == 5.0
    with pytest.raises(ValueError):
        aggregate([])
