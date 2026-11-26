"""Scores for one agent run against its question, and aggregates over a benchmark run.

A run is what the agent did: the tool calls it made (`[{"tool", "args"}]`), its final answer
text, whether it refused, how many model steps it took and how long it ran.

Per question:
- tool_selection: 1 if the set of data tools called equals the gold set (patient search
  included), else 0. For a refusal question the gold set is empty.
- args: share of gold calls matched by a made call with the same tool and matching arguments
  (1.0 when there are no gold calls). Arguments match when every gold argument is present and
  equal after normalisation; a patient search matches if the called name contains every word
  of the gold name; an extra `limit` is allowed if it is at least the gold limit; any other
  extra argument (a filter the question did not ask for) is a mismatch.
- answer: 1 if the final answer is correct for its kind:
  ordered - every gold item appears, in order; set - every gold item appears;
  none - the answer says there is nothing (and lists no item); refuse - the agent refused.
  Items match case-insensitively; numbers and dates match on token boundaries (6.7 does not
  match 6.75).
- refusal_correct: 1 if the agent refused exactly when the question is out of scope.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from statistics import fmean

from .questions import Question

_NONE_PHRASES = re.compile(
    r"\b(no (recorded |known |documented |active )?(allerg\w*|medications?|conditions?|results?|"
    r"values?|records?|data)|none|nothing (is )?recorded|not (have|has|find|found)|"
    r"there are no|there is no)\b",
    re.I,
)
_REFUSAL_PHRASES = re.compile(
    r"\b(can(no|')t|cannot|unable to|not able to|won't|will not|not (permitted|allowed|"
    r"authori[sz]ed)|read-only|outside (my|the) scope|i('m| am) (sorry|afraid))\b",
    re.I,
)
_NUMERIC = re.compile(r"^[\d.\-]+$")


def detect_refusal(text: str) -> bool:
    """Heuristic refusal detector for answers that do not carry an explicit refusal flag."""
    return bool(_REFUSAL_PHRASES.search(text or ""))


def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def _item_positions(text: str, item: str) -> list[int]:
    """Start offsets where `item` occurs in `text` (case-insensitive; numbers and dates on
    token boundaries)."""
    t, it = _norm(text), _norm(item)
    if _NUMERIC.match(it):
        pattern = rf"(?<![\d.]){re.escape(it)}(?![\d]|\.\d)"
    else:
        pattern = re.escape(it)
    return [m.start() for m in re.finditer(pattern, t)]


def answer_correct(question: Question, answer: str, refused: bool) -> bool:
    kind, items = question.answer_kind, question.answer
    if kind == "refuse":
        return refused
    if refused:
        return False
    if kind == "none":
        return bool(_NONE_PHRASES.search(answer or ""))
    found = [_item_positions(answer, it) for it in items]
    if not all(found):
        return False
    if kind == "set":
        return True
    # ordered: choose increasing positions greedily
    last = -1
    for positions in found:
        nxt = [p for p in positions if p > last]
        if not nxt:
            return False
        last = nxt[0]
    return True


def args_match(gold: dict, made: dict) -> bool:
    if gold["tool"] != made["tool"]:
        return False
    g, m = gold["args"], made.get("args") or {}
    for key, gv in g.items():
        if key not in m:
            return False
        mv = m[key]
        if gold["tool"] == "search_patients" and key == "name":
            if not all(w in _norm(mv).split() for w in _norm(gv).split()):
                return False
        elif key == "limit":
            if not isinstance(mv, int) or mv < gv:
                return False
        elif _norm(mv) != _norm(gv):
            return False
    for key, mv in m.items():
        if key in g or mv in ("", None):
            continue
        if key == "limit":
            continue  # a larger page than needed does not change the newest-first prefix
        return False  # an extra filter the question did not ask for
    return True


def score_run(question: Question, run: dict) -> dict:
    calls = run.get("calls", [])
    refused = run.get("refused")
    if refused is None:
        refused = detect_refusal(run.get("answer", "")) and not calls
    gold_tools = {c["tool"] for c in question.gold_calls}
    made_tools = {c["tool"] for c in calls}
    if question.gold_calls:
        matched = sum(any(args_match(g, m) for m in calls) for g in question.gold_calls)
        args = matched / len(question.gold_calls)
    else:
        args = 1.0 if not calls else 0.0
    return {
        "id": question.id,
        "category": question.category,
        "tool_selection": float(gold_tools == made_tools),
        "args": args,
        "answer": float(answer_correct(question, run.get("answer", ""), bool(refused))),
        "refusal_correct": float(bool(refused) == (question.answer_kind == "refuse")),
        "refused": bool(refused),
        "n_calls": len(calls),
        "steps": run.get("steps", 0),
        "latency_ms": run.get("latency_ms", 0.0),
    }


METRICS = ("tool_selection", "args", "answer", "refusal_correct")


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile (same as numpy's default)."""
    xs = sorted(values)
    if not xs:
        return float("nan")
    pos = (len(xs) - 1) * q / 100
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return float(xs[lo] + (xs[hi] - xs[lo]) * (pos - lo))


def aggregate(scores: Iterable[dict]) -> dict:
    """Means overall and per category, refusal error counts, steps and latency percentiles."""
    scores = list(scores)
    if not scores:
        raise ValueError("no scored runs")
    out: dict = {"n": len(scores)}
    out.update({m: float(fmean([s[m] for s in scores])) for m in METRICS})
    should = [s for s in scores if s["category"] == "refusal"]
    out["missed_refusals"] = sum(not s["refused"] for s in should)
    out["false_refusals"] = sum(s["refused"] for s in scores if s["category"] != "refusal")
    out["steps_mean"] = float(fmean([s["steps"] for s in scores]))
    lat = [s["latency_ms"] for s in scores]
    out["latency_ms"] = {"p50": percentile(lat, 50), "p95": percentile(lat, 95)}
    by_cat: dict[str, list] = defaultdict(list)
    for s in scores:
        by_cat[s["category"]].append(s)
    out["by_category"] = {
        cat: {"n": len(rows), **{m: float(fmean([r[m] for r in rows])) for m in METRICS}}
        for cat, rows in sorted(by_cat.items())
    }
    return out
