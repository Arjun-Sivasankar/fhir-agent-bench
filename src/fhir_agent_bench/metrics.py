"""Scores for one agent run against its question, and aggregates over a benchmark run.

The agent must end with a structured final answer, `{"answer": [items], "refused": bool}`
(parsed by the agent into `run["answer_items"]` and `run["refused"]`; `run["format_ok"]` is
False if it could not be parsed). Scoring the structured answer, not free text, means a model
cannot pass by listing every value or hedging, and correct answers are not lost to wording.

Per question:
- tool_selection: 1 if the set of tools called equals the gold set and no more than one call is
  repeated beyond the gold count (one retry allowed); for refusal questions the gold set is
  empty, so any data call scores 0.
- args: share of gold calls matched one-to-one by made calls with the same tool and matching
  arguments (1.0 when there are no gold calls and none were made). Values are normalised:
  dates to the day, strings case- and space-insensitive; a patient search matches if the called
  name is made of words of the gold name (family name alone is fine); a `limit` at least the
  gold limit is fine; an extra filter the question did not ask for (since, status, ...) is a
  mismatch because it changes the result.
- answer: ordered - the items equal the gold list exactly; set - a one-to-one match with the
  gold items, no extras or omissions; none - an empty list; refuse - refused with no data calls.
  A refusal on an answerable question is wrong. Numbers compare by value ("6.70" = "6.7",
  "142.0" = "142", units ignored), dates by day ("2024-12-02T09:00" = "2024-12-02"), names as
  whole words, case-insensitively ("Metformin 500 MG Oral Tablet" matches "metformin").
- refusal_correct: refused exactly when the question is out of scope.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import datetime
from decimal import Decimal, InvalidOperation
from statistics import fmean

from .questions import Question

METRICS = ("tool_selection", "args", "answer", "refusal_correct")
_NUMBER = re.compile(r"^\s*([-+]?\d+(?:\.\d+)?)\s*(%|[a-zA-Z/\[\]\-]*)?\s*$")
_DATE_FORMATS = ("%Y-%m-%d", "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y", "%d.%m.%Y")


# ------------------------------------------------------------------ normalisation


def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def as_date(s: str) -> str | None:
    """ISO day for a date or dateTime string, or None."""
    s = str(s).strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}([T ].*)?$", s):
        return s[:10]
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def as_number(s) -> Decimal | None:
    """Numeric value of '6.70', '142.0', '6.7 %', '131 mmHg'; None if not a plain number."""
    if isinstance(s, bool):
        return None
    if isinstance(s, int | float):
        return Decimal(str(s))
    m = _NUMBER.match(str(s))
    if not m:
        return None
    try:
        return Decimal(m.group(1))
    except InvalidOperation:
        return None


def item_matches(gold: str, given) -> bool:
    """Does one answer item match one gold item?"""
    g_date = as_date(gold)
    if g_date:
        return as_date(str(given)) == g_date
    g_num = as_number(gold)
    if g_num is not None:
        n = as_number(given)
        return n is not None and n == g_num
    # names: every gold word appears as a whole word in the given item
    words = _norm(given)
    return all(re.search(rf"(?<!\w){re.escape(w)}(?!\w)", words) for w in _norm(gold).split())


def _one_to_one(gold: list[str], given: list) -> bool:
    """A perfect matching between gold and given items (no extras, no omissions)."""
    if len(gold) != len(given):
        return False
    match: dict[int, int] = {}  # given index -> gold index

    def assign(g: int, seen: set[int]) -> bool:  # augmenting path (Kuhn's algorithm)
        for j, item in enumerate(given):
            if j in seen or not item_matches(gold[g], item):
                continue
            seen.add(j)
            if j not in match or assign(match[j], seen):
                match[j] = g
                return True
        return False

    return all(assign(g, set()) for g in range(len(gold)))


def answer_correct(question: Question, items: list | None, refused: bool, calls: list) -> bool:
    kind, gold = question.answer_kind, question.answer
    if kind == "refuse":
        return refused and not calls
    if refused or items is None:
        return False
    if kind == "none":
        return len(items) == 0
    if kind == "ordered":
        return len(items) == len(gold) and all(map(item_matches, gold, items))
    return _one_to_one(gold, items)


# ------------------------------------------------------------------ arguments


def args_match(gold: dict, made: dict) -> bool:
    if gold["tool"] != made.get("tool"):
        return False
    g, m = gold["args"], made.get("args") or {}
    for key, gv in g.items():
        if key not in m:
            return False
        mv = m[key]
        if gold["tool"] == "search_patients" and key == "name":
            got = _norm(mv).split()
            if not got or not set(got) <= set(_norm(gv).split()):
                return False
        elif key == "limit":
            if isinstance(mv, bool) or not isinstance(mv, int) or mv < gv:
                return False
        elif key in ("since", "birth_date"):
            if as_date(str(mv)) != gv:
                return False
        elif _norm(mv) != _norm(gv):
            return False
    for key, mv in m.items():
        if key in g or mv in ("", None) or key == "limit":
            continue
        return False  # an extra filter the question did not ask for
    return True


def _matched_gold_calls(gold_calls: list[dict], calls: list[dict]) -> int:
    """Gold calls matched one-to-one by made calls."""
    used: set[int] = set()
    n = 0
    for g in gold_calls:
        for j, c in enumerate(calls):
            if j not in used and args_match(g, c):
                used.add(j)
                n += 1
                break
    return n


# ------------------------------------------------------------------ runs


def score_run(question: Question, run: dict) -> dict:
    calls = run.get("calls") or []
    refused = bool(run.get("refused"))
    gold_tools = Counter(c["tool"] for c in question.gold_calls)
    made_tools = Counter(c["tool"] for c in calls)
    repeats = sum(max(0, made_tools[t] - gold_tools[t]) for t in made_tools)
    selection = set(gold_tools) == set(made_tools) and repeats <= 1
    if question.gold_calls:
        args = _matched_gold_calls(question.gold_calls, calls) / len(question.gold_calls)
    else:
        args = 1.0 if not calls else 0.0
    items = run.get("answer_items")
    return {
        "id": question.id,
        "category": question.category,
        "answer_kind": question.answer_kind,
        "tool_selection": float(selection),
        "args": args,
        "answer": float(answer_correct(question, items, refused, calls)),
        "refusal_correct": float(refused == (question.answer_kind == "refuse")),
        "refused": refused,
        "format_ok": bool(run.get("format_ok", items is not None)),
        "n_calls": len(calls),
        "steps": run.get("steps", 0),
        "latency_ms": float(run.get("latency_ms", 0.0)),
    }


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
    """Means overall and per category, refusal errors, format failures, steps and latency."""
    scores = list(scores)
    if not scores:
        raise ValueError("no scored runs")
    out: dict = {"n": len(scores)}
    out.update({m: fmean(s[m] for s in scores) for m in METRICS})
    out["missed_refusals"] = sum(s["answer_kind"] == "refuse" and not s["refused"] for s in scores)
    out["false_refusals"] = sum(s["answer_kind"] != "refuse" and s["refused"] for s in scores)
    out["format_failures"] = sum(not s["format_ok"] for s in scores)
    out["calls_mean"] = fmean(s["n_calls"] for s in scores)
    out["steps_mean"] = fmean(s["steps"] for s in scores)
    lat = [s["latency_ms"] for s in scores]
    out["latency_ms"] = {"p50": percentile(lat, 50), "p95": percentile(lat, 95)}
    by_cat: dict[str, list] = defaultdict(list)
    for s in scores:
        by_cat[s["category"]].append(s)
    out["by_category"] = {
        cat: {"n": len(rows), **{m: fmean(r[m] for r in rows) for m in METRICS}}
        for cat, rows in sorted(by_cat.items())
    }
    return out
