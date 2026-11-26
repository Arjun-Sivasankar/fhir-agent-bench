"""Benchmark runner: questions -> agent -> traces -> scores.

`fab-run --name <run> --model-kind stub-gold` runs the gold-replay stub on the fixture (the CI
smoke test: it must score perfectly). For a real model:

    fab-run --name qwen3-4b --model-kind openai --base-url http://<host>:8000/v1 \\
        --model Qwen/Qwen3-4B --backend "vllm 0.x" --hardware "RTX 5060 Laptop 8 GB" \\
        [--fhir-url http://<host>:8080/fhir] [--questions questions/synthea.jsonl]

Each question runs in a fresh MCP session against the server in-process; its tool calls go to
an audit log next to the results. Rows are appended to results/<name>.jsonl as they finish and
skipped on a re-run (resume); resuming with different settings is refused. results/<name>.json
holds the aggregate scores and the run's settings, including model, backend and hardware.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from .agent import Agent, make_chat_model
from .metrics import aggregate, score_run
from .questions import read_jsonl
from .server import AuditLog, create_server
from .store import BundleStore, HapiStore
from .stub import gold_replay_model


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def open_rows(rows_path: Path, config: dict) -> dict[str, dict]:
    """Rows already done (by question id). Settings are stored beside the rows and a resume
    with other settings is refused; a partial last line (cut-off run) is dropped."""
    cfg_path = rows_path.with_suffix(".config.json")
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text(encoding="utf-8"))
        if old != config:
            diff = {
                k: (old.get(k), config.get(k))
                for k in old.keys() | config.keys()
                if old.get(k) != config.get(k)
            }
            raise SystemExit(
                f"{rows_path.name} was made with other settings {diff}; use a new --name"
            )
    else:
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8", newline="\n")
    done: dict[str, dict] = {}
    if not rows_path.exists():
        return done
    lines = rows_path.read_text(encoding="utf-8").splitlines()
    good = []
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                break
            raise
        good.append(line)
        done[row["id"]] = row
    rows_path.write_text("".join(g + "\n" for g in good), encoding="utf-8", newline="\n")
    return done


async def run_benchmark(
    questions, store, model_for, rows_path: Path, config: dict, max_steps: int = 8
) -> dict:
    """Run every question not yet done; return the aggregate over all done rows."""
    done = open_rows(rows_path, config)
    audit = AuditLog(rows_path.with_suffix(".audit.jsonl"))
    server = create_server(store, audit)
    with open(rows_path, "a", encoding="utf-8", newline="\n") as f:
        for q in questions:
            if q.id in done:
                continue
            run = await Agent(model_for(q), server, max_steps=max_steps).run(q.question)
            row = {**score_run(q, run), "run": run}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            done[q.id] = row
    return aggregate(done[q.id] for q in questions if q.id in done)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="fab-run", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--name", required=True, help="results/<name>.jsonl and .json")
    ap.add_argument("--questions", type=Path, default=Path("questions/fixture.jsonl"))
    ap.add_argument("--fhir-url", default="", help="HAPI FHIR R4 base URL (default: fixture)")
    ap.add_argument("--model-kind", choices=["stub-gold", "openai"], required=True)
    ap.add_argument("--base-url", default="", help="OpenAI-compatible endpoint (openai kind)")
    ap.add_argument("--model", default="", help="model name on that endpoint")
    ap.add_argument("--backend", default="", help="serving stack, e.g. 'vllm 0.x' or 'mlx-lm'")
    ap.add_argument("--hardware", default="", help="where the model runs")
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="first N questions only")
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    args = ap.parse_args(argv)

    if args.model_kind == "openai" and not (args.base_url and args.model and args.hardware):
        ap.error("--model-kind openai needs --base-url, --model and --hardware")
    questions = read_jsonl(args.questions)
    if args.limit:
        questions = questions[: args.limit]
    store = HapiStore(args.fhir_url) if args.fhir_url else BundleStore.fixture()
    if args.model_kind == "stub-gold":

        def model_for(q):
            return gold_replay_model(q)
    else:
        chat = make_chat_model(args.base_url, args.model)

        def model_for(q):
            return chat

    config = {
        "questions": str(args.questions),
        "questions_sha256": _sha256(args.questions),
        "data": args.fhir_url or "fixture",
        "model_kind": args.model_kind,
        "model": args.model or "gold-replay stub",
        "backend": args.backend,
        "hardware": args.hardware,
        "max_steps": args.max_steps,
    }
    rows_path = args.results_dir / f"{args.name}.jsonl"
    summary = asyncio.run(
        run_benchmark(questions, store, model_for, rows_path, config, args.max_steps)
    )
    out = {
        "meta": {**config, "finished_utc": datetime.now(UTC).isoformat(timespec="seconds")},
        "summary": summary,
    }
    (args.results_dir / f"{args.name}.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "n",
                    "tool_selection",
                    "args",
                    "answer",
                    "refusal_correct",
                    "format_failures",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
