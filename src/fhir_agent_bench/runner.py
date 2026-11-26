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
import os
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from . import __version__
from .agent import SYSTEM_PROMPT, Agent, make_chat_model
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
    """Run every question not yet done; return the aggregate over all done rows.

    A question whose run fails (model server error, malformed reply, recursion limit) is
    recorded as a failed run (no answer, format failure, the error kept) and the run goes on.
    The audit log is an append-only trail: a question re-run after a crash appears again.
    """
    ids = [q.id for q in questions]
    if len(set(ids)) != len(ids):
        raise ValueError("question ids must be unique")
    done = open_rows(rows_path, config)
    audit = AuditLog(rows_path.with_suffix(".audit.jsonl"))
    server = create_server(store, audit)
    with open(rows_path, "a", encoding="utf-8", newline="\n") as f:
        for q in questions:
            if q.id in done:
                continue
            try:
                run = await Agent(model_for(q), server, max_steps=max_steps).run(q.question)
            except Exception as e:  # noqa: BLE001  - one failing question must not end the run
                run = {
                    "answer": "",
                    "answer_items": None,
                    "refused": False,
                    "format_ok": False,
                    "calls": [],
                    "trace": [],
                    "steps": 0,
                    "latency_ms": 0.0,
                    "error": f"{type(e).__name__}: {e}"[:500],
                }
            row = {**score_run(q, run), "run": run}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            done[q.id] = row
    rows = [done[i] for i in ids if i in done]
    summary = aggregate(rows)
    summary["run_errors"] = sum(bool(r["run"].get("error")) for r in rows)
    return summary


USER_AGENT = f"fhir-agent-bench/{__version__}"

# Hosted APIs. OpenCode Go serves its models on an OpenAI Chat Completions route and asks
# clients to send their own user agent and a stable session id per conversation.
PROVIDERS = {
    "opencode-go": {
        "base_url": "https://opencode.ai/inference/go/openai/v1",
        "api_key_env": "OPENCODE_API_KEY",
        "session_header": "x-opencode-session",
        "backend": "OpenCode Go API",
        "hardware": "hosted API (OpenCode Go)",
    },
}


def _public_url(url: str) -> str:
    """The URL without credentials or query string, as stored in run settings."""
    if not url:
        return url
    p = urlsplit(url)
    host = p.hostname or ""
    netloc = host + (f":{p.port}" if p.port else "")
    return urlunsplit((p.scheme, netloc, p.path, "", ""))


def chat_model_factory(base_url: str, model: str, api_key: str, preset: dict | None = None):
    """model_for(question) for an OpenAI-compatible endpoint. Every request carries our user
    agent; with a session header in the preset, each question (one conversation) gets its own
    fresh session id, otherwise one stateless client is shared across questions."""
    preset = preset or {}
    session_header = preset.get("session_header")
    if not session_header:
        chat = make_chat_model(base_url, model, api_key, headers={"User-Agent": USER_AGENT})
        return lambda q: chat

    def model_for(q):
        headers = {"User-Agent": USER_AGENT, session_header: str(uuid4())}
        return make_chat_model(base_url, model, api_key, headers=headers)

    return model_for


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="fab-run", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--name", required=True, help="results/<name>.jsonl and .json")
    ap.add_argument("--questions", type=Path, default=Path("questions/fixture.jsonl"))
    ap.add_argument("--fhir-url", default="", help="HAPI FHIR R4 base URL (default: fixture)")
    ap.add_argument("--model-kind", choices=["stub-gold", "openai"], default="")
    ap.add_argument(
        "--provider",
        choices=sorted(PROVIDERS),
        default="",
        help="preset for a hosted API (base URL, headers, key variable)",
    )
    ap.add_argument("--base-url", default="", help="OpenAI-compatible endpoint (openai kind)")
    ap.add_argument("--model", default="", help="model name on that endpoint")
    ap.add_argument(
        "--api-key-env",
        default="",
        help="name of the environment variable holding the API key (the key is never stored)",
    )
    ap.add_argument("--backend", default="", help="serving stack, e.g. 'vllm 0.x' or 'mlx-lm'")
    ap.add_argument("--hardware", default="", help="where the model runs")
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="first N questions only")
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    args = ap.parse_args(argv)

    if not (args.model_kind or args.provider):
        ap.error("give --model-kind or --provider")
    if args.provider and args.model_kind == "stub-gold":
        ap.error("--provider runs a real model; do not combine it with --model-kind stub-gold")
    preset = PROVIDERS.get(args.provider, {})
    if preset:
        args.model_kind = "openai"
        args.base_url = args.base_url or preset["base_url"]
        args.api_key_env = args.api_key_env or preset["api_key_env"]
        args.backend = args.backend or preset["backend"]
        args.hardware = args.hardware or preset["hardware"]
    if args.model_kind == "openai" and not (args.base_url and args.model and args.hardware):
        ap.error("--model-kind openai needs --base-url, --model and --hardware")
    api_key = "EMPTY"
    if args.api_key_env and args.model_kind == "openai":
        api_key = os.environ.get(args.api_key_env, "")
        if not api_key:
            ap.error(f"environment variable {args.api_key_env} is not set")
    questions = read_jsonl(args.questions)
    if args.limit:
        questions = questions[: args.limit]
    store = HapiStore(args.fhir_url) if args.fhir_url else BundleStore.fixture()
    if args.model_kind == "stub-gold":

        def model_for(q):
            return gold_replay_model(q)
    else:
        model_for = chat_model_factory(args.base_url, args.model, api_key, preset)

    config = {
        "questions": str(args.questions),
        "questions_sha256": _sha256(args.questions),
        "limit": args.limit,
        "question_ids": [q.id for q in questions],
        "data": args.fhir_url or "fixture",
        "model_kind": args.model_kind,
        "model": args.model or "gold-replay stub",
        "provider": args.provider,
        "base_url": _public_url(args.base_url),
        "api_key_env": args.api_key_env,  # the variable's name only, never its value
        "temperature": 0.0,
        "backend": args.backend,
        "hardware": args.hardware,
        "max_steps": args.max_steps,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:16],
        "fhir_agent_bench": __version__,
    }
    rows_path = args.results_dir / f"{args.name}.jsonl"
    try:
        summary = asyncio.run(
            run_benchmark(questions, store, model_for, rows_path, config, args.max_steps)
        )
    finally:
        if isinstance(store, HapiStore):
            store.close()
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
                    "run_errors",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
