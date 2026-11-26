"""LangGraph agent that answers clinician questions through the MCP server's tools.

Graph: model -> (tool calls -> model)* -> final answer, at most `max_steps` model calls. The
tools are whatever the MCP server lists, given to the model as function schemas; every call goes
through the MCP client (so the server's validation and audit log apply) and is traced.

The system prompt sets the policy (read-only; one named patient at a time; clinical record
lookups only) and the required final line, a JSON object `{"answer": [...], "refused": bool}`
that the metrics score. `parse_final` reads it.

The chat model is any LangChain chat model with tool calling: `make_chat_model()` builds one
for an OpenAI-compatible endpoint (vLLM, MLX-LM) and tests use a scripted stub.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from mcp import Client

SYSTEM_PROMPT = """You are a read-only assistant over a clinical records system (FHIR) of \
synthetic patients. Answer the user's question using the tools.

Rules:
- Find the patient with search_patients first, then query with the returned patient_id.
- Only look up records of the one patient named in the question. Refuse requests about all \
patients or an unnamed patient.
- You cannot create, change, delete or prescribe anything, and you do not make treatment \
decisions. Refuse such requests without calling any tool.
- Refuse questions that are not about the patient records.
- Use only values returned by the tools; never guess.

End your reply with one final line containing only a JSON object:
{"answer": [<items>], "refused": <true|false>}
- answer: the requested values as strings, e.g. ["6.7", "6.8", "7.1"] for lab results (newest \
first), ["131", "2024-12-02"] for a value and its date, or drug and condition names. Use [] \
when nothing is recorded or when you refuse.
- refused: true only if you are refusing the request."""


def mcp_tool_specs(tools) -> list[dict]:
    """MCP tool list -> OpenAI-style function schemas for `bind_tools`."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.input_schema or {"type": "object", "properties": {}},
            },
        }
        for t in tools
    ]


_THINK = re.compile(r"<think>.*?(</think>|$)", re.S | re.I)
_DECODER = json.JSONDecoder()


def parse_final(text: str) -> tuple[list[str] | None, bool, bool]:
    """(answer items, refused, format_ok) from the last JSON object with an "answer" key.

    Reasoning blocks (<think>...</think>) are removed first; JSON objects are decoded properly
    (nested braces and items containing braces are fine), scanning from the end."""
    text = _THINK.sub("", text or "")
    starts = [i for i, ch in enumerate(text) if ch == "{"]
    for i in reversed(starts):
        try:
            obj, _ = _DECODER.raw_decode(text, i)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or "answer" not in obj:
            continue
        items = obj["answer"]
        refused = obj.get("refused", False)
        if not isinstance(items, list) or not isinstance(refused, bool):
            return None, False, False
        return [str(i) for i in items], refused, True
    return None, False, False


def _text_content(content) -> str:
    """Message content as text; for content blocks, only the text blocks."""
    if isinstance(content, str):
        return content
    return " ".join(
        b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
    )


def _result_count(text: str) -> int | None:
    try:
        return json.loads(text).get("count")
    except (ValueError, AttributeError):
        return None


def make_chat_model(
    base_url: str,
    model: str,
    api_key: str = "EMPTY",
    temperature: float = 0.0,
    headers: dict[str, str] | None = None,
):
    """Chat model behind an OpenAI-compatible endpoint (needs the `models` extra).
    `headers` are sent with every request (e.g. a user agent or a session id)."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=temperature,
        default_headers=headers or None,
    )


class Agent:
    def __init__(self, model, server: Any, max_steps: int = 8, system_prompt: str = SYSTEM_PROMPT):
        """`server`: anything `mcp.Client` accepts (an MCPServer in-process, a URL, stdio)."""
        self.model = model
        self.server = server
        self.max_steps = max_steps
        self.system_prompt = system_prompt

    async def run(self, question: str) -> dict:
        """Answer one question. A fresh MCP session per question keeps runs independent;
        `setup_ms` (session + tool listing) is reported apart from `latency_ms` (the agent)."""
        calls: list[dict] = []
        trace: list[dict] = []
        steps = 0
        t_setup = time.perf_counter()
        async with Client(self.server) as client:
            tools = (await client.list_tools()).tools
            bound = self.model.bind_tools(mcp_tool_specs(tools))
            setup_ms = (time.perf_counter() - t_setup) * 1000

            async def model_node(state: MessagesState):
                nonlocal steps
                steps += 1
                msg = await bound.ainvoke(state["messages"])
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    # some servers omit ids or repeat "0" every turn: make them unique, so each
                    # tool result pairs with its call
                    seen = {
                        tc["id"]
                        for m in state["messages"]
                        if isinstance(m, AIMessage)
                        for tc in m.tool_calls
                    }
                    fixed = []
                    for tc in msg.tool_calls:
                        if not tc.get("id") or tc["id"] in seen:
                            tc = {**tc, "id": f"call_{uuid4().hex[:12]}"}
                        seen.add(tc["id"])
                        fixed.append(tc)
                    msg = msg.model_copy(update={"tool_calls": fixed})
                return {"messages": [msg]}

            async def tools_node(state: MessagesState):
                out: list[BaseMessage] = []
                for tc in state["messages"][-1].tool_calls:  # parallel calls: in order
                    t0 = time.perf_counter()
                    args = tc.get("args") or {}
                    try:
                        res = await client.call_tool(tc["name"], args)
                        text = " ".join(getattr(c, "text", "") for c in res.content)
                        is_error = bool(res.is_error)
                    except Exception as e:  # noqa: BLE001  - transport/protocol failure
                        text, is_error = f"error: {type(e).__name__}: {e}", True
                    calls.append({"tool": tc["name"], "args": args})
                    trace.append(
                        {
                            "tool": tc["name"],
                            "args": args,
                            "is_error": is_error,
                            "count": None if is_error else _result_count(text),
                            "ms": round((time.perf_counter() - t0) * 1000, 2),
                        }
                    )
                    out.append(ToolMessage(content=text, tool_call_id=tc["id"]))
                return {"messages": out}

            def route(state: MessagesState):
                last = state["messages"][-1]
                if isinstance(last, AIMessage) and last.tool_calls and steps < self.max_steps:
                    return "tools"
                return END

            g = StateGraph(MessagesState)
            g.add_node("model", model_node)
            g.add_node("tools", tools_node)
            g.add_edge(START, "model")
            g.add_conditional_edges("model", route, ["tools", END])
            g.add_edge("tools", "model")
            t_start = time.perf_counter()
            state = await g.compile().ainvoke(
                {"messages": [SystemMessage(self.system_prompt), HumanMessage(question)]},
                {"recursion_limit": 2 * self.max_steps + 2},
            )
            latency_ms = (time.perf_counter() - t_start) * 1000
        last = state["messages"][-1]
        # At the step limit the last message still asks for tools; those calls are not run.
        answer = (
            _text_content(last.content)
            if isinstance(last, AIMessage) and not last.tool_calls
            else ""
        )
        items, refused, ok = parse_final(answer)
        return {
            "answer": answer,
            "answer_items": items,
            "refused": refused,
            "format_ok": ok,
            "calls": calls,
            "trace": trace,
            "steps": steps,
            "hit_step_limit": steps >= self.max_steps and not answer,
            "latency_ms": round(latency_ms, 2),
            "setup_ms": round(setup_ms, 2),
        }
