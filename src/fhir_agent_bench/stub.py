"""Stand-in chat models for tests and the CI smoke run (no LLM).

- `ScriptedModel`: replies with scripted messages in turn (or callables of the conversation).
- `gold_replay_model(question)`: makes the question's gold tool calls one model step at a time,
  then gives the gold answer in the required final format, so a full benchmark run on the
  fixture must score perfectly; any drop means a bug in the agent, server, store or metrics.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

Step = AIMessage | Callable[[list[BaseMessage]], AIMessage]


class ScriptedModel(BaseChatModel):
    script: list[Any] = Field(default_factory=list)
    position: int = 0
    bound_tools: list[dict] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Sequence[dict], **kwargs) -> ScriptedModel:
        self.bound_tools = list(tools)
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        step = self.script[min(self.position, len(self.script) - 1)]
        self.position += 1
        message = step(messages) if callable(step) else step
        # a fresh message each turn (as a real model gives): LangGraph de-duplicates by id
        message = message.model_copy(update={"id": None}, deep=True)
        return ChatResult(generations=[ChatGeneration(message=message)])


def tool_call_message(tool: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": tool, "args": args, "id": call_id}])


def final_message(items: list[str], refused: bool = False, text: str = "") -> AIMessage:
    return AIMessage(content=f"{text}\n{json.dumps({'answer': items, 'refused': refused})}".strip())


def gold_replay_model(question) -> ScriptedModel:
    steps: list[Any] = [
        tool_call_message(c["tool"], c["args"], f"call-{i}")
        for i, c in enumerate(question.gold_calls)
    ]
    refused = question.answer_kind == "refuse"
    steps.append(
        final_message(list(question.answer), refused, "I cannot do that." if refused else "")
    )
    return ScriptedModel(script=steps)
