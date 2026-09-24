"""Test doubles for the model layer.

Agent tests must be deterministic and offline: a real provider would need an API
key or a pulled Ollama model, and would answer differently on every run. The
scripted model below stands in for whatever `get_llm()` would have returned, and
records the prompts it was sent so a test can assert on what reached the model
rather than only on what came back.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field


@dataclass(slots=True)
class RecordedCall:
    """One invocation of the model, as the prompt template rendered it."""

    messages: list[BaseMessage]

    @property
    def text(self) -> str:
        """Every message joined, for `assert "document_qa" in call.text`."""
        return "\n".join(str(m.content) for m in self.messages)

    def __contains__(self, needle: str) -> bool:
        return needle in self.text


@dataclass(slots=True)
class ScriptedToolCall:
    """One tool call for the model to emit on its next turn.

    Several may be queued as a list to script parallel calls in one turn, which
    is what the evidence collector's locking has to cope with.
    """

    name: str
    args: dict[str, Any] = field(default_factory=dict)

    def as_tool_call(self) -> dict[str, Any]:
        return {"name": self.name, "args": dict(self.args),
                "id": f"call_{uuid.uuid4().hex[:8]}", "type": "tool_call"}


def tool_call(name: str, **args: Any) -> ScriptedToolCall:
    """`tool_call("search_documents", query="risks")`, for readable scripts."""
    return ScriptedToolCall(name=name, args=args)


class ScriptedChatModel(BaseChatModel):
    """A chat model that returns queued responses and never touches a network.

    Queue one entry per expected model call. An entry may be:

    - `str`               -> returned as the message content
    - a `ScriptedToolCall`-> returned as a tool-calling turn (a list of them
                             scripts several calls in one turn)
    - a pydantic model    -> returned as-is by `with_structured_output`, or
                             dumped to JSON by a plain call
    - `dict`              -> validated into the structured-output schema
    - an `Exception`      -> raised, which is how failure paths are driven

    `bind_tools` records what it was offered and returns `self`: agents built
    with `create_agent` bind per request rather than at construction, so there
    is nothing to wrap. `with_structured_output` is still overridden rather than
    inherited -- the router and the follow-up rewrite want a schema back, not a
    tool call, and popping a queue says that in one line.
    """

    responses: list[Any] = Field(default_factory=list)
    calls: list[RecordedCall] = Field(default_factory=list)
    bound_tools: list[Any] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    @property
    def profile(self) -> dict[str, Any] | None:
        """`create_agent` reads this unguarded; `BaseChatModel` does not define it."""
        return None

    # -- internals ---------------------------------------------------------

    def _next(self) -> Any:
        if not self.responses:
            raise AssertionError(
                f"ScriptedChatModel ran out of responses after {len(self.calls)} call(s); "
                "queue one entry per expected model call"
            )
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def _record(self, prompt: Any) -> None:
        if isinstance(prompt, BaseMessage):
            messages = [prompt]
        elif hasattr(prompt, "to_messages"):        # ChatPromptValue
            messages = list(prompt.to_messages())
        elif isinstance(prompt, list):
            messages = list(prompt)
        else:
            messages = [AIMessage(content=str(prompt))]
        self.calls.append(RecordedCall(messages=messages))

    # -- BaseChatModel -----------------------------------------------------

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        """Record the offered tools so a test can assert on an agent's skills."""
        self.bound_tools.clear()
        self.bound_tools.extend(tools)
        return self

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None,
                  run_manager: Any = None, **kwargs: Any) -> ChatResult:
        self._record(messages)
        response = self._next()

        scripted = response if isinstance(response, list) else [response]
        if scripted and all(isinstance(item, ScriptedToolCall) for item in scripted):
            message = AIMessage(content="",
                                tool_calls=[item.as_tool_call() for item in scripted])
            return ChatResult(generations=[ChatGeneration(message=message)])

        content = response if isinstance(response, str) else (
            response.model_dump_json() if isinstance(response, BaseModel) else json.dumps(response)
        )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    def with_structured_output(self, schema: Any, *, include_raw: bool = False,
                               **kwargs: Any) -> Any:
        def _invoke(prompt: Any) -> Any:
            self._record(prompt)
            response = self._next()
            if isinstance(response, BaseModel):
                return response
            if isinstance(response, str):
                response = json.loads(response)
            return schema.model_validate(response)

        return RunnableLambda(_invoke)
