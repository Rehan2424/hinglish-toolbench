"""Data model for benchmark examples."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

Action = Literal["call", "clarify", "respond"]
Category = Literal["single", "parallel", "multi_turn", "clarify", "irrelevant"]
Script = Literal["roman", "devanagari", "english"]

CATEGORIES: tuple[str, ...] = ("single", "parallel", "multi_turn", "clarify", "irrelevant")
SCRIPTS: tuple[str, ...] = ("roman", "devanagari", "english")


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Expected(BaseModel):
    """What the assistant should do next.

    ``call``: run ``tool_calls`` (one or more). ``clarify``: ask for a missing
    required detail. ``respond``: no tool fits, reply in text.
    """

    action: Action
    tool_calls: list[ToolCall] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> Expected:
        if self.action == "call" and not self.tool_calls:
            raise ValueError("action 'call' needs at least one tool call")
        if self.action != "call" and self.tool_calls:
            raise ValueError(f"action {self.action!r} must not have tool calls")
        return self


class Context(BaseModel):
    now: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$")
    city: str

    @property
    def dt(self) -> datetime:
        return datetime.strptime(self.now, "%Y-%m-%dT%H:%M")


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class Example(BaseModel):
    id: str
    category: Category
    script: Script
    context: Context
    messages: list[Message]
    expected: Expected
    meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _ends_with_user(self) -> Example:
        if not self.messages or self.messages[-1].role != "user":
            raise ValueError("conversation must end with a user message")
        return self
