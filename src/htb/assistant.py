"""Run the assistant on a live conversation and return validated tool calls.

This is the piece a product would embed. It builds the same prompt the benchmark
uses, sends it to any chat backend (a local llama.cpp server for the small model, or
a hosted API), parses the reply, and validates every call against the tool schemas.
Invalid calls are reported, never silently executed.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

from .eval.parse import parse_prediction
from .prompting import render_system
from .schema import Context
from .tools import ToolCallError, canonicalize

IST = timezone(timedelta(hours=5, minutes=30))


class ChatBackend(Protocol):
    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str: ...


@dataclass
class Result:
    action: str  # call | clarify | respond | error
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    message: str | None = None
    errors: list[str] = field(default_factory=list)
    raw: str = ""
    latency_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.action != "error" and not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "ok": self.ok}


def now_ist() -> str:
    return datetime.now(IST).strftime("%Y-%m-%dT%H:%M")


class Assistant:
    def __init__(self, backend: ChatBackend, *, max_tokens: int = 512) -> None:
        self.backend = backend
        self.max_tokens = max_tokens

    def run(self, messages: list[dict[str, str]], *, city: str = "Delhi",
            now: str | None = None) -> Result:
        """``messages`` is the conversation so far, ending with the user's turn."""
        if not messages or messages[-1].get("role") != "user":
            raise ValueError("conversation must end with a user message")
        ctx = Context(now=now or now_ist(), city=city)
        prompt = [{"role": "system", "content": render_system(ctx)}]
        prompt += [{"role": m["role"], "content": m["content"]} for m in messages]

        start = time.perf_counter()
        raw = self.backend.chat(prompt, temperature=0.0, max_tokens=self.max_tokens)
        latency = (time.perf_counter() - start) * 1000

        pred = parse_prediction(raw)
        if pred is None:
            return Result("error", errors=["model output is not a valid response object"],
                          raw=raw, latency_ms=latency)
        calls, errors = [], []
        for call in pred.tool_calls:
            try:
                args = canonicalize(call["name"], call["arguments"])
                calls.append({"name": call["name"], "arguments": args, "valid": True})
            except ToolCallError as exc:
                errors.append(f"{call['name']}: {str(exc).splitlines()[0]}")
                calls.append({**call, "valid": False})
        return Result(pred.action, calls, pred.message, errors, raw, latency)
