"""Turn raw model text into a structured prediction."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Prediction:
    action: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    message: str | None = None


def extract_json(text: str) -> Any | None:
    """Return the first complete JSON object in ``text``.

    Handles markdown fences, leading chatter and trailing text, which smaller models
    produce often. Braces inside strings are respected.
    """
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def _normalize_call(call: Any) -> dict[str, Any] | None:
    if not isinstance(call, dict):
        return None
    if "function" in call and isinstance(call["function"], dict):  # OpenAI-style wrapper
        call = call["function"]
    name = call.get("name")
    args = call.get("arguments", {})
    if isinstance(args, str):  # OpenAI returns arguments as a JSON string
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None
    if not isinstance(name, str) or not isinstance(args, dict):
        return None
    return {"name": name, "arguments": args}


def parse_prediction(text: str) -> Prediction | None:
    """Parse model output. Returns ``None`` if it is not a usable response object."""
    obj = extract_json(text or "")
    if not isinstance(obj, dict):
        return None
    raw_calls = obj.get("tool_calls") or []
    if not isinstance(raw_calls, list):
        return None
    calls = [_normalize_call(c) for c in raw_calls]
    if any(c is None for c in calls):
        return None
    action = obj.get("action")
    if action is None:  # lenient: infer from content
        action = "call" if calls else "respond"
    if action not in ("call", "clarify", "respond"):
        return None
    if action == "call" and not calls:
        return None
    message = obj.get("message")
    return Prediction(
        action=action, tool_calls=calls if action == "call" else [],
        message=message if isinstance(message, str) else None,
    )
