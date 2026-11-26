"""The assistant's prompt and output format.

This is the single source of truth for how a conversation is shown to a model.
Data verification, API baselines, fine-tuning and evaluation all go through it,
so every model sees exactly the same instructions.
"""

from __future__ import annotations

import json
from functools import lru_cache

from .schema import Example, Expected
from .tools import TOOLS

SYSTEM_TEMPLATE = """\
You are the assistant inside an Indian super-app. Users write in Hinglish (romanized \
Hindi mixed with English), Hindi in Devanagari, or English. Turn each request into \
tool calls.

Now: {now} ({weekday}). User's city: {city}.

Tools:
{tools}

Reply with exactly one JSON object and nothing else:
- {{"action": "call", "tool_calls": [{{"name": "...", "arguments": {{...}}}}]}} when tools \
can do it. Use several calls if the user asks for several things.
- {{"action": "clarify", "message": "..."}} when a required argument is missing and cannot \
be inferred. Ask briefly, in the user's language.
- {{"action": "respond", "message": "..."}} when no tool can help.

Rules:
1. Include only the optional arguments the user actually specified.
2. Resolve dates and times to absolute values from "Now". "kal" means tomorrow for \
plans and yesterday for past records; "parso" is the day after tomorrow; "is <day>" or \
"<day> ko" is the next such day; "agle hafte <day>" is that day in next week (weeks \
start Monday). "pichle N din" means start_date = today minus N days, end_date = today. \
A month name in a past-records request means its most recent full occurrence.
3. A time with no date means its next occurrence from now.
4. If the user names no departure city (trains, flights) or no city for weather, use \
their city. Leave out cab pickup and restaurant location unless named; they default \
to the user's current location.
5. Write titles, notes and labels in English (translate them if needed). Write names of \
people and dishes in Latin script, even if the user wrote in Devanagari. Use standard \
city names.
"""

WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def _type(schema: dict) -> str:
    if "enum" in schema:
        return "|".join(json.dumps(v) for v in schema["enum"])
    t = schema.get("type")
    if t == "array":
        return f"list[{_type(schema.get('items', {}))}]"
    if t == "object":
        req = set(schema.get("required", []))
        fields = []
        for name, sub in schema.get("properties", {}).items():
            opt = "" if name in req else "?"
            default = f" = {json.dumps(sub['default'])}" if "default" in sub else ""
            fields.append(f"{name}{opt}: {_type(sub)}{default}")
        return "{" + ", ".join(fields) + "}"
    return {"number": "number", "integer": "int", "boolean": "bool"}.get(t, "string")


def compact_signature(schema: dict) -> str:
    params = schema["parameters"]
    req = set(params.get("required", []))
    lines = [f"{schema['name']}: {schema['description']}"]
    for name, sub in params.get("properties", {}).items():
        opt = "" if name in req else "?"
        default = f" = {json.dumps(sub['default'])}" if "default" in sub else ""
        desc = f"  # {sub['description']}" if sub.get("description") else ""
        lines.append(f"  {name}{opt}: {_type(sub)}{default}{desc}")
    return "\n".join(lines)


@lru_cache(maxsize=2)
def tools_block(style: str = "compact") -> str:
    """Tool list for the system prompt.

    ``compact`` (default) renders signatures like ``mode?: "upi"|"imps" = "upi"``, which
    is about half the tokens of raw JSON schema. That matters for a small model on CPU,
    where prompt length dominates latency. ``json`` is kept for comparison.
    """
    schemas = [t.json_schema() for t in TOOLS.values()]
    if style == "json":
        return "\n".join(json.dumps(s, ensure_ascii=False, separators=(",", ":"))
                         for s in schemas)
    return "\n".join(compact_signature(s) for s in schemas)


def system_prompt(example: Example) -> str:
    ctx = example.context
    return SYSTEM_TEMPLATE.format(
        now=ctx.now, weekday=WEEKDAYS[ctx.dt.weekday()], city=ctx.city, tools=tools_block()
    )


def agent_messages(example: Example) -> list[dict[str, str]]:
    """Chat messages for a model: system prompt, then the conversation so far."""
    msgs = [{"role": "system", "content": system_prompt(example)}]
    msgs += [m.model_dump() for m in example.messages]
    return msgs


def render_target(expected: Expected, message: str | None = None) -> str:
    """Serialize the expected output in the exact format the model must produce."""
    obj: dict = {"action": expected.action}
    if expected.action == "call":
        obj["tool_calls"] = [c.model_dump() for c in expected.tool_calls]
    else:
        obj["message"] = message or ""
    return json.dumps(obj, ensure_ascii=False)
