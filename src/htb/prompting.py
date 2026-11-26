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
4. If the user does not name a departure city or location, use their city.
5. Write free-text values (titles, notes, labels, dish names) and names of people in \
English or romanized script, even if the user wrote in Devanagari. Use standard city names.
"""

WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


@lru_cache(maxsize=1)
def tools_block() -> str:
    return "\n".join(
        json.dumps(t.json_schema(), ensure_ascii=False, separators=(",", ":"))
        for t in TOOLS.values()
    )


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
