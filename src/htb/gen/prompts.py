"""Prompt that asks the teacher model to phrase a scenario as user messages."""

from __future__ import annotations

import json

from ..prompting import WEEKDAYS, compact_signature
from ..tools import TOOLS
from .sampler import Scenario

SCRIPT_INSTRUCTIONS = {
    "roman": "Romanized Hinglish: Hindi written in Latin letters, mixed naturally with "
             "English words, the way people text on WhatsApp.",
    "devanagari": "Hindi written in DEVANAGARI SCRIPT, for example: "
                  "\"कल सुबह 7 बजे के बाद दिल्ली से जयपुर की ट्रेन दिखाओ\". Romanized Hindi "
                  "(Latin letters) is NOT acceptable here: transliterate every Hindi phrase you "
                  "are given into Devanagari. Only everyday English words (train, bill, UPI, AC) "
                  "and numbers may stay in Latin letters.",
    "english": "Mostly Indian English with a few Hindi words mixed in.",
}

OUTPUT_FORMATS = {
    "one_turn": '{"user": "<the user\'s message>"}',
    "multi_turn": '{"user_1": "<first message, missing one detail>", '
                  '"assistant": "<short question asking for that detail, in the user\'s '
                  'language>", "user_2": "<reply that supplies the detail>"}',
}

TEMPLATE = """\
You write realistic messages that people in India type to an assistant in a phone app.

Today is {weekday}, {date}, time {time}. The user is in {city}.

{task}

Phrasing requirements:
{hints}

Language: {script}
{style}
Rules:
- Do not add any detail that is not listed. Do not leave out any listed detail.
- When a phrase is given in quotes, use it (transliterate it if the language needs another \
script), never the exact date or time it stands for. Do not put quotation marks around it \
in the message.
- Never mention tools, functions, JSON, or field names (nothing with an underscore). Write \
only what the user types.
- Keep it natural and short (one or two sentences per message).
{examples}
Return only JSON in this format:
{fmt}
"""


def _task(s: Scenario) -> str:
    if s.category == "irrelevant":
        return ("Write a message for which none of the app's features help. The user should "
                "sound like a real app user, not a test.")
    target = {"action": s.expected.action,
              "tool_calls": [c.model_dump() for c in s.expected.tool_calls]}
    tools = sorted({c.name for c in s.expected.tool_calls} | set(s.meta.get("tools", [])))
    specs = "\n".join(compact_signature(TOOLS[t].json_schema()) for t in tools)
    lines = [f"App feature(s) involved, for your reference:\n{specs}", ""]
    if s.category == "clarify":
        lines.append("The user asks for this but forgets one required detail, so the assistant "
                     "will have to ask a follow-up question.")
    elif s.category == "multi_turn":
        lines.append("Write a three-message exchange: the user asks but forgets one required "
                     "detail, the assistant asks for it, and the user replies with it. After "
                     "the exchange, the assistant's correct action is exactly:")
        lines.append(json.dumps(target, ensure_ascii=False))
    else:
        lines.append("The assistant's correct action for the user's message is exactly:")
        lines.append(json.dumps(target, ensure_ascii=False))
    return "\n".join(lines)


def _hints(s: Scenario) -> str:
    if len(s.turns) == 1:
        return "\n".join(f"- {h}" for h in s.turns[0]) or "- (none)"
    out = []
    for i, turn in enumerate(s.turns, start=1):
        label = "first message" if i == 1 else "reply after the assistant's question"
        out.append(f"In the {label}:")
        out += [f"  - {h}" for h in turn]
    return "\n".join(out)


def build_teacher_prompt(s: Scenario, style_examples: list[str] | None = None) -> str:
    dt = s.context.dt
    style = ""
    if s.style:
        styles = list(s.style)
        if s.script == "devanagari":  # roman short forms ("kr do", "plz") fight the script
            styles = [("casual texting: informal, short, little punctuation"
                       if st.startswith("casual") else st) for st in styles]
        style = "Style: " + "; ".join(styles) + "\n"
    examples = ""
    if style_examples:
        examples = ("\nFor tone only, real-sounding messages look like:\n"
                    + "\n".join(f"- {e}" for e in style_examples) + "\n")
    fmt = OUTPUT_FORMATS["multi_turn" if s.category == "multi_turn" else "one_turn"]
    if s.script == "devanagari":
        fmt += "\nAll user messages must be in Devanagari script."
    return TEMPLATE.format(
        weekday=WEEKDAYS[dt.weekday()], date=dt.strftime("%d %B %Y"), time=dt.strftime("%H:%M"),
        city=s.context.city, task=_task(s), hints=_hints(s), script=SCRIPT_INSTRUCTIONS[s.script],
        style=style, examples=examples, fmt=fmt,
    )
