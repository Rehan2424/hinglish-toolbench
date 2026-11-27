"""Tag examples with the Hinglish phenomena their messages contain.

Tags come from the message text (and, for implicit cities, the gold label), not from
generation metadata, so they also work on hand-edited or external data. Results are
then sliced by phenomenon: how often does a model resolve "saade 7", "parso" or
"dedh hazaar" correctly?
"""

from __future__ import annotations

import re

from ..lexicon import CITY_ALIASES, CITY_COLLOQUIAL
from ..schema import Example

_DEV = "ऀ-ॿ"


def _dev_word(alternatives: str) -> str:
    """Whole-word match for Devanagari, where \\b is unreliable around vowel signs."""
    return rf"(?<![{_DEV}\w])(?:{alternatives})(?![{_DEV}\w])"


_WEEKDAYS_EN = "monday|tuesday|wednesday|thursday|friday|saturday|sunday"
_WEEKDAYS_HI = ("somvaar|somwar|mangalvaar|mangalwar|budhvaar|budhwar|guruvaar|guruwar|"
                "veervaar|shukravaar|shukrawar|shanivaar|shaniwar|ravivaar|raviwar")

PHENOMENA: dict[str, re.Pattern[str]] = {
    "kal (tomorrow / yesterday)": re.compile(r"\bkal\b|" + _dev_word("कल"), re.I),
    "parso (day after tomorrow)": re.compile(r"\bparso[n]?\b|" + _dev_word("परसों|परसो"), re.I),
    "weekday reference": re.compile(
        rf"\b(?:is|agle hafte|next week|next|coming|this)\s+(?:{_WEEKDAYS_EN})\b"
        rf"|\b(?:{_WEEKDAYS_HI})\b|"
        + _dev_word("सोमवार|मंगलवार|बुधवार|गुरुवार|शुक्रवार|शनिवार|रविवार"), re.I),
    "saade / sava / paune / dedh / dhai time": re.compile(
        r"\b(?:saade|sade|saadhe|sava|savaa|sawa|paune|pone)\s+\d{1,2}\b"
        r"|\b(?:dedh|derh|dhai|dhaai|dhayi)\s+baje\b"
        r"|(?:साढ़े|साढे|सवा|पौने)\s*[\d०-९]|(?:डेढ़|डेढ|ढाई)\s*बजे", re.I),
    "spoken amount (hazaar / lakh / sau / 2k)": re.compile(
        r"\b(?:hazaar|hazar|hajar|hazaar|lakh|lac|sau)\b|\b\d+(?:\.\d+)?\s?k\b|"
        + _dev_word("हज़ार|हजार|लाख|सौ"), re.I),
}

_COLLOQUIAL = re.compile(
    r"\b(?:" + "|".join(a for names in CITY_COLLOQUIAL.values() for a in names) + r")\b", re.I)

IMPLICIT_CITY_FIELDS = {"search_trains": "source", "search_flights": "source",
                        "get_weather": "city"}


def _mentions(text: str, city: str) -> bool:
    names = [city, *CITY_ALIASES.get(city, [])]
    low = text.lower()
    return any(n.lower() in low for n in names)


def tag(example: Example) -> list[str]:
    text = " ".join(m.content for m in example.messages if m.role == "user")
    tags = [name for name, rx in PHENOMENA.items() if rx.search(text)]
    if _COLLOQUIAL.search(text):
        tags.append("colloquial city name (Bombay, Dilli...)")
    for call in example.expected.tool_calls:
        field = IMPLICIT_CITY_FIELDS.get(call.name)
        value = call.arguments.get(field) if field else None
        if value == example.context.city and not _mentions(text, value):
            tags.append("implicit city (user's own)")
            break
    return tags
