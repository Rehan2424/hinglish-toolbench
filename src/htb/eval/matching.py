"""Value and call matching with per-field policies.

Policies come from the tool definitions (``x-match``):

* exact  - enums, numbers (int/float agnostic), booleans, dates, ids.
* entity - case, punctuation and spacing ignored; known aliases map to one canonical
  form ("Bombay" == "Mumbai" == "मुंबई").
* fuzzy  - free text. Equal if F1 over content words (function words removed) is
  >= 0.6, or character similarity >= 0.8. "Call mummy" matches "Mummy ko call",
  "Doctor appointment" matches "Go to doctor", but "Call papa" does not match
  "Call mummy" and "Pay rent" does not match "Pay bill".
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Callable
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any

from pydantic import BaseModel

from ..lexicon import CITY_ALIASES, PEOPLE_ALIASES
from ..tools import ENTITY, FUZZY, TOOLS, field_policy, item_model

TOKEN_F1_THRESHOLD = 0.6
CHAR_RATIO_THRESHOLD = 0.8

_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)


def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).casefold().replace("_", " ")
    s = _PUNCT.sub(" ", s)
    return " ".join(s.split())


@lru_cache(maxsize=1)
def _alias_table() -> dict[str, str]:
    table: dict[str, str] = {}
    for canon, aliases in {**CITY_ALIASES, **PEOPLE_ALIASES}.items():
        key = norm_text(canon)
        table[key] = key
        for a in aliases:
            table[norm_text(a)] = key
    return table


def entity_key(s: str) -> str:
    n = norm_text(s)
    return _alias_table().get(n, n)


# Function words (English + common Hinglish) ignored when comparing free text, so
# "Doctor appointment" ~ "Go to the doctor" but "Call papa" !~ "Call mummy".
STOPWORDS = frozenset(["a", "an", "the", "to", "of", "for", "in", "on", "at", "with", "and", "or", "my", "me", "i", "is", "are", "be", "go", "get", "do", "please", "ko", "ka", "ki", "ke", "se", "me", "mein", "par", "pe", "hai", "hain", "tha", "thi", "karna", "karni", "karne", "kar", "lena", "leni", "lene", "dena", "deni", "dene", "jaana", "jana", "wala", "wali", "wale", "ek", "aur", "bhi"])


def content_tokens(s: str) -> Counter:
    tokens = norm_text(s).split()
    kept = [t for t in tokens if t not in STOPWORDS]
    return Counter(kept or tokens)


def token_f1(a: str, b: str) -> float:
    ta, tb = content_tokens(a), content_tokens(b)
    overlap = sum((ta & tb).values())
    if overlap == 0:
        return 0.0
    p, r = overlap / sum(tb.values()), overlap / sum(ta.values())
    return 2 * p * r / (p + r)


def fuzzy_equal(a: str, b: str) -> bool:
    na, nb = norm_text(a), norm_text(b)
    if na == nb:
        return True
    if token_f1(na, nb) >= TOKEN_F1_THRESHOLD:
        return True
    return SequenceMatcher(None, na, nb).ratio() >= CHAR_RATIO_THRESHOLD


def scalar_match(policy: str, gold: Any, pred: Any) -> bool:
    if isinstance(gold, bool) or isinstance(pred, bool):
        return type(gold) is type(pred) and gold == pred
    if isinstance(gold, (int, float)) and isinstance(pred, (int, float)):
        return abs(gold - pred) < 1e-6
    if isinstance(gold, str) and isinstance(pred, str):
        if policy == ENTITY:
            return entity_key(gold) == entity_key(pred)
        if policy == FUZZY:
            return fuzzy_equal(gold, pred)
        return gold == pred
    return gold == pred


def multiset_match(gold: list, pred: list, eq: Callable[[Any, Any], bool]) -> bool:
    """True if there is a one-to-one pairing of gold and pred items under ``eq``.

    Uses augmenting paths (bipartite matching), so it is exact even when a fuzzy
    item could pair with more than one candidate. Lists here are tiny.
    """
    if len(gold) != len(pred):
        return False
    owner: dict[int, int] = {}  # pred index -> gold index

    def assign(gi: int, seen: set[int]) -> bool:
        for pi in range(len(pred)):
            if pi in seen or not eq(gold[gi], pred[pi]):
                continue
            seen.add(pi)
            if pi not in owner or assign(owner[pi], seen):
                owner[pi] = gi
                return True
        return False

    return all(assign(gi, set()) for gi in range(len(gold)))


def values_match(model: type[BaseModel], key: str, gold: Any, pred: Any) -> bool:
    policy = field_policy(model, key)
    if isinstance(gold, list):
        if not isinstance(pred, list):
            return False
        sub = item_model(model, key)
        if sub is not None:
            return multiset_match(
                gold, pred, lambda g, p: isinstance(g, dict) and isinstance(p, dict)
                and dicts_match(sub, g, p),
            )
        return multiset_match(gold, pred, lambda g, p: scalar_match(policy, g, p))
    return scalar_match(policy, gold, pred)


def dicts_match(model: type[BaseModel], gold: dict, pred: dict) -> bool:
    if set(gold) != set(pred):
        return False
    return all(values_match(model, k, gold[k], pred[k]) for k in gold)


def matching_keys(name: str, gold: dict, pred: dict) -> int:
    """Number of argument keys present in both calls with matching values."""
    model = TOOLS[name].params
    return sum(1 for k in gold if k in pred and values_match(model, k, gold[k], pred[k]))


def calls_equal(gold: dict, pred: dict) -> bool:
    if gold["name"] != pred["name"] or gold["name"] not in TOOLS:
        return False
    return dicts_match(TOOLS[gold["name"]].params, gold["arguments"], pred["arguments"])


def align_calls(gold: list[dict], pred: list[dict]) -> list[tuple[int, int]]:
    """Pair gold and predicted calls of the same tool, best argument overlap first."""
    candidates = []
    for gi, g in enumerate(gold):
        for pi, p in enumerate(pred):
            if g["name"] == p["name"] and g["name"] in TOOLS:
                score = matching_keys(g["name"], g["arguments"], p["arguments"])
                candidates.append((score, gi, pi))
    candidates.sort(key=lambda t: (-t[0], t[1], t[2]))
    used_g: set[int] = set()
    used_p: set[int] = set()
    pairs = []
    for _, gi, pi in candidates:
        if gi not in used_g and pi not in used_p:
            used_g.add(gi)
            used_p.add(pi)
            pairs.append((gi, pi))
    return pairs
