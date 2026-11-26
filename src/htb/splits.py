"""Train/dev splitting, de-duplication, and test-set review."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from typing import Any

from .eval.matching import norm_text
from .schema import Example, Expected
from .tools import canonicalize


def dedupe_key(ex: Example) -> str:
    return " | ".join(norm_text(m.content) for m in ex.messages if m.role == "user")


def dedupe(examples: list[Example], exclude: set[str] | None = None) -> list[Example]:
    """Drop exact duplicates (after normalization) and anything whose key is in ``exclude``."""
    seen = set(exclude or ())
    out = []
    for ex in examples:
        key = dedupe_key(ex)
        if key not in seen:
            seen.add(key)
            out.append(ex)
    return out


def split_train_dev(examples: list[Example], dev_frac: float, seed: int
                    ) -> tuple[list[Example], list[Example]]:
    """Stratified by category so dev mirrors train's task mix."""
    rng = random.Random(seed)
    by_cat: dict[str, list[Example]] = defaultdict(list)
    for ex in examples:
        by_cat[ex.category].append(ex)
    train, dev = [], []
    for cat in sorted(by_cat):
        items = by_cat[cat][:]
        rng.shuffle(items)
        k = round(len(items) * dev_frac)
        dev += items[:k]
        train += items[k:]
    rng.shuffle(train)
    rng.shuffle(dev)
    return train, dev


def apply_review(candidates: list[Example], decisions: dict[str, Any]) -> tuple[list[Example], dict]:
    """Build the test set from reviewer decisions exported by the review page.

    Accepted examples keep their label unless the reviewer edited it; edited labels
    are re-validated strictly. Undecided examples are left out.
    """
    out, stats = [], defaultdict(int)
    for ex in candidates:
        d = decisions.get(ex.id)
        if not d:
            stats["undecided"] += 1
            continue
        if d.get("decision") != "accept":
            stats["rejected"] += 1
            continue
        meta = {**ex.meta, "human_reviewed": True}
        expected = ex.expected
        if d.get("expected") is not None:
            expected = Expected.model_validate(d["expected"])
            for c in expected.tool_calls:
                canonicalize(c.name, c.arguments)
            if expected.model_dump() != ex.expected.model_dump():
                meta["label_edited"] = True
                stats["edited"] += 1
        if d.get("comment"):
            meta["review_comment"] = d["comment"]
        out.append(ex.model_copy(update={"expected": expected, "meta": meta}))
        stats["accepted"] += 1
    return out, dict(stats)


def review_payload(examples: list[Example]) -> list[dict]:
    from .prompting import WEEKDAYS

    rows = []
    for ex in examples:
        attempts = ex.meta.get("verify_attempts", [])
        rows.append({
            "id": ex.id,
            "category": ex.category,
            "script": ex.script,
            "now": ex.context.now,
            "weekday": WEEKDAYS[ex.context.dt.weekday()],
            "city": ex.context.city,
            "messages": [m.model_dump() for m in ex.messages],
            "expected": ex.expected.model_dump(),
            "verified_by": ex.meta.get("verified_by"),
            "attempts": [{"verifier": a["verifier"], "output": a["output"][:600]}
                         for a in attempts if not ex.meta.get("verified_by")],
            "hard_negative": ex.meta.get("hard_negative"),
        })
    return rows


def render_review_html(examples: list[Example], title: str = "Test set review") -> str:
    from importlib.resources import files

    template = files("htb").joinpath("review_template.html").read_text(encoding="utf-8")
    data = json.dumps(review_payload(examples), ensure_ascii=False).replace("</", "<\\/")
    return template.replace("__TITLE__", title).replace("__DATA__", data)
