"""Generation pipeline: scenarios -> teacher phrasing -> lint -> round-trip verification.

Every stage reads and writes JSONL and is resumable: rows already present in the
output (or rejects) file are skipped, so a run that hits a rate limit can simply be
restarted.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Protocol

from ..eval.metrics import score_example
from ..eval.parse import extract_json
from ..io import existing_ids
from ..prompting import agent_messages
from ..schema import Example, Message
from ..tools import TOOLS
from .prompts import build_teacher_prompt
from .sampler import Scenario

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_LATIN = re.compile(r"[A-Za-z]")
# Only snake_case names: plain words like "date" or "amount" occur in normal speech.
_FIELD_NAMES = {k for t in TOOLS.values() for k in t.params.model_fields if "_" in k}


class Chat(Protocol):
    def chat(self, messages: list[dict[str, str]], **kw: Any) -> str: ...


# --------------------------------------------------------------------------- helpers


def _run(items: list[Any], fn: Callable[[Any], tuple[str, dict]], out_path: str | Path,
         reject_path: str | Path | None, workers: int, label: str) -> Counter:
    """Apply ``fn`` to items in a thread pool, appending results as they finish.

    Workers only call the API; all file writes happen on this thread. Items that
    raise (network errors, exhausted retries) are not written anywhere, so the next
    run of the same command retries them.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    stats: Counter = Counter()
    with ExitStack() as stack:
        out = stack.enter_context(open(out_path, "a", encoding="utf-8"))
        rej = (stack.enter_context(open(reject_path, "a", encoding="utf-8"))
               if reject_path else None)
        pool = stack.enter_context(ThreadPoolExecutor(max_workers=workers))
        futures = [pool.submit(fn, it) for it in items]
        for i, fut in enumerate(as_completed(futures), start=1):
            try:
                status, row = fut.result()
            except Exception as exc:  # one failed call should not kill a long run
                status, row = "error", {}
                print(f"[{label}] error: {exc!r}"[:300], file=sys.stderr)
            stats[status] += 1
            target = out if status == "ok" else rej
            if target is not None and row.get("id") is not None:
                target.write(json.dumps(row, ensure_ascii=False) + "\n")
                target.flush()
            if i % 25 == 0 or i == len(items):
                print(f"[{label}] {i}/{len(items)} {dict(stats)}", file=sys.stderr)
    return stats


def lint(example: Example) -> str | None:
    """Cheap checks before spending an API call on verification. Returns a reason or None."""
    user_texts = [m.content for m in example.messages if m.role == "user"]
    for text in user_texts:
        if not text.strip():
            return "empty message"
        if len(text) > 500:
            return "message too long"
        lowered = text.lower()
        if "{" in text or "json" in lowered:
            return "leaked format"
        if any(name in lowered for name in TOOLS) or any(f in lowered for f in _FIELD_NAMES):
            return "leaked tool or field name"
    joined = " ".join(user_texts)
    deva, latin = len(_DEVANAGARI.findall(joined)), len(_LATIN.findall(joined))
    if example.script == "devanagari" and deva < 0.5 * (deva + latin):
        return "expected mostly Devanagari"
    if example.script in ("roman", "english") and deva > 0:
        return "unexpected Devanagari"
    digits = re.sub(r"\D", "", joined)
    for call in example.expected.tool_calls:
        for key in ("pnr", "phone_number", "order_id"):
            value = call.arguments.get(key)
            if value and re.sub(r"\D", "", value) not in digits:
                return f"{key} missing from text"
    return None


def assemble(scenario: Scenario, obj: Any) -> Example:
    if not isinstance(obj, dict):
        raise ValueError("teacher did not return a JSON object")
    if scenario.category == "multi_turn":
        u1, a, u2 = obj.get("user_1"), obj.get("assistant"), obj.get("user_2")
        if not all(isinstance(x, str) and x.strip() for x in (u1, a, u2)):
            raise ValueError("missing multi-turn fields")
        clarify = json.dumps({"action": "clarify", "message": a.strip()}, ensure_ascii=False)
        messages = [Message(role="user", content=u1.strip()),
                    Message(role="assistant", content=clarify),
                    Message(role="user", content=u2.strip())]
    else:
        u = obj.get("user")
        if not isinstance(u, str) or not u.strip():
            raise ValueError("missing user field")
        messages = [Message(role="user", content=u.strip())]
    return Example(id=scenario.id, category=scenario.category, script=scenario.script,
                   context=scenario.context, messages=messages, expected=scenario.expected,
                   meta={**scenario.meta, "style": scenario.style})


# --------------------------------------------------------------------------- stages


def generate(scenarios: Iterable[Scenario], teacher: Chat, out_path: str | Path,
             reject_path: str | Path, *, workers: int = 4, temperature: float = 0.9,
             style_examples: dict[str, list[str]] | None = None,
             teacher_name: str = "") -> Counter:
    done = existing_ids(out_path, reject_path)
    todo = [s for s in scenarios if s.id not in done]

    def work(s: Scenario) -> tuple[str, dict]:
        prompt = build_teacher_prompt(s, (style_examples or {}).get(s.script))
        raw = teacher.chat([{"role": "user", "content": prompt}], temperature=temperature,
                           max_tokens=600)
        try:
            ex = assemble(s, extract_json(raw))
        except ValueError as exc:
            return "bad_output", {"id": s.id, "reason": str(exc), "raw": raw}
        reason = lint(ex)
        if reason:
            return "lint_fail", {"id": s.id, "reason": reason, "example": ex.model_dump()}
        ex.meta["teacher"] = teacher_name
        return "ok", ex.model_dump()

    return _run(todo, work, out_path, reject_path, workers, "generate")


def verify(examples: Iterable[Example], verifier: Chat, out_path: str | Path,
           reject_path: str | Path, *, workers: int = 4, verifier_name: str = "") -> Counter:
    """Round-trip check: a model that never saw the label must recover it from the text.

    If it cannot, the phrasing is ambiguous or unfaithful and the example is dropped.
    """
    done = existing_ids(out_path, reject_path)
    todo = [e for e in examples if e.id not in done]

    def work(ex: Example) -> tuple[str, dict]:
        raw = verifier.chat(agent_messages(ex), temperature=0.0, max_tokens=600)
        s = score_example(ex, raw)
        row = ex.model_dump()
        row["meta"] = {**row["meta"], "verifier": verifier_name, "verifier_output": raw}
        if s.correct:
            return "ok", row
        return "mismatch", {**row, "reason": s.error or "verifier disagreed with label"}

    return _run(todo, work, out_path, reject_path, workers, "verify")


def predict(examples: Iterable[Example], model: Chat, out_path: str | Path, *,
            workers: int = 4, max_tokens: int = 600) -> Counter:
    done = existing_ids(out_path)
    todo = [e for e in examples if e.id not in done]

    def work(ex: Example) -> tuple[str, dict]:
        raw = model.chat(agent_messages(ex), temperature=0.0, max_tokens=max_tokens)
        return "ok", {"id": ex.id, "output": raw}

    return _run(todo, work, out_path, None, workers, "predict")
