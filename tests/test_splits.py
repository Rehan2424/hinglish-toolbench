import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from htb.io import read_jsonl
from htb.schema import Example
from htb.splits import apply_review, dedupe, render_review_html, split_train_dev

SEEDS = Path(__file__).resolve().parents[1] / "data" / "seed" / "examples.jsonl"


@pytest.fixture
def seeds():
    return [Example.model_validate(r) for r in read_jsonl(SEEDS)]


def test_dedupe_normalizes_and_excludes(seeds):
    dup = seeds[0].model_copy(update={"id": "dup", "messages": [
        m.model_copy(update={"content": m.content.upper() + "!!"}) for m in seeds[0].messages]})
    assert len(dedupe(seeds + [dup])) == len(seeds)
    from htb.splits import dedupe_key
    assert len(dedupe(seeds, exclude={dedupe_key(seeds[1])})) == len(seeds) - 1


def test_split_is_stratified_and_disjoint(seeds):
    many = [s.model_copy(update={"id": f"{s.id}-{k}"}) for k in range(10) for s in seeds]
    train, dev = split_train_dev(many, 0.2, seed=1)
    assert len(train) + len(dev) == len(many)
    assert not {e.id for e in train} & {e.id for e in dev}
    assert {e.category for e in dev} == {e.category for e in many}


def test_apply_review_accept_edit_reject(seeds):
    a, b, c = seeds[1], seeds[7], seeds[19]
    edited = a.expected.model_dump()
    edited["tool_calls"][0]["arguments"]["amount"] = 2500
    decisions = {a.id: {"decision": "accept", "expected": edited, "comment": "fixed amount"},
                 b.id: {"decision": "accept"},
                 c.id: {"decision": "reject"}}
    out, stats = apply_review([a, b, c, seeds[3]], decisions)
    assert stats == {"accepted": 2, "edited": 1, "rejected": 1, "undecided": 1}
    assert out[0].expected.tool_calls[0].arguments["amount"] == 2500
    assert out[0].meta["label_edited"] and out[0].meta["human_reviewed"]


def test_apply_review_rejects_invalid_edits(seeds):
    bad = {"action": "call", "tool_calls": []}
    with pytest.raises(ValidationError):
        apply_review([seeds[0]], {seeds[0].id: {"decision": "accept", "expected": bad}})


def test_review_html_embeds_data_safely(seeds):
    tricky = seeds[0].model_copy(update={"messages": [
        seeds[0].messages[0].model_copy(update={"content": "</script><b>x</b>"})]})
    html = render_review_html([tricky] + seeds[1:3])
    assert "</script><b>" not in html
    start = html.index('type="application/json">') + len('type="application/json">')
    payload = json.loads(html[start: html.index("</script>", start)].replace("<\\/", "</"))
    assert payload[0]["messages"][0]["content"] == "</script><b>x</b>" and len(payload) == 3
