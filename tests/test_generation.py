import json
import random
from datetime import date
from pathlib import Path

import pytest

from htb.eval.metrics import evaluate
from htb.gen.hinglish import future_date_phrase, indian_grouping, past_range_phrase, time_phrase
from htb.gen.pipeline import generate, lint, verify
from htb.gen.prompts import build_teacher_prompt
from htb.gen.sampler import CLARIFY_FIELDS, SAMPLERS, build_scenario, build_scenarios
from htb.io import read_jsonl
from htb.prompting import agent_messages, render_target
from htb.schema import Example
from htb.tools import TOOLS, canonicalize

SEEDS = Path(__file__).resolve().parents[1] / "data" / "seed" / "examples.jsonl"


# ------------------------------------------------------------------ hinglish surface forms

@pytest.mark.parametrize("hh,mm,expected", [
    (19, 30, "shaam saade 7 baje"), (13, 30, "dopahar dedh baje"), (2, 30, "raat dhai baje"),
    (8, 15, "subah sava 8 baje"), (8, 45, "subah paune 9 baje"), (22, 0, "raat 10 baje"),
])
def test_time_phrases(hh, mm, expected):
    seen = {time_phrase(random.Random(i), hh, mm) for i in range(40)}
    assert expected in seen


def test_relative_dates():
    today = date(2026, 11, 26)  # Thursday
    phrases = {future_date_phrase(random.Random(i), date(2026, 11, 27), today) for i in range(60)}
    assert "kal" in phrases and "is Friday" in phrases
    nxt = {future_date_phrase(random.Random(i), date(2026, 11, 30), today) for i in range(60)}
    assert "agle hafte Monday" in nxt


def test_past_ranges_are_in_the_past_and_ordered():
    today = date(2027, 1, 2)
    for i in range(200):
        start, end, phrase = past_range_phrase(random.Random(i), today)
        assert start <= end <= today and phrase


def test_indian_grouping():
    assert indian_grouping(1500000) == "15,00,000" and indian_grouping(999) == "999"


# ------------------------------------------------------------------ sampler

def test_every_sampler_emits_valid_calls():
    rng = random.Random(7)
    from htb.gen.sampler import sample_context
    for tool, sampler in SAMPLERS.items():
        for _ in range(150):
            d = sampler(rng, sample_context(rng))
            assert d.tool == tool
            canonicalize(tool, d.args)
            assert set(d.hints) <= set(TOOLS[tool].params.model_fields)


def test_scenarios_are_valid_and_deterministic():
    a = build_scenarios(400, seed=3)
    assert [s.model_dump() for s in a] == [s.model_dump() for s in build_scenarios(400, seed=3)]
    cats = {s.category for s in a}
    assert cats == {"single", "parallel", "multi_turn", "clarify", "irrelevant"}
    for s in a:
        for c in s.expected.tool_calls:
            canonicalize(c.name, c.arguments)
        assert all(s.turns)


def test_clarify_really_drops_a_required_field():
    rng = random.Random(1)
    for i in range(200):
        s = build_scenario(rng, f"c{i}", category="clarify")
        assert s.expected.action == "clarify"
        tool, missing = s.meta["tools"][0], s.meta["missing"]
        assert missing in CLARIFY_FIELDS[tool]
        assert TOOLS[tool].params.model_fields[missing].is_required()


def test_multi_turn_supplies_the_missing_field_in_turn_two():
    rng = random.Random(2)
    for i in range(100):
        s = build_scenario(rng, f"m{i}", category="multi_turn")
        assert len(s.turns) == 2 and s.turns[1]
        assert s.meta["missing"] in s.expected.tool_calls[0].arguments


def test_teacher_prompt_mentions_hints_but_not_iso_answers_as_phrases():
    s = build_scenario(random.Random(5), "p0", category="single")
    prompt = build_teacher_prompt(s, ["kal ki train dikhao"])
    for hint in s.turns[0]:
        assert hint in prompt
    assert "Return only JSON" in prompt


# ------------------------------------------------------------------ seeds

def test_seed_examples_are_valid_and_score_perfectly_against_themselves():
    examples = [Example.model_validate(r) for r in read_jsonl(SEEDS)]
    assert len(examples) >= 20
    for ex in examples:
        for c in ex.expected.tool_calls:
            canonicalize(c.name, c.arguments)
        assert lint(ex) is None, ex.id
        assert agent_messages(ex)[0]["role"] == "system"
    report, _ = evaluate(examples, {e.id: render_target(e.expected) for e in examples})
    assert report["overall"]["task_accuracy"] == 1.0


# ------------------------------------------------------------------ pipeline with a fake LLM

class FakeTeacher:
    """Returns a canned phrasing; good enough to exercise the plumbing."""

    def chat(self, messages, **kw):
        prompt = messages[-1]["content"]
        if '"user_1"' in prompt:
            return json.dumps({"user_1": "ticket dekhni hai", "assistant": "Kab ki?",
                               "user_2": "kal ki"})
        return 'Here you go: {"user": "kal ki train dikhao"}'


class OracleVerifier:
    """Answers with the gold label, found via the conversation text."""

    def __init__(self, gold):
        self.gold = gold

    def chat(self, messages, **kw):
        return render_target(self.gold)


def test_generate_and_verify_end_to_end(tmp_path):
    scenarios = [s for s in build_scenarios(60, seed=11) if s.script != "devanagari"][:10]
    out, rej = tmp_path / "gen.jsonl", tmp_path / "rej.jsonl"
    stats = generate(scenarios, FakeTeacher(), out, rej, workers=2)
    assert sum(stats.values()) == 10
    rows = list(read_jsonl(out)) if out.exists() else []
    assert len(rows) + (len(list(read_jsonl(rej))) if rej.exists() else 0) == 10
    # resumable: a second run does nothing
    assert sum(generate(scenarios, FakeTeacher(), out, rej).values()) == 0

    examples = [Example.model_validate(r) for r in rows]
    if examples:
        ex = examples[0]
        vout, vrej = tmp_path / "ver.jsonl", tmp_path / "vrej.jsonl"
        stats = verify([ex], OracleVerifier(ex.expected), vout, vrej, workers=1)
        assert stats["ok"] == 1
