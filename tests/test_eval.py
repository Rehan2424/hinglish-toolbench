import json

import pytest

from htb.eval.matching import entity_key, fuzzy_equal
from htb.eval.metrics import evaluate, score_example
from htb.eval.parse import extract_json, parse_prediction
from htb.schema import Context, Example, Expected, Message, ToolCall

CTX = Context(now="2026-11-26T10:00", city="Roorkee")


def make(calls=None, action="call", category="single"):
    return Example(
        id="t", category=category, script="roman", context=CTX,
        messages=[Message(role="user", content="x")],
        expected=Expected(action=action, tool_calls=[ToolCall(name=n, arguments=a)
                                                     for n, a in (calls or [])]),
    )


def out(calls=None, action="call", message=None):
    obj = {"action": action}
    if calls is not None:
        obj["tool_calls"] = [{"name": n, "arguments": a} for n, a in calls]
    if message:
        obj["message"] = message
    return json.dumps(obj, ensure_ascii=False)


# ------------------------------------------------------------------ parsing

def test_extract_json_handles_fences_and_chatter():
    text = 'Sure! ```json\n{"action": "respond", "message": "a {brace} inside"}\n``` done'
    assert extract_json(text) == {"action": "respond", "message": "a {brace} inside"}


def test_parse_openai_style_string_arguments():
    p = parse_prediction('{"tool_calls":[{"function":{"name":"check_balance","arguments":"{}"}}]}')
    assert p.action == "call" and p.tool_calls == [{"name": "check_balance", "arguments": {}}]


@pytest.mark.parametrize("text", ["", "no json here", '{"action": "dance"}',
                                  '{"action": "call", "tool_calls": []}', "[1, 2]"])
def test_unusable_outputs(text):
    assert parse_prediction(text) is None


# ------------------------------------------------------------------ matching

def test_entity_aliases():
    assert entity_key("Bombay") == entity_key("mumbai") == entity_key("मुंबई")
    assert entity_key("Mom") == entity_key("Mummy")
    assert entity_key("Pune") != entity_key("Mumbai")


@pytest.mark.parametrize("a,b,same", [
    ("Call mummy", "Mummy ko call", True),
    ("Doctor appointment", "Go to doctor", True),
    ("Take medicine", "take medicines", True),
    ("Call papa", "Call mummy", False),
    ("Pay rent", "Pay electricity bill", False),
])
def test_fuzzy(a, b, same):
    assert fuzzy_equal(a, b) is same


@pytest.mark.parametrize("gold,pred,same", [
    ("dinner split", "Dinner share", True),
    ("dinner split", "dinner ka hissa", True),  # the user's own phrasing
    ("rent", "kiraya", True),
    ("Call mummy", "Call papa", False),
    ("rent", "trip expenses", False),
])
def test_free_text_alternates(gold, pred, same):
    from htb.eval.matching import fuzzy_match
    assert fuzzy_match(gold, pred) is same


# ------------------------------------------------------------------ scoring

TRAIN = ("search_trains", {"source": "Delhi", "destination": "Jaipur", "date": "2026-11-27",
                           "travel_class": "3A"})


def test_exact_match_with_alias_and_explicit_default():
    ex = make([("transfer_money", {"recipient": "Mummy", "amount": 1500})])
    s = score_example(ex, out([("transfer_money", {"recipient": "Mom", "amount": 1500.0,
                                                   "mode": "upi"})]))
    assert s.correct and s.arg_fp == 0 and s.arg_fn == 0


def test_wrong_date_is_partial_credit_not_success():
    ex = make([TRAIN])
    wrong = dict(TRAIN[1], date="2026-11-25")
    s = score_example(ex, out([("search_trains", wrong)]))
    assert not s.correct and s.names_match
    assert (s.arg_tp, s.arg_fp, s.arg_fn) == (3, 1, 1)


def test_extra_optional_argument_is_penalised():
    ex = make([("pay_bill", {"biller": "electricity"})])
    s = score_example(ex, out([("pay_bill", {"biller": "electricity", "amount": 500})]))
    assert not s.correct and s.arg_fp == 1


def test_parallel_calls_order_insensitive():
    calls = [("check_balance", {}), ("transfer_money", {"recipient": "Priya", "amount": 500})]
    ex = make(calls, category="parallel")
    assert score_example(ex, out(list(reversed(calls)))).correct
    assert not score_example(ex, out(calls[:1])).correct


def test_hallucinated_tool_and_schema_invalid():
    ex = make([("check_balance", {})])
    s = score_example(ex, out([("get_balance", {})]))
    assert s.hallucinated_tool and s.schema_valid is False and not s.correct
    s = score_example(ex, out([("check_balance", {"account": "wallet"})]))
    assert s.schema_valid is False and not s.hallucinated_tool


def test_abstention():
    ex = make(action="respond", category="irrelevant")
    assert score_example(ex, out(action="respond", message="Sorry")).correct
    s = score_example(ex, out([("check_balance", {})]))
    assert not s.correct and s.pred_action == "call"
    clar = make(action="clarify", category="clarify")
    assert not score_example(clar, out(action="respond", message="no")).correct


def test_report_aggregates():
    ex_ok = make([("check_balance", {})])
    ex_bad = make(action="respond", category="irrelevant").model_copy(update={"id": "u"})
    report, rows = evaluate([ex_ok, ex_bad], {"t": out([("check_balance", {})]),
                                              "u": out([("check_balance", {})])})
    o = report["overall"]
    assert o["n"] == 2 and o["task_accuracy"] == 0.5 and o["false_call_rate"] == 1.0
    assert set(report["by_category"]) == {"single", "irrelevant"} and len(rows) == 2
