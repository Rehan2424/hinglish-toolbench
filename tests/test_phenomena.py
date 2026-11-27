import pytest

from htb.eval.phenomena import tag
from htb.schema import Context, Example, Expected, Message, ToolCall

CTX = Context(now="2026-11-26T10:00", city="Roorkee")


def ex(text, calls=None, action="call"):
    return Example(id="x", category="single", script="roman", context=CTX,
                   messages=[Message(role="user", content=text)],
                   expected=Expected(action=action, tool_calls=[
                       ToolCall(name=n, arguments=a) for n, a in (calls or [])]))


@pytest.mark.parametrize("text,expected", [
    ("kal subah ki train dikhao", "kal (tomorrow / yesterday)"),
    ("कल का मौसम बताओ", "kal (tomorrow / yesterday)"),
    ("parso Jaipur jaana hai", "parso (day after tomorrow)"),
    ("परसों की फ्लाइट", "parso (day after tomorrow)"),
    ("agle hafte Monday ko meeting", "weekday reference"),
    ("shukravaar ko alarm", "weekday reference"),
    ("शुक्रवार को याद दिलाना", "weekday reference"),
    ("shaam saade 7 baje", "saade / sava / paune / dedh / dhai time"),
    ("dopahar dedh baje", "saade / sava / paune / dedh / dhai time"),
    ("साढ़े 7 बजे", "saade / sava / paune / dedh / dhai time"),
    ("Priya ko dedh hazaar bhejo", "spoken amount (hazaar / lakh / sau / 2k)"),
    ("rent 2k bhejo", "spoken amount (hazaar / lakh / sau / 2k)"),
    ("दो लाख भेजो", "spoken amount (hazaar / lakh / sau / 2k)"),
    ("Bombay ki flight", "colloquial city name (Bombay, Dilli...)"),
])
def test_tags(text, expected):
    assert expected in tag(ex(text, action="respond"))


@pytest.mark.parametrize("text", ["kalkatta", "Kolkata jaana hai", "dedh hazaar", "sava"])
def test_no_false_time_or_day_tags(text):
    tags = tag(ex(text, action="respond"))
    assert "kal (tomorrow / yesterday)" not in tags
    if "hazaar" in text or text == "sava":
        assert "saade / sava / paune / dedh / dhai time" not in tags


def test_implicit_city_only_when_not_named():
    call = [("search_trains", {"source": "Roorkee", "destination": "Delhi", "date": "2026-11-27"})]
    assert "implicit city (user's own)" in tag(ex("Delhi ki train dikhao", call))
    assert "implicit city (user's own)" not in tag(ex("Roorkee se Delhi ki train", call))
    assert "implicit city (user's own)" not in tag(ex("रुड़की से दिल्ली", call))
