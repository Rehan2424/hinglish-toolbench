import json
import random

import pytest

from htb import mock_backend
from htb.assistant import Assistant
from htb.gen.sampler import SAMPLERS, sample_context
from htb.tools import TOOLS


class Canned:
    """Backend that returns a fixed reply and records the prompt it was sent."""

    def __init__(self, reply):
        self.reply = reply
        self.prompt = None

    def chat(self, messages, **kw):
        self.prompt = messages
        return self.reply


CALL = json.dumps({"action": "call", "tool_calls": [
    {"name": "transfer_money", "arguments": {"recipient": "Priya", "amount": 1500, "mode": "upi"}},
    {"name": "transfer_money", "arguments": {"recipient": "Amit", "amount": "lots"}},
]})


def test_assistant_validates_calls_and_builds_the_prompt():
    backend = Canned("<think>hmm</think>" + CALL)
    r = Assistant(backend).run([{"role": "user", "content": "Priya ko 1500 bhejo"}],
                               city="Pune", now="2026-11-26T10:00")
    assert r.action == "call" and not r.ok
    good, bad = r.tool_calls
    assert good == {"name": "transfer_money", "valid": True,
                    "arguments": {"recipient": "Priya", "amount": 1500.0}}  # default dropped
    assert bad["valid"] is False and len(r.errors) == 1
    system = backend.prompt[0]["content"]
    assert "2026-11-26T10:00 (Thursday)" in system and "User's city: Pune" in system


def test_plain_text_assistant_turns_are_shown_to_the_model_as_json():
    backend = Canned(CALL)
    Assistant(backend).run([
        {"role": "user", "content": "Priya ko paise bhejo"},
        {"role": "assistant", "content": "Kitne paise?"},
        {"role": "user", "content": "1500"},
    ], now="2026-11-26T10:00")
    turn = backend.prompt[2]["content"]
    assert json.loads(turn) == {"action": "clarify", "message": "Kitne paise?"}
    already_json = json.dumps({"action": "clarify", "message": "Kitne?"})
    Assistant(backend).run([{"role": "user", "content": "a"},
                            {"role": "assistant", "content": already_json},
                            {"role": "user", "content": "b"}], now="2026-11-26T10:00")
    assert backend.prompt[2]["content"] == already_json


def test_assistant_reports_unparseable_output():
    r = Assistant(Canned("sorry, no idea")).run([{"role": "user", "content": "x"}])
    assert r.action == "error" and not r.ok


def test_mock_backend_covers_every_tool_deterministically():
    rng = random.Random(0)
    for name, sampler in SAMPLERS.items():
        for _ in range(20):
            args = sampler(rng, sample_context(rng)).args
            a, b = mock_backend.execute(name, args), mock_backend.execute(name, args)
            assert a == b and a["summary"]
    assert set(SAMPLERS) == set(TOOLS)


def test_http_api():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from htb.server import create_app

    client = TestClient(create_app(Canned(CALL), model_name="fake"))
    assert client.get("/health").json()["model"] == "fake"
    assert len(client.get("/api/tools").json()) == len(TOOLS)
    assert "Hinglish ToolBench" in client.get("/").text

    r = client.post("/api/chat", json={"messages": [{"role": "user", "content": "x"}],
                                       "now": "2026-11-26T10:00"}).json()
    assert r["action"] == "call" and len(r["executions"]) == 1  # the invalid call is not run
    assert "Priya" in r["executions"][0]["summary"]

    bad = client.post("/api/chat", json={"messages": [{"role": "assistant", "content": "x"}]})
    assert bad.status_code == 422


def test_mcp_server_registers_tools():
    pytest.importorskip("mcp")
    import asyncio

    from htb.mcp_server import build_server

    server = build_server(Canned(CALL))
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert names == {"understand_request", "run_mock_tool"}
