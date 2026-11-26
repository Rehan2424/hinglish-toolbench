"""HTTP API and demo page.

    htb serve --provider local --base-url http://localhost:8080/v1   # small model via llama.cpp
    htb serve --provider groq --model openai/gpt-oss-120b            # or any hosted model

Endpoints:
    POST /api/chat    {"messages": [...], "city": "Delhi", "now": null, "execute": true}
    GET  /api/tools   tool schemas
    GET  /health
    GET  /            demo page

The model only proposes calls. They are validated against the schemas, and only valid
calls are passed to the (mock) backend.
"""

from __future__ import annotations

from importlib import resources
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import mock_backend
from .assistant import Assistant, ChatBackend
from .lexicon import CITIES
from .tools import TOOLS


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=2000)


class ChatRequest(BaseModel):
    messages: list[Turn] = Field(min_length=1, max_length=12)
    city: str = "Delhi"
    now: str | None = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$")
    execute: bool = True


def create_app(backend: ChatBackend, *, model_name: str = "model") -> FastAPI:
    app = FastAPI(title="Hinglish ToolBench", version="0.1.0",
                  description="Hinglish requests in, validated tool calls out.")
    assistant = Assistant(backend)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": model_name}

    @app.get("/api/tools")
    def tools() -> list[dict]:
        return [t.json_schema() for t in TOOLS.values()]

    @app.get("/api/meta")
    def meta() -> dict:
        return {"model": model_name, "cities": sorted(CITIES)}

    @app.post("/api/chat")
    def chat(req: ChatRequest) -> dict:
        if req.messages[-1].role != "user":
            raise HTTPException(422, "the last message must be from the user")
        try:
            result = assistant.run([m.model_dump() for m in req.messages], city=req.city,
                                   now=req.now)
        except Exception as exc:  # backend down, timeout, quota
            raise HTTPException(502, f"model backend error: {exc}") from exc
        executions = []
        if req.execute and result.action == "call":
            executions = [mock_backend.execute(c["name"], c["arguments"])
                          for c in result.tool_calls if c["valid"]]
        return {**result.to_dict(), "executions": executions}

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return resources.files("htb").joinpath("demo.html").read_text(encoding="utf-8")

    return app
