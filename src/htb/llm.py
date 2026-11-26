"""Minimal client for OpenAI-compatible chat completion APIs.

Works with Gemini, Groq, OpenRouter, OpenAI, and local servers (llama.cpp, vLLM,
Ollama) through the same ``/chat/completions`` endpoint, so swapping the teacher
or a baseline is a command-line flag rather than a code change.
"""

from __future__ import annotations

import os
import random
import re
import threading
import time
from pathlib import Path

import httpx

PROVIDERS: dict[str, tuple[str, str | None]] = {
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "local": ("http://localhost:8080/v1", None),
}

# Request fields applied by default for some model families (overridable with --extra).
# gpt-oss reasons at "medium" by default; "low" keeps answers fast and within
# per-minute token limits without hurting this task.
MODEL_DEFAULT_EXTRA: dict[str, dict] = {
    "openai/gpt-oss": {"reasoning_effort": "low"},
}


def default_extra(model: str) -> dict:
    for prefix, extra in MODEL_DEFAULT_EXTRA.items():
        if model.startswith(prefix):
            return dict(extra)
    return {}
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


def load_dotenv(path: str | Path = ".env") -> None:
    """Load KEY=VALUE lines into the environment without overriding existing vars."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


class LLMError(RuntimeError):
    pass


class QuotaExhausted(LLMError):
    """The provider's daily quota is used up; retrying today is pointless."""


# Groq says "requests per day (RPD)"; Gemini names quotas like "...PerDay...".
_DAILY_QUOTA = re.compile(r"per day|RPD|PerDay|TPD", re.IGNORECASE)


class ChatClient:
    def __init__(self, provider: str, model: str, *, base_url: str | None = None,
                 api_key: str | None = None, timeout: float = 90.0, max_retries: int = 6,
                 requests_per_minute: float | None = None,
                 extra_body: dict | None = None) -> None:
        if provider not in PROVIDERS and base_url is None:
            raise ValueError(f"unknown provider {provider!r}; pass base_url")
        default_url, key_env = PROVIDERS.get(provider, (base_url, None))
        self.base_url = (base_url or default_url or "").rstrip("/")
        self.model = model
        self.api_key = api_key or (os.environ.get(key_env) if key_env else None)
        if key_env and not self.api_key:
            raise LLMError(f"{key_env} is not set (add it to .env)")
        self.max_retries = max_retries
        self.extra_body = default_extra(model) if extra_body is None else dict(extra_body)
        self._http = httpx.Client(timeout=timeout)
        self._min_gap = 60.0 / requests_per_minute if requests_per_minute else 0.0
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def _throttle(self) -> None:
        if not self._min_gap:
            return
        with self._lock:
            now = time.monotonic()
            wait = self._next_slot - now
            self._next_slot = max(now, self._next_slot) + self._min_gap
        if wait > 0:
            time.sleep(wait)

    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.0,
             max_tokens: int = 1024, json_mode: bool = False) -> str:
        body: dict = {"model": self.model, "messages": messages, "temperature": temperature,
                      "max_tokens": max_tokens, **self.extra_body}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                r = self._http.post(f"{self.base_url}/chat/completions", json=body,
                                    headers=headers)
            except httpx.HTTPError as exc:
                if attempt == self.max_retries:
                    raise LLMError(f"request failed: {exc}") from exc
            else:
                if r.status_code == 200:
                    data = r.json()
                    try:
                        return data["choices"][0]["message"]["content"] or ""
                    except (KeyError, IndexError) as exc:
                        raise LLMError(f"unexpected response: {str(data)[:300]}") from exc
                if r.status_code == 429 and _DAILY_QUOTA.search(r.text):
                    raise QuotaExhausted(f"daily quota used up for {self.model}: {r.text[:200]}")
                if r.status_code not in RETRY_STATUS or attempt == self.max_retries:
                    raise LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
                retry_after = r.headers.get("retry-after")
                if retry_after and retry_after.replace(".", "", 1).isdigit():
                    delay = max(delay, float(retry_after))
            time.sleep(delay + random.uniform(0, 1))
            delay = min(delay * 2, 120)
        raise LLMError("unreachable")
