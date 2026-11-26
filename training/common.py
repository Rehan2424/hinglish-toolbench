"""Prompt/target construction shared by training and inference.

Both sides must build the prompt identically, otherwise the model is evaluated on a
format it was not trained on. Kept dependency-free apart from a tokenizer.
"""

from __future__ import annotations

import json
from pathlib import Path

# Passed to chat templates that support it (Qwen3: answer directly, no <think> block).
TEMPLATE_KWARGS = {"enable_thinking": False}


def dtype_kwarg(dtype) -> dict:
    """transformers 5 renamed ``torch_dtype`` to ``dtype``; support both."""
    import transformers

    major = int(transformers.__version__.split(".")[0])
    return {"dtype": dtype} if major >= 5 else {"torch_dtype": dtype}


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def prompt_text(tokenizer, messages: list[dict]) -> str:
    """Chat-formatted prompt ending where the assistant's answer starts."""
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         **TEMPLATE_KWARGS)


def split_prompt_completion(tokenizer, messages: list[dict]) -> tuple[str, str]:
    """Return (prompt, completion) for a conversation whose last turn is the target.

    The completion is taken from the chat template itself so it carries the model's
    own end-of-turn markers. If the template rewrites earlier turns (some do), fall
    back to the bare target plus the EOS token.
    """
    prompt = prompt_text(tokenizer, messages[:-1])
    full = tokenizer.apply_chat_template(messages, tokenize=False, **TEMPLATE_KWARGS)
    target = messages[-1]["content"]
    if full.startswith(prompt) and target in full[len(prompt):]:
        completion = full[len(prompt):].rstrip("\n")
    else:
        completion = target + (tokenizer.eos_token or "")
    return prompt, completion
