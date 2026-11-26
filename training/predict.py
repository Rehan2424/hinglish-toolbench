"""Generate predictions with a base model or a LoRA adapter.

Reads chat JSONL (from `htb export-sft`), drops the final assistant turn, and writes
``{"id", "output", "latency_s"}`` rows that `htb eval` scores directly.

    python training/predict.py --model Qwen/Qwen3-1.7B --adapter outputs/qwen3-1.7b-lora/adapter \
        --data data/sft/test.jsonl --out preds/qwen3-1.7b-lora.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import dtype_kwarg, prompt_text, read_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.adapter or args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cuda = torch.cuda.is_available()
    dtype = (torch.bfloat16 if cuda and torch.cuda.is_bf16_supported()
             else torch.float16 if cuda else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(args.model, **dtype_kwarg(dtype),
                                                 device_map="auto" if cuda else None)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.eval()

    rows = read_jsonl(args.data)[: args.limit]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    done = {r["id"] for r in read_jsonl(args.out)} if Path(args.out).exists() else set()
    rows = [r for r in rows if r["id"] not in done]

    with open(args.out, "a", encoding="utf-8") as f:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start: start + args.batch_size]
            prompts = [prompt_text(tok, r["messages"][:-1]) for r in batch]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            t0 = time.time()
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            per_example = (time.time() - t0) / len(batch)
            new_tokens = gen[:, enc["input_ids"].shape[1]:]
            for r, ids in zip(batch, new_tokens, strict=True):
                text = tok.decode(ids, skip_special_tokens=True).strip()
                f.write(json.dumps({"id": r["id"], "output": text,
                                    "latency_s": round(per_example, 3)}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{start + len(batch)}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
