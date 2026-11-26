"""LoRA fine-tuning on Hinglish ToolBench chat data.

Fits a single 16 GB GPU (Kaggle T4/P100, Colab T4) for models up to ~2B parameters;
add --load-in-4bit (QLoRA) for larger ones. Loss is computed on the assistant's
answer only, never on the system prompt or user turns.

    python training/finetune.py --model Qwen/Qwen3-1.7B \
        --train data/sft/train.jsonl --dev data/sft/dev.jsonl --out outputs/qwen3-1.7b-lora

Smoke test on CPU with a tiny model:

    python training/finetune.py --model HuggingFaceTB/SmolLM2-135M-Instruct \
        --train data/sft/train.jsonl --dev data/sft/dev.jsonl --out outputs/smoke \
        --max-steps 2 --max-train 16 --max-dev 4
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import dtype_kwarg, read_jsonl, split_prompt_completion  # noqa: E402

IGNORE = -100


def encode(tokenizer, rows: list[dict], max_len: int) -> tuple[list[dict], int]:
    """Tokenize to input_ids/labels with the prompt masked. Over-long rows are dropped
    rather than truncated, since a truncated target teaches broken JSON."""
    out, dropped = [], 0
    for row in rows:
        prompt, completion = split_prompt_completion(tokenizer, row["messages"])
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
        if len(p_ids) + len(c_ids) > max_len:
            dropped += 1
            continue
        out.append({"input_ids": p_ids + c_ids, "labels": [IGNORE] * len(p_ids) + c_ids})
    return out, dropped


class Collator:
    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict:
        width = max(len(b["input_ids"]) for b in batch)
        ids, labels, mask = [], [], []
        for b in batch:
            pad = width - len(b["input_ids"])
            ids.append(b["input_ids"] + [self.pad_id] * pad)
            labels.append(b["labels"] + [IGNORE] * pad)
            mask.append([1] * len(b["input_ids"]) + [0] * pad)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(mask)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=2560)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--max-train", type=int, default=None, help="subsample (ablations, smoke)")
    ap.add_argument("--max-dev", type=int, default=None)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--wandb-project", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_rows, dev_rows = read_jsonl(args.train), read_jsonl(args.dev)
    if args.max_train:
        train_rows = random.sample(train_rows, min(args.max_train, len(train_rows)))
    if args.max_dev:
        dev_rows = dev_rows[: args.max_dev]
    train, dropped_t = encode(tokenizer, train_rows, args.max_len)
    dev, dropped_d = encode(tokenizer, dev_rows, args.max_len)
    lengths = sorted(len(x["input_ids"]) for x in train)
    print(f"train {len(train)} (dropped {dropped_t} over {args.max_len} tokens), "
          f"dev {len(dev)} (dropped {dropped_d}); median length {lengths[len(lengths) // 2]}, "
          f"max {lengths[-1]}")

    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else (torch.float16 if cuda else torch.float32)
    quant = None
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, **dtype_kwarg(dtype),
                                                 quantization_config=quant,
                                                 device_map="auto" if cuda else None)
    if quant is not None:
        model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules="all-linear", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    steps_per_epoch = math.ceil(len(train) / (args.batch_size * args.grad_accum))
    total_steps = args.max_steps if args.max_steps > 0 else math.ceil(steps_per_epoch * args.epochs)
    eval_every = max(1, min(steps_per_epoch // 4, total_steps))
    # Argument names differ between transformers 4.x and 5.x; pass whichever exists.
    accepted = inspect.signature(TrainingArguments.__init__).parameters
    compat = {}
    if "group_by_length" in accepted:
        compat["group_by_length"] = True
    elif "train_sampling_strategy" in accepted:
        compat["train_sampling_strategy"] = "group_by_length"
    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, round(0.03 * total_steps)),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        eval_strategy="steps",
        eval_steps=eval_every,
        save_strategy="steps",
        save_steps=eval_every,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=max(1, eval_every // 5),
        bf16=bf16,
        fp16=cuda and not bf16,
        report_to=["wandb"] if args.wandb_project else [],
        run_name=Path(args.out).name,
        seed=args.seed,
        remove_unused_columns=False,
        **compat,
    )
    if args.wandb_project:
        import os
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    trainer = Trainer(model=model, args=targs, train_dataset=train, eval_dataset=dev,
                      data_collator=Collator(tokenizer.pad_token_id))
    t0 = time.time()
    trainer.train()
    minutes = (time.time() - t0) / 60
    metrics = trainer.evaluate()

    adapter_dir = Path(args.out) / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    summary = {"base_model": args.model, "train_examples": len(train), "dev_examples": len(dev),
               "train_minutes": round(minutes, 1), "eval_loss": metrics.get("eval_loss"),
               "args": vars(args)}
    (Path(args.out) / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
