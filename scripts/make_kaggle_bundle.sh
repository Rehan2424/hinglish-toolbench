#!/usr/bin/env bash
# Package chat-format data and the training scripts into one folder to upload as a
# Kaggle dataset (Kaggle cannot pip-install from a private repo).
#
#   scripts/make_kaggle_bundle.sh data/splits data/test.jsonl
set -euo pipefail

SPLITS=$1          # folder with train.jsonl and dev.jsonl (htb split output)
TEST=${2:-}        # hand-verified test set (htb apply-review output), optional
OUT=${OUT:-dist/kaggle_bundle}
HTB=${HTB:-htb}

rm -rf "$OUT" && mkdir -p "$OUT"
$HTB export-sft --examples "$SPLITS/train.jsonl" --out "$OUT/train.jsonl"
$HTB export-sft --examples "$SPLITS/dev.jsonl" --out "$OUT/dev.jsonl"
[ -n "$TEST" ] && $HTB export-sft --examples "$TEST" --out "$OUT/test.jsonl"
cp training/common.py training/finetune.py training/predict.py "$OUT/"
echo "bundle ready in $OUT; upload it as a Kaggle dataset named hinglish-toolbench-bundle"
