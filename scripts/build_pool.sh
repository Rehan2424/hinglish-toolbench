#!/usr/bin/env bash
# Build one data pool: sample scenarios, phrase them with the teacher, verify.
# Every step is resumable, so rerunning after a quota stop continues where it left off.
#
#   scripts/build_pool.sh <name> <n> <seed> <prefix>
#   scripts/build_pool.sh test_pool 800 2 t
#   TEACHER=gemini-3.1-flash-lite scripts/build_pool.sh train_a 3000 11 a
#
# Free-tier notes (Nov 2026): Gemini Flash-Lite allows ~500 requests/day per model;
# Gemma 4 on the Gemini API has a much larger quota but is slow (it reasons in
# plain text), so it is the primary verifier, run with many workers. Groq models
# allow ~200K tokens/day, so they serve only as fallbacks on mismatches.
set -euo pipefail

NAME=$1; N=$2; SEED=$3; PREFIX=$4
TEACHER=${TEACHER:-gemini-3.5-flash-lite}
VERIFIER=${VERIFIER:-gemma-4-31b-it}
DIR=data/$NAME
HTB=${HTB:-htb}

mkdir -p "$DIR"
[ -f "$DIR/scenarios.jsonl" ] || $HTB scenarios --n "$N" --seed "$SEED" --prefix "$PREFIX" \
    --out "$DIR/scenarios.jsonl"

$HTB generate --scenarios "$DIR/scenarios.jsonl" --out "$DIR/generated.jsonl" \
    --rejects "$DIR/rejects_gen.jsonl" --provider gemini --model "$TEACHER" \
    --rpm "${TEACHER_RPM:-12}" --workers 3 || echo "generation stopped early; verifying what exists"

$HTB verify --examples "$DIR/generated.jsonl" --out "$DIR/verified.jsonl" \
    --rejects "$DIR/rejects_verify.jsonl" --provider gemini --model "$VERIFIER" \
    --workers "${VERIFY_WORKERS:-8}" --rpm "${VERIFY_RPM:-14}" \
    --fallback groq:openai/gpt-oss-120b --fallback groq:qwen/qwen3.8-27b

$HTB stats --examples "$DIR/verified.jsonl"
