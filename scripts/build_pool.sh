#!/usr/bin/env bash
# Build one data pool: sample scenarios, phrase them with the teacher, verify.
# Every step is resumable, so rerunning after a quota stop continues where it left off.
#
#   scripts/build_pool.sh <name> <n> <seed> <prefix> [primary-verifier]
#   scripts/build_pool.sh test_pool 800 2 t
#   scripts/build_pool.sh train_a 3000 11 a openai/gpt-oss-20b
set -euo pipefail

NAME=$1; N=$2; SEED=$3; PREFIX=$4
PRIMARY=${5:-openai/gpt-oss-120b}
TEACHER=${TEACHER:-gemini-3.5-flash-lite}
DIR=data/$NAME
HTB=${HTB:-htb}

mkdir -p "$DIR"
[ -f "$DIR/scenarios.jsonl" ] || $HTB scenarios --n "$N" --seed "$SEED" --prefix "$PREFIX" \
    --out "$DIR/scenarios.jsonl"

$HTB generate --scenarios "$DIR/scenarios.jsonl" --out "$DIR/generated.jsonl" \
    --rejects "$DIR/rejects_gen.jsonl" --provider gemini --model "$TEACHER" \
    --rpm "${TEACHER_RPM:-12}" --workers 3

$HTB verify --examples "$DIR/generated.jsonl" --out "$DIR/verified.jsonl" \
    --rejects "$DIR/rejects_verify.jsonl" --provider groq --model "$PRIMARY" \
    --extra '{"reasoning_effort":"low"}' --fallback-model qwen/qwen3.8-27b \
    --rpm "${VERIFY_RPM:-3.5}" --workers 2

$HTB stats --examples "$DIR/verified.jsonl"
