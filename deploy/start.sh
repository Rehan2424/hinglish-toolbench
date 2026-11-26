#!/bin/sh
# Start llama.cpp on localhost, wait until the model is loaded, then start the API.
set -e

/app/llama-server -hf "$MODEL_HF" --host 127.0.0.1 --port 8080 -c 4096 --jinja &

echo "loading $MODEL_HF ..."
until curl -sf http://127.0.0.1:8080/health > /dev/null; do sleep 2; done

exec /venv/bin/htb serve --provider local --model "$MODEL_NAME" \
    --host 0.0.0.0 --port "$PORT" \
    --extra '{"chat_template_kwargs": {"enable_thinking": false}}'
