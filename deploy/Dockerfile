# CPU-only image: llama.cpp serves the quantized model, FastAPI serves the API and demo.
# Works as a Hugging Face Docker Space (port 7860).
#
#   docker build -f deploy/Dockerfile -t hinglish-toolbench .
#   docker run -p 7860:7860 -e MODEL_HF=<user>/<repo>:Q4_K_M hinglish-toolbench
FROM ghcr.io/ggml-org/llama.cpp:server

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv curl \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /venv

WORKDIR /srv
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN /venv/bin/pip install --no-cache-dir ".[serve]"
COPY deploy/start.sh /start.sh

# GGUF on the Hugging Face Hub, as repo[:quant]. Downloaded on first start.
ENV MODEL_HF=Qwen/Qwen3-0.6B-GGUF:Q8_0 \
    MODEL_NAME=qwen3-0.6b \
    LLAMA_CACHE=/tmp/llama-cache \
    PORT=7860

EXPOSE 7860
ENTRYPOINT ["/bin/sh", "/start.sh"]
