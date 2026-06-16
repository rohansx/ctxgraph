#!/usr/bin/env bash
# Run llama.cpp's `llama-server` as an OpenAI-compatible backend for ctxgraph.
#
# ctxgraph's LlmExtractor speaks plain OpenAI-compatible HTTP, so anything that
# exposes POST /v1/chat/completions works as a drop-in backend. This serves a
# local GGUF with llama-server and prints the env vars that point ctxgraph at it.
#
# By default it REUSES the GGUF that Ollama already downloaded (no second 7GB
# copy on disk): it resolves the blob path from an Ollama model tag via
# `ollama show <tag> --modelfile`. Override with GGUF=/path/to/model.gguf.
#
# Prereq: a `llama-server` binary on PATH. Build one (CUDA) with:
#   git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/.local/src/llama.cpp
#   cd ~/.local/src/llama.cpp
#   CUDACXX=/opt/cuda/bin/nvcc cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
#     -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DLLAMA_CURL=OFF
#   cmake --build build --target llama-server -j"$(nproc)"
#   cp build/bin/llama-server ~/.local/bin/
#
# Usage:
#   scripts/run-llama-server.sh                                  # default coder model
#   OLLAMA_MODEL=qwen2.5:7b-instruct-q4_K_M scripts/run-llama-server.sh
#   GGUF=/path/to/model.gguf NGL=33 PORT=8080 scripts/run-llama-server.sh
set -euo pipefail

PORT="${PORT:-8080}"
HOST="${HOST:-127.0.0.1}"
NGL="${NGL:-12}"     # GPU layers to offload. Tune to FREE VRAM, not total:
                     # ~12 layers of a 12B Q4_K_M fits when a desktop already
                     # uses ~2GB of a 6GB GPU. Raise it when more VRAM is free.
CTX="${CTX:-2048}"   # context window (KV cache grows with this)
OLLAMA_MODEL="${OLLAMA_MODEL:-hf.co/yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF:Q4_K_M}"

# --- resolve the GGUF path -------------------------------------------------
if [ -n "${GGUF:-}" ]; then
  MODEL_PATH="$GGUF"
else
  echo "Resolving GGUF for ollama model: $OLLAMA_MODEL" >&2
  MODEL_PATH="$(ollama show "$OLLAMA_MODEL" --modelfile | awk '/^FROM /{print $2; exit}')"
fi
[ -r "$MODEL_PATH" ] || { echo "ERROR: GGUF not readable: $MODEL_PATH" >&2; exit 1; }
command -v llama-server >/dev/null 2>&1 || {
  echo "ERROR: llama-server not on PATH — build it first (see header)." >&2; exit 1; }

echo "Serving: $MODEL_PATH" >&2
echo >&2
echo "Point ctxgraph at this backend (in another shell):" >&2
cat >&2 <<EOF
  export CTXGRAPH_LLM_KEY=llamacpp
  export CTXGRAPH_LLM_URL=http://${HOST}:${PORT}/v1/chat/completions
  export CTXGRAPH_LLM_MODEL=local
  export CTXGRAPH_LLM_TIMEOUT=600
EOF
echo >&2

# --jinja uses the GGUF's embedded chat template; --alias fixes the model name
# the OpenAI endpoint reports (so CTXGRAPH_LLM_MODEL=local lines up).
exec llama-server \
  -m "$MODEL_PATH" \
  --host "$HOST" --port "$PORT" \
  -ngl "$NGL" -c "$CTX" \
  --jinja \
  --alias local
