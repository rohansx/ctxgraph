#!/usr/bin/env bash
# Download ONNX models for ctxgraph extraction pipeline.
#
# Downloads:
#   1. GLiNER Large v2.1 INT8 (NER) — 653MB
#   2. GLiNER Large v2.1 tokenizer
#
# Models are saved to ~/.cache/ctxgraph/models/

set -euo pipefail

CACHE_DIR="${CTXGRAPH_MODELS_DIR:-$HOME/.cache/ctxgraph/models}"
GLINER_DIR="$CACHE_DIR/gliner_large-v2.1"
ONNX_DIR="$GLINER_DIR/onnx"

HF_BASE="https://huggingface.co/onnx-community/gliner_large-v2.1/resolve/main"

mkdir -p "$ONNX_DIR"

echo "=== ctxgraph model download ==="
echo "Cache dir: $CACHE_DIR"
echo

# Download INT8 model
MODEL_PATH="$ONNX_DIR/model_int8.onnx"
if [ -f "$MODEL_PATH" ]; then
    echo "[skip] model_int8.onnx already exists ($(du -h "$MODEL_PATH" | cut -f1))"
else
    echo "[download] GLiNER Large v2.1 INT8 (~653MB)..."
    curl -L --progress-bar "$HF_BASE/onnx/model_int8.onnx" -o "$MODEL_PATH"
    echo "[done] $(du -h "$MODEL_PATH" | cut -f1)"
fi

# Download tokenizer
TOKENIZER_PATH="$GLINER_DIR/tokenizer.json"
if [ -f "$TOKENIZER_PATH" ]; then
    echo "[skip] tokenizer.json already exists"
else
    echo "[download] tokenizer.json..."
    curl -L --progress-bar "$HF_BASE/tokenizer.json" -o "$TOKENIZER_PATH"
    echo "[done] tokenizer.json"
fi

echo
echo "=== Download complete ==="
echo "NER model:  $MODEL_PATH"
echo "Tokenizer:  $TOKENIZER_PATH"
echo
echo "To enable relation extraction, also run:"
echo "  pip install gliner onnx && python scripts/convert_model.py"
