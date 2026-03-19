#!/usr/bin/env python3
"""Convert GLiNER multitask model to ONNX format for ctxgraph.

This script converts the `knowledgator/gliner-multitask-large-v0.5` model
from PyTorch to ONNX format, ready for use with gline-rs.

Usage:
    pip install gliner onnx
    python scripts/convert_model.py

Output:
    ~/.cache/ctxgraph/models/gliner-multitask-large-v0.5/onnx/model.onnx
    ~/.cache/ctxgraph/models/gliner-multitask-large-v0.5/tokenizer.json
"""

import os
import shutil
from pathlib import Path

def main():
    try:
        from gliner import GLiNER
    except ImportError:
        print("Error: gliner package not installed.")
        print("Run: pip install gliner onnx")
        return 1

    model_name = "knowledgator/gliner-multitask-large-v0.5"
    cache_dir = Path.home() / ".cache" / "ctxgraph" / "models" / "gliner-multitask-large-v0.5"
    onnx_dir = cache_dir / "onnx"

    print(f"Loading model: {model_name}")
    model = GLiNER.from_pretrained(model_name)

    print(f"Converting to ONNX...")
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = str(onnx_dir / "model.onnx")
    model.to_onnx(onnx_path)
    print(f"Saved ONNX model to: {onnx_path}")

    # Copy tokenizer
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    tokenizer_src = None

    # Try to find tokenizer.json in HF cache
    for d in hf_cache.glob("models--knowledgator--gliner-multitask-large-v0.5/snapshots/*/tokenizer.json"):
        tokenizer_src = d
        break

    if tokenizer_src and tokenizer_src.exists():
        tokenizer_dst = cache_dir / "tokenizer.json"
        shutil.copy2(tokenizer_src, tokenizer_dst)
        print(f"Copied tokenizer to: {tokenizer_dst}")
    else:
        print("Warning: tokenizer.json not found in HF cache.")
        print(f"Please manually copy tokenizer.json to: {cache_dir / 'tokenizer.json'}")

    print()
    print("Done! The model is ready for ctxgraph relation extraction.")
    print(f"Model dir: {cache_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
