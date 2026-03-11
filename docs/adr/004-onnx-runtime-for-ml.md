# ADR-004: ONNX Runtime for ML Inference

**Status**: Accepted
**Date**: 2026-03-11

## Context

ctxgraph runs three ML models locally: GLiNER2 (NER), GLiREL (RE), and MiniLM (embeddings). Options for inference:

1. **PyTorch via FFI** — Call Python from Rust. Maximum model compatibility.
2. **candle (Rust-native)** — Pure Rust ML framework from HuggingFace.
3. **ONNX Runtime** — Cross-platform inference engine with Rust bindings (`ort` crate).

## Decision

Use **ONNX Runtime** via the `ort` Rust crate for all ML inference.

## Rationale

- **No Python dependency.** PyTorch FFI would require a Python runtime, violating the "single binary" constraint.
- **candle is immature for these models.** GLiNER2 and GLiREL have custom architectures that aren't yet well-supported in candle. ONNX export exists and is tested.
- **ONNX Runtime is production-grade.** Microsoft-backed, optimized for CPU, supports INT8 quantization out of the box.
- **Cross-platform.** Works on Linux, macOS, Windows without modification.
- **The `ort` crate is mature.** v2+ with good Rust ergonomics, auto-downloads ONNX Runtime binaries.

## Consequences

- **Positive**: Single Rust binary, no Python, no GPU required.
- **Positive**: INT8 quantization reduces model sizes (~200MB for GLiNER2) and improves CPU inference speed.
- **Positive**: Well-tested inference path for standard model architectures.
- **Negative**: ONNX export may not support all model features. GLiNER2 and GLiREL export must be validated.
- **Negative**: `ort` downloads ONNX Runtime shared libraries (~50MB). Increases first-build time.
- **Mitigation**: Pre-built binaries include ONNX Runtime. Models are converted and tested during development.
