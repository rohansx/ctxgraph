# ADR-006: Unified GLiNER2 over Separate GLiNER2 + GLiREL

**Status**: Accepted
**Date**: 2026-03-11

## Context

The original design used two separate ONNX models:
- **GLiNER2** for entity extraction (NER)
- **GLiREL** for relationship extraction (RE)

This meant ~350MB in downloads, two inference passes per episode, and more complex pipeline code.

The 2025 EMNLP version of GLiNER2 now supports unified entity + relationship extraction with structured JSON output in a single forward pass.

## Decision

Use **unified GLiNER2** as the default extraction model. Retain **GLiREL as an optional precision mode** for users who need better relation quality on complex text.

## Rationale

| Factor | Two models (GLiNER2 + GLiREL) | Unified GLiNER2 |
|---|---|---|
| Download size | ~350MB | ~200MB |
| Inference passes | 2 | 1 |
| Pipeline complexity | Entity extraction → relation extraction | Single call |
| Crate structure | More complex orchestration | Simpler |
| Relation quality | Higher (dedicated model) | Good (sufficient for 90%+ of use cases) |

The unified approach:
- Saves ~150MB in user downloads
- Cuts extraction latency roughly in half
- Simplifies the pipeline code significantly
- Merges what was v0.2 + v0.3 in the original plan into a single version

GLiREL is not removed — it's available as `extraction.precision_mode = true` for users who need it. But the default path is simpler and faster.

## Consequences

- **Positive**: Faster downloads, simpler pipeline, faster extraction, merged build phases.
- **Positive**: Users only download ~200MB instead of ~350MB on first run.
- **Negative**: Relation quality may be slightly lower than dedicated GLiREL on complex text.
- **Mitigation**: GLiREL available as optional precision mode. Monitor extraction benchmark F1 — if relation quality drops below 0.75, reconsider.
