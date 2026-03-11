# ADR-002: Three-Tier Extraction Pipeline

**Status**: Accepted
**Date**: 2026-03-11

## Context

ctxgraph needs to extract structured entities and relationships from unstructured text. Options:

1. **LLM-only** — Send all text to an LLM (GPT-4, Llama, etc.) for extraction.
2. **Traditional NER** — Fixed entity types (PERSON, ORG, LOC) via spaCy/similar.
3. **Tiered approach** — Local ONNX models first, LLM only when needed.

## Decision

Use a **three-tier extraction pipeline** where each tier is additive and optional:

- **Tier 1**: GLiNER2 + GLiREL (ONNX, local, always-on) — $0, 2-10ms
- **Tier 2**: Coreference + dedup + enhanced temporal (local, default-on) — $0, 15-50ms
- **Tier 3**: LLM (Ollama or API, opt-in) — $0-0.05, 500-2000ms

## Rationale

- **LLM-only fails the core value prop.** ctxgraph's differentiator is zero infrastructure, zero cost. Requiring an LLM API key eliminates this.
- **Traditional NER is too limited.** Fixed entity types (PERSON, ORG, LOC) can't extract domain-specific concepts like Decision, Reason, Constraint.
- **GLiNER2 is the breakthrough.** Zero-shot extraction with custom labels, matching GPT-4o on NER benchmarks, running on CPU in <10ms. This makes Tier 1 viable as the default.
- **Progressive enhancement.** Each tier adds quality. Users who need 85% accuracy pay $0. Users who need 95% can opt into Tier 3.
- **Self-regulating escalation.** Tier 3 only fires when Tier 1 appears to miss entities (low density heuristic), so costs stay near-zero on well-structured text.

## Consequences

- **Positive**: Zero cost and offline operation by default. LLM is strictly opt-in.
- **Positive**: ~85% accuracy on semi-structured text without any external services.
- **Positive**: Graceful degradation — if ONNX models aren't downloaded, basic storage still works.
- **Negative**: Two ONNX model downloads (~350MB) required for Tier 1.
- **Negative**: Tier 1-2 struggles with highly conversational/casual text (~58% accuracy).
- **Mitigation**: Tier 3 handles the long tail. Auto-escalation ensures messy text gets LLM treatment.
