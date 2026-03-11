# ADR-005: Reciprocal Rank Fusion for Search

**Status**: Accepted
**Date**: 2026-03-11

## Context

ctxgraph has three search modes: FTS5 keyword, semantic embedding, and graph traversal. These need to be combined into a single ranked result list.

Options:

1. **Linear combination** — Weighted sum of normalized scores. Requires score normalization across modes.
2. **Learning to rank** — ML model trained on relevance judgments. Requires training data.
3. **Reciprocal Rank Fusion (RRF)** — Rank-based fusion that doesn't require score normalization.

## Decision

Use **Reciprocal Rank Fusion** with constant k=60.

```
rrf_score(d) = Σ 1/(k + rank_i(d))  for each mode i where d appears
```

## Rationale

- **No score normalization needed.** FTS5 BM25 scores, cosine similarities, and graph depth are on completely different scales. RRF uses ranks, not scores, so normalization is unnecessary.
- **No training data needed.** Unlike learning-to-rank, RRF works out of the box.
- **Proven in IR research.** RRF is widely used in hybrid search systems (Elasticsearch, Vespa, Weaviate all support it).
- **Simple to implement.** ~20 lines of code.
- **Naturally rewards multi-mode hits.** A result appearing in all three modes gets a higher fused score than one appearing in only one mode — which is exactly the desired behavior.

## Consequences

- **Positive**: Simple, robust, no tuning required.
- **Positive**: Results appearing across multiple modes are naturally boosted.
- **Negative**: Equal weighting of all modes. Can't prioritize e.g. graph results over FTS without modification.
- **Mitigation**: Can add per-mode weight multipliers if needed post-v1.0. For now, equal weighting works well.
