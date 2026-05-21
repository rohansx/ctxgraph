# Extraction Benchmark (v0.6.0, ARCHIVED)

> ⚠️ **DEPRECATED — superseded by `../BENCHMARKS.md`.**
> This is the original 50-tech-episode F1 comparison vs. Graphiti + GPT-4o (combined F1 0.800 vs. 0.337). Tech text is ctxgraph's home turf; the canonical headline now uses the 29-episode cross-domain v2 fixture with same-LLM-vs-Graphiti scoring. The 0.800 / 0.337 numbers below remain valid; they're just **secondary** evidence now, not the headline.
>
> **Current authoritative benchmark: `../BENCHMARKS.md`**

---

# Original content below

ctxgraph's extraction pipeline is benchmarked against 50 gold-labeled episodes covering real-world architectural decision records (ADRs), incident reports, PR descriptions, and migration plans.

## Benchmark Setup

- **50 episodes**, each with hand-labeled expected entities and relations
- **10 entity types**: Person, Component, Service, Language, Database, Infrastructure, Decision, Constraint, Metric, Pattern
- **9 relation types**: chose, rejected, replaced, depends_on, fixed, introduced, deprecated, caused, constrained_by
- **Scoring**: Per-episode entity F1 and relation F1 (exact match on name/type), averaged across all episodes
- **Combined F1** = (avg entity F1 + avg relation F1) / 2

### Running the Benchmark

```bash
# Run with ONNX models (requires GLiNER model cached locally)
CTXGRAPH_MODELS_DIR=~/.cache/ctxgraph/models \
  cargo test --package ctxgraph-extract --test benchmark_test -- --ignored
```

## Results: ctxgraph v0.6.0

| Metric | Score |
|--------|:-----:|
| Avg Entity F1 | 0.837 |
| Avg Relation F1 | 0.763 |
| **Combined F1** | **0.800** |

All extraction runs **fully locally** with zero API calls, using:
- GLiNER Large v2.1 (INT8 ONNX) for named entity recognition
- Heuristic keyword + proximity scoring for relation extraction
- Schema-typed entity constraints for candidate filtering

## Comparison with Graphiti v0.28.2

We ran the same 50 episodes through [Graphiti](https://github.com/getzep/graphiti) (by Zep AI), a popular knowledge graph framework that uses LLM-based extraction.

### Setup Differences

| | ctxgraph | Graphiti |
|---|---|---|
| **Extraction method** | Local ONNX models + heuristics | OpenAI gpt-4o API calls |
| **Entity typing** | Schema-typed (10 fixed types) | Free-form names |
| **Relation typing** | Schema-typed (9 fixed types) | Free-form verbs (e.g., `CONNECTS_TO`, `WAS_REWRITTEN_FROM`) |
| **Graph storage** | SQLite (embedded) | Neo4j (external service) |
| **Dependencies** | None (fully local) | Neo4j + OpenAI API key |
| **Internet required** | No | Yes |

### Head-to-Head Results

| Metric | ctxgraph (local) | Graphiti (gpt-4o) | Graphiti (mapped)\* |
|--------|:----:|:----:|:----:|
| Avg Entity F1 | **0.837** | 0.570 | 0.570 |
| Avg Relation F1 | **0.763** | 0.000 | 0.104 |
| **Combined F1** | **0.800** | 0.285 | 0.337 |
| API calls | **0** | ~200+ | ~200+ |
| Cost per run | **$0** | ~$2-5 | ~$2-5 |
| Latency (50 eps) | **~2s** | ~8min | ~8min |

\* "Mapped" = Graphiti's free-form relation names mapped to ctxgraph's taxonomy using generous keyword heuristics (e.g., `CHOSE_FOR` → `chose`, `MIGRATED_TO` → `replaced`). This gives Graphiti maximum benefit of the doubt.

### Why Graphiti Scores Lower

**Entity extraction (0.570 vs 0.837):**
Graphiti extracts multi-word descriptive phrases instead of canonical names. For example:
- Gold: `"Postgres"` → Graphiti: `"primary Postgres cluster"`
- Gold: `"SOAP endpoint"` → Graphiti: `"legacy SOAP endpoint in UserService"`
- Gold: `"p99 latency"` → Graphiti: `"100ms SLA at the p99 level"`

These are semantically correct but don't match the expected canonical entity names, reducing precision.

**Relation extraction (0.000 raw, 0.104 mapped vs 0.763):**
1. **Incompatible output format**: Graphiti uses free-form relation verbs (`CONNECTS_TO`, `WAS_REWRITTEN_FROM`, `CHOSE_FOR`) while ctxgraph uses a fixed schema. Even with generous keyword mapping, only 10/50 episodes produce any matching relations.
2. **Different decomposition**: Graphiti decomposes the same information differently. For example, "Migrate from Redis to Postgres" becomes `(AuthService, CONNECTS_TO, primary Postgres cluster)` instead of `(Postgres, replaced, Redis)` + `(AuthService, depends_on, Postgres)`.
3. **Entity name mismatch cascades**: Since Graphiti's entity names differ from gold labels, even correctly-typed relations fail to match because the head/tail names are wrong.
4. **UUID leakage**: Some Graphiti edges reference nodes by UUID rather than name (e.g., `('SearchService', 'REPLACED_BY', 'e173b631-...')`), making those relations unmatchable.

### What Graphiti Does Well

Graphiti is designed for a different use case:
- **Temporal knowledge graphs**: Tracks how facts change over time (valid_at, expired_at)
- **Cross-episode deduplication**: Merges entities across episodes into a unified graph
- **Open-domain exploration**: Free-form facts are useful for Q&A and natural language search
- **Rich edge metadata**: Each edge stores a full "fact" sentence, not just a type label

ctxgraph is optimized for **structured, schema-typed extraction** from technical documents where you need precise entity types and relation types for downstream querying — without any API calls or internet connectivity.

### Reproducing the Comparison

```bash
# 1. Start Neo4j
docker run -d --name neo4j-graphiti -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/graphiti123 -e NEO4J_PLUGINS='["apoc"]' \
  neo4j:5.26-community

# 2. Install Graphiti
pip install graphiti-core

# 3. Run Graphiti benchmark
OPENAI_API_KEY=sk-... python scripts/benchmark_graphiti.py

# 4. Run comparison with semantic mapping
python scripts/compare_benchmarks.py
```

Raw results are saved in `scripts/graphiti_benchmark_results.json` and `scripts/benchmark_comparison.json`.
