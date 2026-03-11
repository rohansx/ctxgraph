# ctxgraph — Roadmap

---

## Timeline

```
        Week 1-2         Week 3-4          Week 5         Week 6
      ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
      │  v0.1    │    │  v0.2    │    │  v0.3    │    │  v0.4    │
      │  Core    │───▶│ GLiNER2  │───▶│ MCP +    │───▶│ Tier 2 + │
      │  Engine  │    │ Extract  │    │ Search   │    │ Git Watch│
      │          │    │          │    │          │    │          │
      │  DONE    │    │ HARD     │    │ DEMO     │    │ LAUNCH   │
      └──────────┘    └──────────┘    └──────────┘    └──────────┘
           │                                               │
           │                                               │
           │         ── open source, HN post, get feedback ─
           │                                               │
        Week 7-8          Week 9
      ┌──────────┐    ┌──────────┐
      │  v0.5    │    │  v1.0    │
      │ Tier 3 + │───▶│Production│
      │ Ingest   │    │ Ready    │
      │          │    │          │
      │ USER-    │    │ SHIP     │
      │ GUIDED   │    │          │
      └──────────┘    └──────────┘
```

Estimated total: **~9 weeks** (solo, focused).

**Key change from original 10-phase plan**: Compressed from 10 versions to 6. GLiNER2's unified model (entities + relations in one pass) eliminates a full version. Search and MCP are merged because MCP without search is useless. Real user feedback after v0.4 guides the remaining work.

---

## Version Detail

### v0.1 — Core Engine (DONE)

| Deliverable | Description | Status |
|---|---|---|
| `ctxgraph-core` crate | Types, SQLite storage, FTS5, bi-temporal logic | Done |
| `ctxgraph-cli` crate | CLI: init, log, query, entities, decisions, stats | Done |
| 24 tests | Episode/Entity/Edge CRUD, FTS5, traversal, temporal | Done |

**Shipped**: Working context graph that stores and retrieves episodes. No extraction — structured input only. Storage and query model proven.

---

### v0.2 — GLiNER2 Unified Extraction (~2 weeks)

| Deliverable | Description |
|---|---|
| `ctxgraph-extract` crate | ONNX-based extraction pipeline |
| Unified GLiNER2 | Entity + relation extraction in one model pass |
| Temporal heuristics | 5-layer date parser |
| Extraction benchmark | 50 annotated episodes, F1 ≥ 0.80 |
| Model management | Download, cache, verify, license check |

**Key milestone**: `ctxgraph log "Chose Postgres over SQLite for billing"` automatically extracts entities (Postgres, SQLite, billing) AND relations (chose, rejected) in a single ~10ms inference pass.

**Risk**: Medium-High. This is the hardest phase. ONNX integration with GLiNER2's specific tensor format requires careful engineering. The `ort` crate works but has rough edges with custom architectures. Budget 2 full weeks.

**What could go wrong**:
- GLiNER2 ONNX export doesn't exist or doesn't work → fallback to separate GLiNER2 (entities only) + GLiREL (relations), reverting to the two-model approach
- Extraction quality < 0.80 F1 → investigate, tune threshold, consider different checkpoint
- `ort` crate issues with GLiNER2 architecture → file issue upstream, work around

---

### v0.3 — MCP Server + Search (~1 week)

| Deliverable | Description |
|---|---|
| `ctxgraph-embed` crate | all-MiniLM-L6-v2 ONNX for 384-dim embeddings |
| `ctxgraph-mcp` crate | MCP server with stdio transport |
| RRF fusion | FTS5 + semantic + graph traversal merged |
| 5 MCP tools | add_episode, search, get_decision, traverse, find_precedents |

**Key milestone**: DEMO READY. Can show ctxgraph working as memory for Claude Desktop/Cursor. This is when you write the "SQLite for context graphs" blog post.

**Risk**: Low. MCP is well-documented. Embedding model is straightforward. RRF is ~20 lines of code.

---

### v0.4 — Tier 2 + Git Watch (~1 week)

| Deliverable | Description |
|---|---|
| Coreference resolution | Rule-based pronoun → entity mapping |
| Fuzzy entity dedup | Jaro-Winkler + alias groups |
| `ctxgraph watch --git` | Auto-capture git commits as episodes |

**Key milestone**: LAUNCH READY. Extraction quality hits ~90%. Passive capture solves ingestion friction. Open source, post on HN, get real users.

**Risk**: Low. All techniques well-understood. Git watch is simple subprocess + episode creation.

**Why git watch matters**: This solves the #1 tool adoption killer. People stop manually logging by day 3. `ctxgraph watch --git` captures context passively from something they're already doing (committing code). Low effort, high capture rate.

---

### v0.5 — Tier 3 + Bulk Ingest (~2 weeks, user-guided)

| Deliverable | Description |
|---|---|
| LLM provider abstraction | Ollama + OpenAI-compatible APIs |
| Contradiction detection | Invalidate conflicting edges via LLM |
| Community summarization | Cluster-level summaries |
| Bulk ingest | JSONL/CSV import, stdin piping |
| Built-in schemas | default, developer, support, finance |
| Export | JSON/CSV graph export |

**Key milestone**: Handles messy, unstructured text. Quality hits ~93-95% with Tier 3.

**Risk**: Medium. LLM output parsing is unpredictable. Mitigated by structured JSON mode and fallback to Tier 1.

**What to build here depends on user feedback from the v0.4 launch.** If nobody asks for Tier 3 but everyone wants bulk ingest, prioritize accordingly. Don't build features in a vacuum.

---

### v1.0 — Production Ready (~1 week)

| Deliverable | Description |
|---|---|
| Benchmarks | criterion at 1K/10K/100K scale |
| Documentation | Rustdoc, user guide, MCP setup guide |
| Pre-built binaries | Linux, macOS, Windows via CI |
| crates.io publish | All crates published |

**Key milestone**: Ship it. Stable APIs, full docs, pre-built binaries.

---

## Dependency Graph

```
v0.1 (Core) ── DONE
 │
 └──▶ v0.2 (GLiNER2 Unified Extraction)
       │
       └──▶ v0.3 (MCP + Search) ── DEMO MILESTONE
             │
             └──▶ v0.4 (Tier 2 + Git Watch) ── LAUNCH MILESTONE
                   │
                   └──▶ v0.5 (Tier 3 + Ingest) ── user-guided
                         │
                         └──▶ v1.0 (Production)
```

Linear dependency chain. Each version builds on the previous. No parallel tracks to manage.

---

## Success Metrics

| Metric | Target | Measured at |
|---|---|---|
| Extraction F1 (entities) | ≥ 0.80 on benchmark corpus | v0.2 |
| Extraction F1 (Tier 1+2) | ≥ 0.90 on semi-structured text | v0.4 |
| Episode ingestion latency (Tier 1) | < 15ms | v0.2 |
| Search latency (RRF fused) | < 100ms at 10K episodes | v0.3 |
| Binary size | < 50MB (without models) | v1.0 |
| Cold startup | < 2s (model load) | v0.2 |
| Test count | ≥ 80 | v1.0 |
| Zero external services | True for Tier 1+2 | Always |
| Time to first demo | ≤ 5 weeks from start | v0.3 |

---

## What's NOT in v1.0

Explicitly deferred:

- **HNSW index for embeddings** — Brute-force cosine is fine under 100K episodes. `usearch` crate is ready when needed.
- **Multi-user / auth** — ctxgraph is single-user. Multi-user requires server mode.
- **GPU acceleration** — CPU-only. GPU via ort features is possible but not prioritized.
- **Graph visualization** — No built-in UI. Export to JSON, use external tools.
- **Schema marketplace** — Ship 4 built-in schemas. Let community emerge organically from adoption, not from an empty store.
- **India fintech / government schemas** — Real verticals, but premature without 100+ users of the default schema. Build the tool, get adoption, then pursue verticals.
- **Full PR parsing / DevTrace integration** — `ctxgraph watch --git` covers commit messages. Full PR analysis is DevTrace territory.
- **Plugin system** — Custom extractors are Rust traits only. No dynamic loading.

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| GLiNER2 ONNX export doesn't work | Blocks v0.2 | Fallback to two-model approach (GLiNER2 + GLiREL) |
| `ort` crate issues with GLiNER2 tensors | Delays v0.2 by 1 week | File upstream issue, implement workaround |
| Extraction quality < 0.80 F1 | Weak value prop | Tune threshold, try different checkpoint, add GLiREL precision mode |
| Nobody uses it after HN post | Wasted effort | Low cost — it's a useful internal tool regardless |
| Users want features not on roadmap | Scope creep | v0.5 is explicitly user-guided. Listen and adapt. |
