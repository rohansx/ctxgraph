# ctxgraph — Implementation Plan

Compressed build plan optimized for speed-to-demo. Ship a working, demoable product (CLI + extraction + MCP) in 5-6 weeks, then iterate with real users.

---

## Philosophy

> Don't let the perfect 10-version roadmap be the enemy of shipping v0.3 in 5-6 weeks.

The original 10-phase plan was thorough but risked over-planning. This revised plan merges the detailed roadmap with a speed-first instinct:

1. Get the storage model right (v0.1).
2. Get extraction working — this is the hard engineering sprint (v0.2).
3. Ship MCP so you can demo it (v0.3).
4. Then iterate with real user feedback instead of building in a vacuum.

---

## Phase Overview

```
Phase 1 │ v0.1  Core Engine                      │ 2 weeks │ DONE
Phase 2 │ v0.2  GLiNER2 Extraction (NER + RE)     │ 2 weeks │ Hard sprint
Phase 3 │ v0.3  MCP Server + Search               │ 1 week  │ DEMO MILESTONE
Phase 4 │ v0.4  Tier 2 + Git Watch                │ 1 week  │ Quality + capture
        │       ─── open source, post HN, get feedback ───
Phase 5 │ v0.5  Tier 3 + Bulk Ingest              │ 2 weeks │ User-guided
Phase 6 │ v1.0  Production Ready                  │ 1 week  │ Ship
                                                    ────────
                                                    ~9 weeks total
```

**Key difference from original plan**: v0.2 and v0.3 (old plan) are merged into a single version because GLiNER2 now handles both entity and relation extraction in one model. Old v0.5 (search) and v0.6 (MCP) are merged into v0.3 because search is needed for MCP to be useful.

---

## Phase 1: v0.1 — Core Engine (DONE)

**Status**: Complete. 24 tests passing, 0 clippy warnings.

**What shipped**:
- `ctxgraph-core` crate: types, SQLite storage, FTS5, bi-temporal logic, graph traversal
- `ctxgraph-cli` crate: init, log, query, entities, decisions, stats
- Episode builder pattern, Entity/Edge CRUD, edge invalidation
- Recursive CTE graph traversal, FTS5 search
- UUID v7 IDs, embedded migrations

### Definition of Done

- [x] `ctxgraph init` creates `.ctxgraph/` with SQLite DB
- [x] `ctxgraph log` stores episodes
- [x] `ctxgraph query` returns episodes via FTS5
- [x] `ctxgraph entities list/show` works
- [x] `ctxgraph stats` shows correct counts
- [x] Bi-temporal invalidation works
- [x] 24 tests passing
- [x] `cargo clippy` clean

---

## Phase 2: v0.2 — GLiNER2 Unified Extraction (~2 weeks)

**Goal**: Episodes auto-extract entities AND relations in a single model pass. This is the hardest engineering sprint — getting GLiNER2 running in Rust via `ort`, handling tokenization, span decoding, entity-type prompts concatenated with text tokens, and post-processing.

### Why This Is Hard

The `ort` crate works but has rough edges with certain model architectures. GLiNER2's input format (entity type prompts concatenated with text tokens) requires custom tensor construction. This is not a weekend project — it's 1-2 weeks of careful engineering:

1. Verify GLiNER2 ONNX export exists and works
2. Understand the exact input tensor format (label encoding scheme)
3. Implement tokenization with proper attention masks
4. Decode span scores back to character offsets
5. Handle edge cases: long text truncation, overlapping spans, empty inputs

### New Crate

| Crate | Files | Purpose |
|---|---|---|
| `ctxgraph-extract` | lib.rs, tier1/{mod,gliner,models}.rs, schema.rs, temporal.rs | Extraction pipeline |

### Build Steps

1. **Model manager** — Download GLiNER2 ONNX + tokenizer from HuggingFace. SHA256 verification. Cache at `~/.ctxgraph/models/`.
2. **GLiNER2 wrapper** — Load ONNX via `ort`, tokenize via `tokenizers` crate. Build input tensors with label encoding. Decode span scores to entity + relation lists.
3. **Schema loading** — Parse `ctxgraph.toml` for entity labels and relation types.
4. **Temporal heuristics** — Five-layer date parser (ISO, written, relative, fiscal, duration).
5. **Pipeline integration** — `Graph::add_episode()` calls extractor, stores entities, creates edges.
6. **Extraction benchmark** — 50 manually annotated PR descriptions and commit messages. Measure F1 score against annotations. Threshold: entity F1 ≥ 0.80 for launch.
7. **CLI updates** — `ctxgraph log` shows extracted entities/relations. `ctxgraph models download/list`.
8. **Cold start UX** — Sparse graph hints and bootstrap suggestions.

### Key Technical Details

- GLiNER2 input format: `[CLS] label1 [SEP] label2 [SEP] ... [SEP] text [SEP]`
- Output tensor: `[1, seq_len, seq_len, num_labels]` — span start × span end × label type
- Post-processing: threshold filter (0.5), overlap resolution, token-to-char mapping
- Model files: `gliner2-large-q8.onnx` (~200MB INT8), `tokenizer.json` (~2MB)
- **Model licensing**: Verify Apache 2.0 on specific checkpoint before shipping

### Dependencies Added

```toml
# ctxgraph-extract
ort = { version = "2", features = ["download-binaries"] }
tokenizers = "0.19"
ndarray = "0.15"
reqwest = { version = "0.12", features = ["stream"] }
indicatif = "0.17"
sha2 = "0.10"
strsim = "0.11"
regex = "1"
```

### Tests (15+)

- GLiNER2 model loads from ONNX
- Extract entities from "Chose Postgres for billing" → Component, Service
- Extract relations from "Chose Postgres for billing" → (Postgres, chosen_for, billing)
- Extract from "Priya approved 30% discount" → Person, Amount
- Custom schema labels work
- Empty text → zero entities (no crash)
- Long text (>512 tokens) → graceful truncation
- Model download + cache works
- Re-download skipped when cached
- Threshold filtering removes low-confidence spans
- Overlapping spans resolved correctly
- Temporal: "yesterday" → correct date
- Temporal: "Q3 2025" → 2025-07-01
- Temporal: "3 weeks ago" → correct offset
- **Extraction benchmark: F1 ≥ 0.80 on annotated corpus**

### Definition of Done

- [ ] GLiNER2 ONNX loads and runs via `ort` (entities + relations)
- [ ] `ctxgraph log` auto-extracts entities and relations
- [ ] Edges auto-created between extracted entities
- [ ] Schema configurable via `ctxgraph.toml`
- [ ] `ctxgraph models download/list` works
- [ ] Temporal heuristics parse 80%+ of common date patterns
- [ ] Extraction benchmark: entity F1 ≥ 0.80
- [ ] Model license verified (Apache 2.0)
- [ ] Cold start UX: sparse graph hints
- [ ] 15+ new tests passing
- [ ] Extraction latency < 15ms on CPU (single episode)

---

## Phase 3: v0.3 — MCP Server + Search (~1 week)

**Goal**: AI agents can use ctxgraph. This is the demo milestone — the point where you can show it working in Claude Desktop/Cursor. Write the "SQLite for context graphs" blog post and ship.

### What Ships

- MCP server with 5 tools
- Local embedding model for semantic search
- RRF fusion across FTS5 + semantic + graph traversal
- `ctxgraph mcp start` command

### New Crates

| Crate | Files | Purpose |
|---|---|---|
| `ctxgraph-embed` | lib.rs, model.rs, similarity.rs | Local embedding generation |
| `ctxgraph-mcp` | lib.rs, server.rs, tools.rs, config.rs | MCP server |

### Build Steps

1. **Embedding model** — Load all-MiniLM-L6-v2 ONNX. 384-dim vectors. Mean pooling.
2. **Semantic search** — Brute-force cosine similarity. Return top-K.
3. **Multi-hop traversal** — Recursive CTE graph walk with cycle detection.
4. **RRF fusion** — Merge FTS5 + semantic + graph results. k=60.
5. **MCP tools** — 5 tools: add_episode, search, get_decision, traverse, find_precedents.
6. **Stdio transport** — JSON-RPC over stdin/stdout.
7. **Integration testing** — Test with Claude Desktop, Cursor, Claude Code.

### Tests (12+)

- Embedding generates 384-dim vector
- Cosine similarity: identical text → ~1.0
- Semantic search finds paraphrases
- Multi-hop traversal: 1-hop, 2-hop, 3-hop correct
- RRF fusion: multi-mode hit ranks higher
- MCP initialize handshake
- tools/list returns all 5 tools
- add_episode stores and returns entity count
- search returns ranked results
- traverse returns graph neighbors
- Invalid tool call returns proper error
- Query latency < 100ms on < 10K episodes

### Definition of Done

- [ ] `ctxgraph query` returns RRF-fused ranked results
- [ ] `ctxgraph mcp start` runs MCP server on stdio
- [ ] All 5 MCP tools functional
- [ ] Tested with Claude Desktop
- [ ] 12+ new tests passing
- [ ] **DEMO READY**: Can show end-to-end flow in a screencast

---

## Phase 4: v0.4 — Tier 2 + Git Watch (~1 week)

**Goal**: Quality improvement + passive capture. After this, open source and post on HN.

### What Ships

- Tier 2: coreference resolution, fuzzy entity dedup, context-aware temporal
- `ctxgraph watch --git` for auto-capturing commit messages
- User-defined alias groups in config

### Build Steps

1. **Coreference resolver** — Rule-based pronoun resolution.
2. **Fuzzy dedup** — Jaro-Winkler + alias groups from config.
3. **Context temporal** — Resolve dates relative to events in the graph.
4. **Git watch** — `ctxgraph watch --git --last N` imports commit messages as episodes. Post-commit hook setup.
5. **Config** — `[tier2]` and `[watch]` sections in TOML.

### Tests (10+)

- "She approved it" after "Priya reviewed..." → She = Priya
- "P. Sharma" dedup matches "Priya Sharma"
- "PostgreSQL" matches alias "Postgres"
- "three weeks after the migration" → resolved date
- Alias groups loaded from config
- Dedup: same type only
- Git watch: imports last N commits as episodes
- Git watch: sets source to "git-commit"
- Git watch: includes commit hash in metadata
- Quality benchmark: 88-90% on semi-structured corpus

### Definition of Done

- [ ] Coreference resolves pronouns
- [ ] Jaro-Winkler dedup merges similar names
- [ ] `ctxgraph watch --git` imports commits
- [ ] 10+ new tests passing
- [ ] **LAUNCH READY**: Open source, HN post

---

## Phase 5: v0.5 — Tier 3 + Bulk Ingest (~2 weeks, user-guided)

**Goal**: Handle messy text and large imports. Build this based on user feedback from the v0.4 launch.

### What Ships

- Tier 3: Ollama/API integration for contradiction detection + summarization
- Auto-escalation when Tier 1 misses entities
- Bulk ingest from JSONL/CSV/stdin
- Built-in schemas (default, developer, support, finance)
- JSON/CSV export

### Build Steps

1. **LLM provider abstraction** — `LlmProvider` trait with Ollama + OpenAI implementations.
2. **Contradiction detection** — Check conflicting edges, invalidate via LLM judgment.
3. **Community summarization** — Cluster detection + LLM summary.
4. **Auto-escalation** — Low entity density triggers Tier 3.
5. **Bulk ingest** — JSONL/CSV file import, stdin piping, batch processing.
6. **Built-in schemas** — Ship 4 schemas with the binary.
7. **Export** — JSON and CSV export commands.

### Definition of Done

- [ ] Ollama integration works
- [ ] Contradiction detection invalidates conflicting edges
- [ ] Bulk ingest: JSONL/CSV/stdin
- [ ] 4 built-in schemas
- [ ] Export works
- [ ] 15+ new tests passing

---

## Phase 6: v1.0 — Production Ready (~1 week)

**Goal**: Stable, documented, published.

### Build Steps

1. **Benchmarks** — `criterion` at 1K/10K/100K episodes.
2. **Documentation** — Rustdoc, user guide, MCP setup guide.
3. **Error messages** — Every error has actionable guidance.
4. **Publish** — crates.io, GitHub release with pre-built binaries.

### Definition of Done

- [ ] Benchmarks published
- [ ] All public APIs documented
- [ ] Pre-built binaries for Linux/macOS/Windows
- [ ] Published to crates.io
- [ ] 80+ total tests passing
- [ ] `cargo clippy` clean
- [ ] No `unsafe` code
