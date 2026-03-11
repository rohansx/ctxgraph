# ctxgraph — Architecture

> Local-first context graph engine for AI agents and human teams.

---

## System Overview

ctxgraph is an embedded, temporal context graph engine that stores decision traces and makes them searchable. It runs as a single Rust binary backed by a single SQLite file — no external services required.

```
┌─────────────────────────────────────────────────────────┐
│                    Consumer Layer                        │
│  CLI  │  MCP Server (AI agents)  │  Rust SDK (embedded) │
└───┬───┴──────────┬───────────────┴──────────┬───────────┘
    │              │                          │
    ▼              ▼                          ▼
┌─────────────────────────────────────────────────────────┐
│                  ctxgraph-core (Engine)                  │
│                                                         │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ Graph API│  │ Query Engine │  │ Temporal Engine    │  │
│  │          │  │              │  │                    │  │
│  │ add      │  │ FTS5         │  │ bi-temporal        │  │
│  │ search   │  │ semantic     │  │ invalidation       │  │
│  │ traverse │  │ graph walk   │  │ time-travel query  │  │
│  │          │  │ RRF fusion   │  │                    │  │
│  └──────────┘  └──────────────┘  └───────────────────┘  │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │              Storage (SQLite + FTS5)             │    │
│  │  episodes │ entities │ edges │ aliases │ communities │
│  └─────────────────────────────────────────────────┘    │
└──────────────────────────┬──────────────────────────────┘
                           │
    ┌──────────────────────┼──────────────────────┐
    │                      │                      │
    ▼                      ▼                      ▼
┌──────────────┐  ┌────────────────┐  ┌───────────────────┐
│ctxgraph-extract│ │ ctxgraph-embed │ │ ctxgraph-mcp      │
│               │  │                │  │                   │
│ Tier 1: ONNX  │  │ all-MiniLM-L6  │  │ stdio transport   │
│  GLiNER2      │  │ 384-dim vectors │  │ 5 tools exposed   │
│  (unified     │  │ cosine sim     │  │ JSON-RPC          │
│   NER + RE)   │  │                │  │                   │
│               │  └────────────────┘  └───────────────────┘
│ Tier 2: Local │
│  coref+dedup  │
│  +temporal    │
│               │
│ Tier 3: LLM  │
│  ollama/API   │
│  (optional)   │
└──────────────┘
```

---

## Crate Dependency Graph

```
ctxgraph-cli ──────┐
ctxgraph-mcp ──────┤
ctxgraph-sdk ──────┤
                   ▼
             ctxgraph-core
                   │
          ┌────────┼────────┐
          ▼        ▼        ▼
  ctxgraph-extract  ctxgraph-embed
          │
     ort + tokenizers
     (ONNX Runtime)
```

| Crate | Purpose | Key deps |
|---|---|---|
| `ctxgraph-core` | Engine: types, storage, query, temporal logic | rusqlite, chrono, uuid, serde |
| `ctxgraph-extract` | Three-tier extraction pipeline | ort, tokenizers, strsim, reqwest |
| `ctxgraph-embed` | Local embedding generation | ort, ndarray |
| `ctxgraph-cli` | Binary with clap-based CLI | clap, colored, indicatif |
| `ctxgraph-mcp` | MCP server for AI agents | tokio, serde_json |
| `ctxgraph-sdk` | Re-export crate for embedding ctxgraph in other Rust apps | — |

---

## Data Model

### Core Entities

```
Episode ──< episode_entities >── Entity
                                    │
                               Edge (source ──relation──▶ target)
                                    │
                               bi-temporal: valid_from / valid_until / recorded_at
```

**Episode** — The atomic unit of information. Represents "something happened."
- Content (free text), source tag, metadata JSON, optional embedding vector.

**Entity** — A thing mentioned in episodes. Extracted automatically (Tier 1+) or referenced via metadata.
- Name, type (Person, Component, Decision, etc.), optional summary.

**Edge** — A relationship between two entities. The core of the graph.
- Typed relation (chose, rejected, approved, etc.).
- Bi-temporal: `valid_from`/`valid_until` (real-world truth) + `recorded_at` (system time).
- Linked back to the episode that produced it.

**Alias** — Maps variant names to a canonical entity (fuzzy dedup).

**Community** — A cluster of related entities with an LLM-generated summary.

### Bi-Temporal Model

Every edge carries two time dimensions:

| Dimension | Column | Meaning |
|---|---|---|
| Valid time | `valid_from`, `valid_until` | When was this fact true in the real world? |
| Transaction time | `recorded_at` | When was this fact recorded in ctxgraph? |

This enables:
- **Current view**: `WHERE valid_until IS NULL` — only facts that are still true.
- **Time-travel**: `WHERE valid_from <= ?t AND (valid_until IS NULL OR valid_until > ?t)` — what was true at time `t`.
- **Audit trail**: `WHERE recorded_at BETWEEN ?start AND ?end` — what was recorded in a time window.

Facts are never deleted — they are invalidated by setting `valid_until`.

---

## Extraction Pipeline

The extraction pipeline converts raw episode text into structured graph nodes and edges.

### Key Design Decision: Unified GLiNER2 Model

The 2025 EMNLP version of GLiNER2 handles **both** entity extraction and relationship extraction in a single model via structured JSON output. This is a significant simplification over the original two-model design (GLiNER2 + GLiREL):

| Approach | Models | Download | Inference passes | Pipeline complexity |
|---|---|---|---|---|
| **Old: GLiNER2 + GLiREL** | 2 ONNX files | ~350MB | 2 | Entity extraction → relation extraction |
| **New: Unified GLiNER2** | 1 ONNX file | ~200MB | 1 | Single extraction pass |

GLiREL is retained as an **optional precision mode** for users who need better relation quality on complex text, but the default pipeline uses GLiNER2 alone.

```
Episode text
    │
    ▼
┌─────────────────────────────────────────┐
│ Tier 1: Schema-Driven Local (always on) │
│ Cost: $0 | Latency: 2-10ms             │
│                                         │
│ GLiNER2 (ONNX) → entities + relations  │
│ Regex/dateparser → temporal expressions │
│                                         │
│ Optional: GLiREL precision mode         │
│ (separate ONNX, better relation quality │
│  on complex text, +150MB download)      │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│ Tier 2: Enhanced Local (default on)     │
│ Cost: $0 | Latency: 15-50ms            │
│                                         │
│ Coreference → pronoun resolution        │
│ Jaro-Winkler → fuzzy entity dedup       │
│ Context temporal → relative-to-event    │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│ Tier 3: LLM-Enhanced (opt-in)           │
│ Cost: $0 (Ollama) / $0.01+ (API)       │
│ Latency: 500-2000ms                    │
│                                         │
│ Contradiction detection                 │
│ Complex temporal reasoning              │
│ Community summarization                 │
│ Unstructured text extraction            │
└─────────────────────────────────────────┘
```

### Tier Selection Logic

Tier escalation is automatic and self-regulating:

1. **Tier 1** always runs.
2. **Tier 2** runs if enabled (default: yes). Adds ~10-40ms.
3. **Tier 3** only fires when triggered:
   - Conflicting edges detected → contradiction check.
   - Tier 2 couldn't resolve a date → complex temporal.
   - Community cluster exceeds size threshold → summarization.
   - Entity density suspiciously low for text length → full LLM extraction.

```
trigger_heuristic:
  expected = word_count / 15
  actual   = tier1_entities.len()
  if actual < expected * 0.4 → escalate to Tier 3
```

---

## Query Architecture

Three search modes, fused via Reciprocal Rank Fusion (RRF):

```
Query: "why did we choose Postgres?"
         │
    ┌────┼────────────────────┐
    ▼    ▼                    ▼
  FTS5  Semantic          Graph Walk
  │     │                    │
  │  cosine sim on        rCTE traversal
  │  384-dim embeddings   from matched entity
  │     │                    │
  └──┬──┘                    │
     ▼                       │
  ┌──────────────────────────┘
  │
  ▼
 RRF Fusion
  score = Σ 1/(k + rank_i) across all modes
  │
  ▼
 Ranked SearchResults
```

| Mode | Catches | Misses |
|---|---|---|
| FTS5 | Exact keyword matches | Synonyms, paraphrases |
| Semantic | Meaning similarity | Exact names, IDs |
| Graph walk | Structural relationships | Disconnected episodes |

RRF fusion ensures that a result appearing in multiple modes is ranked highest.

### Scaling Strategy

Brute-force cosine similarity is sufficient for graphs under 100K episodes (<50ms). For larger graphs, an optional HNSW index via `usearch` crate can be enabled. This is a post-v1.0 concern — no solo dev or small team will hit 100K episodes in their first year.

### Time-Travel Queries

```rust
graph.search_at("who works at Google?", as_of: "2024-06-15")
```

Filters edges by `valid_from <= as_of AND (valid_until IS NULL OR valid_until > as_of)`, returning the state of the graph at that point in time.

---

## Storage Architecture

Single SQLite file: `.ctxgraph/graph.db`

### Why SQLite (not Neo4j)

| Factor | SQLite | Neo4j |
|---|---|---|
| Deployment | Embedded, zero config | Requires Docker or server |
| Graph traversal | Recursive CTEs (rCTEs) | Native Cypher |
| Full-text search | FTS5 (built-in) | Requires Lucene plugin |
| Concurrency | Single-writer, multi-reader | Full ACID |
| Scale ceiling | ~100K-1M episodes | Millions+ |
| Operational cost | Zero | $$$$ |

For ctxgraph's target use case (solo devs, small teams, <100K episodes), SQLite with rCTEs provides sufficient graph traversal performance without any infrastructure overhead.

### Graph Traversal via Recursive CTEs

Multi-hop traversal uses `WITH RECURSIVE`:

```sql
WITH RECURSIVE traversal(entity_id, depth, path) AS (
    SELECT id, 0, json_array(id) FROM entities WHERE name = ?
    UNION ALL
    SELECT
        CASE WHEN e.source_id = t.entity_id THEN e.target_id ELSE e.source_id END,
        t.depth + 1,
        json_insert(t.path, '$[#]', ...)
    FROM traversal t
    JOIN edges e ON (e.source_id = t.entity_id OR e.target_id = t.entity_id)
    WHERE t.depth < ?max_hops
      AND e.valid_until IS NULL
)
SELECT DISTINCT ent.*, t.depth FROM traversal t
JOIN entities ent ON ent.id = t.entity_id ORDER BY t.depth;
```

---

## ONNX Model Strategy

All ML models run locally via ONNX Runtime (`ort` crate). No GPU required.

| Model | Purpose | Size (INT8) | Latency (CPU) | Required |
|---|---|---|---|---|
| GLiNER2-large | Unified entity + relation extraction | ~200MB | 2-10ms | Yes (Tier 1) |
| GLiREL-large | Precision relation extraction | ~150MB | 5-15ms | Optional |
| all-MiniLM-L6-v2 | Embedding generation (384-dim) | ~80MB | 3-5ms | Yes (search) |

Models are downloaded on first use and cached at `~/.ctxgraph/models/`. Checksums are verified on download. For air-gapped environments, models can be pre-downloaded and pointed to via `CTXGRAPH_MODELS_DIR`.

### Model Licensing

GLiNER2 models are Apache 2.0 licensed. The specific checkpoint used (verified: `gliner_large-v2.1` or `fastino/gliner2-large-v1`) must have a license compatible with ctxgraph's MIT license. This is verified before each release.

---

## Interface Layer

### CLI

```
ctxgraph init               — initialize .ctxgraph/ in current directory
ctxgraph log                 — store an episode (with auto-extraction)
ctxgraph query               — search the graph (FTS + semantic + graph)
ctxgraph entities            — list/show entities
ctxgraph decisions           — list/show decision traces
ctxgraph stats               — graph statistics
ctxgraph watch --git         — auto-capture git commits as episodes
ctxgraph models              — download/list/verify ONNX models
ctxgraph export              — export graph as JSON/CSV
ctxgraph config              — show/set configuration
ctxgraph mcp start           — run as MCP server
ctxgraph ingest              — bulk import from JSONL/CSV/stdin
```

### Auto-Capture: `ctxgraph watch --git`

Solves the ingestion friction problem: people forget to `ctxgraph log` manually after day 3. `ctxgraph watch --git` monitors the git log and auto-imports commit messages as episodes. Low effort, high capture rate. This is a lightweight alternative to full DevTrace integration — just commit messages, not PR parsing.

```bash
# One-time setup: import last 50 commits
ctxgraph watch --git --last 50

# Ongoing: run as post-commit hook
# .git/hooks/post-commit
ctxgraph watch --git --last 1
```

### Cold Start UX

When the graph has fewer than 20 episodes, queries and traversals will return sparse results. ctxgraph handles this gracefully:

- Search results include a hint: "Graph has 5 episodes. Add 20+ for best results."
- `ctxgraph stats` suggests bootstrapping: "Tip: run `ctxgraph watch --git --last 50` to import recent commits."
- Never shows "0 results found" without context — always explains and suggests next steps.

### MCP Server

Exposes 5 tools over stdio JSON-RPC for AI agents:

| Tool | Purpose |
|---|---|
| `ctxgraph_add_episode` | Record a new decision/event |
| `ctxgraph_search` | Search for relevant decisions and precedents |
| `ctxgraph_get_decision` | Get full decision trace by ID |
| `ctxgraph_traverse` | Walk the graph from an entity |
| `ctxgraph_find_precedents` | Find similar past decisions for a scenario |

### Rust SDK

`ctxgraph-sdk` re-exports core + extract + embed for embedding in other Rust applications:

```rust
let graph = ctxgraph::Graph::init(".ctxgraph")?;
let result = graph.add_episode(Episode::builder("chose Postgres for billing").build())?;
let results = graph.search("why Postgres?", 10)?;
```

---

## Configuration

Single TOML file: `ctxgraph.toml` (or `.ctxgraph/config.toml`)

```toml
[schema]
name = "default"

[schema.entities]
Person = "A person involved in a decision"
Component = "A software component or technology"
# ... (fully customizable)

[schema.relations]
chose = { head = "Person", tail = "Component" }
# ...

[extraction]
precision_mode = false    # enable GLiREL for better relation quality

[tier2]
enabled = true

[tier2.dedup]
threshold = 0.85
same_source_threshold = 0.75

[tier2.dedup.aliases]
"Postgres" = ["PostgreSQL", "PG", "psql"]

[watch]
git = false               # auto-capture git commits

[llm]
enabled = false           # opt-in only

[llm.provider.ollama]
base_url = "http://localhost:11434"
model = "llama3.2:8b"
```

---

## Design Principles

1. **Zero infrastructure** — One binary, one SQLite file. No Docker, no API keys, no Python.
2. **Offline-first** — Everything runs locally. Internet only needed for model download (once) and optional Tier 3 API.
3. **Privacy by default** — Nothing leaves the machine unless you explicitly enable Tier 3 with a remote LLM provider.
4. **Progressive enhancement** — Tier 1 works great alone. Tier 2 improves quality. Tier 3 handles edge cases. Each tier is additive and optional.
5. **Schema-driven** — Extraction labels are user-defined, not hardcoded. Adapt to any domain.
6. **Embeddable** — ctxgraph is a Rust library first, a CLI second. Other tools can embed it directly.
7. **Append-only history** — Facts are never deleted, only invalidated. The full temporal history is preserved.
8. **Ship fast, iterate with users** — Get to a demoable product (MCP server) in 5-6 weeks. Let real usage guide the rest of the roadmap instead of building in a vacuum.
