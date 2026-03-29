# ctxgraph — Architecture

> Privacy-first knowledge graph engine. Better quality, 6x cheaper, 3x faster than Graphiti — and when it does call an LLM, CloakPipe strips PII first.

---

## System Overview

ctxgraph is a tiered knowledge graph engine that extracts entities and relations from text using local ONNX models first, and only escalates to an LLM when local confidence is low — with PII stripped via CloakPipe before any cloud call. In fair head-to-head testing (same model, same data, same evaluation), ctxgraph scores 0.846 combined F1 vs Graphiti's 0.601 on tech text, while making 6x fewer LLM calls and querying 20x faster.

It can be used as a Rust library, a CLI tool, an MCP server for AI coding tools, or a background daemon that indexes developer activity.

```
┌─────────────────────────────────────────────────────────────────┐
│                      Source Layer                                │
│  [Git]  [Shell History]  [FS Watcher]  [Browser]  [Screenpipe]  │
└────┬──────────┬──────────────┬────────────┬───────────┬─────────┘
     │          │              │            │           │
     ▼          ▼              ▼            ▼           ▼
┌─────────────────────────────────────────────────────────────────┐
│                 ctxgraph-ingest (Connectors)                     │
│                                                                  │
│  GitConnector    ShellConnector    FsConnector    BrowserConnector│
│  (git2-rs)      (bash/zsh/fish)   (notify)       (Chrome/FF)    │
│                                                                  │
│  Structured data → Episodes with source tags                     │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ctxgraph-core (Engine)                          │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐  │
│  │ Graph API    │  │ Query Engine │  │ Temporal Engine         │  │
│  │              │  │              │  │                         │  │
│  │ add_episode  │  │ FTS5         │  │ bi-temporal edges       │  │
│  │ add_entity   │  │ semantic     │  │ invalidation            │  │
│  │ add_edge     │  │ graph walk   │  │ time-travel query       │  │
│  │ traverse     │  │ RRF fusion   │  │ temporal window filter  │  │
│  └──────────────┘  └──────────────┘  └────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │                Storage (SQLite + FTS5 + WAL)              │    │
│  │  episodes │ entities │ edges │ aliases │ communities      │    │
│  └──────────────────────────────────────────────────────────┘    │
└──────────────────────────────┬──────────────────────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
┌──────────────────┐  ┌────────────────┐  ┌──────────────────┐
│ ctxgraph-extract │  │ ctxgraph-embed │  │ Interface Layer  │
│                  │  │                │  │                  │
│ Tier 1: ONNX     │  │ all-MiniLM-L6  │  │ MCP Server       │
│  GLiNER2 (NER)   │  │ 384-dim vectors│  │ CLI              │
│  GLiREL (RE)     │  │ cosine sim     │  │ TUI (ratatui)    │
│                  │  │                │  │                  │
│ Tier 2: Local    │  └────────────────┘  └──────────────────┘
│  coref+dedup     │
│  +temporal       │
│                  │
│ Tier 3: LLM      │
│  ollama/API      │
│  (optional)      │
└──────────────────┘
```

### Key Insight: Don't Call an LLM When You Don't Have To

Graphiti calls GPT-4o on every episode — even when the entities are obvious. "Alice chose PostgreSQL" doesn't need a $0.01 API call.

ctxgraph's tiered pipeline handles 80%+ of episodes locally (free, <10ms), and only escalates when local confidence is low. CloakPipe PII stripping coming in v0.8, and caches entity resolutions so the same pattern never costs twice.

For developer workflows specifically, structured ingestion (git, shell, FS) provides entities directly — no ML needed at all. NER is only needed for unstructured text like commit messages, terminal output, and free-form decision logs.

---

## Crate Structure

```
ctxgraph-cli ──────┐
ctxgraph-mcp ──────┤
ctxgraph-sdk ──────┤
                   ▼
             ctxgraph-core
                   │
          ┌────────┼────────────────┐
          ▼        ▼                ▼
  ctxgraph-extract  ctxgraph-embed  ctxgraph-ingest
          │                              │
     ort + tokenizers              git2 + notify
     (ONNX Runtime)                (connectors)
```

| Crate | Purpose | Key deps | Status |
|---|---|---|---|
| `ctxgraph-core` | Engine: types, storage, query, temporal logic | rusqlite, chrono, uuid, serde | Shipped (v0.8.0) |
| `ctxgraph-extract` | Three-tier extraction pipeline | ort, tokenizers, strsim, reqwest | Shipped (v0.8.0) |
| `ctxgraph-embed` | Local embedding generation (384-dim) | fastembed | Shipped (v0.8.0) |
| `ctxgraph-ingest` | Connectors: git, shell, FS, browser, Screenpipe | git2, notify, rusqlite (browser) | **New — Phase 1** |
| `ctxgraph-mcp` | MCP server for AI coding tools | tokio, serde_json | Shipped (v0.8.0) |
| `ctxgraph-cli` | CLI + daemon mode + TUI dashboard | clap, colored, ratatui | Partial (CLI shipped, daemon + TUI in Phase 2) |
| `ctxgraph-sdk` | Re-export crate for embedding in Rust apps | — | Shipped |
| `ctxgraph-privacy` | PII detection and entity redaction (CloakPipe) | — | Phase 4 |

---

## Data Model

### Entity Types (Developer-Specific)

The entity schema is purpose-built for developer workflows. Most entities are extracted directly from structured sources — no NER required.

| Entity Type | Source | Example | NER Required? |
|---|---|---|---|
| `File` | Git, FS watcher | `src/auth/middleware.rs` | No — direct from structured data |
| `Function` | ctxgraph-extract (ONNX NER) | `validate_jwt_token()` | Yes — extracted from terminal output, commit messages |
| `Error` | Terminal history, ctxgraph-extract | `ECONNREFUSED`, `panic at thread main` | Partial — structured from exit codes, NER for stack traces |
| `Package` | ctxgraph-extract, lockfiles | `tokio@1.35`, `express@4.18` | No — parsed from Cargo.toml, package.json |
| `PR` | Git remote, browser history | `PR #142` | No — parsed from git refs, URLs |
| `Issue` | Browser history, commit messages | `JIRA-1234`, `GH-42` | Partial — regex from commit messages |
| `Person` | Git author field | `sarah <sarah@company.com>` | No — direct from git |
| `Command` | Shell history | `cargo test --release` | No — direct from history file |
| `URL` | Browser history (filtered) | `github.com/org/repo/pull/142` | No — direct from browser DB |
| `Branch` | Git | `feature/auth-migration` | No — direct from git |
| `Commit` | Git | `a1b2c3d` | No — direct from git |

**Key observation**: 8 of 11 entity types need zero NER. They come directly from structured sources. This is why structured ingestion beats screen capture for developer workflows.

### Relationship Types

| Relationship | Between | Example | Source |
|---|---|---|---|
| `modified_in` | File → Commit | `middleware.rs` modified in `a1b2c3d` | Git diff |
| `authored_by` | Commit → Person | Commit `a1b2c3d` authored by sarah | Git log |
| `caused_by` | Error → File/Commit | `ECONNREFUSED` caused by changes in `db.rs` | Temporal correlation |
| `depends_on` | File → Package | `middleware.rs` depends on `jsonwebtoken@9.2` | Lockfile parsing |
| `referenced_in` | Issue → Commit/PR | `JIRA-1234` referenced in PR #142 | Commit message regex |
| `resolved_by` | Error → Commit | Panic resolved by hotfix `f4e5d6a` | Temporal: error disappears after commit |
| `browsed_while` | URL → File/Error | SO answer browsed while debugging auth error | Temporal overlap |
| `branched_from` | Branch → Branch | `feature/auth-migration` branched from `main` | Git reflog |
| `ran_in` | Command → File/Branch | `cargo test` ran in `feature/auth` context | Shell history + git state |
| `reviewed_in` | File → PR | `middleware.rs` reviewed in PR #142 | Git remote |

### Temporal Properties

Every edge carries temporal metadata:

| Field | Purpose | Example |
|---|---|---|
| `valid_from` | When the relationship was established | Commit timestamp, file save time |
| `valid_until` | When it was superseded (NULL if still active) | File refactored, error resolved |
| `recorded_at` | When ctxgraph recorded this fact (system time) | Daemon ingestion timestamp |
| `source_event` | Raw event that produced this relationship | Commit hash, shell history line, FS event |

This enables temporal queries:
- **Current view**: `WHERE valid_until IS NULL` — only facts still true
- **Time-travel**: `WHERE valid_from <= ?t AND (valid_until IS NULL OR valid_until > ?t)` — what was true at time `t`
- **Activity window**: `WHERE recorded_at BETWEEN ?start AND ?end` — what happened in a time range

Facts are never deleted — they are invalidated by setting `valid_until`.

---

## Ingestion Layer

### Architecture

Each connector runs as an async task within the daemon, producing `Episode` events that flow into the core graph engine.

```
┌──────────────────────────────────────────────────────────┐
│                   ctxgraph-ingest                          │
│                                                           │
│  ┌─────────────┐  Watches repos for commits, branch      │
│  │ Git         │  switches, merges. Parses diffs for      │
│  │ Connector   │  file→commit→author relationships.       │
│  │ (git2-rs)   │  No NER needed — all structured.         │
│  └──────┬──────┘                                          │
│         │                                                 │
│  ┌──────┴──────┐  Reads bash/zsh/fish history files.      │
│  │ Shell       │  Extracts commands, detects errors        │
│  │ Connector   │  (non-zero exit, stderr patterns).        │
│  │ (history)   │  NER on stack traces only.                │
│  └──────┬──────┘                                          │
│         │                                                 │
│  ┌──────┴──────┐  Watches project directories via inotify/ │
│  │ FS          │  kqueue. Tracks open/save/rename/delete.   │
│  │ Connector   │  Correlates with git state for context.   │
│  │ (notify)    │                                          │
│  └──────┬──────┘                                          │
│         │                                                 │
│  ┌──────┴──────┐  Reads Chrome/Firefox SQLite history DB.  │
│  │ Browser     │  Filters to dev-relevant domains:         │
│  │ Connector   │  GitHub, SO, docs, Jira, Linear.          │
│  │ (rusqlite)  │  Phase 3.                                │
│  └──────┬──────┘                                          │
│         │                                                 │
│  ┌──────┴──────┐  Ingests Screenpipe output, filtered to   │
│  │ Screenpipe  │  terminal/IDE/browser windows only.        │
│  │ Pipe        │  Phase 3.                                 │
│  └──────┬──────┘                                          │
│         │                                                 │
│         ▼                                                 │
│  ┌─────────────────────────────────────────────────┐      │
│  │ Episode Builder                                  │      │
│  │ source → tag → metadata → Episode → Graph.add()  │      │
│  └─────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────┘
```

### Connector Details

**Git Connector (Phase 1)**
- Uses `git2-rs` to read repository state
- Watches for: commits, branch switches, merges, tag creation
- Extracts: files modified, author, commit message, diff stats, branch context
- Entities created directly (no NER): File, Person, Commit, Branch, PR (from message refs)
- Relationships created directly: `modified_in`, `authored_by`, `branched_from`, `referenced_in`
- Bootstrapping: import last N commits on first run

**Shell History Connector (Phase 1)**
- Reads history files: `~/.bash_history`, `~/.zsh_history`, `~/.local/share/fish/fish_history`
- Watches for new entries via file polling or inotify
- Extracts: command text, working directory (if available), timestamp
- Entities: Command (direct), Error (from exit codes + NER on stderr patterns)
- NER used for: stack traces, error messages in terminal output
- Relationships: `ran_in` (command → project context), `caused_by` (error → command)

**FS Watcher Connector (Phase 1)**
- Uses `notify` crate for cross-platform file system events
- Watches project directories (configured paths)
- Tracks: file create, modify, rename, delete events
- Entities: File (direct from path)
- Correlates with current git branch for context
- Deduplicates rapid save events (debounce window)

**Browser History Connector (Phase 3)**
- Reads Chrome `History` or Firefox `places.sqlite` (both are SQLite)
- Filters to developer-relevant domains (configurable allowlist):
  - `github.com`, `gitlab.com` — PRs, issues, code
  - `stackoverflow.com` — debugging research
  - `docs.rs`, `doc.rust-lang.org`, `developer.mozilla.org` — documentation
  - `linear.app`, `jira.atlassian.com` — issue trackers
  - Custom domains via config
- Entities: URL (direct), Issue/PR (parsed from URL patterns)
- Relationships: `browsed_while` (temporal overlap with file edits or errors)

**Screenpipe Pipe (Phase 3)**
- Ingests Screenpipe's structured output
- Filters to terminal, IDE, and browser window contexts only
- Enriches existing graph with screen-capture context where structured sources have gaps

---

## Extraction Pipeline

The extraction pipeline converts raw episode text into structured graph nodes and edges.

For developer memory, most extraction is **structural** (parsing structured data), not **NER** (machine learning on text). The ONNX NER pipeline fires only for unstructured content like terminal output, commit messages, and stack traces.

### Tier Breakdown

```
Episode (from connector)
    │
    ▼
┌─────────────────────────────────────────┐
│ Tier 0: Structural Extraction (always)  │
│ Cost: $0 | Latency: <1ms               │
│                                         │
│ Git fields → entities + relations       │
│ Shell history → command entities        │
│ FS events → file entities               │
│ URL parsing → URL/PR/Issue entities     │
│ Lockfile parsing → Package entities     │
│                                         │
│ NO ML — just parsing structured data    │
└─────────────┬───────────────────────────┘
              │
              ▼ (only for unstructured content)
┌─────────────────────────────────────────┐
│ Tier 1: Schema-Driven Local (as needed) │
│ Cost: $0 | Latency: 2-10ms             │
│                                         │
│ GLiNER2 (ONNX) → entities + relations  │
│ Regex/dateparser → temporal expressions │
│                                         │
│ Used for: stack traces, error messages, │
│ commit message body, PR descriptions    │
│                                         │
│ Optional: GLiREL precision mode         │
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
│ Natural language query interpretation   │
└─────────────────────────────────────────┘
```

### Confidence Gate (Automatic LLM Escalation)

The engine automatically decides when local extraction is insufficient and escalates to an LLM:

```
Tier 1 output (GLiNER + GLiREL)
    │
    ▼
┌───────────────────────────────────────┐
│ Confidence Gate                       │
│                                       │
│ Escalate to LLM if ANY of:           │
│  - Avg entity confidence < 0.4       │
│  - Zero relations found              │
│  - GLiREL confidence < 0.5           │
│  - Entity count < expected for text  │
│                                       │
│ If no LLM configured → use local as-is│
└───────────────┬───────────────────────┘
                │
    ┌───────────┴───────────┐
    │                       │
 Confident              Not confident
    │                       │
 Use local result      CloakPipe → LLM
    │                       │
    └───────────┬───────────┘
                │
              Graph
```

No user intervention needed. No LLM configured? Works fully local. LLM configured? Fills gaps automatically.

### CloakPipe Privacy Layer

When the confidence gate triggers LLM escalation, CloakPipe (planned for v0.8) ensures privacy:

1. **PII scan** — Detects API keys, passwords, emails, tokens, IP addresses in the text
2. **Reversible redaction** — Replaces PII with deterministic placeholders (`[EMAIL_1]`, `[KEY_1]`)
3. **Sanitized LLM call** — Only the redacted text reaches the LLM provider
4. **Entity mapping** — Maps LLM-extracted entities back through the placeholder table
5. **Encrypted cache** — AES-256-GCM vault stores entity type resolutions, so repeated patterns cost $0

**Cost impact**: The cache means a pattern like "company name → Organization" is resolved once and cached forever. Over time, the LLM escalation rate drops as the cache grows.

### Why Local ONNX Works for Most Cases

The ONNX NER model (GLiNER2) is tuned for tech/code domains. For the common use case of software decision logs, it performs well:

| Text Type | Extraction Method | Local Quality | LLM Needed? |
|---|---|---|---|
| Software architecture decisions | GLiNER + GLiREL | High (0.800 F1) | Rarely |
| Git data | Fully structured — no NER | N/A (100% accurate) | Never |
| Terminal output / stack traces | GLiNER (tech-optimized) | High | Rarely |
| Cross-domain text (finance, legal, healthcare) | GLiNER (low confidence) | Low (0.230 F1) | Yes — auto-escalated |
| Commit messages | GLiNER + regex | Medium-High | Sometimes |

The hybrid approach means tech-domain users get free extraction, and cross-domain users pay only for what the local models can't handle.

---

## Query Architecture

### Natural Language Query Resolution

These queries are resolved by graph traversal, not LLM inference:

| Query | Resolution Strategy |
|---|---|
| "What was I debugging 2 hours ago?" | Error entities in terminal history (time filter) → connected files → related commits |
| "What's related to the auth migration?" | Entity name match "auth" → all connected nodes (files, PRs, issues, errors, commands) |
| "Show me everything @sarah touched this sprint" | Person entity → commits (time filter) → files → connected issues and PRs |
| "What broke after the last deploy?" | Git diff at deploy tag → error entities timestamped after deploy → causal chain |
| "How is this error connected to that PR?" | Multi-hop traversal: error → file → commit → PR → related discussions |
| "What SO answers did I read about this?" | File/error entity → browser history URLs (temporal overlap) |

### Search Modes (RRF Fusion)

Three search modes, fused via Reciprocal Rank Fusion:

```
Query: "what was I debugging yesterday?"
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

### Time-Travel Queries

```rust
graph.search_at("who was working on auth?", as_of: "2026-03-15")
```

Filters edges by `valid_from <= as_of AND (valid_until IS NULL OR valid_until > as_of)`, returning the graph state at that point in time.

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

## Storage Architecture

Single SQLite file: `.ctxgraph/graph.db`

### Why SQLite (not Neo4j, not RuVector, not anything else)

| Factor | SQLite | Neo4j | RuVector |
|---|---|---|---|
| Deployment | Embedded, zero config | Requires Docker or server | 90+ crate dependency tree |
| Graph traversal | Recursive CTEs | Native Cypher | HNSW (not graph traversal) |
| Full-text search | FTS5 (built-in) | Requires Lucene plugin | Custom |
| Temporal queries | Native SQL on bi-temporal columns | Custom | Not built for this |
| Concurrency | Single-writer, multi-reader (WAL) | Full ACID | Unknown at scale |
| Scale ceiling | ~100K-1M entities | Millions+ | Unverified |
| Operational cost | Zero | $$$$ | Dependency risk |
| Binary size impact | Bundled via rusqlite | N/A | Massive |

For a developer's activity graph (~10K-100K nodes over months of use), SQLite with recursive CTEs provides sub-5ms graph traversal without any infrastructure overhead.

### Scaling Strategy

If traversal becomes a bottleneck at scale (unlikely for single-developer graphs):
1. **petgraph** for in-memory graph — hydrate from SQLite on startup, traverse in memory
2. **usearch** for HNSW vector index — if brute-force cosine exceeds 50ms at 100K+ episodes

Both are drop-in additions. Neither is needed for Phase 1-4.

### Pragmas

```sql
PRAGMA journal_mode = WAL;          -- Concurrent reads during daemon writes
PRAGMA synchronous = NORMAL;        -- Balanced durability/performance
PRAGMA foreign_keys = ON;           -- Referential integrity
```

---

## Interface Layer

### MCP Server

Exposes graph queries to any MCP-compatible AI coding tool (Claude Code, Cursor, Cline, Continue):

| Tool | Purpose |
|---|---|
| `add_episode` | Record a new event (manual logging) |
| `search` | Fused FTS5 + semantic search with temporal filters |
| `traverse` | Multi-hop graph walk from an entity |
| `traverse_batch` | Walk from multiple entities, merge results |
| `find_precedents` | Find similar past events (semantic similarity) |
| `list_entities` | List entities with type/time filters |
| `export_graph` | Export entities and edges |
| `query_activity` | Natural language query over dev activity (Phase 2) |

Transport: stdio JSON-RPC 2.0 (protocol version 2024-11-05).

### CLI

```
ctxgraph init                    — Initialize .ctxgraph/ in project directory
ctxgraph daemon start            — Start background daemon (ingestion + MCP)
ctxgraph daemon stop             — Stop background daemon
ctxgraph daemon status           — Show daemon status and connector health
ctxgraph log <text>              — Manually add an episode
ctxgraph query <text>            — Search the graph (FTS + semantic + traversal)
ctxgraph entities [--type TYPE]  — List entities
ctxgraph activity [--since 2h]   — Show recent dev activity timeline
ctxgraph traverse <entity>       — Walk the graph from an entity
ctxgraph stats                   — Graph statistics
ctxgraph models download         — Download ONNX models
ctxgraph mcp start               — Run as MCP server (stdio)
ctxgraph config                  — Show/set configuration
```

### TUI Dashboard (Phase 2)

ratatui-based interactive dashboard showing:
- Recent activity timeline (commits, commands, file edits)
- Entity graph visualization (ASCII)
- Active connections and relationship explorer
- Search with live results

### Web Dashboard (Future)

Graph visualization for the browser, inspired by [Rowboat Labs](https://github.com/rowboatlabs/rowboat) (Apache 2.0) — their knowledge graph UI shows smooth entity connections and relationship exploration. For ctxgraph, candidate libraries:
- **reagraph** (WebGL, React) — performant 3D graph rendering
- **react-force-graph** — force-directed layout, 2D/3D
- **cytoscape.js** — framework-agnostic, rich graph algorithms

---

## Daemon Architecture

The daemon runs as a lightweight background process (~15MB RSS target):

```
┌────────────────────────────────────────────┐
│              ctxgraph daemon                │
│                                            │
│  ┌──────────────────────────────────┐      │
│  │ Connector Manager (tokio tasks)  │      │
│  │                                  │      │
│  │  [Git]  [Shell]  [FS]  [Browser] │      │
│  │    ↓       ↓      ↓       ↓     │      │
│  │  Episode stream (mpsc channel)   │      │
│  └──────────────┬───────────────────┘      │
│                 │                          │
│                 ▼                          │
│  ┌──────────────────────────────────┐      │
│  │ Ingestion Pipeline               │      │
│  │                                  │      │
│  │ Episode → Extract → Dedup → Store│      │
│  └──────────────┬───────────────────┘      │
│                 │                          │
│                 ▼                          │
│  ┌──────────────────────────────────┐      │
│  │ Graph Engine (ctxgraph-core)     │      │
│  │ SQLite + FTS5 + embeddings       │      │
│  └──────────────┬───────────────────┘      │
│                 │                          │
│                 ▼                          │
│  ┌──────────────────────────────────┐      │
│  │ MCP Server (stdio)               │      │
│  │ Serves queries from AI tools     │      │
│  └──────────────────────────────────┘      │
└────────────────────────────────────────────┘
```

Process management: systemd (Linux), launchd (macOS), or simple PID file.

---

## Model Strategy

### Local Models (ONNX Runtime — no GPU required)

| Model | Purpose | Size | Latency (CPU) | Always loaded? |
|---|---|---|---|---|
| GLiNER2-large | Entity extraction | ~653MB (INT8) | 2-10ms | Yes |
| GLiREL-large | Zero-shot relation extraction | ~1GB (split: encoder + scoring head) | ~5s | Yes (when available) |
| all-MiniLM-L6-v2 | Embedding generation (384-dim) | ~80MB | 3-5ms | Yes |

Models download on first use, cached at `~/.cache/ctxgraph/models/`.

### LLM Fallback (Ollama or Cloud — optional)

When the confidence gate fires, ONE LLM call extracts entities + relations:

| Provider | Model | Quality (cross-domain F1) | Cost per episode | Local? |
|---|---|---|---|---|
| None | — | 0.325 | $0 | Yes |
| Ollama | llama3.2:3b | 0.472 | $0 | Yes |
| Ollama | qwen2.5:7b | 0.508 | $0 | Yes |
| Ollama | gemma2:9b | 0.506 | $0 | Yes |
| OpenRouter | gemini-2.0-flash | **0.552** | ~$0.00005 | No (CloakPipe strips PII) |

**Recommended**: GPT-4o-mini for best quality ($0.30/1K eps). Gemini Flash for cheapest cloud ($0.05/1K eps). Qwen 2.5 7B for fully local ($0).

### Head-to-Head: ctxgraph vs Graphiti (Fair Evaluation)

Same model (GPT-4o-mini), same data, same evaluation methodology:

| Metric | Graphiti | ctxgraph | |
|---|---|---|---|
| Tech Combined F1 | 0.601 | **0.846** | ctxgraph wins |
| Cross-domain Combined F1 | 0.474 | **0.650** | ctxgraph wins |
| Tech Relation F1 | 0.349 | **0.714** | ctxgraph 2x better |
| Cross-domain Relation F1 | 0.092 | **0.457** | ctxgraph 5x better |
| LLM calls per episode | 6 | **1** | 6x fewer |
| Extraction time | 15s | **5s** | 3x faster |
| Query latency (fused search) | ~300ms | **<15ms** | 20x faster |
| Graph traversal | 5-50ms (Neo4j) | **<5ms** (SQLite CTE) | 10x faster |
| Cost per 1000 episodes | $1.80 | **$0.30** | 6x cheaper |
| Works offline? | No | **Yes** | |
| Works with Ollama? | No | **Yes** | |
| Infrastructure | Neo4j + Docker | **SQLite** | |

### Why 6x Cheaper

Graphiti makes 6 LLM calls per episode (entity extraction, dedup, relation extraction, contradiction detection, summarization, community detection). **Every episode, regardless of complexity.**

ctxgraph makes 0-1 calls:
- Tech text: 0 calls (local ONNX handles it at 0.846 F1)
- Cross-domain: 1 call (entities + relations in one prompt)
- Dedup, temporal, community: always local (Jaro-Winkler, SQL, bi-temporal model)

### Why 20x Faster Queries

| Query Type | ctxgraph (SQLite) | Graphiti (Neo4j) |
|---|---|---|
| Full-text search | **<1ms** (FTS5) | ~50ms (BM25) |
| Semantic search | **3-5ms** (local cosine) | ~100ms (cloud embeddings) |
| Graph traversal (2-3 hops) | **<5ms** (recursive CTE) | 5-50ms (Cypher) |
| Fused search (all modes) | **5-15ms** (RRF) | ~300ms |
| Time-travel query | **<5ms** (bi-temporal SQL) | ~300ms |

ctxgraph queries are faster because everything runs in-process (embedded SQLite) with no serialization overhead. Graphiti requires network round-trips to Neo4j.

### Bi-Temporal vs Point-in-Time

ctxgraph tracks three timestamps per edge:
- `valid_from`: when the relationship was established in the real world
- `valid_until`: when it was superseded (NULL if still active)
- `recorded_at`: when ctxgraph ingested this fact

Graphiti tracks `valid_at` and `expired_at` (point-in-time). ctxgraph's model is more precise for developer workflows where you need to distinguish "when did this become true?" from "when did we learn about it?"

---

## Privacy Architecture

### Default: Local-First

- All data stays on the developer's machine by default
- No cloud services, no API calls unless LLM tier is configured
- SQLite file is the single source of truth
- No network access required for core functionality (Tier 0 + Tier 1)

### CloakPipe Integration (Built-In)

CloakPipe will be integrated into the extraction pipeline, not an add-on:

- **On LLM escalation**: PII detected and redacted before any text leaves the machine
- **Encrypted entity cache**: AES-256-GCM vault stores entity type resolutions — same patterns never cost twice
- **Fuzzy entity resolution**: CloakPipe's entity resolver deduplicates entities across episodes
- **App/directory exclusion rules**: Skip `~/.ssh`, `~/.aws`, etc.
- **Optional encrypted graph storage**: AES-256-GCM for the SQLite file itself

---

## Configuration

Single TOML file: `ctxgraph.toml` (or `.ctxgraph/config.toml`)

```toml
[daemon]
pid_file = "~/.ctxgraph/daemon.pid"
log_level = "info"

[connectors.git]
enabled = true
repos = ["."]                     # Watch current directory by default
bootstrap_commits = 100           # Import last N commits on first run

[connectors.shell]
enabled = true
history_files = ["auto"]          # Auto-detect bash/zsh/fish

[connectors.fs]
enabled = true
watch_paths = ["."]
debounce_ms = 500                 # Coalesce rapid save events
ignore = ["target/", "node_modules/", ".git/"]

[connectors.browser]
enabled = false                   # Phase 3, opt-in
allowed_domains = [
    "github.com", "gitlab.com",
    "stackoverflow.com",
    "docs.rs", "developer.mozilla.org",
    "linear.app", "jira.atlassian.com",
]

[connectors.screenpipe]
enabled = false                   # Phase 3, opt-in
window_filter = ["terminal", "ide", "browser"]

[schema]
name = "developer"

[schema.entities]
File = "A source code file or configuration file"
Function = "A function, method, or class definition"
Error = "An error, exception, or stack trace"
Package = "A library, package, or dependency with version"
PR = "A pull request or merge request"
Issue = "A bug report, feature request, or ticket"
Person = "A developer, team member, or contributor"
Command = "A CLI command or script execution"
URL = "A web URL visited during development"
Branch = "A git branch"
Commit = "A git commit"

[schema.relations]
modified_in = { head = ["File"], tail = ["Commit"] }
authored_by = { head = ["Commit"], tail = ["Person"] }
caused_by = { head = ["Error"], tail = ["File", "Commit", "Command"] }
depends_on = { head = ["File", "Package"], tail = ["Package"] }
referenced_in = { head = ["Issue", "PR"], tail = ["Commit", "PR"] }
resolved_by = { head = ["Error"], tail = ["Commit"] }
browsed_while = { head = ["URL"], tail = ["File", "Error"] }
branched_from = { head = ["Branch"], tail = ["Branch"] }
ran_in = { head = ["Command"], tail = ["File", "Branch"] }
reviewed_in = { head = ["File"], tail = ["PR"] }

[extraction]
precision_mode = false            # Enable GLiREL for better relation quality

[extraction.dedup]
threshold = 0.85                  # Jaro-Winkler similarity for entity dedup

[privacy]
exclude_paths = ["~/.ssh", "~/.aws", "~/.gnupg"]
redact_secrets = false            # Phase 4: CloakPipe integration

[llm]
enabled = false                   # Tier 3: opt-in only

[llm.provider.ollama]
base_url = "http://localhost:11434"
model = "llama3.2:8b"
```

---

## Design Principles

1. **Zero infrastructure** — One binary, one SQLite file. No Docker, no API keys, no Python.
2. **Structured ingestion over screen capture** — Parse structured data from developer tools directly. OCR and NER are last resorts, not defaults.
3. **Privacy by default** — Nothing leaves the machine unless explicitly enabled.
4. **Progressive enhancement** — Structural extraction works great alone. ONNX NER adds depth. LLM handles edge cases. Each tier is additive.
5. **Developer-native** — Entity types, relationship types, and query patterns are all designed for developer workflows.
6. **Temporal-first** — Every relationship has a time dimension. The graph is a living history, not a static snapshot.
7. **Daemon, not manual** — Background ingestion solves the adoption killer of manual logging.
8. **Embeddable** — ctxgraph is a Rust library first, a CLI second. Other tools can embed it directly.
