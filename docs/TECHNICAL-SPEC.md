# ctxgraph — Technical Specification

---

## 1. Storage Schema

All data resides in a single SQLite database at `.ctxgraph/graph.db`.

### 1.1 Tables

```sql
-- Episodes: raw events, the atomic unit of information
CREATE TABLE episodes (
    id          TEXT PRIMARY KEY,       -- UUID v7 (time-sortable)
    content     TEXT NOT NULL,
    source      TEXT,                   -- "manual", "git-commit", "slack", etc.
    recorded_at TEXT NOT NULL,          -- ISO-8601, when recorded in ctxgraph
    metadata    TEXT,                   -- JSON blob (author, tags, confidence, etc.)
    embedding   BLOB                   -- 384-dim f32 vector (optional, from ctxgraph-embed)
);

-- Entities: extracted or referenced graph nodes
CREATE TABLE entities (
    id          TEXT PRIMARY KEY,       -- UUID v7
    name        TEXT NOT NULL,
    entity_type TEXT NOT NULL,          -- matches schema labels: "Person", "Component", etc.
    summary     TEXT,                   -- LLM-generated summary (Tier 3)
    created_at  TEXT NOT NULL,
    metadata    TEXT                    -- JSON blob
);

-- Edges: relationships between entities (the graph)
CREATE TABLE edges (
    id          TEXT PRIMARY KEY,
    source_id   TEXT NOT NULL REFERENCES entities(id),
    target_id   TEXT NOT NULL REFERENCES entities(id),
    relation    TEXT NOT NULL,          -- "chose", "rejected", "approved", etc.
    fact        TEXT,                   -- human-readable: "rohan chose Postgres"
    valid_from  TEXT,                   -- when this became true (real-world)
    valid_until TEXT,                   -- when this stopped being true (NULL = current)
    recorded_at TEXT NOT NULL,          -- when recorded in ctxgraph
    confidence  REAL DEFAULT 1.0,      -- extraction confidence score
    episode_id  TEXT REFERENCES episodes(id),
    metadata    TEXT
);

-- Episode-Entity junction table
CREATE TABLE episode_entities (
    episode_id  TEXT REFERENCES episodes(id),
    entity_id   TEXT REFERENCES entities(id),
    span_start  INTEGER,               -- character offset in episode content
    span_end    INTEGER,
    PRIMARY KEY (episode_id, entity_id)
);

-- Entity aliases for deduplication
CREATE TABLE aliases (
    canonical_id TEXT REFERENCES entities(id),
    alias_name   TEXT NOT NULL,
    similarity   REAL,                 -- Jaro-Winkler score at merge time
    UNIQUE(canonical_id, alias_name)
);

-- Community clusters
CREATE TABLE communities (
    id          TEXT PRIMARY KEY,
    summary     TEXT,                  -- LLM-generated (Tier 3)
    entity_ids  TEXT,                  -- JSON array of entity IDs
    created_at  TEXT NOT NULL,
    updated_at  TEXT
);
```

### 1.2 Full-Text Search Indexes

```sql
CREATE VIRTUAL TABLE episodes_fts USING fts5(
    content, source, metadata,
    content=episodes, content_rowid=rowid
);

CREATE VIRTUAL TABLE entities_fts USING fts5(
    name, entity_type, summary,
    content=entities, content_rowid=rowid
);

CREATE VIRTUAL TABLE edges_fts USING fts5(
    fact, relation,
    content=edges, content_rowid=rowid
);
```

FTS5 triggers must be maintained on INSERT/UPDATE/DELETE to keep indexes in sync with base tables.

### 1.3 Performance Indexes

```sql
CREATE INDEX idx_edges_source   ON edges(source_id);
CREATE INDEX idx_edges_target   ON edges(target_id);
CREATE INDEX idx_edges_relation ON edges(relation);
CREATE INDEX idx_edges_valid    ON edges(valid_from, valid_until);
CREATE INDEX idx_entities_type  ON entities(entity_type);
CREATE INDEX idx_episode_entities ON episode_entities(entity_id);
CREATE INDEX idx_episodes_source ON episodes(source);
CREATE INDEX idx_episodes_recorded ON episodes(recorded_at);
```

### 1.4 ID Strategy

All IDs use UUID v7 (RFC 9562). UUID v7 is time-sortable — the first 48 bits encode a Unix millisecond timestamp. This means:
- Natural chronological ordering without an extra `created_at` sort.
- No sequential ID guessing.
- Safe for distributed generation (no coordination needed).

Rust crate: `uuid` with feature `v7`.

### 1.5 Migration Strategy

Schema migrations are embedded in the binary and run on database open:

```rust
const MIGRATIONS: &[(&str, &str)] = &[
    ("001_initial", include_str!("...")),
];
```

A `_migrations` table tracks which migrations have been applied:

```sql
CREATE TABLE IF NOT EXISTS _migrations (
    version TEXT PRIMARY KEY,
    applied_at TEXT NOT NULL
);
```

On `Graph::open()`, any unapplied migrations run automatically.

---

## 2. Extraction Pipeline

### 2.1 Tier 1 — Schema-Driven Local Extraction

#### Unified GLiNER2 Model

**Key simplification**: The 2025 EMNLP version of GLiNER2 now handles entity extraction, relationship extraction, and structured JSON output in a single forward pass. This eliminates the need for a separate GLiREL model in the default pipeline.

| Aspect | Old (GLiNER2 + GLiREL) | New (Unified GLiNER2) |
|---|---|---|
| ONNX files | 2 (~350MB) | 1 (~200MB) |
| Inference passes | 2 | 1 |
| Pipeline code | Entity extraction → relation extraction | Single extraction call |
| Relation quality | Higher (dedicated model) | Good (sufficient for 90%+ of use cases) |

**Default**: Unified GLiNER2 handles both entities and relations.
**Optional**: GLiREL can be enabled via `extraction.precision_mode = true` for users who need better relation quality on complex text.

#### Entity + Relation Extraction (GLiNER2)

Input format for GLiNER2 ONNX:
```
[CLS] label1 [SEP] label2 [SEP] ... [SEP] text tokens [SEP]
```

Output: span score tensor `[batch, seq_len, seq_len, num_labels]` where `scores[0][start][end][label_idx]` is the confidence that `text[start..end]` is an entity of type `labels[label_idx]`.

Post-processing:
1. Filter spans where `score > threshold` (default: 0.5).
2. Resolve overlapping spans — keep highest confidence.
3. Map token offsets back to character offsets via tokenizer.
4. Deduplicate within-episode (same text + same type = one entity).
5. Extract relation triples from the structured output.

#### Precision Mode: GLiREL (Optional)

When enabled, GLiREL runs as a second pass after GLiNER2 entity extraction:

Input: text with entity markers + candidate relation types.
Output: `(head_entity, relation, tail_entity, score)` tuples.

For each entity pair, GLiREL scores all candidate relation types. Pairs with no score above threshold are discarded.

#### Temporal Heuristics

Five parsing layers, each handling one pattern class:

| Layer | Pattern | Example | Resolution |
|---|---|---|---|
| 1 | ISO-8601 | `2026-03-11` | Direct parse |
| 2 | Written dates | `March 11, 2026` | Regex + chrono |
| 3 | Relative to now | `yesterday`, `3 days ago` | Offset from `episode.recorded_at` |
| 4 | Fiscal/quarter | `Q3 2025`, `FY26` | Mapped to date ranges |
| 5 | Duration | `for 3 months` | Duration struct |

### 2.2 Extraction Quality Benchmark

**Critical requirement**: Extraction quality must be measured, not assumed.

**Benchmark dataset**: 50 real PR descriptions and commit messages, manually annotated with:
- Expected entities (name, type, span)
- Expected relationships (head, relation, tail)

**Metrics**:
- Entity F1 score (precision + recall)
- Relation F1 score
- End-to-end latency per episode

**Thresholds**:
- Entity F1 ≥ 0.80: acceptable for launch
- Entity F1 ≥ 0.85: target quality
- Entity F1 < 0.80: investigate and improve before shipping

This benchmark is part of v0.2, not an afterthought. It ships as `tests/fixtures/benchmark_episodes.json` and runs in CI.

### 2.3 Tier 2 — Enhanced Local

#### Coreference Resolution

Rule-based pronoun resolution:

| Pronoun | Matches |
|---|---|
| he/him/his | Nearest PERSON |
| she/her | Nearest PERSON |
| they/them | Nearest PERSON or ORG |
| it/this/that | Nearest non-PERSON entity |
| the company/org/team | Nearest ORG |
| the project/service | Nearest SERVICE or COMPONENT |

Resolution strategy: for each pronoun, find the nearest compatible entity **before** the pronoun in text (left-to-right, closest wins).

#### Fuzzy Entity Deduplication

Algorithm:
1. **Alias check** — if the new entity name matches a configured alias, map to canonical.
2. **Jaro-Winkler similarity** — compare against all existing entities of the same type.
   - Default threshold: 0.85
   - Same-source threshold: 0.75 (more aggressive merging within one source)
3. On match, reuse existing entity ID and add alias record.

#### Context-Aware Temporal

For references like "three weeks after the migration incident":
1. Parse the event reference ("migration incident").
2. Search the graph for matching entities/episodes.
3. Resolve the temporal offset relative to the found event's timestamp.

### 2.4 Tier 3 — LLM-Enhanced

Strictly opt-in. Four capabilities:

| Capability | Trigger | LLM task |
|---|---|---|
| Contradiction detection | Conflicting edges on same entity pair | "Does new fact X contradict existing fact Y?" |
| Complex temporal | Tier 2 couldn't resolve a date | "What date does 'a few months before Series B' refer to?" |
| Community summarization | Cluster exceeds size threshold | "Summarize this cluster of related decisions" |
| Unstructured extraction | Entity density < 40% of expected | Full structured extraction from text |

**Provider abstraction**: `LlmProvider` trait with implementations for Ollama (local) and OpenAI-compatible APIs (remote).

**Auto-escalation heuristic**:
```
expected_density = word_count / 15
if actual_entities < expected_density * 0.4 → trigger Tier 3
```

---

## 3. Query System

### 3.1 Search Modes

#### FTS5 Keyword Search

Uses SQLite FTS5 with BM25 ranking across all three FTS tables (episodes, entities, edges). Supports:
- Boolean operators: `postgres AND NOT mysql`
- Phrase search: `"chose Postgres"`
- Prefix search: `post*`

#### Semantic Embedding Search

1. Embed query text using all-MiniLM-L6-v2 (384-dim).
2. Brute-force cosine similarity against all episode embeddings stored as BLOBs.
3. For graphs > 100K episodes, optional HNSW index via `usearch` crate (post-v1.0).

Cosine similarity:
```
sim(a, b) = (a · b) / (|a| × |b|)
```

#### Graph Traversal

Starting from entities matched by FTS or semantic search, walk the graph via recursive CTEs up to `max_hops` depth (default: 3). Only follows edges where `valid_until IS NULL` (current facts) unless `include_invalidated` is set.

### 3.2 Reciprocal Rank Fusion (RRF)

Merges results from all three modes into a single ranked list:

```
rrf_score(d) = Σ  1 / (k + rank_i(d))
               i∈{fts, semantic, graph}
```

Where `k = 60` (standard constant). A document appearing at rank 1 in two modes scores higher than rank 1 in one mode.

### 3.3 Search Filters

```rust
pub struct SearchFilter {
    pub after: Option<String>,           // ISO date
    pub before: Option<String>,
    pub source: Option<String>,          // e.g., "manual", "git-commit"
    pub entity_type: Option<String>,     // e.g., "Decision"
    pub max_hops: usize,                 // graph traversal depth (default: 3)
    pub include_invalidated: bool,       // include edges with valid_until set
}
```

### 3.4 Time-Travel Queries

```rust
graph.search_at("who works at Google?", as_of: "2024-06-15")
```

Generates SQL where clause:
```sql
WHERE (valid_from IS NULL OR valid_from <= '2024-06-15')
  AND (valid_until IS NULL OR valid_until > '2024-06-15')
```

---

## 4. Auto-Capture: Git Watch

### Problem

Manual `ctxgraph log` has high ingestion friction. Users are excited for 2 days, forget by day 3, uninstall by day 10. The tool must capture context passively.

### Solution

`ctxgraph watch --git` auto-imports git commit messages as episodes:

```bash
# Bootstrap: import last 50 commits
ctxgraph watch --git --last 50

# Ongoing: as a post-commit hook
echo 'ctxgraph watch --git --last 1' >> .git/hooks/post-commit
```

Each commit becomes an episode with:
- `content`: commit message
- `source`: "git-commit"
- `metadata`: `{ "hash": "abc123", "author": "rohan", "branch": "main" }`

This is a lightweight capture mechanism — not full PR parsing (that's DevTrace territory). Just commit messages, which are already semi-structured decision records.

---

## 5. Cold Start Handling

### Problem

When someone runs `ctxgraph init` and logs 5 episodes, the graph is too sparse for meaningful traversal or precedent matching. Showing "0 results found" makes the tool feel useless.

### Solution

1. **Sparse graph hints**: When episode count < 20, search results include: "Graph has {n} episodes. Add 20+ for best results."
2. **Bootstrap suggestions**: `ctxgraph stats` on a sparse graph suggests: "Tip: run `ctxgraph watch --git --last 50` to import recent commits."
3. **Never empty errors**: "No results" always includes an explanation and a suggested next action.
4. **Graceful degradation**: FTS5 search works from episode 1. Semantic search and graph traversal become useful at ~20+ episodes. RRF fusion adapts — if only one mode returns results, it still works.

---

## 6. Configuration Schema

Full `ctxgraph.toml` reference:

```toml
[schema]
name = "default"                        # schema name

[schema.entities]
# Label = "description for GLiNER2"
Person = "A person who made or was involved in a decision"
Component = "A software component, tool, library, framework, or technology"
Service = "A service, system, or application"
Decision = "An explicit choice or judgment that was made"
Reason = "The justification or rationale behind a decision"
Alternative = "An option that was considered but not chosen"
Policy = "A rule, guideline, or policy that was referenced"
Amount = "A monetary value, percentage, or quantifiable metric"
Constraint = "A limitation, requirement, or condition"

[schema.relations]
chose = { head = "Person", tail = "Component", description = "person chose a component" }
rejected = { head = "Person", tail = "Alternative", description = "person rejected an alternative" }
approved = { head = "Person", tail = "Decision", description = "person approved a decision" }
reason_for = { head = "Reason", tail = "Decision", description = "reason justifying a decision" }
applies_to = { head = "Decision", tail = "Service", description = "decision applies to a service" }
constrained_by = { head = "Decision", tail = "Constraint", description = "decision limited by constraint" }
references = { head = "Decision", tail = "Policy", description = "decision references a policy" }
supersedes = { head = "Decision", tail = "Decision", description = "newer decision replaces older one" }

[extraction]
precision_mode = false                  # enable GLiREL for better relation quality

[tier2]
enabled = true

[tier2.coreference]
enabled = true
max_distance = 500

[tier2.dedup]
threshold = 0.85
same_source_threshold = 0.75

[tier2.dedup.aliases]
"Postgres" = ["PostgreSQL", "PG", "psql"]
"K8s" = ["Kubernetes", "kube"]

[tier2.temporal]
fiscal_year_start = "april"
timezone = "Asia/Kolkata"

[watch]
git = false                             # auto-capture git commits

[llm]
enabled = false

[llm.provider.ollama]
base_url = "http://localhost:11434"
model = "llama3.2:8b"
timeout_seconds = 30

[llm.provider.openai]
api_key_env = "OPENAI_API_KEY"
model = "gpt-4o-mini"
max_tokens = 500

[llm.tasks]
contradiction_detection = true
temporal_reasoning = true
community_summarization = true
unstructured_extraction = false

[embeddings]
enabled = true
model = "all-MiniLM-L6-v2"
```

---

## 7. Error Handling Strategy

All public API functions return `Result<T, CtxGraphError>`:

```rust
#[derive(thiserror::Error, Debug)]
pub enum CtxGraphError {
    #[error("storage error: {0}")]
    Storage(#[from] rusqlite::Error),

    #[error("model not found: {0}. Run `ctxgraph models download`")]
    ModelNotFound(String),

    #[error("extraction failed: {0}")]
    Extraction(String),

    #[error("ONNX runtime error: {0}")]
    Onnx(String),

    #[error("schema error: {0}")]
    Schema(String),

    #[error("LLM provider error: {0}")]
    Llm(String),

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}
```

---

## 8. Performance Targets

| Operation | Target | Graph size |
|---|---|---|
| `add_episode` (Tier 1 only) | < 15ms | any |
| `add_episode` (Tier 1+2) | < 60ms | any |
| FTS5 search | < 10ms | < 100K episodes |
| Semantic search (brute-force) | < 50ms | < 100K episodes |
| Graph traversal (3-hop) | < 30ms | < 100K entities |
| RRF fused search | < 100ms | < 100K episodes |
| Model cold load | < 2s | — |
| Model warm inference | < 10ms | — |

---

## 9. Model Licensing

All models used by ctxgraph must have licenses compatible with MIT distribution:

| Model | License | Status |
|---|---|---|
| GLiNER2 (gliner_large-v2.1) | Apache 2.0 | Verified compatible |
| GLiREL | Apache 2.0 | Verified compatible |
| all-MiniLM-L6-v2 | Apache 2.0 | Verified compatible |

**Pre-release checklist**: Before each release, verify that the specific checkpoint being bundled/referenced has not changed its license. Check `fastino/gliner2-large-v1` vs `urchade/gliner_large-v2.1` — both Apache 2.0 currently, but must be re-verified.

---

## 10. File System Layout

```
project/
├── ctxgraph.toml                # extraction schema + config
└── .ctxgraph/
    ├── graph.db                 # SQLite database
    └── config.toml              # local config overrides

~/.ctxgraph/
├── models/
│   ├── gliner2-large-q8.onnx   # ~200MB (unified entity + relation)
│   ├── gliner2-tokenizer.json   # ~2MB
│   ├── glirel-large.onnx        # ~150MB (optional, precision mode)
│   └── minilm-l6-v2.onnx       # ~80MB
└── config.toml                  # global config
```

Model cache is shared across all projects at `~/.ctxgraph/models/`. Override with `CTXGRAPH_MODELS_DIR`.
