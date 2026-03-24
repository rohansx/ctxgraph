# ctxgraph Explained (Simple + Detailed)

This page explains ctxgraph in everyday language, while still covering the technical architecture.

## What ctxgraph is

ctxgraph is a local system that turns messy team notes and decisions into a searchable graph.

If you already have text like:

- "We moved billing to Postgres for concurrent writes."
- "Redis cache was rejected for this service."
- "Starting June 2025, Alice owns on-call."

ctxgraph converts that into structured memory so you can query it later.

In short, it is:

- a **memory system** (stores decisions/events),
- a **graph** (stores relationships),
- a **timeline** (keeps historical truth over time),
- and a **search engine** (keyword + semantic + graph search),

all in one local SQLite file.

## Big picture architecture

You can think of ctxgraph as 4 layers:

1. **Input layer** (CLI, MCP server, Rust SDK)  
   Where notes come from (humans or AI agents).
2. **Extraction layer** (`ctxgraph-extract`)  
   Reads plain text and extracts entities, relationships, and time hints.
3. **Engine + storage layer** (`ctxgraph-core` + SQLite)  
   Stores episodes/entities/edges and supports temporal logic.
4. **Query layer**  
   Finds answers using keyword search, semantic similarity, and graph traversal.

## Step-by-step: what happens when you add a note

Example input:

> "We chose Postgres over SQLite for billing due to concurrent writes."

### Step 1: Episode is created

The raw text becomes an **episode** (one atomic event in memory).

### Step 2: Entity + relation extraction

The extraction pipeline detects:

- entities: `Postgres`, `SQLite`, `billing`
- relation candidates: `chose(billing, Postgres)`, `rejected(billing, SQLite)`

Tier system:

- **Tier 1 (always on):** local ONNX model + rule-based parsing
- **Tier 2 (default on):** better cleanup (dedup, coreference, temporal improvements)
- **Tier 3 (optional):** LLM-based enhancement for hard/ambiguous cases

### Step 3: Temporal tagging

Each relation gets time fields:

- `valid_from` / `valid_until`: when the fact is true in the real world
- `recorded_at`: when ctxgraph stored it

This makes "what did we believe at date X?" queries possible.

### Step 4: Persist to SQLite graph

Data is stored in tables (episodes, entities, edges, aliases, communities) inside one SQLite database file.

### Step 5: Query and ranking

When you search, ctxgraph runs multiple retrieval methods and merges them:

- **FTS5** for exact words,
- **semantic search** for similar meaning,
- **graph traversal** for connected context.

Results are fused using Reciprocal Rank Fusion so items that appear in multiple methods are ranked higher.

## Core technical ideas in simple terms

### 1) Why SQLite for a graph?

Instead of running a heavy graph database server, ctxgraph uses SQLite plus recursive SQL queries.

That gives:

- easy setup (no server),
- portability (single file),
- enough graph capability for small/medium team memory workloads.

### 2) What "bi-temporal" really means

Normal systems store one timestamp.
ctxgraph stores two time views:

- **reality time** (when it was true),
- **recording time** (when we learned/stored it).

So if someone logs a late update, history still remains accurate.

### 3) Why multiple search methods?

No single search style is perfect:

- keywords catch exact terms,
- semantic search catches paraphrases,
- graph traversal catches relationship context.

Combining them gives more reliable answers.

### 4) Local-first model execution

Models run through ONNX Runtime on your machine.
That means:

- no API keys required for core features,
- private data stays local,
- predictable cost (mostly zero after download).

## Crates and responsibilities

- `ctxgraph-core`: graph API, storage, query, temporal logic
- `ctxgraph-extract`: extraction pipeline (entities/relations/time)
- `ctxgraph-embed`: embedding generation for semantic search
- `ctxgraph-cli`: command-line interface
- `ctxgraph-mcp`: MCP server tools for agent integration
- `ctxgraph-sdk`: easy embedding in Rust applications

## Practical mental model

If you want one sentence:

> ctxgraph is a local "decision brain" that continuously turns plain text into a time-aware relationship graph you can query from CLI, code, or AI tools.

If you want one workflow:

1. Log decisions/events regularly.
2. Let extraction structure the data.
3. Query by keyword/meaning/relationships.
4. Traverse and audit historical reasoning.

---

For deeper internals and implementation details, see [`ARCHITECTURE.md`](./ARCHITECTURE.md).
