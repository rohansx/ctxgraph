# ctxgraph

**Local-first context graph engine for AI agents and human teams.**

Zero infrastructure. Zero API cost. Single Rust binary.

---

ctxgraph stores decision traces — the full story behind every choice — and makes them searchable. It runs entirely on your machine. No Neo4j. No OpenAI API key. No Docker. No Python. One binary, one SQLite file, instant startup.

When someone (or an AI agent) asks *"why did we do X?"*, ctxgraph traverses the graph of past decisions, finds relevant precedents, and returns the full context — who decided, when, what alternatives were considered, and what the outcome was.

## Why ctxgraph

Every context graph tool today requires heavy infrastructure. ctxgraph doesn't.

| | Graphiti (Zep) | ctxgraph |
|---|---|---|
| Graph database | Neo4j / FalkorDB (Docker) | SQLite (embedded) |
| LLM API key | Required (OpenAI/Anthropic) | Not required |
| Runtime | Python 3.10+ | Single Rust binary |
| Cost per episode | ~$0.01-0.05 (LLM tokens) | $0.00 |
| Setup time | 15-30 minutes | 5 seconds |
| Internet required | Yes (always) | No (fully offline) |
| Privacy | Data sent to OpenAI | Nothing leaves your machine |

## Quick Start

```bash
# Install
cargo install ctxgraph

# Initialize in your project
ctxgraph init

# Log decisions
ctxgraph log "Chose Postgres over SQLite for billing. Reason: concurrent writes."
ctxgraph log --source slack "Priya approved the discount for Reliance"
ctxgraph log --tags "architecture,database" "Switched from REST to gRPC"

# Search
ctxgraph query "why Postgres?"
ctxgraph query "discount precedents" --limit 5

# Auto-capture git commits
ctxgraph watch --git --last 50

# Stats
ctxgraph stats
```

## How It Works

```
Your App / CLI / AI Agent
         |
    ctxgraph engine
         |
    ┌─────────────────────────────────┐
    │  Extraction                     │
    │  GLiNER2 (ONNX) — local, $0    │
    │  entities + relations + dates   │
    └─────────────────────────────────┘
         |
    ┌─────────────────────────────────┐
    │  Storage                        │
    │  SQLite + FTS5                  │
    │  Bi-temporal timestamps         │
    │  Graph via recursive CTEs       │
    └─────────────────────────────────┘
         |
    ┌─────────────────────────────────┐
    │  Search                         │
    │  FTS5 + Semantic + Graph Walk   │
    │  Fused via Reciprocal Rank      │
    └─────────────────────────────────┘
```

### Extraction

ctxgraph automatically extracts entities and relationships from plain text using a local ONNX model. No API calls, no cost.

```
Input:  "Chose Postgres over SQLite for billing. Reason: concurrent writes."

Output: Entities  → Postgres (Component), SQLite (Component), billing (Service)
        Relations → chose(Postgres, billing), rejected(SQLite, billing)
        Temporal  → recorded now, valid indefinitely
```

Entity types and relation labels are fully configurable via `ctxgraph.toml`. Define what matters to your domain — Person, Component, Decision, Policy, whatever fits.

For higher accuracy, optional tiers add coreference resolution, fuzzy dedup, and LLM-powered contradiction detection. Each tier is additive — Tier 1 alone covers 85%+ of structured text.

### Bi-Temporal History

Every relationship tracks two time dimensions:

- **valid_from / valid_until** — when was this true in the real world?
- **recorded_at** — when was this recorded?

Facts are never deleted — they are invalidated. You can query the graph as it existed at any point in time.

```
Alice →[works_at]→ Google   (2020-01 to 2025-06)
Alice →[works_at]→ Meta     (2025-06 to now)
```

### Search

Three search modes fused via Reciprocal Rank Fusion:

- **FTS5** — keyword matching across episodes, entities, edges
- **Semantic** — 384-dim embeddings via all-MiniLM-L6-v2 (local)
- **Graph traversal** — multi-hop walk via recursive CTEs

A result appearing in multiple modes is ranked highest.

## MCP Server

ctxgraph runs as an MCP server for AI agents (Claude Desktop, Cursor, Claude Code):

```json
{
  "mcpServers": {
    "ctxgraph": {
      "command": "ctxgraph",
      "args": ["mcp", "start"]
    }
  }
}
```

| Tool | Description |
|---|---|
| `ctxgraph_add_episode` | Record a decision or event |
| `ctxgraph_search` | Search for relevant decisions and precedents |
| `ctxgraph_get_decision` | Get full decision trace by ID |
| `ctxgraph_traverse` | Walk the graph from an entity |
| `ctxgraph_find_precedents` | Find similar past decisions |

## Rust SDK

Embed ctxgraph directly in your Rust application:

```rust
use ctxgraph::{Graph, Episode};

let graph = Graph::init(".ctxgraph")?;

// Log a decision
graph.add_episode(
    Episode::builder("Chose Postgres for billing — concurrent writes required")
        .source("architecture-review")
        .tag("database")
        .build()
)?;

// Search
let results = graph.search("why Postgres?", 10)?;

// Traverse from an entity
let neighbors = graph.traverse("Postgres", 2)?;
```

## CLI Reference

```
ctxgraph init [--name <name>]                         Initialize .ctxgraph/ in current directory
ctxgraph log <text> [--source <src>] [--tags <t1,t2>] Log a decision or event
ctxgraph query <text> [--limit <n>]                   Search the context graph
ctxgraph entities list [--type <type>]                List entities
ctxgraph entities show <id>                           Show entity with relationships
ctxgraph decisions list                               List episodes
ctxgraph decisions show <id>                          Show full decision trace
ctxgraph stats                                        Graph statistics
ctxgraph watch --git [--last <n>]                     Auto-capture git commits
ctxgraph models download                              Download ONNX models
ctxgraph export --format json|csv                     Export graph data
ctxgraph mcp start                                    Run as MCP server
```

## Configuration

```toml
# ctxgraph.toml

[schema]
name = "default"

[schema.entities]
Person = "A person involved in a decision"
Component = "A software component or technology"
Decision = "An explicit choice that was made"
Reason = "The justification behind a decision"

[schema.relations]
chose = { head = "Person", tail = "Component" }
rejected = { head = "Person", tail = "Alternative" }
approved = { head = "Person", tail = "Decision" }

[tier2]
enabled = true

[tier2.dedup.aliases]
"Postgres" = ["PostgreSQL", "PG", "psql"]

[llm]
enabled = false   # opt-in, works with Ollama
```

## Project Structure

```
crates/
├── ctxgraph-core/       Core engine: types, storage, query, temporal
├── ctxgraph-extract/    Extraction pipeline (GLiNER2 ONNX)
├── ctxgraph-embed/      Local embedding generation
├── ctxgraph-cli/        CLI binary
├── ctxgraph-mcp/        MCP server for AI agents
└── ctxgraph-sdk/        Re-export crate for embedding in Rust apps
```

## Design Principles

1. **Zero infrastructure** — One binary, one SQLite file
2. **Offline-first** — No internet required after model download
3. **Privacy by default** — Nothing leaves your machine
4. **Progressive enhancement** — Each tier is additive and optional
5. **Schema-driven** — Extraction labels are user-defined, not hardcoded
6. **Embeddable** — Rust library first, CLI second
7. **Append-only history** — Facts invalidated, never deleted

## License

MIT
