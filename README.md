# ctxgraph

**Local-first context graph engine for AI agents and human teams.**

Zero infrastructure. Zero API cost. Single Rust binary.

---

### What is a context graph?

A **context graph** is a knowledge graph of decisions. Every time your team (or an AI agent) makes a choice — picking a database, approving a discount, switching architectures — ctxgraph records it along with the who, why, when, and what alternatives were considered. Over time, you build a searchable, traversable history of institutional knowledge that answers "why did we do X?" in seconds.

ctxgraph stores decision traces — the full story behind every choice — and makes them searchable. It runs entirely on your machine. No Neo4j. No OpenAI API key. No Docker. No Python. One binary, one SQLite file, instant startup.

When someone (or an AI agent) asks *"why did we do X?"*, ctxgraph traverses the graph of past decisions, finds relevant precedents, and returns the full context — who decided, when, what alternatives were considered, and what the outcome was.

## Why ctxgraph

Every context graph tool today requires heavy infrastructure and sends your data to OpenAI. ctxgraph doesn't.

We benchmarked ctxgraph against [Graphiti](https://github.com/getzep/graphiti) (by Zep) on the same 50 software-engineering episodes. ctxgraph extracts higher-quality entities and relations using only local ONNX models — no API calls at all.

### Extraction Quality (50-episode benchmark)

| System | Entity F1 | Relation F1 | Combined F1 | API Calls | Cost | Latency |
|---|---|---|---|---|---|---|
| **ctxgraph** (local-only) | **0.837** | **0.763** | **0.800** | 0 | $0.00 | ~40ms/ep |
| Graphiti (gpt-4o) | 0.570 | 0.104\* | 0.337 | ~200+ | ~$2-5 | ~10s/ep |

\*Graphiti's free-form relations mapped to ctxgraph's taxonomy using generous keyword heuristics.

ctxgraph achieves **2.4x higher combined F1** than Graphiti while being **250x faster** and **100% free**.

### Infrastructure

| | Graphiti (Zep) | ctxgraph |
|---|---|---|
| Graph database | Neo4j / FalkorDB (Docker) | SQLite (embedded) |
| LLM API key | Required (OpenAI) | Not required |
| Runtime | Python 3.10+ | Single Rust binary |
| Models | Cloud API (gpt-4o) | Local ONNX (~623 MB) |
| RAM usage | Neo4j: 512MB+ | ~150 MB (inference) |
| Cost per episode | ~$0.01-0.05 | $0.00 |
| Setup time | 15-30 min (Neo4j + pip) | `cargo install` |
| Internet required | Always (LLM calls) | Only for initial model download |
| Privacy | Data sent to OpenAI | Nothing leaves your machine |

### Why Graphiti Scores Lower

Graphiti makes 6+ GPT-4o calls per episode (entity extraction, deduplication, relation extraction, contradiction detection, summarization, community detection). Despite this:

- **Entity names are verbose**: Graphiti extracts `"primary Postgres cluster"` instead of `"Postgres"`, `"legacy SOAP endpoint in UserService"` instead of `"SOAP endpoint"`. Semantically correct, but doesn't match canonical names.
- **Relations are free-form**: Produces verbs like `COMMUNICATES_ENCRYPTED_WITH` and `PREVENTS_CASCADING_FAILURES_WHEN_DOWN` that don't map to a typed taxonomy. Even with generous keyword mapping, only 10/50 episodes produce any matching relations.
- **Different decomposition**: "Migrate from Redis to Postgres" becomes `(AuthService, CONNECTS_TO, primary Postgres cluster)` instead of `(Postgres, replaced, Redis)` + `(AuthService, depends_on, Postgres)`.

ctxgraph uses domain-specific heuristics for software engineering patterns — keyword matching, proximity scoring, coreference resolution, and schema-aware type validation — that produce structured, queryable knowledge without any API calls.

See [docs/benchmark.md](docs/benchmark.md) for the full comparison methodology and per-episode results.

## Quick Start

```bash
# Install
# Via cargo (requires Rust toolchain)
cargo install ctxgraph

# Via Homebrew (macOS / Linux)
brew install rohansx/tap/ctxgraph

# Or download a prebuilt binary from GitHub Releases
# https://github.com/rohansx/ctxgraph/releases

# Download ONNX models (~600 MB, one-time)
ctxgraph models download

# Initialize in your project
ctxgraph init

# Log decisions
ctxgraph log "Chose Postgres over SQLite for billing. Reason: concurrent writes."
ctxgraph log --source slack "Priya approved the discount for Reliance"
ctxgraph log --tags "architecture,database" "Switched from REST to gRPC"

# Search
ctxgraph query "why Postgres?"
ctxgraph query "discount precedents" --limit 5

# Stats
ctxgraph stats
```

## What It Looks Like

```
$ ctxgraph log "Chose Postgres over SQLite for billing. Reason: concurrent writes."
Episode stored: a1b2c3d4
  Extracted 3 entities
  Created 2 edges
```

```
$ ctxgraph query "why Postgres?"
Found 2 result(s) for 'why Postgres?':

  [a1b2c3d4] (cli, 2025-03-23 14:05) score=0.92
    Chose Postgres over SQLite for billing. Reason: concurrent writes.

  [e5f6a7b8] (slack, 2025-03-20 09:12) score=0.71
    Priya confirmed Postgres handles our write volume — benchmarked at 10k TPS.
```

```
$ ctxgraph entities show Postgres
Entity: Postgres (Database)
ID: 9f8e7d6c-...
Created: 2025-03-23 14:05

Relationships:
  --[chose]--> billing
  --[rejected]--> SQLite (invalidated)
  <--[depends_on]-- payment-service

Neighbors:
  billing (Service)
  SQLite (Database)
  payment-service (Component)
```

```
$ ctxgraph stats
ctxgraph stats
------------------------------
Episodes:  127
Entities:  89
Edges:     214
Sources:   cli (45), git (72), slack (10)
DB size:   2.4 MB
```

### Real-World Scenario

Your team has been logging decisions for three months — architecture choices, vendor evaluations, incident responses. A new engineer joins and asks: "Why are we using Postgres instead of MongoDB for the billing service?"

```bash
$ ctxgraph query "why Postgres for billing?"
```

ctxgraph returns the original decision episode, the benchmark data that supported it, and the Slack discussion where the team evaluated MongoDB and rejected it for lack of ACID transactions. The new engineer gets the full context in seconds instead of asking three people and reading old Slack threads.

## How It Works

```
Your App / CLI / AI Agent
         |
    ctxgraph engine
         |
    +---------------------------------+
    |  Extraction                     |
    |  GLiNER v2.1 (ONNX) - local    |
    |  Entities: 0.837 F1             |
    |  Relations: 0.763 F1            |
    |  Temporal: date/time parsing    |
    +---------------------------------+
         |
    +---------------------------------+
    |  Storage                        |
    |  SQLite + FTS5                  |
    |  Bi-temporal timestamps         |
    |  Graph via recursive CTEs       |
    +---------------------------------+
         |
    +---------------------------------+
    |  Search                         |
    |  FTS5 + Semantic + Graph Walk   |
    |  Fused via Reciprocal Rank      |
    +---------------------------------+
```

### Extraction Pipeline

ctxgraph extracts entities and relationships from plain text using local ONNX models. No API calls, no cost, no internet required.

```
Input:  "Chose Postgres over SQLite for billing. Reason: concurrent writes."

Output: Entities  -> Postgres (Database), SQLite (Database), billing (Service)
        Relations -> chose(billing, Postgres), rejected(billing, SQLite)
        Temporal  -> recorded now, valid indefinitely
```

The pipeline:
1. **NER** — GLiNER v2.1 span-based extraction (10 entity types)
2. **Coreference** — Pronoun resolution to preceding entities
3. **Entity supplement** — Dictionary-based detection for names GLiNER missed
4. **Type remapping** — Fix common misclassifications using domain knowledge
5. **Relation extraction** — Keyword + proximity + schema-aware heuristics (9 relation types)
6. **Conflict resolution** — Resolve contradictory relations per entity pair
7. **Temporal parsing** — Date/time extraction with relative date support

Entity types and relation labels are fully configurable via `ctxgraph.toml`.

### Bi-Temporal History

Every relationship tracks two time dimensions:

- **valid_from / valid_until** — when was this true in the real world?
- **recorded_at** — when was this recorded?

Facts are never deleted — they are invalidated. You can query the graph as it existed at any point in time.

```
Alice -[works_at]-> Google   (2020-01 to 2025-06)
Alice -[works_at]-> Meta     (2025-06 to now)
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
      "command": "ctxgraph-mcp"
    }
  }
}
```

Install the MCP server separately: `cargo install ctxgraph-mcp`

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
ctxgraph models download                              Download ONNX models
ctxgraph watch --git [--last <n>]                     Auto-capture git commits (planned)
ctxgraph export --format json|csv                     Export graph data (planned)
ctxgraph-mcp                                           Run as MCP server (separate binary)
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
chose = { head = ["Person"], tail = ["Component"], description = "person chose" }
rejected = { head = ["Person"], tail = ["Component"], description = "person rejected" }
depends_on = { head = ["Component"], tail = ["Component"], description = "dependency" }
```

## Benchmark

The extraction pipeline is evaluated against 50 software-engineering episodes covering all 10 entity types and 9 relation types. Scores are macro-averaged F1.

The full benchmark corpus of 50 episodes and ground-truth annotations is [available in the repository](crates/ctxgraph-extract/tests/fixtures/benchmark_episodes.json) for inspection and reproduction. The corpus was authored by us — not cherry-picked to favor our heuristics, but we're transparent about this. If you find cases where ctxgraph gets it wrong, [open an issue](https://github.com/rohansx/ctxgraph/issues) or submit new episodes to make the benchmark more robust.

```bash
cargo test --test benchmark_test -- --ignored --nocapture
```

Requires ONNX models (`ctxgraph models download`).

### Results (v0.6.0 — GLiNER v2.1 INT8, fully local)

| Metric | Score |
|---|---|
| Entity F1 | 0.837 |
| Relation F1 | 0.763 |
| **Combined F1** | **0.800** |
| Latency | ~40ms/episode |

### Comparison with Graphiti

Both systems were tested on the same 50 episodes with identical ground truth.

| | ctxgraph | Graphiti (gpt-4o) |
|---|---|---|
| Entity F1 | **0.837** | 0.570 |
| Relation F1 | **0.763** | 0.104\* |
| Combined F1 | **0.800** | 0.337 |
| API calls | 0 | ~200+ |
| Cost | $0 | ~$2-5 |
| Per episode | ~40ms | ~10s |
| Infrastructure | SQLite | Neo4j (Docker) |
| Privacy | 100% local | Data sent to OpenAI |

\*With generous semantic mapping of Graphiti's free-form relations to ctxgraph's taxonomy.

## Project Structure

```
crates/
+-- ctxgraph-core/       Core engine: types, storage, query, temporal
+-- ctxgraph-extract/    Extraction pipeline (GLiNER ONNX, heuristics)
+-- ctxgraph-embed/      Local embedding generation
+-- ctxgraph-cli/        CLI binary
+-- ctxgraph-mcp/        MCP server for AI agents
+-- ctxgraph-sdk/        Re-export crate for embedding in Rust apps
```

## Design Principles

1. **Zero infrastructure** — One binary, one SQLite file
2. **Offline-first** — No internet required after model download
3. **Privacy by default** — Nothing leaves your machine
4. **Schema-driven** — Extraction labels are user-defined, not hardcoded
5. **Embeddable** — Rust library first, CLI second
6. **Append-only history** — Facts invalidated, never deleted

## License

MIT
