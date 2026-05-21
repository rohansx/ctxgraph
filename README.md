# ctxgraph

**Typed knowledge graph for AI agents. Single Rust binary. Single SQLite file. One LLM call per write. Zero LLM calls for 90% of reads.**

```bash
brew install rohansx/tap/ctxgraph
ctxgraph init
ctxgraph log "Migrated auth from Redis sessions to JWT. Chose JWT for stateless scaling."
ctxgraph query "why did we move away from Redis?"
```

> **Working spec**: [`docs/CLARITY.md`](docs/CLARITY.md) — product, decisions, the 5 pieces to build, launch pitch.
> **Architecture**: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — as-built (§1-4) + v0.3 target (§5-14).
> **Roadmap**: [`docs/ROADMAP.md`](docs/ROADMAP.md) — 5 pieces + 12-week schedule + this-weekend todo.
> **Benchmarks**: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) — measured F1 numbers + hostile-reader audit.

---

## Headline result (measured 2026-05-13)

Same LLM, same fixture, same scoring code. ctxgraph's single-call schema-typed prompt vs Graphiti's 6-call pipeline:

| Metric (Gemma 4 26B-A4B in both columns) | ctxgraph | Graphiti | Δ |
|---|---|---|---|
| entity F1 (pair-fuzzy) | 0.819 | 0.824 | -0.005 |
| **relation F1 (pair-fuzzy)** | **0.555** | 0.096 | **+0.459 (5.8×)** |
| **combined F1** | **0.687** | 0.460 | **+0.227** |

The win **replicates** with Gemma 4 31B (ctxgraph 0.739 vs Graphiti 0.467, +0.272 combined / +0.499 relation). It's architectural, not model-specific. Graphiti's 6-call pipeline tops out at combined F1 ≈ 0.46 regardless of which Gemma you feed it.

Fixture: 29 hand-labeled cross-domain episodes covering 25 domains (`crates/ctxgraph-extract/tests/fixtures/cross_domain_v2.json`). Scoring code: `scripts/openrouter_bench.py` + `scripts/graphiti_openrouter_bench.py`. Raw per-episode outputs: `scripts/results/v0.9_cross_domain_v2/*.json`. Total cost to reproduce: ~$0.15.

[Full benchmark detail → docs/BENCHMARKS.md](docs/BENCHMARKS.md)

---

## How it works

```
                 ┌──────────────────────────────────────┐
                 │       WRITE PATH (one LLM call)       │
                 │  Tier 1: GLiNER2 ONNX (CPU, ~30ms)    │
                 │  Tier 2: NuExtract 2.0 (local Ollama) │
                 │  Tier 3: Cloud (only if needed)       │
                 │    Mode B default: Cerebras free       │
                 │    Paid: DeepInfra Gemma-4-26B-A4B    │
                 └──────────────────────────────────────┘
                                  │
                                  ▼
                 ┌──────────────────────────────────────┐
                 │       SQLite + FTS5 + sqlite-vec      │
                 │       bi-temporal edges, RRF search   │
                 └──────────────────────────────────────┘
                                  ▲
                                  │
                 ┌──────────────────────────────────────┐
                 │   READ PATH (zero LLM in 90% cases)   │
                 │  Simple (90%):                        │
                 │    verb → typed relation via cosine   │
                 │    embedding match (~30 LOC)          │
                 │    then deterministic SQL             │
                 │  Complex (10%):                       │
                 │    local Qwen3-1.5B parses NL →       │
                 │    graph op, then SQL                 │
                 │  NO cloud LLM ever in read path       │
                 └──────────────────────────────────────┘
```

Two architectural bets:

1. **One LLM call per write.** Tiered escalation: local ONNX handles ~70% of episodes, local LLM another 25%, cloud only when both fail. Compare to Graphiti's 6 calls per episode.
2. **Zero LLM calls in the read path for 90% of queries.** The universal schema's 10 typed relations are a closed set — your user verb cosine-matches to one of them, then SQL runs deterministically. Only multi-hop / time-filter / conjunction queries (~10%) call a tiny **local** Qwen3-1.5B. No cloud LLM ever sees a read.

This is the bit competitors can't match. Graphiti, Mem0, Letta all need an LLM at read time because their relation types are free-form text the SQL engine can't reason about.

---

## The universal schema (v0.3 target)

9 entity types, 10 relations, hardcoded. Users **never write a schema**.

| Entity types | Relation types |
|---|---|
| Person, Place, Organization, Concept, Artifact, Event, Time, Idea, Fact | mentions, located_at, related_to, caused, preceded, references, owned_by, part_of, depends_on, participated_in |

Broad enough to handle personal wikis, work notes, research, recipes, code, journal entries — anything text-shaped. Edge-case domains (recipes need "Ingredient", scientific datasets need "Measurement") get handled by an **automatic schema-improvement loop**: the LLM logs suggestions to a side-table; a nightly cron promotes types that show up across ≥ 5 distinct episodes with cosine-similarity < 0.85 to any existing type. Users see this as a one-line notice the next time they invoke the CLI.

[Full schema rationale → docs/CLARITY.md § 3](docs/CLARITY.md)

---

## Three modes

You pick one at `ctxgraph init`. All three keep reads local.

| Mode | Writes | Cost / 1k eps | Best for |
|---|---|---|---|
| **`local-only`** | GLiNER2 → NuExtract 2.0 → Qwen3-8B (all local) | $0 | Privacy / offline / sensitive data |
| **`cloud-fallback`** (default) | Local first; Cerebras free tier when local is stuck | $0 in practice* | Most users |
| **`cloud-quality`** | Skip local; every episode goes to Cerebras Qwen3-32B or DeepInfra Gemma-4-26B-A4B | $0–$0.11 | Long-form text, research papers |

\* Cerebras free tier = 1M tokens/day, 30 RPM. Enough for ~1 250 episodes/day. DeepInfra Gemma-4-26B-A4B ($0.07 in / $0.34 out, ~$0.11/1k eps) is the paid fallback when Cerebras rate-limits.

`allow_cloud = false` in `~/.ctxgraph/config.toml` forces Mode A regardless of `mode` — the privacy override.

---

## Competitive landscape

| | ctxgraph | Graphiti / Zep | Mem0 | Letta | Cognee |
|---|---|---|---|---|---|
| Distribution | **single Rust binary** | Python + Neo4j + Docker | Python SDK | Python | Python + Neo4j |
| Local-only mode | **yes** | no | no | yes (Apache 2.0) | no |
| LLM calls per write | **1** | 6 | N | varies | varies |
| LLM in read path | **no (90% of queries)** | yes | yes | yes | yes |
| Schema-typed extraction | **yes (universal 9/10)** | free-form verbs | free-form | typed but manual | typed but manual |
| Bi-temporal edges | **yes** | yes | no | no | no |
| Verified $/1k eps (Gemma 4 26B) | **$0.11** | ~$0.66 (6×) | N/A | N/A | N/A |
| Apples-to-apples combined F1 vs ctxgraph (same model) | **0.687** | 0.460 | not measured | not measured | not measured |
| Stars (rough, May 2026) | early | ~20K | ~50K | ~30K | ~15K |

[More competitor analysis → docs/ROADMAP.md § "Competitive landscape"](docs/ROADMAP.md)

---

## What's in the box today (v0.8.0)

| Component | Status | Lines |
|---|---|---|
| `ctxgraph-core` — SQLite + FTS5 + bi-temporal graph | shipped | ~2 000 |
| `ctxgraph-extract` — tiered extraction (current: GLiNER + GLiREL + LLM gate) | shipped | ~8 500 |
| `ctxgraph-embed` — fastembed wrapper, all-MiniLM-L6-v2 (384-dim) | shipped | ~70 |
| `ctxgraph-cli` — init, log, query, entities, stats, models, mcp start | shipped | ~600 |
| `ctxgraph-mcp` — MCP server, 6 tools | shipped | ~870 |

**v0.3 is the next launch** — see [`docs/ROADMAP.md`](docs/ROADMAP.md). It swaps GLiNER + GLiREL for GLiNER2 (single forward pass), adopts the universal schema, adds the no-LLM read path, defaults to Cerebras free tier, and re-runs the 29-episode benchmark to confirm the headline lands at ≥ 0.745 combined F1 with a fully local stack.

---

## Install

```bash
# macOS + Linux (prebuilt binaries via Homebrew)
brew install rohansx/tap/ctxgraph

# or from source (Rust 1.85+)
cargo install ctxgraph-cli
```

## Quick start

```bash
ctxgraph init
ctxgraph log "Alice chose PostgreSQL over MySQL for the new billing service."
ctxgraph log "PostgreSQL replaced MySQL in prod on 2026-04-12."
ctxgraph query "what did Alice choose?"
ctxgraph query "what was replaced?"
```

## MCP server (Claude Code / Cursor / Cline)

```json
{
  "mcpServers": {
    "ctxgraph": { "command": "ctxgraph-mcp" }
  }
}
```

| Tool | Description |
|---|---|
| `ctxgraph_add_episode` | Record an event or decision |
| `ctxgraph_search` | Fused FTS5 + semantic + graph search |
| `ctxgraph_traverse` | Walk the graph from an entity |
| `ctxgraph_find_precedents` | Find similar past events |
| `ctxgraph_list_entities` | List entities with filters |
| `ctxgraph_export_graph` | Export entities and edges |

## Rust SDK

```rust
use ctxgraph::{Graph, Episode};

let mut graph = Graph::init(".ctxgraph")?;
graph.add_episode(
    Episode::builder("Chose Postgres over Mongo for the billing rewrite").build()
)?;
let results = graph.search("why Postgres?", 10)?;
```

## Project structure

```
crates/
├── ctxgraph-core/      types, storage, query, temporal
├── ctxgraph-extract/   tiered extraction (ONNX + LLM)
├── ctxgraph-embed/     local embeddings (384-dim)
├── ctxgraph-cli/       CLI binary
└── ctxgraph-mcp/       MCP server
```

## Reproducing the benchmark

```bash
export OPENROUTER_API_KEY=sk-or-...
python scripts/openrouter_bench.py \
  --model google/gemma-4-26b-a4b-it \
  --out bench.json \
  --skip-tech \
  --cd-fixture crates/ctxgraph-extract/tests/fixtures/cross_domain_v2.json

# Spin up Neo4j for Graphiti
docker run -d --name neo4j-bench -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/benchpass123 neo4j:5.26-community

# Graphiti through OpenRouter (needs Python 3.12 venv)
python3.12 -m venv /tmp/graphiti_venv
/tmp/graphiti_venv/bin/pip install graphiti-core neo4j openai
/tmp/graphiti_venv/bin/python scripts/graphiti_openrouter_bench.py \
  --model google/gemma-4-26b-a4b-it \
  --out graphiti.json

python scripts/compare_v2.py
```

Total cost: ~$0.15. Total wall-clock: ~90 minutes.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). For design discussions, [`docs/CLARITY.md`](docs/CLARITY.md) is the working doc — propose changes against it.

## License

MIT
