# ctxgraph

**Privacy-first knowledge graph engine for AI agents.**

Extracts entities and relations from any text. Builds a temporal knowledge graph. Works locally with zero API keys — and when it does call an LLM, it makes one call per episode instead of Graphiti's six.

```bash
brew install rohansx/tap/ctxgraph
ctxgraph init && ctxgraph models download
ctxgraph log "Migrated auth from Redis sessions to JWT. Chose JWT for stateless scaling."
ctxgraph query "Why did we move away from Redis?"
```

---

## Why ctxgraph?

Every knowledge graph engine requires an LLM for every operation. Graphiti makes 6 API calls per episode. Mem0 calls GPT-4 on every add/search. Microsoft GraphRAG is so expensive they put a cost warning in their README.

ctxgraph runs a tiered extraction pipeline: local ONNX models handle most episodes at zero cost, and only escalates to an LLM when local confidence is low. One call, not six. PII stripped before it leaves your machine.

**Benchmarked on 20 random real-world texts (GPT-4o judge, 10-point scale):**

| | ctxgraph (local) | ctxgraph + Ollama | Graphiti (Neo4j+GPT-4o) |
|---|---|---|---|
| **Quality score** | 5.0/10 | **7.6/10** | 8.2/10 |
| **With cloud LLM (26B)** | — | — | **8.4/10** |
| LLM calls per episode | **0** | **0-1** | **5-6** |
| Works without LLM? | **Yes** | **Yes** | No |
| Works offline? | **Yes** | **Yes** | No |
| Query latency | **<15ms** | **<15ms** | ~300ms |
| Infrastructure | **SQLite** | **SQLite + Ollama** | Neo4j+Docker |
| Cost per 1000 episodes | **$0** | **$0** | ~$2-5 |
| Language | **Rust** | **Rust** | Python |
| Privacy (PII protection) | **Full** | **Full** | None |

ctxgraph's local ONNX tier handles tech-domain text at 0.800 F1. For cross-domain, the Ollama tier (auto-detected) uses Gemma 3n E4B at zero cost. Cloud escalation to Gemma 4 26B MoE hits 8.4/10 quality at $0.13/1M tokens.

---

## How It Works

```
Text comes in
    │
    ▼
[Tier 1] GLiNER + GLiREL (local ONNX, FREE, ~30ms)
    │
    ▼ confidence < threshold?
    │
[Tier 2] Ollama auto-detected (Gemma 3n E4B, FREE, ~18s, 3.8GB VRAM)
    │
    ▼ Ollama not available?
    │
[Tier 3] Cloud API + CloakPipe PII stripping ($0.13/1M tokens)
```

| Tier | What | Cost | Latency | Quality |
|---|---|---|---|---|
| **Local ONNX** | GLiNER (entities) + GLiREL (relations) | $0 | ~30ms | 5.0/10 cross-domain, 0.800 F1 tech |
| **Local Ollama** | Gemma 3n E4B (auto-detected if installed) | $0 | ~18s | 7.6/10 cross-domain |
| **Cloud LLM** | Gemma 4 26B MoE via OpenRouter | $0.13/1M | ~5s | 8.4/10 cross-domain |
| **Dedup** | Jaro-Winkler similarity + alias table | $0 | <1ms | — |
| **Search** | FTS5 + semantic + graph walk, fused via RRF | $0 | <15ms | — |

**Auto-schema inference**: Log 3 episodes → ctxgraph infers domain-specific entity/relation types automatically. No manual schema definition needed.

Graphiti does ALL of these via LLM: entity extraction, deduplication, relation extraction, contradiction detection, summarization, community detection. Five calls. Every episode.

---

## Competitive Landscape

### Knowledge Graph Engines

| | ctxgraph | Graphiti | Cognee | WhyHow.AI |
|---|---|---|---|---|
| **Extraction** | Local ONNX + LLM fallback | LLM only (6 calls/ep) | LLM only | LLM only (OpenAI) |
| **Graph DB** | SQLite (embedded) | Neo4j/FalkorDB | Neo4j/Kuzu | MongoDB Atlas |
| **Works offline?** | **Yes** | No | No | No |
| **Temporal queries** | **Bi-temporal** | Bi-temporal | No | No |
| **MCP support** | Yes | Yes | Yes | No |
| **Language** | Rust (single binary) | Python | Python | Python |
| **Schema-driven** | Yes (ctxgraph.toml) | Yes (prescribed ontology) | Yes | Yes |
| **Cost/1000 eps** | **$0.30** | $1.80 | ~$1.50 | ~$2+ |
| **Stars** | Early | 24K | 15K | 900 |

### Agent Memory Systems

| | ctxgraph | Mem0 | Basic Memory | mcp-memory-service |
|---|---|---|---|---|
| **Entity extraction** | **Automated (ONNX+LLM)** | LLM-only | None (manual) | None (manual) |
| **Relation extraction** | **Automated (GLiREL+LLM)** | Limited | None | Manual typed edges |
| **Knowledge graph** | **Yes (temporal)** | Optional (Neptune) | Semantic links | Basic typed edges |
| **Works without LLM?** | **Yes** | No | Yes | Yes |
| **Query latency** | **<15ms** | ~100ms | ~10ms | ~5ms |
| **Dedup** | Jaro-Winkler + aliases | LLM-based | None | None |
| **Cost** | **$0 (local) / $0.30 (hybrid)** | LLM cost per op | $0 | $0 |
| **Stars** | Early | 51K | 2.7K | 1.6K |

### Graph-Enhanced RAG

| | ctxgraph | LightRAG | Microsoft GraphRAG | nano-graphrag |
|---|---|---|---|---|
| **Purpose** | Knowledge graph engine | RAG retrieval | Document summarization | Lightweight GraphRAG |
| **Incremental updates** | **Yes** | Yes | **No** (batch only) | No |
| **Temporal awareness** | **Yes (bi-temporal)** | No | No | No |
| **LLM per query** | **No** | Yes | Yes | Yes |
| **Offline capable** | **Yes** | Partial (Ollama) | No | Partial |
| **Cost/1000 docs** | **$0.30** | ~$1-5 | ~$10-50 | ~$1-5 |
| **Stars** | Early | 31K | 32K | 3.8K |

### What Makes ctxgraph Unique

**No other tool has all of these:**
1. **Zero-config schema inference** — Log 3 episodes, ctxgraph auto-infers domain-specific entity/relation types. No manual schema definition.
2. **Tiered local-first pipeline** — ONNX handles ~70% of extractions at $0. Ollama auto-detection adds cross-domain quality at $0. Cloud LLM only when needed.
3. **5x fewer LLM calls** — 0-1 calls per episode vs Graphiti's 5. Same quality, fraction of the cost.
4. **Runs on a laptop GPU** — Gemma 3n E4B fits in 3.8GB VRAM (RTX 4050), scores 7.6/10 on cross-domain extraction.
5. **Bi-temporal history** — Only ctxgraph and Graphiti have this. Every other tool is current-state only.
6. **PII protection** — CloakPipe strips PII before cloud LLM calls (`--features cloakpipe`). No competitor offers this.
7. **Single Rust binary** — Every competitor is Python with pip/Docker/Neo4j. ctxgraph is `cargo install`.

---

## Benchmarks

### Real-World Extraction (GPT-4o judge, 20 random texts, 10-point scale)

| System | Score | Time | Cost | Infrastructure |
|---|---|---|---|---|
| ctxgraph + Gemma 4 26B (cloud) | **8.4/10** | ~25s/ep | $0.13/1M | SQLite |
| Graphiti + GPT-4o | 8.2/10 | ~16s/ep | ~$2-5/batch | Neo4j + Docker |
| ctxgraph + Gemma 3n E4B (local) | 7.6/10 | ~18s/ep | **$0** | **SQLite only** |
| ctxgraph local ONNX only | 5.0/10 | ~30ms/ep | **$0** | **SQLite only** |

### LLM Model Comparison (10 random texts, GPT-4o judge)

| LLM | Hostable locally? | Score | Cost/1M tokens |
|---|---|---|---|
| Gemma 4 26B MoE | 24GB GPU | **8.4/10** | $0.13 |
| Gemma 4 31B | 24GB GPU | 8.2/10 | $0.14 |
| GPT-4o-mini | Cloud only | 8.2/10 | $0.15 |
| **Gemma 3n E4B** | **6GB GPU** | **7.6/10** | **$0 (local)** |
| GPT-4o | Cloud only | 7.0/10 | $2.50 |

### Schema-Typed Extraction (50 tech-domain episodes, F1 scores)

| System | Entity F1 | Relation F1 | Combined F1 |
|---|---|---|---|
| ctxgraph local ONNX | **0.837** | **0.763** | **0.800** |
| Gemma 4 31B (best LLM) | 0.658 | 0.374 | 0.516 |
| GPT-4o-mini | 0.625 | 0.191 | 0.408 |
| Graphiti + GPT-4o | 0.570 | 0.104 | 0.337 |

### Query Performance

| | ctxgraph | Graphiti |
|---|---|---|
| Full-text search | **<1ms** | ~50ms |
| Semantic search | **3-5ms** | ~100ms |
| Graph traversal (2-3 hops) | **<5ms** | 5-50ms |
| Fused search (RRF) | **<15ms** | ~300ms |

---

## Key Features

- **Tiered extraction** — Local ONNX first → Ollama auto-detected → Cloud LLM fallback
- **Auto-schema inference** — Domain-specific entity/relation types inferred from first 3 episodes
- **Privacy** — CloakPipe strips PII before cloud LLM calls (enable with `--features cloakpipe`)
- **Zero infrastructure** — One binary, one SQLite file. No Docker, no Neo4j
- **Ollama auto-detection** — Finds local Ollama and picks the best model automatically
- **Any LLM** — Ollama (Gemma 3n/4), OpenRouter, OpenAI, Anthropic
- **Bi-temporal** — Time-travel queries, fact invalidation
- **Schema-driven** — Auto-inferred or manual via `ctxgraph.toml`
- **MCP server** — Claude Code, Cursor, Cline, any MCP client
- **Embeddable** — Rust library, CLI, or MCP server
- **Entity dedup** — Jaro-Winkler + alias table across episodes

---

## Installation

```bash
# Homebrew
brew install rohansx/tap/ctxgraph

# Or build from source (Rust 1.85+)
cargo install ctxgraph-cli
```

## Quick Start

```bash
ctxgraph models download              # one-time ONNX model download
ctxgraph init                         # initialize graph in current dir
ctxgraph log "Alice chose PostgreSQL"  # extract + store
ctxgraph query "why PostgreSQL?"       # search the graph
```

### Optional: Local LLM for cross-domain quality

```bash
# Install Ollama + Gemma 3n E4B (one-time, 7.5GB download)
ollama pull gemma3n:e4b

# ctxgraph auto-detects Ollama — no config needed!
# You'll see: [ctxgraph] Ollama detected with model 'gemma3n:e4b' (local, free)
```

### Optional: Cloud LLM for maximum quality

```toml
# ctxgraph.toml
[llm]
provider = "openrouter"
model = "google/gemma-4-26b-a4b-it"   # 8.4/10 quality, $0.13/1M tokens
api_key_env = "OPENROUTER_API_KEY"
```

### Auto-schema inference

No need to define entity types manually. After 3 episodes, ctxgraph automatically infers domain-specific types:

```bash
ctxgraph log "Pfizer's Ozempic generated $28B revenue..."
ctxgraph log "FDA approved Casgevy gene therapy..."
ctxgraph log "NHS launched Federated Data Platform by Palantir..."
# [ctxgraph] Schema inferred: domain='Healthcare', 9 entity types, 10 relation types
# Saved to .ctxgraph/schema.toml — all future extractions use this schema
```

---

## MCP Server

```json
{
  "mcpServers": {
    "ctxgraph": { "command": "ctxgraph-mcp" }
  }
}
```

| Tool | Description |
|---|---|
| `ctxgraph_add_episode` | Record a decision or event |
| `ctxgraph_search` | Fused FTS5 + semantic + graph search |
| `ctxgraph_traverse` | Walk the graph from an entity |
| `ctxgraph_find_precedents` | Find similar past events |
| `ctxgraph_list_entities` | List entities with filters |
| `ctxgraph_export_graph` | Export entities and edges |

## Rust SDK

```rust
let graph = ctxgraph::Graph::init(".ctxgraph")?;
graph.add_episode(Episode::builder("Chose Postgres for billing").build())?;
let results = graph.search("why Postgres?", 10)?;
```

## Project Structure

```
crates/
+-- ctxgraph-core/       Types, storage, query, temporal
+-- ctxgraph-extract/    Tiered extraction (ONNX + LLM)
+-- ctxgraph-embed/      Local embeddings
+-- ctxgraph-cli/        CLI binary
+-- ctxgraph-mcp/        MCP server
+-- ctxgraph-sdk/        Rust SDK
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT
