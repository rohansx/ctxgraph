# ctxgraph — Roadmap

> Privacy-first knowledge graph engine. Better quality, 6x cheaper, 3x faster than Graphiti.

---

## Timeline

```
     Phase 1 (Wk 1-3)      Phase 2 (Wk 4-6)     Phase 3 (Wk 7-9)     Phase 4 (Wk 10-12)
    +------------------+  +------------------+  +------------------+  +------------------+
    | Smart Extraction |  | Privacy +        |  | Agent Features   |  | Polish + Launch  |
    |                  |  | Cross-Domain     |  |                  |  |                  |
    | Confidence gate  |  | CloakPipe integ  |  | Reflect API      |  | Web dashboard    |
    | LLM fallback     |  | Cross-domain     |  | (agent reflection|  | Python SDK       |
    | Ollama + cloud   |  | benchmarks       |  |  + memory)       |  | Docs + guides    |
    | GLiREL primary   |  | Entity cache     |  | Dev memory       |  | Homebrew update  |
    | Cost tracking    |  | Threshold tuning |  | (connectors)     |  | Blog + HN        |
    |                  |  |                  |  |                  |  |                  |
    | CORE ENGINE      |  | PRIVACY + QUALITY|  | AGENT LAYER      |  | SHIP + LAUNCH    |
    +--------+---------+  +--------+---------+  +--------+---------+  +------------------+
             |                     |                     |
             |    Benchmarks ------+                     |
             |    vs Graphiti                            |
             |                                          |
         Library usable as Graphiti alternative     Screenpipe pipe PR
```

---

## What Exists (v0.8.0 — Shipped)

| Component | Status | Notes |
|---|---|---|
| `ctxgraph-core` | Shipped | SQLite + FTS5 + WAL, bi-temporal edges, RRF search, BFS traversal |
| `ctxgraph-extract` | Shipped | GLiNER2 + GLiREL + LLM fallback pipeline, confidence gate, temporal parsing |
| `ctxgraph-embed` | Shipped | fastembed, all-MiniLM-L6-v2, 384-dim vectors |
| `ctxgraph-mcp` | Shipped | 8 MCP tools (add_episode, search, traverse, find_precedents, etc.) |
| `ctxgraph-cli` | Shipped | init, log, query, entities, stats, models, mcp start |
| Homebrew tap | Shipped | macOS + Linux prebuilt binaries |
| GLiREL integration | Shipped | Zero-shot relation extraction, schema-aware direction resolution |
| LLM fallback | Shipped | Confidence gate, OpenRouter/Ollama, entity+relation extraction |
| Cross-domain benchmarks | Shipped | 10 episodes across 6 domains, 0.552 F1 with Gemini Flash |

---

## Phase 1: Smart Extraction Engine (Weeks 1-3) — CORE ENGINE [DONE]

The tiered extraction pipeline is built and benchmarked:

- Confidence gate detects when local models aren't confident
- LLM fallback extracts entities + relations in ONE call (vs Graphiti's 6)
- GLiREL handles relations locally when entities are good
- Schema-aware direction resolution fixes GLiREL head/tail errors
- Entity canonicalization handles verbose LLM names
- Fuzzy matching for evaluation

### Benchmark Results

| Mode | Tech F1 | Cross-domain F1 | Cost per 1000 eps |
|---|---|---|---|
| Local only | **0.800** | 0.325 | $0 |
| + Llama 3.2 3B (Ollama) | **0.800** | 0.472 | $0 |
| + Qwen 2.5 7B (Ollama) | **0.800** | 0.508 | $0 |
| + Gemini Flash (cloud) | **0.800** | **0.552** | ~$0.05 |
| Graphiti (GPT-4o) | 0.337 | varies | ~$30 |

---

## Phase 2: Privacy + Cross-Domain Quality (Weeks 4-6)

### CloakPipe Integration

CloakPipe (already built as a separate product) gets integrated into the extraction pipeline:

| Deliverable | Description |
|---|---|
| PII scanner in extraction pipeline | Detect API keys, passwords, emails, tokens, names before LLM call |
| Reversible redaction | Replace PII with deterministic placeholders, map back after LLM response |
| Encrypted entity cache | AES-256-GCM vault for entity type resolutions — same pattern never costs twice |
| Config flag | `[privacy] cloakpipe = true` in ctxgraph.toml |

**Why this matters**: Even with Gemini Flash at $0.05/1000 episodes, enterprise users won't send raw text to a cloud LLM. CloakPipe makes the cloud tier enterprise-safe.

### Cross-Domain Quality Improvements

| Deliverable | Description |
|---|---|
| Expand cross-domain benchmarks | 50+ episodes across 10+ domains |
| Entity cache warming | Pre-populate common entity resolutions to reduce initial LLM costs |
| Threshold tuning | Optimize confidence thresholds per entity type and domain |
| Better LLM prompts | Domain-specific prompt variants for finance, healthcare, legal |
| Cost dashboard | Track local vs LLM episodes, tokens used, cost per domain |

### Target Metrics

| Metric | Target |
|---|---|
| Tech F1 (local only) | >= 0.800 (maintain) |
| Cross-domain F1 (Gemini Flash) | >= 0.650 |
| Cross-domain F1 (Ollama 7B) | >= 0.550 |
| LLM escalation rate (mixed workload) | <= 30% |

---

## Phase 3: Agent Features (Weeks 7-9)

### Reflect API — Reflection for Agents

Add a `reflect` operation to the MCP server and Rust SDK. Agents can ask ctxgraph to **reflect** on its knowledge — finding contradictions, gaps, patterns, and insights across the graph.

| MCP Tool | Description |
|---|---|
| `ctxgraph_reflect` | Analyze the graph for contradictions, patterns, and gaps |
| `ctxgraph_reflect_on` | Reflect on a specific entity or topic |
| `ctxgraph_suggest` | Suggest related context the agent should consider |

**How it works**: Reflect traverses the graph, finds clusters of related entities, detects temporal contradictions (e.g., "chose X" followed by "replaced X" without "deprecated X"), and surfaces patterns the agent might not have asked about.

**Why this matters**: Current MCP tools are reactive — the agent asks, ctxgraph answers. Reflect is proactive — ctxgraph tells the agent what it should know. This is the "memory layer for agents" use case.

```rust
// Agent asks: "what should I know about the auth system?"
let insights = graph.reflect_on("auth")?;
// Returns:
// - "JWT was chosen over Redis sessions (March 2026)"
// - "Contradiction: JWT was marked as 'stateless' but SessionManager still depends on Redis"
// - "Gap: no test coverage recorded for auth middleware since migration"
```

### Developer Memory (Use Case Layer)

The dev memory engine features from the earlier product doc, now scoped as a **use case** on top of the core engine:

| Deliverable | Description | Crate |
|---|---|---|
| `ctxgraph-ingest` | Connector framework + git/shell/FS connectors | New |
| Git connector | git2-rs: commits, branches, diffs, authors -> entities + edges | ctxgraph-ingest |
| Shell history connector | bash/zsh/fish parsing -> command + error entities | ctxgraph-ingest |
| FS watcher connector | notify crate: file events | ctxgraph-ingest |
| Daemon mode | Background process: connectors + MCP server | ctxgraph-cli |

---

## Phase 4: Polish + Launch (Weeks 10-12)

| Deliverable | Description |
|---|---|
| Web dashboard | Graph visualization (inspired by Rowboat Labs graph UI) |
| Python SDK | `pip install ctxgraph` via PyO3 bindings |
| Documentation | User guide, API docs, MCP setup for Claude Code + Cursor |
| Homebrew update | Updated formula with LLM tier support |
| Blog post | "How ctxgraph Gives You Graphiti-Level Quality at 6x Lower Cost" |
| HN Show HN | Launch with benchmark comparison and cost analysis |

---

## Competitive Positioning

| | ctxgraph | Graphiti (GPT-4o-mini + Neo4j) | Screenpipe | CodeYam |
|---|---|---|---|---|
| Approach | Local ONNX + 1 LLM call fallback | 6 LLM calls per episode | Screen recording + OCR | Session transcript rules |
| Tech F1 (fair h2h) | **0.846** | 0.601 | No extraction | No extraction |
| Cross-domain F1 (fair h2h) | **0.650** | 0.474 | No extraction | No extraction |
| Tech relation F1 | **0.714** | 0.349 | — | — |
| Cross-domain relation F1 | **0.457** | 0.092 | — | — |
| Cost per 1000 episodes | **$0.30** | $1.80 | $0 (no extraction) | $0 (no extraction) |
| Cost (tech, local-only) | **$0** | $1.80 | — | — |
| Query latency (fused) | **<15ms** | ~300ms | ~1ms (FTS5 only) | N/A |
| Graph traversal | **<5ms** (recursive CTE) | 5-50ms (Cypher) | None | None |
| Privacy | **CloakPipe PII stripping** | Data sent to OpenAI | Local-first | Local (repo-scoped) |
| LLM calls per episode | **0 (tech) / 1 (cross-domain)** | **6 (always)** | 0 | 0 |
| Infrastructure | SQLite (single file) | Neo4j + Docker + Python | Desktop app | npm package |
| Works offline? | **Yes** | No | Yes | Yes |
| Works with Ollama? | **Yes** | No | N/A | N/A |
| Entity dedup | **Jaro-Winkler + alias table** | Neo4j merge | None | None |
| Temporal model | **Bi-temporal** | Point-in-time | None | None |
| MCP Support | Native | Yes | Yes | No |

---

## Success Metrics

### Phase 2 (Week 6)
- CloakPipe integration started
- Cross-domain F1 >= 0.650 with Gemini Flash
- Cost tracking showing 6x cost savings vs Graphiti

### Month 3
- 100+ GitHub stars
- Reflect API shipped and documented
- Python SDK on PyPI
- Blog post + HN submission

### Month 6
- 500+ GitHub stars
- Featured in developer newsletter or podcast
- Recognized as privacy-first Graphiti alternative
- Screenpipe pipe published

---

## Business Model

Three-tier open-core:

| Tier | Price | Includes |
|---|---|---|
| OSS (Free) | $0 | Core engine, local extraction, Ollama support, MCP server, CLI, Rust SDK |
| Pro | $15/month | Managed CloakPipe cloud, Gemini Flash integration, entity cache sync, web dashboard, priority support |
| Enterprise | Custom | SSO, audit logging, compliance export, team graph federation, dedicated support |

The free tier handles tech text at full quality with zero cost. Pro adds cloud LLM quality + managed privacy for cross-domain enterprise use.

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Graphiti adds local-first tier | High | CloakPipe privacy layer is the moat, not just cost |
| Local model quality improves (GLiNER v3?) | Low — this is a win | Reduces LLM dependency, lowers cost further |
| LLM providers raise prices | Medium | Ollama default is free. Entity cache amortizes cloud costs. |
| Cross-domain quality doesn't reach 0.650 | Medium | Focus on cost story (6x cheaper) even at 0.550 |
| CloakPipe integration complexity | Medium | CloakPipe already exists. Integration is wiring, not building. |
