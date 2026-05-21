# ctxgraph — Architecture

> **Status**: Authoritative architecture document as of 2026-05-14.
> **Master working doc**: `CLARITY.md` — this file is the technical deep-dive that elaborates it.
> **Supersedes**: `archive/ARCHITECTURE_v1.md` (aspirational pre-research version, kept for history).
> **Synthesizes**: actual codebase state (v0.8.0 in Cargo.toml, v0.9.0 features unmerged), measured benchmarks (`docs/research_brief.md`), and deep-research findings (`docs/deep-research/FINAL.md`).
>
> Two architectures live in this doc, clearly labeled:
> - **§ 1–4 — As-built**: what's in `crates/` today.
> - **§ 5–14 — Target v0.3**: what the post-research-synthesis architecture moves to before HN. Mirrors the **5 pieces to build** in `CLARITY.md` § 3.
>
> If a section header is **§-T** it's target. **§-A** is as-built.

---

## TL;DR

ctxgraph is a **single-binary, SQLite-only typed knowledge-graph engine** for AI agents. Two architectural bets:

1. **One LLM call per write** (tiered escalation: local ONNX → local LLM → cloud only as fallback). Measured headline: same model, same fixture — ctxgraph's single-call prompt beats Graphiti's 6-call pipeline by **+0.227 combined F1 (Gemma 4 26B) and +0.272 (Gemma 4 31B)**.
2. **Zero LLM calls in the read path for ~90% of queries.** Simple lookups embed-match the user's verb to one of 10 typed relations, then run deterministic SQL. Only multi-hop / conjunction / time-filter queries (~10%) call a tiny local Qwen3-1.5B to parse NL → graph op. **No cloud LLM ever touches a read in any mode.**

The v0.3 target architecture: replace GLiNER + GLiREL with GLiNER2 (single forward pass for NER + RE), introduce NuExtract 2.0 as the Tier-2 local extractive decoder, swap the schema from the legacy 10/9 tech taxonomy to a **universal 9/10 schema** (Person, Place, Organization, Concept, Artifact, Event, Time, Idea, Fact + 10 broad relations), default cloud writes to DeepInfra `google/gemma-4-26b-a4b-it` ($0.07 in / $0.34 out, with Cerebras free tier as the practical default), and treat host-memory prompt caching as a first-class concern.

The product spec lives in [`CLARITY.md`](./CLARITY.md). The five concrete pieces to build are listed there.

---

## 1. As-built (§-A): crate layout

```
crates/                                   ~12 000 lines of Rust
├── ctxgraph-core/         SQLite + FTS5 + bi-temporal graph engine
│   ├── graph.rs            685 lines — add_episode, search, traverse, time-travel
│   ├── storage/sqlite.rs   648 lines — DDL, migrations, query backends
│   └── types.rs                       — Episode, Entity, Edge, schema enums
├── ctxgraph-extract/      Tiered extraction pipeline (current)
│   ├── pipeline.rs         438 lines — NER → coref → remap → LLM gate → relations
│   ├── ner.rs                         — GLiNER ONNX wrapper
│   ├── rel.rs             1792 lines — relation extraction (GLiREL ONNX + heuristics)
│   ├── glirel.rs           717 lines — GLiREL zero-shot RE
│   ├── llm_extract.rs     1012 lines — Ollama/OpenRouter/OpenAI/Anthropic client + tiered autodetect
│   ├── schema.rs           596 lines — entity/relation taxonomy + auto-inference
│   ├── remap.rs           1262 lines — dictionary-based entity-type fixups
│   ├── coref.rs                       — pronoun resolution
│   ├── temporal.rs                    — date / duration parsing
│   └── model_manager.rs               — ONNX model download + cache
├── ctxgraph-embed/        fastembed wrapper, all-MiniLM-L6-v2 (384-dim)
├── ctxgraph-cli/          init, log, query, entities, stats, models, mcp start
└── ctxgraph-mcp/          MCP server, 6 tools
```

No `ctxgraph-ingest`, no `ctxgraph-sdk` separate crate, no `ctxgraph-privacy` — those exist only in the older `ARCHITECTURE.md` as aspirational. CloakPipe is wired in via a `cloakpipe` feature flag on `ctxgraph-extract`, not a separate crate.

### 1.1 Distribution

- `brew install rohansx/tap/ctxgraph` (macOS + Linux prebuilt)
- `cargo install ctxgraph-cli` (Rust 1.85+)
- `ctxgraph-mcp` binary for MCP clients (Claude Code, Cursor, Cline)

### 1.2 Active schema (taxonomy, legacy tech-focused)

10 entity types, 9 relation types, hard-coded in `crates/ctxgraph-extract/src/schema.rs`. This is the **tech-decision-record taxonomy**, tuned for ADRs, postmortems, and migration plans:

| Entity types | Relation types |
|---|---|
| Person, Component, Service, Language, Database, Infrastructure, Decision, Constraint, Metric, Pattern | chose, rejected, replaced, depends_on, fixed, introduced, deprecated, caused, constrained_by |

**Auto-schema inference** (v0.9, currently unmerged in commit `9dcb574` + `83f3487`): after the first 3 episodes, an LLM call infers a domain-specific taxonomy and writes it to `.ctxgraph/schema.toml`. All subsequent extractions use the inferred schema.

> **v0.3 target replaces this with a single universal schema** — see § 6. The auto-schema-inference behavior is replaced by a "track unmapped entities/relations, promote conservatively over time" mechanism (§ 9 / Piece 5).

---

## 2. As-built (§-A): extraction pipeline (current)

```
Episode text in
     │
     ▼
┌── Tier 1 — Local ONNX (always runs) ─────────────────────────────┐
│  GLiNER NER (~30 ms)                                              │
│   ↓ coref → dictionary supplement → entity-type remap →           │
│   canonicalization → de-overlap → article stripping               │
└───────────────────────────────────────────────────────────────────┘
     │
     ▼
[Confidence gate — pipeline.rs:242–246]
   Escalate if any of:
     - entity density < 1.5 per 10 words
     - avg confidence < 0.4
     - <60% of entities map to valid schema types
     - >25 words & <5 unique entities
     - text contains complexity markers (@, v2, ::, outage, …)
     │
     ▼
┌── Tier 2 — Ollama (auto-detected, free, local) ─────────────────┐
│  detect_ollama() probes http://localhost:11434/api/tags          │
│  Preferred models in order:                                       │
│   gemma3n:e4b → gemma4:e4b → gemma3n:e2b → gemma4:e2b →           │
│   llama3.2:3b                                                     │
│  CloakPipe strips PII before send (feature "cloakpipe")           │
└──────────────────────────────────────────────────────────────────┘
     │
     ▼
┌── Tier 3 — Cloud LLM (only if Ollama absent + cloud key set) ───┐
│  OpenAI / Anthropic / OpenRouter (OpenAI-compat endpoint)         │
│  Default in code (llm_extract.rs:15): gpt-4o-mini                 │
│  graph.rs:657 uses google/gemma-4-31b-it for schema inference     │
│  CloakPipe strips PII before any cloud call                       │
└──────────────────────────────────────────────────────────────────┘
     │
     ▼
[Merge] LLM entities not already in local results are added;
        LLM relations merged with GLiREL relations
     │
     ▼
[Relation extraction layer] GLiREL ONNX runs over all entities
     │
     ▼
[Schema validation] entity types and relation types filtered
        against active schema (manual or auto-inferred)
     │
     ▼
Episode + entities + edges → SQLite (bi-temporal: valid_from, valid_until)
```

---

## 3. As-built (§-A): storage model

SQLite + WAL + FTS5. Single file, embeddable.

| Table | Key columns |
|---|---|
| `episodes` | text, source, tags, created_at |
| `entities` | name, entity_type, attributes JSON |
| `aliases` | canonical entity ↔ Jaro-Winkler fuzzy match table |
| `edges` | head, relation, tail, fact, **valid_from**, **valid_until** (bi-temporal) |
| `embeddings` | episode_id → 384-dim vector blob |

FTS5 virtual tables on episode text + entity names. Search is **RRF-fused**: FTS5 + cosine semantic + recursive-CTE graph walk, combined via Reciprocal Rank Fusion. Median fused-search latency **<15 ms** on a million-row graph.

### 3.1 Temporal model

Every edge carries `valid_from` and `valid_until`:

- **Current view**: `WHERE valid_until IS NULL`
- **Time-travel**: `WHERE valid_from <= ?t AND (valid_until IS NULL OR valid_until > ?t)`
- **Activity window**: `WHERE created_at BETWEEN ?start AND ?end`

Facts are never deleted. They are invalidated by setting `valid_until`.

---

## 4. As-built (§-A): interfaces

### 4.1 MCP tools (shipped)

`ctxgraph_add_episode`, `ctxgraph_search`, `ctxgraph_traverse`, `ctxgraph_find_precedents`, `ctxgraph_list_entities`, `ctxgraph_export_graph`. Roadmapped but not yet built: `ctxgraph_reflect`, `ctxgraph_reflect_on`, `ctxgraph_suggest`.

### 4.2 CLI subcommands (shipped)

`init`, `log`, `query`, `entities`, `decisions`, `stats`, `models download`, `mcp start`.

### 4.3 Rust SDK

`ctxgraph-core` is the public crate. `Graph::init(path)`, `graph.add_episode(Episode)`, `graph.search(query, limit)`, `graph.traverse(entity, depth)`.

---

## 5. Target v0.3 (§-T): new write pipeline

> Source: `CLARITY.md` § 4 + `deep-research/FINAL.md` § 2.

```
Episode text in
     │
     ▼
┌── Tier 1 — GLiNER2 ONNX (single pass) ───────────────────────────┐
│  Replaces GLiNER + GLiREL split                                   │
│  ~205M params, single forward pass for NER + typed RE +           │
│  hierarchical JSON                                                │
│  CPU-runnable, <20 ms                                             │
│  Schema-aware: emits the universal 9/10 taxonomy natively (§6)    │
│  Model: fastino-ai/GLiNER2 (Apache-2.0)                           │
└───────────────────────────────────────────────────────────────────┘
     │
     ▼
[Confidence gate — unchanged from §-A]
     │
     ▼
┌── Tier 2 — Local extractive decoder ────────────────────────────┐
│  VRAM-autodetect routing:                                         │
│    < 4 GB → GLiNER2 only (skip Tier 2)                            │
│    4–8 GB → NuExtract 2.0-2B (Apache 2.0)                         │
│    8–16 GB → NuExtract 2.0-4B or Qwen3-8B-LoRA                    │
│    16–24 GB → Hermes-4-14B-LoRA Q4_K_M                            │
│  Host-memory prompt cache enabled by default                      │
│    Ollama: keep_alive=-1                                          │
│    llama.cpp: --cram 256 --system-prompt-file ./schema.txt        │
│    SWA models: --override-kv to disable sliding window            │
└──────────────────────────────────────────────────────────────────┘
     │
     ▼
┌── Tier 3 — Cloud (mode-dependent, see § 8) ──────────────────────┐
│  Mode B default: Cerebras free tier — Qwen3-32B → gpt-oss-120B    │
│    Free up to 1 M tok/day, 30 RPM                                 │
│  Paid fallback: DeepInfra google/gemma-4-26b-a4b-it               │
│    $0.07/M in, $0.34/M out (~$0.11/1k episodes)                   │
│  Premium / IE-quality: OpenRouter Hermes 4 70B                    │
│  CloakPipe PII stripping pre-call (feature "cloakpipe")           │
└──────────────────────────────────────────────────────────────────┘
     │
     ▼
┌── Tier 4 — Graph Judge (offline, nightly cron) ─────────────────┐
│  arXiv 2411.17388 — binary keep/reject on each stored triple      │
│  ~1.5B model fine-tuned on (text, triple, gold) pairs             │
│  Runs on the previous day's writes, marks low-confidence edges    │
└──────────────────────────────────────────────────────────────────┘
     │
     ▼
Bi-temporal SQLite + FTS5 + (planned) sqlite-vec
```

**Why this structure:**

1. **GLiNER2 replaces the GLiNER + GLiREL split.** One model, schema-aware, CPU-runnable. The encoder forward pass produces typed RE natively. Retires `rel.rs` (1792 lines) and `glirel.rs` (717 lines) — about 2 500 lines of code.
2. **NuExtract 2.0 trained with negative sampling** (empty-string outputs for absent facts) — eliminates an entire class of JSON-validation bugs that the current pipeline hand-codes around.
3. **Cerebras free tier** is the recommended default for non-privacy users (Mode B). 1 M tokens/day is enough for ~1 250 episodes/day; effectively free at any reasonable personal scale.
4. **DeepInfra Gemma 4 26B-A4B** as paid fallback: clean cloud / local parity — the same model family runs in Tier 2 (locally quantized) and Tier 3 (DeepInfra-hosted), so behavior is consistent across tiers.
5. **Graph Judge** is the quality safety net. Runs offline so it never adds latency to writes.

### 5.1 The single-call schema, bi-temporally aware

A v0.3 addition: the LLM emits **`invalidates:`** directly, as part of the same JSON, instead of a separate post-hoc pass. Full prompt + JSON contract is `CLARITY.md` § 3 / Piece 2. Summary:

```json
{
  "entities":    [{"id": "e1", "name": "...", "type": "Person|Place|...", "attributes": {}}],
  "relations":   [{"head": "e1", "relation": "depends_on", "tail": "e2",
                   "confidence": 0.0, "valid_from": null, "valid_to": null}],
  "invalidates": ["natural-language description of what this episode contradicts"],
  "suggestions": [...],
  "confidence":  0.87
}
```

The current-facts context is bounded by retrieving top-K facts touching each entity mentioned in the episode (5–10 facts × N entities, well within 4 k context). This is what competitors literally cannot copy without retraining their models — bi-temporal-aware single-call extraction is unique to ctxgraph.

---

## 6. Target v0.3 (§-T): the universal schema (Piece 1)

> Source: `CLARITY.md` § 3 / Piece 1. **Replaces the legacy tech-focused 10/9 schema from § 1.2.**

A single hardcoded taxonomy that ships with ctxgraph. Broad enough to handle personal wikis, work notes, technical content, recipes — anything. Lives at `crates/ctxgraph-extract/schemas/universal.toml`.

### 6.1 The 9 entity types

| Type | Description |
|---|---|
| Person | humans, named individuals |
| Place | locations, regions, venues, addresses |
| Organization | companies, teams, institutions, groups |
| Concept | ideas, theories, methodologies, technologies, terms |
| Artifact | concrete made objects: tools, systems, documents, code, products |
| Event | occurrences with a time: meetings, deploys, conferences, decisions |
| Time | explicit temporal anchors: dates, periods, durations |
| Idea | personal thoughts, hypotheses, plans, intentions |
| Fact | verified statements, measurements, observations |

### 6.2 The 10 relation types

| Type | Description |
|---|---|
| mentions | X is named in the context of Y |
| located_at | X is physically or conceptually at Y |
| related_to | X is associated with Y (fallback when nothing else fits) |
| caused | X led to Y |
| preceded | X happened before Y |
| references | X cites, links, or builds on Y |
| owned_by | Y owns or controls X |
| part_of | X is a component of Y |
| depends_on | X requires Y to function |
| participated_in | X was an actor in Y |

### 6.3 Why these specific types

- **Personal-wiki coverage**: People you meet, places you go, organizations you interact with, concepts you encounter, artifacts you build/use, events you participate in, dates, your own ideas, and verified facts. Nine entity types map to the nine things a journal or wiki actually accumulates.
- **Universal relations**: Eight specific verbs plus two utility verbs (`mentions` for co-occurrence with no other relation, `related_to` as the explicit "I know they're connected but don't know how" fallback). Cover ~95% of natural English connecting verbs.
- **Resist expansion**: V0.3 ships with exactly these 9 + 10. Piece 5 (§ 9) is the mechanism by which new types get added over time, conservatively, without users having to think about it.

### 6.4 Schema-invisibility commitment

Users in v0.3 **never write a schema and never see a schema file**. `ctxgraph init` doesn't prompt for types. The TOML file exists for power users (`ctxgraph schema edit`) but is hidden from the default flow. This is non-negotiable per `CLARITY.md` § 2.

---

## 7. Target v0.3 (§-T): the read path

> Source: `CLARITY.md` § 5. **Currently the read path is unspecified — this is new architecture.**

### 7.1 The crucial property

**No cloud LLM in the read path, ever, in any mode.** Even cloud-quality mode (§ 8 Mode C) keeps reads local. Cloud is only for writes.

This is the bit competitors can't match. Graphiti, Mem0, Letta all need an LLM at read time because their relation types are free-form text the SQL engine can't reason about. ctxgraph's typed relations make this unnecessary.

### 7.2 Query classification

```
USER ASKS NATURAL LANGUAGE QUERY
   │
   ▼
┌──────────────────────────────────────────────────┐
│  Step 1: Classify (heuristic, no LLM)            │
│    - count graph operations needed               │
│    - ≤ 1 → simple. > 1 → complex.                │
└──────────────────────────────────────────────────┘
   │
   ├── SIMPLE PATH (~90% of queries) ─────────────────────────────┐
   │                                                              │
   │  2a. Extract entity + verb from query                        │
   │      (regex + lightweight NER; local; instant)               │
   │  2b. Match verb → typed relation                             │
   │      (embedding cosine match against the 10 relations —      │
   │       Piece 3, § 9.3)                                        │
   │  2c. Run deterministic SQL                                   │
   │      (WHERE head=? AND relation=?  — bi-temporal predicates  │
   │       handled by index)                                      │
   │                                                              │
   │  Total: ~10–50 ms per query. Zero LLM calls.                 │
   │                                                              │
   └──────────────────────────────────────────────────────────────┘
   │
   └── COMPLEX PATH (~10% of queries) ────────────────────────────┐
                                                                  │
       2a. Send query to local Qwen3-1.5B via Ollama              │
           (few-shot prompt — Piece 4, § 9.4)                     │
           outputs structured graph-operation JSON                │
                                                                  │
       2b. Dispatch op to SQL handlers                            │
           op=traverse → recursive CTE                            │
           op=filter   → WHERE + time predicates                  │
           op=compare  → JOIN with grouping                       │
           op=list     → flat scan with type filter               │
                                                                  │
       Total: ~200–500 ms. One small local LLM call.              │
                                                                  │
   ──────────────────────────────────────────────────────────────┘
   │
   ▼
RESULTS with provenance + confidence
```

### 7.3 Why this works architecturally

Three properties combine:

1. **Typed relations are a finite known set.** A user verb only ever needs to map to one of 10 strings. That's a closed-set classification problem, perfectly suited to embedding cosine match.
2. **SQL handles graph traversal natively.** Recursive CTEs (already used in the FTS5+RRF fused search) cover multi-hop with no LLM involvement.
3. **Bi-temporal predicates are SQL.** Time-travel queries (`as of 2025-Q3`) compile to `WHERE valid_from <= ?t AND (valid_until IS NULL OR valid_until > ?t)` — no LLM reasoning needed.

The local LLM (Qwen3-1.5B) only enters when the query has *structure* the deterministic parser can't infer: multi-hop, conjunctions, time filters, comparisons. Even there, the LLM emits *a graph operation*, not a result — it never sees the data.

---

## 8. Target v0.3 (§-T): three modes

> Source: `CLARITY.md` § 4. **Replaces the implicit single-mode "always tier-up to whatever's available" behavior in current code.**

The user picks one mode at `ctxgraph init`. All three keep reads local; they differ only in the write-path tier chain.

### 8.1 Mode A — `local-only` (privacy mode)

Zero data leaves the machine.

```
Tier 1: GLiNER2 ONNX (CPU)              ~30 ms,  ~70% of episodes
Tier 2: NuExtract 2.0-4B (Ollama)       ~2–4 s,  ~25% of episodes
Tier 3: Qwen3-8B Q4 (Ollama)            ~5–8 s,  ~5% of episodes

Embedding model: all-MiniLM-L6-v2 (local, 384-dim)
Query parser:    Qwen3-1.5B (Ollama, loaded on demand)
```

Default if `[privacy] allow_cloud = false` is set.

### 8.2 Mode B — `cloud-fallback` (recommended default)

Local handles the easy 95%, cloud (free tier) handles the hard 5%.

```
Tier 1: GLiNER2 ONNX (local CPU)
Tier 2: NuExtract 2.0-4B (local Ollama, if available)
Tier 3: Cerebras free — Qwen3-32B  ($0 ≤ 1 M tok/day, 30 RPM)
Tier 4: DeepInfra paid — google/gemma-4-26b-a4b-it
        (~$0.11 / 1k episodes; fires only when Cerebras rate-limited)
```

Recommended for most users. `ctxgraph init` should default to this if a Cerebras key is detected, otherwise `local-only`.

### 8.3 Mode C — `cloud-quality` (power users)

Skip local tiers; every episode goes to a high-quality hosted model.

```
Default: Cerebras Qwen3-32B (free, fast — 2000+ tok/s)
Paid alt: DeepInfra Gemma-4-26B-A4B
Premium: OpenRouter Hermes 4 70B (highest IE quality)
```

Useful for long-form text (papers, transcripts) where local extraction quality degrades.

### 8.4 Config

`~/.ctxgraph/config.toml`:

```toml
[extraction]
mode = "cloud-fallback"   # or "local-only" | "cloud-quality"

[cerebras]
api_key = "..."
model   = "qwen3-32b"     # auto-migrates to gpt-oss-120b after deprecation

[deepinfra]
api_key = "..."
model   = "google/gemma-4-26b-a4b-it"

[privacy]
pii_scrubbing = true      # CloakPipe-style PII strip before any cloud call
allow_cloud   = true      # false ⇒ forced into Mode A regardless of `mode`
```

`allow_cloud = false` is the privacy override. Flip it and ctxgraph cannot leak data even if misconfigured.

---

## 9. Target v0.3 (§-T): host-memory prompt caching

> Source: `deep-research/FINAL.md` § 6. Measured 4.2 s → 0.3 s prefill TTFT on RTX 3090 with an 8 k-token system prompt.

ctxgraph's system prompt (entity types, relation types, schema rules, format constraints) is **identical across episodes** — perfect for KV-cache reuse.

### 6.1 llama-server config

```bash
llama-server \
  --model ./models/qwen3-8b.q4_k_m.gguf \
  --ctx-size 32768 \
  --np 4 \
  --cram 256 \
  --flash-attn \
  --system-prompt-file ./ctxgraph_schema.txt \
  --debug-slot
```

Key params:
- `--cram 256` — 256 MB host-memory cache for pre-computed KV blocks
- `--system-prompt-file` — static schema/ontology block
- `--flash-attn` — required for stable prefix caching

### 6.2 Ollama config

```json
{
  "model": "qwen3:8b",
  "keep_alive": "-1",
  "options": { "num_ctx": 16384 }
}
```

`keep_alive: -1` is the critical bit — without it, the model unloads on idle and the KV cache is purged.

### 6.3 Prefix-isolation requirement

For cache hits, the schema/ontology/few-shot block must be **byte-for-byte stable**. The dynamic episode text and current-facts retrieval go at the *end* of the prompt sequence. This is enforced by ctxgraph's prompt-builder in `llm_extract.rs`.

### 6.4 SWA gotcha

Some Qwen variants use Sliding Window Attention, which conflicts with static KV caches. Force global attention with `--override-kv` to disable sliding-window restrictions.

### 6.5 Measured impact

Combined with the GLiNER2 swap and NuExtract Tier 2 default, expected local-tier latency:

| Current | Target v0.3 |
|---|---|
| ~17 s/episode (Gemma 4 26B via Ollama, no caching) | **3–5 s/episode** |

---

## 10. Target v0.3 (§-T): cloud routing (verified pricing)

> Source: `deep-research/FINAL.md` § 4. All prices verified by the External-A pass; some had to be corrected from earlier passes' rounded numbers.

| Tier | Provider | Model | Price | Notes |
|---|---|---|---|---|
| **2.5 Dev (free)** | Cerebras | Qwen3-32B → gpt-oss-120B | $0 ≤ 1 M tok/day | Qwen3-32B deprecating Feb 2026 |
| **2.5 Dev alt** | Groq | Llama 3.3 70B | $0 ≤ 1 K req/day | 30 RPM cap |
| **3 Paid default** | **DeepInfra** | **Gemma-4-26B-A4B** | **$0.07 in / $0.34 out** | Verified; clean local / cloud parity |
| **3 Paid alt cheap** | DeepInfra | Qwen3-32B | $0.08 in / $0.28 out | Split-priced (corrected from "$0.08 flat") |
| **3 Paid high-quality** | OpenRouter | Hermes 4 70B | $0.13 in / $0.40 out | IE quality leader, slowest s/ep |
| **3 Premium + fine-tune** | Together AI | Llama 3.3 70B | $0.88 in / $0.88 out | Best when training a LoRA on the same platform |

**Cost per 1 000 episodes** (800 tokens: 600 in + 200 out):

| Stack | $/1k eps |
|---|---|
| Cerebras free (up to 1 250 eps/day) | $0 |
| DeepInfra Gemma-4-26B-A4B | **$0.11** |
| Current default (OpenRouter gpt-4o-mini) | $0.24 |
| Graphiti + GPT-4o-mini | $1.80 (≈22× more) |

The swap from OpenRouter gpt-4o-mini → DeepInfra Gemma 4 26B is **2.2× cheaper**.

---

## 11. Target v0.3 (§-T): improvements to `llm_extract.rs`

These map onto specific lines in the current code:

1. **`OLLAMA_PREFERRED_MODELS`** (currently `llm_extract.rs:20–26`): add `aravhawk/gemma4`, `nuextract:2.0-4b`, `qwen3:8b`, `hermes-4:8b`, expand to ~8 variants. Order from most VRAM to least.
2. **GPU autodetect**: probe `nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits` / Apple Metal / ROCm. Route by free VRAM (table in § 5). Add a `CTXGRAPH_VRAM_OVERRIDE` env for testing.
3. **Prompt cache enforcement**: stable-prefix prompt builder; serialize the schema/few-shot/format block before any episode-specific text.
4. **Default cloud model**: change `DEFAULT_MODEL` constant from `gpt-4o-mini` to `google/gemma-4-26b-a4b-it`. Change `graph.rs:657` schema-inference default from `google/gemma-4-31b-it` to the same — one model across both code paths.
5. **Speculative decoding**: do *not* enable by default. Per pass-3 audit, naïve model-based SpS with a sub-1B draft is often net-negative on 8B targets; EAGLE-3 still helps but requires measurement. Make it a `--speculate` flag, document the caveat.

---

## 12. Target v0.3 (§-T): schema-improvement loop (Piece 5)

> Source: `CLARITY.md` § 3 / Piece 5. The mechanism by which the universal 9/10 schema grows over time without users having to think about it.

The universal schema covers most cases but not all. Without an evolution mechanism, edge-case domains (recipes, scientific datasets, gaming, etc.) would force `Concept` or `related_to` as the fallback for everything. Three-layer loop:

### 12.1 Layer A — suggestion logging (always on)

The extraction prompt (§ 5.1) is extended to allow the LLM to *suggest* a more specific type when the universal label feels too generic:

```json
"suggestions": [
  {"kind": "entity_type", "name": "Ingredient",
   "supporting_entity_ids": ["e1", "e3"], "rationale": "..."},
  {"kind": "relation_type", "name": "contains",
   "head_id": "e2", "tail_id": "e3", "rationale": "..."}
]
```

These go into a `schema_suggestions` table — never directly into the schema. Each row: timestamp, episode_id, rationale, confidence.

### 12.2 Layer B — promotion threshold (background job)

A nightly cron (`ctxgraph schema review`) scans `schema_suggestions`. A suggestion is promoted to the schema only if:

- It has appeared in **≥ K distinct episodes** (K = 5 default, configurable)
- It has appeared across **≥ M distinct sources/contexts** (M = 3 default)
- Average confidence > 0.7
- It is **not semantically near an existing type** (cosine similarity < 0.85 against all current type embeddings — prevents "Place" / "Location" / "Region" duplication)

When promoted: added to the schema TOML with a `provisional: true` flag. User is notified at next CLI invocation:

```
$ ctxgraph add "..."
[ctxgraph] note: 2 schema additions in last review:
  + entity:   Ingredient   (seen 7 times across 5 episodes)
  + relation: contains     (seen 12 times across 8 episodes)
  to revert: ctxgraph schema revert
```

### 12.3 Layer C — user override

Power users can edit the schema directly via `ctxgraph schema edit` (opens TOML in `$EDITOR`). Provisional additions can be confirmed, rejected, or renamed. Manual additions are fully supported. Most users never touch this.

---

## 13. Target v0.3 (§-T): bi-temporal storage tweaks

The existing schema already supports `valid_from` / `valid_until`. The v0.3 change is **wiring the LLM's `invalidates:` output to the storage layer**:

```rust
// pseudo-code; lives in ctxgraph-core/src/graph.rs
fn add_episode(&mut self, episode: Episode) -> Result<EpisodeResult> {
    let extraction = pipeline.extract(&episode.text)?;
    for invalid_edge_id in &extraction.invalidates {
        self.invalidate_edge(invalid_edge_id, episode.reference_time)?;
    }
    for new_edge in extraction.relations {
        self.insert_edge(new_edge, episode.reference_time)?;
    }
    // …
}
```

The current-facts retrieval that feeds the LLM context (so it knows what to invalidate) is a new helper: `Graph::current_facts_touching(entities: &[Entity], k_per_entity: usize) -> Vec<Edge>`.

---

## 14. Target v0.3 (§-T): not in scope

These show up in the older `ARCHITECTURE.md` as future work; per the deep-research synthesis, they are **not v0.3 work**:

- `ctxgraph-ingest` crate (git / shell / FS / browser / Screenpipe connectors) — interesting, but the headline F1 win doesn't depend on it. Defer to v0.4+.
- Reflect API (`ctxgraph_reflect*` MCP tools) — defer to v0.4 or v0.5.
- Python SDK (PyO3 bindings) — defer to post-launch.
- Web dashboard (graph visualization) — defer.
- Daemon mode + TUI — defer.
- A-MEM-style append-only memory notes — v0.4 (per `ROADMAP.md`).
- mistral.rs embedded inference (single-binary moat) — v0.4 spike.
- Graph Judge nightly pass — v0.5.

What **is** v0.3 is the four-tier extraction pipeline, host-memory caching, the bi-temporal `invalidates:` prompt, the cloud-routing swap, and a re-run of the 29-episode benchmark to confirm the headline number lands at ≥ 0.745 combined F1 with the new local stack.

---

## 15. Cross-references

- **Master working doc** (product + the 5 pieces + non-negotiable decisions): `docs/CLARITY.md`
- **Headline benchmark + raw numbers**: `docs/research_brief.md` (session-measured results, 7 model runs, two Graphiti runs)
- **All measured numbers + hostile-reader audit**: `docs/BENCHMARKS.md`
- **Roadmap & 12-week schedule**: `docs/ROADMAP.md`
- **Deep-research source material**: `docs/deep-research/FINAL.md` (synthesized), plus `claude-dr.md`, `chatgpt-dr.md`, `claude-dr-2.md`, `grok.md`, `gemini-dr.md` (per-source detail)
- **ADRs** (still authoritative for past decisions): `docs/adr/001-sqlite-over-neo4j.md` through `006-unified-gliner2-model.md`

---

*End of architecture v2. Re-verify all pricing 24 h before any HN-facing claim.*
