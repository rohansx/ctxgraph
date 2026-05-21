# ctxgraph v0.3 — Clarity Doc

**The product, decisions, architecture, and the five pieces that need to be built. Written after four research passes + multiple design conversations. This is the working document; everything else is reference.**

> **Build status (as of 2026-05-14)** — all 5 pieces have working Rust code + tests, all four CLARITY thresholds pass on OpenRouter Gemma 4 26B (25-ep universal_smoke fixture):
> - **Piece 1** (universal TOML): `crates/ctxgraph-extract/schemas/universal.toml` ✓
> - **Piece 2** (extract prompt + JSON contract): `crates/ctxgraph-extract/prompts/extract.txt` — combined F1 = **0.559 PASS** (≥0.55 target) ✓
> - **Piece 3** (relation embeddings): `crates/ctxgraph-extract/src/relation_match.rs` — accuracy **83.3% PASS** (≥80% target) ✓
> - **Piece 4** (NL query parser prompt): `crates/ctxgraph-cli/prompts/query_parse.txt` — JSON valid **100% PASS** (≥95%), op classification **100% PASS** (≥75%) ✓
> - **Piece 5 Layer A** (suggestion logging): `UniversalPipeline::with_suggestions_log()` ✓
> - **Piece 5 Layer B** (promotion job): `crates/ctxgraph-extract/src/schema_review.rs` — 7 unit tests + 1 live-data test pass ✓
> - **CLI wire-up**: `ctxgraph log --universal "text"` runs the end-to-end pipeline against real SQLite ✓
> - **Cerebras free tier validated as Mode B default**: same Piece 4 quality (100% / 100%), $0 cost; Piece 2 lags Gemma 4 26B by 0.11 F1 — acceptable trade for free
>
> Detailed measurements in [`BENCHMARKS.md`](BENCHMARKS.md).

---

## 1. What ctxgraph is, in one paragraph

ctxgraph is a typed knowledge graph engine for AI agents. It runs as a single Rust binary against a single SQLite file. You throw text at it — emails, journal entries, slack threads, code commits, meeting notes, anything — and it produces a structured, queryable graph of entities and typed relations. Agents (Claude Code, Cursor, Cline, custom) read from it via MCP. Compared to Graphiti, it makes one LLM call per episode instead of six, achieves ~5.8× higher relation-extraction F1 on the same fixture with the same model, and queries the graph deterministically in ~10ms without an LLM in the read path.

---

## 2. The non-negotiable decisions

These are settled. Don't relitigate.

1. **Keep typed extraction.** Drop typing and we become "slightly cheaper Graphiti" — a price war we can't win. Typed output is what makes the graph queryable without an LLM at read time. This is the moat.
2. **Make the schema invisible to users.** Ship one universal schema (9 entity types, 10 relations) hardcoded. Users never write or see a schema in v0.3.
3. **Local model first, cloud as fallback, Cerebras as the "free high-quality" tier.** Privacy users get fully local. Everyone else gets the option to use Cerebras free tier for quality boost.
4. **One LLM call per write. Zero LLM calls for ~90% of reads.** Reads use embedding-based relation matching + deterministic SQL. Small local LLM only for complex multi-hop queries.
5. **Schema evolution is automatic but conservative.** Track unmapped entities/relations as observations; promote to schema only when they appear repeatedly across episodes. Never auto-edit without a confidence threshold.
6. **Ship the launch in 12 weeks.** Brain-inspired memory DB is a real opportunity but it's phase 3, not now.

---

## 3. The five pieces to build

This is the actual unblock. Each piece is small, independently testable, and you can build all five over the next 2-3 weekends.

### Piece 1 — The universal schema (TOML)

A single hardcoded taxonomy. Broad enough to handle personal wikis, work notes, technical content, recipes, anything. Lives at `crates/ctxgraph-extract/schemas/universal.toml`.

```toml
# 9 entity types
[entities]
Person          = "humans, named individuals"
Place           = "locations, regions, venues, addresses"
Organization    = "companies, teams, institutions, groups"
Concept         = "ideas, theories, methodologies, technologies, terms"
Artifact        = "concrete made objects: tools, systems, documents, code, products"
Event           = "occurrences with a time: meetings, deploys, conferences, decisions"
Time            = "explicit temporal anchors: dates, periods, durations"
Idea            = "personal thoughts, hypotheses, plans, intentions"
Fact            = "verified statements, measurements, observations"

# 10 relation types
[relations]
mentions          = "X is named in the context of Y"
located_at        = "X is physically or conceptually at Y"
related_to        = "X is associated with Y (fallback when nothing else fits)"
caused            = "X led to Y"
preceded          = "X happened before Y"
references        = "X cites, links, or builds on Y"
owned_by          = "Y owns or controls X"
part_of           = "X is a component of Y"
depends_on        = "X requires Y to function"
participated_in   = "X was an actor in Y"
```

That's the schema. Don't add more types to v0.3. Resist the urge.

### Piece 2 — The extraction prompt + JSON contract

One system prompt, ~500 tokens, schema baked in. Lives at `crates/ctxgraph-extract/prompts/extract.txt`.

```
You extract structured knowledge from text into a typed graph.

Entity types (use exactly one per entity):
  Person | Place | Organization | Concept | Artifact |
  Event | Time | Idea | Fact

Relation types (use exactly one per relation):
  mentions | located_at | related_to | caused | preceded |
  references | owned_by | part_of | depends_on | participated_in

Rules:
- Output strict JSON matching the schema below. No prose. No markdown fences.
- Use 'related_to' only when no other relation fits.
- Use 'Concept' as the entity type for anything ambiguous.
- Empty arrays are valid; do not invent facts not in the text.
- Each entity must be referenced by id in the relations.
- If the text contains a fact that supersedes a prior fact, emit it in 'invalidates' as a description (we will resolve later).

JSON schema:
{
  "entities": [
    { "id": "e1", "name": "string", "type": "Person|Place|...", "attributes": {} }
  ],
  "relations": [
    { "head": "e1", "relation": "depends_on", "tail": "e2",
      "confidence": 0.0,
      "valid_from": "ISO date or null",
      "valid_to": "ISO date or null" }
  ],
  "invalidates": [
    "natural-language description of what this episode contradicts"
  ],
  "confidence": 0.0
}

Episode text:
{episode_text}
```

Iterate this prompt by testing on 10 real wiki episodes. The first version won't be perfect. Tweak based on what breaks.

### Piece 3 — Relation-vocabulary embeddings (the synonym layer)

At ctxgraph startup, embed each of the 10 relation type names + their descriptions, cache vectors. At query time, embed the user's verb, cosine match, pick top-1. ~30 lines of Rust in `crates/ctxgraph-extract/src/relation_match.rs`.

```rust
use ctxgraph_embed::EmbedModel;
use ndarray::Array1;

pub const RELATIONS: &[(&str, &str)] = &[
    ("mentions",        "X is named in the context of Y"),
    ("located_at",      "X is physically or conceptually at Y"),
    ("related_to",      "X is associated with Y"),
    ("caused",          "X led to Y"),
    ("preceded",        "X happened before Y"),
    ("references",      "X cites or builds on Y"),
    ("owned_by",        "Y owns or controls X"),
    ("part_of",         "X is a component of Y"),
    ("depends_on",      "X requires Y to function"),
    ("participated_in", "X was an actor in Y"),
];

pub struct RelationMatcher {
    name_vectors: Vec<(String, Array1<f32>)>,
}

impl RelationMatcher {
    pub fn build(model: &EmbedModel) -> Self {
        let name_vectors = RELATIONS
            .iter()
            .map(|(name, desc)| {
                let text = format!("{}: {}", name, desc);
                (name.to_string(), model.encode(&text))
            })
            .collect();
        Self { name_vectors }
    }

    /// Returns the best-matching relation name and its cosine score.
    pub fn resolve(&self, model: &EmbedModel, user_verb: &str) -> (String, f32) {
        let query = model.encode(user_verb);
        self.name_vectors
            .iter()
            .map(|(name, vec)| (name.clone(), cosine(&query, vec)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
    }
}

fn cosine(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.dot(a).sqrt();
    let nb: f32 = b.dot(b).sqrt();
    dot / (na * nb + 1e-9)
}
```

This handles "relies on" → `depends_on` (cosine ~0.91), "is part of" → `part_of`, "mentioned in" → `mentions`. No LLM at query time. Deterministic. Local. ~5ms per call.

### Piece 4 — The natural-language query parser

For the ~10% of queries too structured for plain embeddings (multi-hop, conjunctions, time filters). Small local LLM (Qwen3-1.5B or similar) + few-shot prompt. Lives at `crates/ctxgraph-cli/prompts/query_parse.txt`.

```
You convert natural language queries to graph operations.

Output strict JSON:
{
  "op": "lookup" | "traverse" | "filter" | "list" | "compare",
  "head": "entity name or null",
  "relation": "one of the 10 relation types or null",
  "tail": "entity name or null",
  "filters": {
    "time_after":  "ISO date or null",
    "time_before": "ISO date or null",
    "entity_type": "one of the 9 entity types or null"
  }
}

Examples:

"what does Vernon CMS depend on?"
→ { "op": "traverse", "head": "Vernon CMS", "relation": "depends_on",
    "tail": null, "filters": {} }

"who did I meet at PyCon?"
→ { "op": "traverse", "head": "PyCon", "relation": "participated_in",
    "tail": null, "filters": { "entity_type": "Person" } }

"what did I learn this week?"
→ { "op": "list", "head": null, "relation": null, "tail": null,
    "filters": { "entity_type": "Concept", "time_after": "{today_minus_7}" } }

"what concepts are connected to Letta?"
→ { "op": "traverse", "head": "Letta", "relation": null,
    "tail": null, "filters": { "entity_type": "Concept" } }

Query: {user_query}
```

200 LOC of Rust to wire qwen3-1.5b via Ollama, parse the JSON, dispatch to deterministic SQL handlers in `ctxgraph-core/src/query.rs`. The LLM only converts NL → operation; it never touches the data.

### Piece 5 — Automatic schema improvement

The universal schema covers most cases but not all. When the LLM emits an entity it labels `Concept` but the context suggests something more specific (e.g., recipes have "Ingredient"-like patterns showing up repeatedly), we want to *eventually* learn that. Without ever forcing the user to think about it.

Three-layer mechanism:

**Layer A — Suggestion logging (always on)**

The extraction prompt is extended slightly to allow the LLM to *suggest* a more specific entity type or relation when `related_to`/`Concept` feels too generic. These go into a side-table, never directly into the schema:

```
Add to extraction JSON:
  "suggestions": [
    { "kind": "entity_type", "name": "Ingredient",
      "supporting_entity_ids": ["e1", "e3"], "rationale": "..." },
    { "kind": "relation_type", "name": "contains",
      "head_id": "e2", "tail_id": "e3", "rationale": "..." }
  ]
```

These get logged to `schema_suggestions` table with: timestamp, episode_id, rationale, and a confidence score.

**Layer B — Promotion threshold (background job)**

A nightly job (`ctxgraph schema review`) scans `schema_suggestions`:

```
A suggestion is promoted to the schema if:
  - It has appeared in ≥ K distinct episodes (K = 5 default, configurable)
  - It has appeared across ≥ M distinct sources/contexts (M = 3 default)
  - Average confidence > 0.7
  - It is not semantically near an existing type (cosine sim < 0.85)
    against all current type embeddings
```

When promoted: added to the schema TOML with a `provisional: true` flag. Future extractions can use it. User is notified at next CLI invocation:

```
$ ctxgraph add "..."
[ctxgraph] note: 2 schema additions in last review:
  + entity: Ingredient    (seen 7 times across 5 episodes)
  + relation: contains    (seen 12 times across 8 episodes)
  to revert: ctxgraph schema revert
```

**Layer C — User override**

Power users can edit the schema directly via `ctxgraph schema edit`, which opens the TOML in `$EDITOR`. Provisional additions can be confirmed, rejected, or renamed. Manual additions are fully supported. Most users never touch this.

This layer is the answer to "what if my domain doesn't fit your 10 types?" The schema grows with your data, conservatively, without ever surprising you. The universal schema is the *starting point*, not the *final word*.

---

## 4. Model strategy — local first, cloud as fallback, Cerebras as the "free quality boost"

This is the spec for `crates/ctxgraph-extract/src/llm_extract.rs`. Three modes:

### Mode A: `local-only` (default — privacy mode)

Everything runs on the user's machine. Zero data leaves.

```
Tier 1: GLiNER2 ONNX (CPU)              ~30ms,  ~70% of episodes
   ↓ (escalate on low confidence)
Tier 2: NuExtract 2.0-4B (Ollama)       ~2-4s,  ~25% of episodes
   ↓ (escalate on JSON validation fail)
Tier 3: Qwen3-8B Q4 (Ollama)            ~5-8s,  ~5% of episodes

Embedding model: all-MiniLM-L6-v2 (local, 384-dim)
Query parser:    Qwen3-1.5B (Ollama, loaded on demand)
```

VRAM autodetect picks the largest Tier 2 model that fits. On a 6GB GPU: NuExtract 2.0-2B as Tier 2, Qwen3-1.5B as query parser. On 16GB+: NuExtract 2.0-4B + Qwen3-8B comfortably.

This mode is the default if Ollama is detected and no `cloud` flag is passed.

### Mode B: `cloud-fallback` (the practical default for most users)

Local Tiers 1+2 still handle the easy cases. Hard cases escalate to Cerebras free tier.

```
Tier 1: GLiNER2 ONNX (local CPU)
Tier 2: NuExtract 2.0-4B (local Ollama, if available)
Tier 3: Cerebras free tier — Qwen3-32B
        (or gpt-oss-120B after Feb 2026 deprecation)
        Free up to 1M tokens / day, 30 req / min
Tier 4: DeepInfra Gemma-4-26B-A4B paid
        ($0.07 / 1M in, $0.34 / 1M out, ~$0.11 per 1k episodes)
        Only fires if Cerebras rate-limited

Embedding model: local all-MiniLM-L6-v2 (PII never leaves machine)
Query parser:    local Qwen3-1.5B
```

In practice, a personal wiki at ~50 episodes/day uses Tier 1 + 2 locally for everything cheap, and Tier 3 (Cerebras free) for the hard ~5%. Total cost: $0/month.

This is the recommended mode for most users. `ctxgraph init` should default to it.

### Mode C: `cloud-quality` (for power users who want best output)

Skip local tiers. Every episode goes to a hosted high-quality model.

```
Default: Cerebras Qwen3-32B (free, fast, 2000+ tok/s)
Paid alt: DeepInfra Gemma-4-26B-A4B
Premium: OpenRouter Hermes 4 70B IE-tuned (highest IE quality)

Embedding model: still local
Query parser:    still local
```

Useful when the user has long episodes (research papers, transcripts) where local extraction quality degrades.

### Config

In `~/.ctxgraph/config.toml`:

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
pii_scrubbing = true      # strip PII before any cloud call (CloakPipe-style)
allow_cloud   = true      # if false: never call cloud regardless of mode
```

The `allow_cloud = false` switch is the privacy override. Flip it and ctxgraph cannot leak data even if misconfigured.

---

## 5. The read path — how queries actually work

```
USER ASKS NATURAL LANGUAGE QUERY
   │
   ▼
┌──────────────────────────────────────────────────┐
│  Step 1: Classify the query                      │
│    - simple lookup ("what does X depend on?")    │
│    - complex (multi-hop, time, conjunction)      │
│                                                  │
│  Heuristic: count graph operations needed.       │
│  ≤ 1 → simple. > 1 → complex.                    │
└──────────────────────────────────────────────────┘
   │
   ├── SIMPLE PATH (90%) ───────────────────────────┐
   │                                                │
   │  Step 2a: Extract entity + verb from query     │
   │    - regex + NER (local, instant)              │
   │                                                │
   │  Step 2b: Match verb → typed relation          │
   │    - embedding cosine match (Piece 3)          │
   │                                                │
   │  Step 2c: Run deterministic SQL                │
   │    - WHERE head=? AND relation=?               │
   │                                                │
   │  Total: ~10-50 ms. Zero LLM calls.             │
   │                                                │
   └────────────────────────────────────────────────┘
   │
   └── COMPLEX PATH (10%) ──────────────────────────┐
                                                    │
      Step 2a: Send query to local Qwen3-1.5B       │
        - few-shot prompt (Piece 4)                 │
        - outputs structured graph op JSON          │
                                                    │
      Step 2b: Dispatch op to SQL handlers          │
        - op=traverse → recursive CTE               │
        - op=filter   → WHERE + time predicates     │
        - op=compare  → JOIN with grouping          │
                                                    │
      Total: ~200-500 ms. One small local LLM call. │
                                                    │
   ────────────────────────────────────────────────┘
   │
   ▼
RESULTS RETURNED with provenance + confidence
```

The crucial property: **no cloud LLM in the read path, ever, in any mode.** Even the cloud-quality mode keeps reads local. Cloud is only for writes (extraction).

This is the bit competitors can't match. Graphiti, Mem0, Letta all need an LLM at read time because their relation types are free-form text the SQL engine can't reason about.

---

## 6. Architecture diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         ctxgraph (single Rust binary)            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────┐    ┌────────────────┐    ┌────────────────┐│
│  │   WRITE PATH    │    │   READ PATH    │    │ BACKGROUND     ││
│  │                 │    │                │    │                ││
│  │ episode →       │    │ query →        │    │ schema review  ││
│  │ Tier 1/2/3 LLM  │    │ classify       │    │ promotion job  ││
│  │ → typed JSON    │    │ ├ simple → SQL │    │ contradiction  ││
│  │ → SQLite        │    │ └ complex →    │    │ detection      ││
│  │                 │    │   Qwen3-1.5b   │    │ embedding      ││
│  │                 │    │   → SQL        │    │ rebuild        ││
│  └────────────────┘    └────────────────┘    └────────────────┘│
│         │                      │                      │         │
│         ▼                      ▼                      ▼         │
│  ┌────────────────────────────────────────────────────────────┐│
│  │  SQLite + FTS5 + sqlite-vec                                ││
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐ ││
│  │  │ episodes │  │ entities │  │  edges   │  │ embeddings │ ││
│  │  │  raw txt │  │  typed   │  │ bi-temp  │  │  384-dim   │ ││
│  │  │          │  │          │  │ + types  │  │            │ ││
│  │  └──────────┘  └──────────┘  └──────────┘  └────────────┘ ││
│  │  ┌──────────────────────┐    ┌────────────────────────┐   ││
│  │  │ schema_suggestions   │    │ schema (TOML mirror)   │   ││
│  │  └──────────────────────┘    └────────────────────────┘   ││
│  └────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ┌──────────┐  ┌─────────────────────────────────────────────┐ │
│  │   CLI    │  │       MCP server (for Claude Code etc.)     │ │
│  └──────────┘  └─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
            │                          │
            ▼                          ▼
   ┌────────────────┐         ┌────────────────────┐
   │     Ollama     │         │   Cerebras / DeepInfra  │
   │  (local LLM)   │         │   (cloud LLM, write-path only)  │
   └────────────────┘         └────────────────────┘
```

---

## 7. What we explicitly rejected

| Idea | Why rejected |
|---|---|
| Schema-less / freeform output | Same as Graphiti. We win on typed extraction. |
| Manual schema TOML in v0.3 | UX cliff. Most users never write a schema. Layer C automatic improvement covers it. |
| Pivot to brain-inspired DB now | Real long-term opportunity but 12-18 months. Phase 3 after ship. |
| Ship LoRA fine-tunes before universal schema launch | LoRA is icing. The universal schema + relation matching is the cake. |
| OpenRouter gpt-4o-mini as cloud default | DeepInfra Gemma-4-26B-A4B is cheaper *and* same family as local Tier 2. |
| Speculative decoding for inference acceleration | Two of three cited papers don't support the claim. Skip or measure. |
| LongMemEval / LoCoMo as headline benchmark | Saturated at 85-95%. Lead with the 29-ep fixture instead. |

---

## 9. The 12-week plan

| Week | Deliverable |
|---|---|
| W1 | GLiNER2 wired into Tier 1; old GLiNER+GLiREL retired; re-run 29-ep fixture |
| W2 | Host-memory prompt caching defaults; NuExtract 2.0-4B Tier 2; VRAM autodetect |
| W3 | Universal schema TOML + extraction prompt + relation embeddings (**Pieces 1, 2, 3**) |
| W4 | Bi-temporal `invalidates:` in extraction prompt; schema suggestion logging (**Piece 5 layer A**) |
| W5 | NL query parser via Qwen3-1.5B (**Piece 4**) |
| W6 | Cerebras + DeepInfra integration; mode switching; config UX |
| W7 | Benchmark expansion: 29-ep + Re-DocRED subset + LongMemEval-S |
| W8 | **HN launch v0.3**: typed local KG, single binary, free-at-scale |
| W9 | mistral.rs embedded inference (eliminate Ollama HTTP boundary) |
| W10 | Schema promotion job (**Piece 5 layers B + C**); A-MEM memory notes |
| W11 | MCP polish; Claude Code integration demo; awesome-mcp-servers PR |
| W12 | **HN launch v0.4** + arXiv preprint |

---

## 10. What to do this weekend

1. Create `crates/ctxgraph-extract/schemas/universal.toml` with **Piece 1**.
2. Create `crates/ctxgraph-extract/prompts/extract.txt` with **Piece 2**.
3. Test the extraction prompt against Cerebras free tier with 10 real episodes from your personal wiki. Iterate the prompt 2-3 times.
4. Implement **Piece 3** (`relation_match.rs`, ~30 lines).
5. Run end-to-end: add 10 episodes, query each with at least 2 variations of the verb ("depends on" vs "relies on" vs "needs"). Make sure the embedding match resolves correctly.

If pieces 1-3 work on real data, you have something to demo. Piece 4 (NL query parser) is week 5. Piece 5 (schema improvement) is week 4 + week 10.

---

## 11. The launch pitch (final)

> **ctxgraph: typed knowledge graph for AI agents.** Single Rust binary. Single SQLite file. One LLM call per write (vs Graphiti's six). Reads run locally without an LLM in 90% of cases. Free at any reasonable scale via Cerebras. Plugs into Claude Code via MCP. On the same fixture with the same model, hits 5.8× higher relation extraction F1 than Graphiti.

Four sharp claims. Every one defensible by the four research passes. No hand-waving.

---

*This is the working document. Reference for everything else: `deep-research/FINAL.md` (synthesized research), `ARCHITECTURE.md` (technical deep dive), `ROADMAP.md` (week-by-week), `BENCHMARKS.md` (measured numbers).*
