# ctxgraph

**Local-first context graph engine for AI agents and human teams.**

Zero infrastructure. Zero API cost. Single Rust binary.

---

## What ctxgraph Is

ctxgraph is an embedded, temporal context graph engine that stores decision traces — the full story behind every choice — and makes them searchable. It runs entirely on your machine. No Neo4j. No OpenAI API key. No Docker. No Python. One binary, one SQLite file, instant startup.

When someone (or an AI agent) asks "why did we do X?", ctxgraph traverses the graph of past decisions, finds relevant precedents, and returns the full context — who decided, when, what alternatives were considered, which policies applied, and what the outcome was.

```
Your App / CLI / AI Agent
         |
    ctxgraph engine
         |
    ┌─────────────────────────────┐
    │  Extraction Pipeline        │
    │  ┌───────┐ ┌──────┐        │
    │  │GLiNER2│ │GLiREL│ (ONNX) │   Tier 1: Schema-driven, local
    │  └───────┘ └──────┘        │
    │  ┌──────────────────┐      │
    │  │Coreference + Dedup│      │   Tier 2: Enhanced local
    │  └──────────────────┘      │
    │  ┌──────────────────┐      │
    │  │Ollama / LLM API  │      │   Tier 3: LLM-enhanced (optional)
    │  └──────────────────┘      │
    └─────────────────────────────┘
         |
    ┌─────────────────────────────┐
    │  Storage Layer              │
    │  SQLite + FTS5              │
    │  Bi-temporal timestamps     │
    │  Adjacency graph structure  │
    └─────────────────────────────┘
         |
    ┌─────────────────────────────┐
    │  Query Layer                │
    │  FTS5 full-text search      │
    │  Graph traversal (rCTEs)    │
    │  Local embedding similarity │
    │  MCP server for AI agents   │
    └─────────────────────────────┘
```

---

## Why ctxgraph Exists

The context graph concept — storing not just what happened but why — is proven. Foundation Capital called it "AI's trillion-dollar opportunity." Graphiti (by Zep) has 20K+ GitHub stars building exactly this.

But every existing implementation requires heavy infrastructure:

| Requirement | Graphiti | ctxgraph |
|---|---|---|
| Graph database | Neo4j / FalkorDB (Docker) | SQLite (embedded) |
| LLM API key | Required (OpenAI/Anthropic) | Not required |
| Runtime | Python 3.10+ | Single Rust binary |
| Cost per episode | ~$0.01-0.05 (LLM tokens) | $0.00 |
| Setup time | 15-30 minutes | 5 seconds |
| Internet required | Yes (always) | No (fully offline) |
| Embeddable | No (Python library + services) | Yes (Rust crate) |
| Privacy | Data sent to OpenAI | Nothing leaves machine |

ctxgraph is for the 90% of use cases where you don't need Neo4j-scale infrastructure. Solo developers, small teams, privacy-constrained organizations, and anyone who wants context graphs without the operational overhead.

---

## Problems ctxgraph Solves

### Problem 1: The Repetition Tax — "Every new session, you re-explain yourself"

**Solved.** This is ctxgraph's core purpose. The MCP server gives Claude/Cursor persistent memory across sessions. You log "I use Rust, my project is a privacy proxy, I chose axum over actix" once. Every future session, the agent queries ctxgraph and already knows. No more pasting context files.

### Problem 2: Long Sessions Degrade — "Lost-in-the-middle problem"

**Partially solved.** ctxgraph helps because instead of dumping your entire history into the context window, the agent retrieves only the relevant decision traces via search. Selective retrieval instead of "paste everything." But ctxgraph doesn't fix the fundamental attention degradation inside the model — that's a model architecture problem. What ctxgraph does is reduce the need for long sessions in the first place, because context persists across short sessions.

### Problem 3: RAG Solves Retrieval, Not Understanding — "You lose the decision graph, not just the data"

**Directly solved.** This is the whole thesis. RAG retrieves text chunks. ctxgraph retrieves decision traces — the reasoning state, the rejected alternatives, the tradeoffs, the constraints. When the agent queries "why Postgres?", it doesn't get a chunk of text mentioning Postgres. It gets a structured subgraph: who decided, what was rejected, what constraint drove it, what condition would invalidate it. That's understanding, not retrieval.

### Problem 4: No Shared Context Across Tools — "Every tool is an island"

**Solved via MCP.** ctxgraph runs as a single local process. Claude Desktop, Cursor, Claude Code, any MCP-compatible agent — they all connect to the same graph. You log a decision in your terminal via `ctxgraph log`, and 30 seconds later Claude Desktop can reference it. One graph, many consumers. The `.ctxgraph/graph.db` file is the shared state between all your tools.

### Problem 5: The Re-orientation Overhead — "Re-orienting a model to a complex project takes time"

**Partially solved.** ctxgraph reduces re-orientation because the agent can query "what are the key architecture decisions in this project?" and get a structured answer in one MCP call instead of you manually explaining. But "managing the model's understanding, checking if it got the nuances, correcting drift" — ctxgraph doesn't fix that. That's still on you. ctxgraph gives the model better input, but it can't guarantee the model processes it correctly.

### The "What Good Would Look Like" Checklist

| Criterion | Status | How |
|---|---|---|
| Session-persistent reasoning state | Yes | Decisions, rejections, tradeoffs stored as first-class graph entities with temporal validity |
| Cross-tool context propagation | Yes | MCP server is the bridge. One graph, all tools |
| Selective, structured recall | Yes | RRF search + graph traversal retrieves what's relevant, not everything. Schema tells ctxgraph what matters, temporal model tells it when |
| Decay-aware memory | Yes | Bi-temporal model with valid_from/valid_until. Old facts get invalidated, not deleted. The graph knows what's current vs historical |

**The honest gap:** ctxgraph solves the storage and retrieval side of these problems. It doesn't solve the model comprehension side. You can give Claude perfect context via ctxgraph, and Claude might still misunderstand the nuances. But that's a model problem, not a memory problem.

**Score:** 3 out of 5 fully solved, 2 out of 5 partially solved, 4 out of 4 on the "what good looks like" checklist.

---

## Competitive Landscape

| Problem | Graphiti | Mem0 | Claude Memory | CLAUDE.md | Windsurf/Cursor Memory | ctxgraph |
|---|---|---|---|---|---|---|
| 1. Repetition tax | Yes | Partial | Partial | Partial | Partial | **Yes** |
| 2. Long session decay | Partial | No | No | No | No | **Partial** |
| 3. Reasoning state, not facts | Yes | No | No | No | No | **Yes** |
| 4. Cross-tool context | Yes (MCP) | No | No | No | No | **Yes (MCP)** |
| 5. Re-orientation overhead | Partial | No | No | Partial | Partial | **Partial** |
| Zero infrastructure | No | No | N/A | Yes | N/A | **Yes** |
| Zero API cost | No | No | N/A | Yes | N/A | **Yes** |
| Works offline | No | No | No | Yes | No | **Yes** |

**Graphiti (Zep)** — The closest competitor. Solves problems 1, 3, and 4. But requires Neo4j + OpenAI API key + Docker + Python. Solves the memory problem while creating an infrastructure problem.

**Mem0** — Key-value memory, not a graph. Remembers "user prefers Rust" but not "user chose Rust over Go because of memory safety requirements, rejecting Go's simpler concurrency model." Flat memory, not structured reasoning state.

**Claude's built-in memory** — Not a graph, not queryable, not shareable across tools, and you don't control what it remembers.

**CLAUDE.md / Cursor rules** — Manual documentation for your AI assistant — backwards. Creates maintenance overhead. Should be auto-captured.

**Windsurf/Cursor Memory** — Product-specific memory locked inside one tool. Switch editors and your context is gone.

**Microsoft GraphRAG** — Batch-oriented document understanding. No bi-temporal model, no decision trace concept, no MCP server. Built for document understanding, not decision memory.

**LangMem / LangGraph** — Tied to LangChain ecosystem. Not cross-tool. Not standalone.

**Positioning:** ctxgraph solves what Graphiti solves, but without the infrastructure tax. Same capability, radically different operational model.

---

## Core Concepts

### Episodes

An episode is the fundamental unit of information. It represents "something happened" — a decision was made, a conversation occurred, an event was logged. ctxgraph doesn't care what domain the episode comes from.

```rust
Episode {
    content: "Chose Postgres over SQLite for billing. 
              Reason: need concurrent writes from 3 microservices.",
    source: "manual",           // just a string tag
    timestamp: "2026-03-11T10:30:00Z",
    metadata: {
        "author": "rohan",
        "tags": ["architecture", "database"],
        "confidence": 0.95
    }
}
```

### Entities

Entities are the things mentioned in episodes — people, organizations, components, policies, amounts, decisions. ctxgraph extracts them automatically using the tiered extraction pipeline.

```
From the episode above, ctxgraph extracts:
  COMPONENT: "Postgres"
  COMPONENT: "SQLite"
  SERVICE:   "billing"
  REASON:    "concurrent writes from 3 microservices"
  PERSON:    "rohan" (from metadata)
```

### Relationships (Edges)

Relationships connect entities and are the core of the context graph. They encode how things relate — causally, temporally, and structurally.

```
rohan →[decided]→ DECISION_7
DECISION_7 →[chose]→ Postgres
DECISION_7 →[rejected]→ SQLite
DECISION_7 →[reason]→ "concurrent writes"
DECISION_7 →[applies_to]→ billing service
```

### Temporal Model (Bi-temporal)

Every entity and relationship has two time dimensions:

- **valid_from / valid_until**: When was this fact true in the real world?
- **recorded_at**: When was this fact recorded in ctxgraph?

This matters because facts change. "Alice works at Google" was true from 2020-2025. She joined Meta in 2025. The old edge doesn't get deleted — it gets invalidated with a `valid_until` timestamp. The full history is preserved.

```
Edge: Alice →[works_at]→ Google
  valid_from:  2020-01-15
  valid_until: 2025-06-01     ← invalidated when new fact arrived
  recorded_at: 2026-03-11

Edge: Alice →[works_at]→ Meta
  valid_from:  2025-06-01
  valid_until: null            ← currently true
  recorded_at: 2026-03-11
```

### Decision Traces

A decision trace is a subgraph — a collection of related nodes and edges that together represent the full story of one decision. It's the core value proposition of a context graph.

```
DECISION_7 (chose Postgres for billing)
  ├── decided_by → rohan
  ├── timestamp → 2026-03-11
  ├── chose → Postgres
  ├── rejected → SQLite
  ├── reason → "concurrent writes from 3 microservices"
  ├── context → "team size 3, limited ops capacity"
  ├── referenced_by → DECISION_12 (later decision about Redis)
  └── invalidation_condition → "reconsider at 6+ team members"
```

---

## The Extraction Pipeline — Three Tiers

This is the core technical differentiator. ctxgraph uses a tiered extraction system where each tier adds capability and cost. Tier 1 is always available at zero cost. Tiers 2 and 3 are additive.

### Tier 1: Schema-Driven Local Extraction

**Cost: $0 | Latency: 2-10ms | Quality: ~85% on semi-structured text**

Tier 1 uses GLiNER2 (via ONNX Runtime) for entity extraction and GLiREL for relationship extraction. Both models run locally on CPU. No GPU required. No internet required.

#### How GLiNER2 Works

GLiNER2 is a zero-shot extraction model from NAACL 2024 / EMNLP 2025. Unlike traditional NER that only recognizes fixed entity types (PERSON, ORG, LOC), GLiNER2 accepts any labels you define at runtime. You give it a schema, and it finds matching entities.

```
Traditional NER:
  Fixed labels: [PERSON, ORG, LOCATION, DATE]
  Input: "Chose Postgres because concurrent writes"
  Output: nothing useful (no PERSON/ORG/LOC in this text)

GLiNER2 (zero-shot):
  Custom labels: [Component, Decision, Reason, Alternative, Service]
  Input: "Chose Postgres over SQLite for billing. Reason: concurrent writes."
  Output:
    "Postgres"          → Component
    "SQLite"            → Alternative
    "billing"           → Service
    "concurrent writes" → Reason
```

This is the breakthrough. GLiNER2 matches or outperforms GPT-4o on NER benchmarks while running on CPU in under 10ms. The model is under 500M parameters, converts to ONNX, and quantizes to INT8.

#### How GLiREL Works

GLiREL extends the same architecture to relationship extraction. Given entities and text, it classifies the relationships between entity pairs.

```
Input text: "Chose Postgres over SQLite for billing"
Input entities: [Postgres (Component), SQLite (Alternative), billing (Service)]
Relation types: [chose, rejected, applies_to, reason_for, depends_on]

Output:
  (Postgres, chose, billing)           → "Postgres was chosen for billing"
  (SQLite, rejected, billing)          → "SQLite was rejected for billing"
```

#### Schema Definition

Users define extraction schemas in TOML. ctxgraph ships with sensible defaults but schemas are fully customizable.

```toml
# ctxgraph.toml — extraction schema

[schema]
name = "default"

[schema.entities]
# Label = description (helps GLiNER2 understand context)
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
```

#### ONNX Integration Architecture

```
                    ┌────────────────────────────────┐
                    │       ctxgraph-extract          │
                    │                                │
  Episode text ───► │  ┌──────────────────────────┐  │
                    │  │  Tokenizer (HF tokenizer) │  │
                    │  │  Rust: tokenizers crate    │  │
                    │  └────────────┬───────────────┘  │
                    │               │                  │
                    │  ┌────────────▼───────────────┐  │
                    │  │  GLiNER2 ONNX Model        │  │
                    │  │  Rust: ort crate (ONNX RT) │  │
                    │  │  ~200MB quantized INT8     │  │
                    │  │  Input: tokens + labels    │  │
                    │  │  Output: entity spans      │  │
                    │  └────────────┬───────────────┘  │
                    │               │                  │
                    │  ┌────────────▼───────────────┐  │
                    │  │  GLiREL ONNX Model         │  │
                    │  │  Input: text + entities     │  │
                    │  │  Output: relation triples   │  │
                    │  └────────────┬───────────────┘  │
                    │               │                  │
                    │  ┌────────────▼───────────────┐  │
                    │  │  Temporal Heuristics        │  │
                    │  │  Regex + dateparser         │  │
                    │  │  "last week" → 2026-03-04   │  │
                    │  │  "Q3 2025" → 2025-07-01     │  │
                    │  └────────────┬───────────────┘  │
                    │               │                  │ ──► Entities + Relations
                    └───────────────┘──────────────────┘
```

#### Model Loading Strategy

ONNX models are downloaded once on first use and cached locally. The download is ~200MB for GLiNER2-large quantized + ~150MB for GLiREL.

```bash
# First run downloads models
ctxgraph init
# Downloading GLiNER2-large (INT8)... 198MB
# Downloading GLiREL-large... 147MB  
# Models cached at ~/.ctxgraph/models/

# Subsequent runs: instant startup
ctxgraph log "chose Postgres for billing"
# Extraction: 8ms | Storage: 2ms | Total: 10ms
```

For air-gapped environments:

```bash
# Pre-download on connected machine
ctxgraph models download --output ./models/

# Copy to air-gapped machine, point ctxgraph at them
export CTXGRAPH_MODELS_DIR=/path/to/models
ctxgraph init
```

#### What Tier 1 Is Good At

- Structured and semi-structured text with clear intent
- Manual decision logs ("chose X because Y")
- Git commit messages and PR descriptions
- Support ticket resolutions with clear outcomes
- Meeting notes with explicit decisions
- API/webhook events with structured payloads
- Any text where entities follow predictable patterns

#### What Tier 1 Struggles With

- Highly conversational/casual text ("yeah just do it lol")
- Implicit decisions (no explicit "I decided" language)
- Complex coreference chains across long text
- Contradiction detection between old and new facts
- Relative temporal reasoning ("a few weeks after the incident")
- Summarization of entity clusters

---

### Tier 2: Enhanced Local Extraction

**Cost: $0 | Latency: 15-50ms | Quality: ~80% on semi-structured, ~70% on unstructured**

Tier 2 adds three capabilities on top of Tier 1, all running locally:

#### 2a. Coreference Resolution

Resolves pronouns and references back to their entity.

```
Input (2 sentences):
  "Priya approved the discount."
  "She said it was within policy limits."

Tier 1 output:
  Sentence 1: Priya (Person), discount (Decision)
  Sentence 2: "She" (missed), "it" (missed), policy limits (Policy)

Tier 2 output:
  Sentence 1: Priya (Person), discount (Decision)
  Sentence 2: She → Priya, it → discount, policy limits (Policy)
```

Implementation: within-episode heuristic resolution. For each pronoun, find the nearest entity of matching type in the preceding context window. This uses a lightweight rules engine, not an LLM:

```
Rules:
  "he/him/his" → nearest PERSON (male, if gender known)
  "she/her"    → nearest PERSON (female, if gender known)
  "they/them"  → nearest PERSON or ORG
  "it"         → nearest non-PERSON entity
  "the company/org/team" → nearest ORG
  "the project/service"  → nearest SERVICE or COMPONENT
```

For cases where gender isn't known or rules are ambiguous, Tier 2 uses entity proximity — the closest matching entity type in the text wins. Not perfect, but catches 70-80% of coreferences in structured/semi-structured text.

#### 2b. Fuzzy Entity Deduplication

Determines if two entity mentions refer to the same real-world entity.

```
Episode 1: "Priya Sharma approved it"      → PERSON: "Priya Sharma"
Episode 5: "P. Sharma signed off"           → PERSON: "P. Sharma"
Episode 9: "Priya reviewed the proposal"    → PERSON: "Priya"

Tier 2 resolution:
  "Priya Sharma" = "P. Sharma" = "Priya"
  → All mapped to same node: PERSON_8
```

Implementation uses the same approach as CloakPipe v0.6:

- **Jaro-Winkler similarity** for name matching (threshold: 0.85)
- **Alias groups** — user-defined aliases in config
- **Source-aware dedup** — entities from the same source/author get lower thresholds (more aggressive merging)
- **Type-constrained** — only compare entities of the same type (PERSON vs PERSON, not PERSON vs ORG)

```toml
# ctxgraph.toml — entity resolution config

[resolution]
similarity_threshold = 0.85        # Jaro-Winkler threshold
same_source_threshold = 0.75       # Lower threshold within same source

[resolution.aliases]
"Priya" = ["Priya Sharma", "P. Sharma", "PS"]
"Postgres" = ["PostgreSQL", "PG", "psql"]
```

#### 2c. Contextual Temporal Extraction

Enhanced time parsing that uses episode context to resolve ambiguous dates.

```
Episode (recorded 2026-03-11):
  "We decided this last Tuesday"
  
Tier 1: regex finds "last Tuesday" but can't resolve it
Tier 2: episode.recorded_at = 2026-03-11 (Wednesday)
        "last Tuesday" → 2026-03-04
        
Episode:
  "Three weeks after the migration incident"
  
Tier 1: can't parse relative-to-event time
Tier 2: searches graph for "migration incident" 
        → finds DECISION_3 at 2026-02-01
        → "three weeks after" → 2026-02-22
```

Implementation: a chain of parsers, each handling one pattern:

1. **Absolute dates**: ISO-8601, "March 11, 2026", "11/03/2026" → direct parse
2. **Relative to now**: "yesterday", "last week", "3 days ago" → offset from episode.recorded_at
3. **Fiscal/quarter**: "Q3 2025", "FY26" → mapped to date ranges
4. **Relative to event**: "after the migration", "before the audit" → graph lookup + offset

#### Tier 2 Quality Assessment

| Text Type | Tier 1 | Tier 2 | Improvement |
|---|---|---|---|
| Manual decision log | 88% | 90% | +2% (already good) |
| Git PR description | 85% | 89% | +4% |
| Meeting notes | 75% | 82% | +7% |
| Semi-structured Slack | 65% | 74% | +9% |
| Casual conversation | 50% | 58% | +8% (still not great) |

The biggest gains are in semi-structured text where coreference and dedup matter most.

---

### Tier 3: LLM-Enhanced Extraction

**Cost: $0.01-0.05/episode (API) or $0 (local Ollama) | Latency: 500-2000ms | Quality: ~93-95%**

Tier 3 adds LLM calls for tasks that fundamentally require language understanding. It is strictly optional and opt-in.

#### 3a. Contradiction Detection

When a new episode is ingested, Tier 3 checks if it conflicts with existing facts in the graph.

```
Existing edge: Alice →[works_at]→ Google (valid_from: 2023)
New episode: "Alice just started at Meta last week"

Tier 1-2: creates new edge Alice →[works_at]→ Meta
          but DOES NOT invalidate the Google edge
          → graph now has contradictory edges

Tier 3: LLM call with context:
  "Existing fact: Alice works at Google (since 2023).
   New fact: Alice started at Meta last week.
   Question: Does the new fact contradict the existing fact?"
  
  LLM response: "Yes. Starting at Meta implies leaving Google."
  
  → Invalidates Google edge: valid_until = 2026-03-04
  → Creates Meta edge: valid_from = 2026-03-04
```

Implementation:

```
On episode ingestion:
1. Extract entities (Tier 1)
2. For each new relationship edge:
   a. Query graph for existing edges between same entity pair
   b. If existing edges found with overlapping validity:
      - If edge types are compatible: skip (e.g., two "works_on" edges)
      - If edge types conflict: send both to LLM for contradiction check
3. LLM returns: { contradicts: true/false, explanation: "..." }
4. If contradicts: invalidate old edge, set valid_until
```

The LLM call only fires when potentially conflicting edges exist — most episodes won't trigger it. Typical rate: 5-15% of episodes need contradiction checks.

#### 3b. Complex Temporal Reasoning

For temporal references that can't be resolved by rules.

```
Episode: "A few months before the Series B, we pivoted the product."

Tier 2 can find "Series B" in the graph (DECISION_45, dated 2025-09-15).
But "a few months before" is ambiguous — 2? 3? 4 months?

Tier 3 LLM call:
  "Event: Series B on 2025-09-15. 
   Reference: 'a few months before the Series B'.
   What date does this most likely refer to?"
  
  LLM: "Approximately June-July 2025 (2-3 months prior)"
  → valid_from: 2025-06-15 (midpoint estimate)
  → confidence: 0.6 (lower due to ambiguity)
```

#### 3c. Community Summarization

When clusters of related entities grow large, Tier 3 generates human-readable summaries.

```
Cluster of 12 nodes: Alice, Bob, authentication-service, OAuth, 
  DECISION_3, DECISION_7, DECISION_12, JWT, security-review, 
  RBAC, permissions-redesign, DECISION_18

Tier 3 LLM call:
  "Summarize this cluster of related decisions: [context]"
  
  LLM: "The authentication system underwent a major redesign 
        between Jan-Mar 2026. Alice led the migration from 
        session-based auth to JWT+OAuth, with Bob handling 
        the RBAC permissions layer. Key decisions: chose JWT 
        over sessions (DECISION_3), adopted OAuth 2.0 
        (DECISION_7), and redesigned RBAC (DECISION_12)."
```

These summaries are stored as community nodes in the graph and serve as high-level entry points for traversal.

#### 3d. Unstructured Text Extraction

For messy, conversational text that Tier 1-2 can't handle well.

```
Slack message:
  "yeah so basically I talked to finance and they said it's fine 
   as long as we don't go over the Q3 budget which tbh I'm not 
   even sure what that is anymore lol. Priya said she'd handle it 
   but I think she meant next week? anyway just do it"

Tier 1 output:
  Entities: finance (ORG), Q3 budget (Amount), Priya (Person)
  Relations: almost nothing useful

Tier 3 LLM call:
  "Extract decision trace from this message: [text]
   Schema: [Person, Decision, Constraint, Timeline, Approval]"
  
  LLM structured output:
  {
    "entities": [
      {"text": "finance", "type": "Department", "role": "approver"},
      {"text": "Priya", "type": "Person", "role": "executor"},
      {"text": "Q3 budget", "type": "Constraint"},
      {"text": "next week", "type": "Timeline"}
    ],
    "decisions": [{
      "action": "proceed with plan (implicit approval)",
      "approved_by": "finance (verbal)",
      "constraint": "must not exceed Q3 budget",
      "executor": "Priya",
      "timeline": "next week (uncertain)",
      "confidence": 0.6
    }],
    "relationships": [
      ["finance", "approved", "proceed with plan"],
      ["Priya", "will_execute", "proceed with plan"],
      ["proceed with plan", "constrained_by", "Q3 budget"]
    ]
  }
```

#### LLM Provider Configuration

```toml
# ctxgraph.toml — Tier 3 config

[llm]
enabled = false                    # opt-in only

# Local (Ollama) — free, private, slower
[llm.provider.ollama]
base_url = "http://localhost:11434"
model = "llama3.2:8b"
timeout_seconds = 30

# Remote API — faster, costs money, data leaves machine
[llm.provider.openai]
api_key_env = "OPENAI_API_KEY"
model = "gpt-4o-mini"              # cheapest option
max_tokens = 500

# Which tasks use LLM
[llm.tasks]
contradiction_detection = true
temporal_reasoning = true
community_summarization = true
unstructured_extraction = false    # only enable if needed
```

#### Tier Selection Logic

ctxgraph automatically selects the appropriate tier per-episode based on content analysis:

```
On episode ingestion:
1. Always run Tier 1 (GLiNER2 + GLiREL)
2. If Tier 2 enabled (default: yes):
   - Run coreference resolution
   - Run fuzzy dedup against existing entities
   - Run enhanced temporal parsing
3. If Tier 3 enabled AND triggered:
   - Contradiction check: only if conflicting edges found
   - Temporal reasoning: only if Tier 2 couldn't resolve a date
   - Summarization: only if community cluster exceeds threshold
   - Full LLM extraction: only if Tier 1 entity count is 
     suspiciously low for the text length (signals missed entities)
```

The trigger heuristic for full LLM extraction:

```
expected_entity_density = text.word_count / 15  # ~1 entity per 15 words
actual_entity_count = tier1_results.entities.len()

if actual_entity_count < expected_entity_density * 0.4 {
    // Tier 1 probably missed a lot — trigger Tier 3
    run_llm_extraction()
}
```

This means Tier 3 is self-regulating. On well-structured text, it almost never fires. On messy text, it fires automatically to compensate for Tier 1's limitations.

---

## Storage Architecture

### SQLite Schema

ctxgraph stores everything in a single SQLite file (`.ctxgraph/graph.db`). The schema uses adjacency lists to represent the graph, with FTS5 for full-text search.

```sql
-- Episodes (raw events)
CREATE TABLE episodes (
    id          TEXT PRIMARY KEY,    -- UUID
    content     TEXT NOT NULL,
    source      TEXT,                -- "manual", "git-commit", "slack", etc.
    recorded_at TEXT NOT NULL,       -- when recorded in ctxgraph
    metadata    TEXT,                -- JSON blob
    embedding   BLOB                -- local embedding vector (optional)
);

-- Entities (extracted nodes)
CREATE TABLE entities (
    id          TEXT PRIMARY KEY,    -- UUID
    name        TEXT NOT NULL,       -- display name
    entity_type TEXT NOT NULL,       -- "Person", "Component", etc.
    summary     TEXT,                -- community summary (Tier 3)
    created_at  TEXT NOT NULL,
    metadata    TEXT                 -- JSON blob
);

-- Relationships (edges between entities)
CREATE TABLE edges (
    id          TEXT PRIMARY KEY,
    source_id   TEXT NOT NULL REFERENCES entities(id),
    target_id   TEXT NOT NULL REFERENCES entities(id),
    relation    TEXT NOT NULL,       -- "chose", "rejected", "approved", etc.
    fact        TEXT,                -- human-readable fact string
    valid_from  TEXT,                -- when this fact became true
    valid_until TEXT,                -- when this fact stopped being true (null = current)
    recorded_at TEXT NOT NULL,
    confidence  REAL DEFAULT 1.0,
    episode_id  TEXT REFERENCES episodes(id),  -- which episode produced this edge
    metadata    TEXT
);

-- Episode-Entity links (which episodes mention which entities)
CREATE TABLE episode_entities (
    episode_id  TEXT REFERENCES episodes(id),
    entity_id   TEXT REFERENCES entities(id),
    span_start  INTEGER,            -- character offset in episode content
    span_end    INTEGER,
    PRIMARY KEY (episode_id, entity_id)
);

-- Entity aliases (for deduplication)
CREATE TABLE aliases (
    canonical_id TEXT REFERENCES entities(id),
    alias_name   TEXT NOT NULL,
    similarity   REAL,              -- Jaro-Winkler score
    UNIQUE(canonical_id, alias_name)
);

-- Community clusters (groups of related entities)
CREATE TABLE communities (
    id          TEXT PRIMARY KEY,
    summary     TEXT,               -- LLM-generated summary (Tier 3)
    entity_ids  TEXT,               -- JSON array of entity IDs
    created_at  TEXT NOT NULL,
    updated_at  TEXT
);

-- Full-text search indexes
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

-- Performance indexes
CREATE INDEX idx_edges_source ON edges(source_id);
CREATE INDEX idx_edges_target ON edges(target_id);
CREATE INDEX idx_edges_relation ON edges(relation);
CREATE INDEX idx_edges_valid ON edges(valid_from, valid_until);
CREATE INDEX idx_entities_type ON entities(entity_type);
CREATE INDEX idx_episode_entities ON episode_entities(entity_id);
```

### Graph Traversal via Recursive CTEs

Multi-hop graph traversal in SQLite uses recursive Common Table Expressions. This is the key technique that eliminates the need for Neo4j.

```sql
-- Find all decisions within 3 hops of "Postgres"
WITH RECURSIVE traversal(entity_id, depth, path) AS (
    -- Start node
    SELECT id, 0, json_array(id)
    FROM entities 
    WHERE name = 'Postgres'
    
    UNION ALL
    
    -- Traverse edges
    SELECT 
        CASE WHEN e.source_id = t.entity_id THEN e.target_id
             ELSE e.source_id END,
        t.depth + 1,
        json_insert(t.path, '$[#]', 
            CASE WHEN e.source_id = t.entity_id THEN e.target_id
                 ELSE e.source_id END)
    FROM traversal t
    JOIN edges e ON (e.source_id = t.entity_id OR e.target_id = t.entity_id)
    WHERE t.depth < 3  -- max hops
      AND e.valid_until IS NULL  -- only current facts
      AND json_array_length(t.path) = t.depth + 1  -- prevent cycles
)
SELECT DISTINCT e.*, t.depth
FROM traversal t
JOIN entities e ON e.id = t.entity_id
ORDER BY t.depth;
```

### Local Embeddings (Optional)

For semantic search beyond keyword matching, ctxgraph can generate embeddings locally using a small ONNX model (all-MiniLM-L6-v2, ~80MB).

```
Episode: "Migrated from S3 to R2 for cost savings"
Embedding: [0.012, -0.034, 0.089, ...] (384 dimensions)

Query: "cloud storage cost optimization"
Query embedding: [0.015, -0.031, 0.092, ...]

Cosine similarity: 0.91 → high relevance
```

Embeddings are stored as BLOBs in SQLite. Search uses a brute-force cosine similarity scan. For graphs under 100K episodes (the vast majority of ctxgraph use cases), this is fast enough (<50ms). For larger graphs, an optional HNSW index (via `usearch` crate) can be enabled.

---

## Query System

### Search Modes

ctxgraph supports three search modes, fused via Reciprocal Rank Fusion (RRF):

```
Query: "why did we choose Postgres?"

Mode 1: FTS5 keyword search
  → Matches episodes/entities/edges containing "Postgres", "chose", "choose"
  → Fast, exact matching

Mode 2: Semantic embedding search  
  → Finds episodes semantically similar to the query
  → Catches "selected PostgreSQL for the database layer" (no keyword overlap)

Mode 3: Graph traversal
  → Finds the Postgres entity node
  → Traverses edges: chose_by, reason_for, rejected_alternative, applies_to
  → Returns the full decision subgraph

RRF fusion merges results from all three modes:
  score = Σ (1 / (k + rank_in_mode_i))  for each result across modes
```

### Query API

```rust
// Simple text query
let results = graph.search("why Postgres?").await?;

// Filtered query
let results = graph.search_with_filter(
    "database decisions",
    SearchFilter {
        after: Some("2026-01-01"),
        before: None,
        source: Some("manual"),
        entity_type: Some("Decision"),
        max_hops: 3,
        include_invalidated: false,  // only current facts
    }
).await?;

// Direct entity lookup + traversal
let entity = graph.get_entity_by_name("Postgres").await?;
let subgraph = graph.traverse(entity.id, max_depth: 3).await?;

// Time-travel query (what was true at a specific point?)
let results = graph.search_at(
    "who works at Google?",
    as_of: "2024-06-15",  // historical query
).await?;
```

### Query Results

```rust
SearchResult {
    // The decision trace
    decision: Decision {
        id: "DECISION_7",
        summary: "Chose Postgres over SQLite for billing service",
        timestamp: "2026-03-11T10:30:00Z",
        confidence: 0.95,
    },
    
    // Connected entities
    entities: [
        Entity { name: "Postgres", type: "Component", role: "chosen" },
        Entity { name: "SQLite", type: "Alternative", role: "rejected" },
        Entity { name: "rohan", type: "Person", role: "decider" },
        Entity { name: "billing", type: "Service", role: "target" },
    ],
    
    // The reasoning chain
    edges: [
        Edge { from: "rohan", relation: "decided", to: "DECISION_7" },
        Edge { from: "DECISION_7", relation: "chose", to: "Postgres" },
        Edge { from: "DECISION_7", relation: "rejected", to: "SQLite" },
        Edge { from: "DECISION_7", relation: "reason", to: "concurrent writes" },
    ],
    
    // Source episode
    episode: Episode {
        content: "Chose Postgres over SQLite for billing...",
        source: "manual",
    },
    
    // Related decisions (graph neighbors)
    related: [
        Decision { id: "DECISION_12", summary: "Rejected Redis, kept Postgres" },
    ],
    
    // Relevance scores
    relevance: 0.94,
    fts_score: 0.88,
    semantic_score: 0.91,
    graph_score: 1.0,  // direct match
}
```

---

## Crate Structure

```
ctxgraph/
├── Cargo.toml                    # workspace root
├── README.md
├── ctxgraph.toml.example         # default schema + config
│
├── crates/
│   ├── ctxgraph-core/            # The engine
│   │   ├── src/
│   │   │   ├── lib.rs            # public API: Graph, Episode, Entity, Edge
│   │   │   ├── graph.rs          # Graph struct: add_episode, search, traverse
│   │   │   ├── episode.rs        # Episode type + builder
│   │   │   ├── entity.rs         # Entity type + resolution
│   │   │   ├── edge.rs           # Edge type + temporal logic
│   │   │   ├── schema.rs         # Schema definition + TOML parsing
│   │   │   ├── storage/
│   │   │   │   ├── mod.rs
│   │   │   │   ├── sqlite.rs     # SQLite driver (rusqlite)
│   │   │   │   ├── migrations.rs # schema migrations
│   │   │   │   └── fts.rs        # FTS5 helpers
│   │   │   ├── query/
│   │   │   │   ├── mod.rs
│   │   │   │   ├── search.rs     # unified search (FTS + semantic + graph)
│   │   │   │   ├── traverse.rs   # recursive CTE graph traversal
│   │   │   │   ├── rrf.rs        # Reciprocal Rank Fusion
│   │   │   │   └── filter.rs     # SearchFilter types
│   │   │   └── temporal.rs       # bi-temporal logic, invalidation
│   │   ├── Cargo.toml
│   │   └── tests/
│   │
│   ├── ctxgraph-extract/         # Extraction pipeline
│   │   ├── src/
│   │   │   ├── lib.rs            # ExtractorPipeline: tier orchestration
│   │   │   ├── tier1/
│   │   │   │   ├── mod.rs
│   │   │   │   ├── gliner.rs     # GLiNER2 ONNX wrapper
│   │   │   │   ├── glirel.rs     # GLiREL ONNX wrapper
│   │   │   │   ├── temporal.rs   # regex + dateparser
│   │   │   │   └── models.rs     # model download + cache management
│   │   │   ├── tier2/
│   │   │   │   ├── mod.rs
│   │   │   │   ├── coreference.rs # pronoun resolution
│   │   │   │   ├── dedup.rs      # Jaro-Winkler entity dedup
│   │   │   │   └── temporal.rs   # context-aware date resolution
│   │   │   └── tier3/
│   │   │       ├── mod.rs
│   │   │       ├── provider.rs   # LLM provider abstraction
│   │   │       ├── ollama.rs     # Ollama client
│   │   │       ├── openai.rs     # OpenAI-compatible client
│   │   │       ├── contradiction.rs  # edge conflict detection
│   │   │       ├── temporal.rs   # complex date reasoning
│   │   │       ├── summarize.rs  # community summarization
│   │   │       └── extract.rs    # full LLM extraction fallback
│   │   ├── Cargo.toml            # depends on: ort, tokenizers
│   │   └── tests/
│   │
│   ├── ctxgraph-embed/           # Local embedding generation
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── model.rs          # all-MiniLM-L6-v2 ONNX wrapper
│   │   │   └── similarity.rs    # cosine similarity, optional HNSW
│   │   └── Cargo.toml
│   │
│   ├── ctxgraph-cli/             # CLI binary
│   │   ├── src/
│   │   │   ├── main.rs           # clap CLI entrypoint
│   │   │   ├── commands/
│   │   │   │   ├── init.rs       # ctxgraph init
│   │   │   │   ├── log.rs        # ctxgraph log "..."
│   │   │   │   ├── query.rs      # ctxgraph query "..."
│   │   │   │   ├── ingest.rs     # ctxgraph ingest --file/--stdin
│   │   │   │   ├── entities.rs   # ctxgraph entities list/show
│   │   │   │   ├── decisions.rs  # ctxgraph decisions list/show
│   │   │   │   ├── stats.rs      # ctxgraph stats
│   │   │   │   ├── models.rs     # ctxgraph models download/list
│   │   │   │   └── export.rs     # ctxgraph export --format json/csv
│   │   │   └── display.rs        # terminal output formatting
│   │   └── Cargo.toml
│   │
│   ├── ctxgraph-mcp/             # MCP server
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── server.rs         # MCP protocol implementation
│   │   │   ├── tools.rs          # add_episode, search, traverse, etc.
│   │   │   └── config.rs
│   │   └── Cargo.toml
│   │
│   └── ctxgraph-sdk/             # Rust SDK for embedding in other apps
│       ├── src/
│       │   └── lib.rs            # re-exports from core + extract + embed
│       └── Cargo.toml
│
├── models/                       # ONNX model configs (not the weights)
│   ├── gliner2-large.toml        # model URL, checksum, quantization
│   ├── glirel-large.toml
│   └── minilm-l6-v2.toml
│
├── schemas/                      # Built-in schemas
│   ├── default.toml              # generic decision tracking
│   ├── developer.toml            # software architecture decisions
│   ├── support.toml              # customer support decisions
│   └── finance.toml              # financial/lending decisions
│
└── tests/
    ├── integration/
    │   ├── tier1_extraction.rs
    │   ├── tier2_dedup.rs
    │   ├── tier3_contradiction.rs
    │   ├── search_rrf.rs
    │   ├── graph_traversal.rs
    │   └── temporal_queries.rs
    └── fixtures/
        ├── sample_episodes.json
        └── expected_graphs.json
```

---

## CLI Reference

```bash
# Initialize ctxgraph in current directory
ctxgraph init [--name <project-name>] [--schema <schema-file>]
# Creates .ctxgraph/ with graph.db and config

# Log a decision manually
ctxgraph log <text> [--source <source>] [--tags <tag1,tag2>]
ctxgraph log "Chose Postgres because concurrent writes needed"
ctxgraph log --source slack "Priya approved the discount"

# Ingest from file or stdin
ctxgraph ingest --file decisions.jsonl
cat events.json | ctxgraph ingest --stdin --source webhook

# Search the graph
ctxgraph query <text> [--after <date>] [--source <src>] [--hops <n>]
ctxgraph query "why Postgres?"
ctxgraph query "discount precedents" --after 2026-01-01 --hops 3

# List entities
ctxgraph entities list [--type <type>] [--limit <n>]
ctxgraph entities show <entity-id>

# List decisions
ctxgraph decisions list [--after <date>] [--source <src>]
ctxgraph decisions show <decision-id>   # full decision trace

# Graph statistics
ctxgraph stats
# Episodes: 1,247 | Entities: 3,891 | Edges: 8,234
# Sources: manual (423), git-commit (612), slack (212)
# Tier usage: T1: 89% | T2: 9% | T3: 2%

# Model management
ctxgraph models download           # download ONNX models
ctxgraph models list               # show cached models + sizes
ctxgraph models verify             # verify checksums

# Export
ctxgraph export --format json > graph.json
ctxgraph export --format csv --entities > entities.csv

# MCP server
ctxgraph mcp start [--port <port>]

# Configuration
ctxgraph config show               # show current config
ctxgraph config set llm.enabled true
```

---

## MCP Server Tools

When running as an MCP server, ctxgraph exposes these tools to AI assistants:

```json
{
  "tools": [
    {
      "name": "ctxgraph_add_episode",
      "description": "Add a new decision or event to the context graph",
      "parameters": {
        "content": "string — the decision/event text",
        "source": "string — where this came from",
        "tags": "string[] — optional tags"
      }
    },
    {
      "name": "ctxgraph_search",
      "description": "Search for relevant decisions and precedents",
      "parameters": {
        "query": "string — natural language search query",
        "max_results": "number — max results (default 5)",
        "after": "string — only decisions after this date",
        "source": "string — filter by source"
      }
    },
    {
      "name": "ctxgraph_get_decision",
      "description": "Get full decision trace by ID",
      "parameters": {
        "decision_id": "string — the decision ID"
      }
    },
    {
      "name": "ctxgraph_traverse",
      "description": "Traverse the graph from an entity",
      "parameters": {
        "entity_name": "string — starting entity",
        "max_depth": "number — max hops (default 3)",
        "relation_filter": "string — only follow these relation types"
      }
    },
    {
      "name": "ctxgraph_find_precedents",
      "description": "Find similar past decisions for a given scenario",
      "parameters": {
        "scenario": "string — describe the current situation",
        "max_results": "number — how many precedents to return"
      }
    }
  ]
}
```

---

## Dependency Tree

```
ctxgraph-core:
  rusqlite        — SQLite driver with FTS5
  uuid            — entity/episode IDs
  chrono          — temporal logic
  serde + serde_json — serialization
  thiserror       — error types

ctxgraph-extract:
  ort             — ONNX Runtime bindings (GLiNER2, GLiREL)
  tokenizers      — HuggingFace tokenizers (Rust native)
  strsim          — Jaro-Winkler similarity
  regex           — temporal pattern matching
  reqwest         — model downloading + LLM API calls (Tier 3)

ctxgraph-embed:
  ort             — ONNX Runtime for embedding model
  ndarray         — vector math for cosine similarity

ctxgraph-cli:
  clap            — CLI argument parsing
  colored         — terminal output
  indicatif       — progress bars (model downloads)

ctxgraph-mcp:
  axum            — HTTP server
  tokio           — async runtime
  serde_json      — MCP protocol serialization
```

---

## Roadmap

| Version | Feature | Tier | Status |
|---|---|---|---|
| v0.1 | Core engine: SQLite storage, basic schema, manual log/query | — | First |
| v0.2 | Tier 1: GLiNER2 ONNX entity extraction | T1 | |
| v0.3 | Tier 1: GLiREL relationship extraction + temporal heuristics | T1 | |
| v0.4 | Tier 2: coreference + fuzzy dedup + enhanced temporal | T2 | |
| v0.5 | Search: FTS5 + local embeddings + RRF fusion | — | |
| v0.6 | MCP server | — | |
| v0.7 | Tier 3: Ollama/API integration for contradiction + summarization | T3 | |
| v0.8 | `ctxgraph ingest` for bulk import (JSONL, CSV) | — | |
| v0.9 | Schema marketplace: community-contributed schemas | — | |
| v1.0 | Production-ready: benchmarks, docs, stability guarantees | — | |

---

## What ctxgraph Is NOT

- **Not a replacement for Graphiti.** If you need enterprise-scale graph operations with Neo4j, use Graphiti. ctxgraph is for different constraints.
- **Not a database.** Don't store your application data in ctxgraph. Store decision traces — the why behind your data.
- **Not an AI agent.** ctxgraph is memory infrastructure. It stores and retrieves context. The agent (Claude, your custom code, DevTrace) decides what to do with it.
- **Not a silver bullet for messy data.** Tier 1 extraction works great on semi-structured text. For highly unstructured text, you need Tier 3 (LLM) or a different tool.