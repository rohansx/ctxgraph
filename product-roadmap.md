# ctxgraph Implementation Roadmap: v0.1 → v1.0

---

## Version Map

```
v0.1 ─── Core Engine ──────────── SQLite storage, manual log/query, basic types
  │                                "The foundation. Can store and retrieve episodes."
  │                                Timeline: ~10 days
  │
v0.2 ─── Tier 1a: Entities ────── GLiNER2 ONNX entity extraction
  │                                "Episodes now auto-extract entities."
  │                                Timeline: ~8 days
  │
v0.3 ─── Tier 1b: Relations ───── GLiREL + temporal heuristics
  │                                "Entities are now connected. Graph exists."
  │                                Timeline: ~7 days
  │
v0.4 ─── Tier 2: Enhanced ─────── Coreference, dedup, enhanced temporal
  │                                "Quality jumps from 85% to 90%."
  │                                Timeline: ~8 days
  │
v0.5 ─── Search ───────────────── FTS5 + local embeddings + RRF + graph traversal
  │                                "Queries actually return good results."
  │                                Timeline: ~8 days
  │
v0.6 ─── MCP Server ──────────── Claude/Cursor can use ctxgraph
  │                                "AI agents have memory."
  │                                Timeline: ~5 days
  │
v0.7 ─── Tier 3: LLM ─────────── Ollama/API for contradictions + summarization
  │                                "Handles messy text. Quality hits 95%."
  │                                Timeline: ~7 days
  │
v0.8 ─── Bulk Ingest ─────────── JSONL/CSV import, webhooks, stdin piping
  │                                "Can ingest thousands of episodes fast."
  │                                Timeline: ~5 days
  │
v0.9 ─── Schemas + Export ─────── Community schemas, JSON/CSV/Neo4j export
  │                                "Ecosystem starts forming."
  │                                Timeline: ~5 days
  │
v1.0 ─── Production Ready ─────── Benchmarks, docs, stability, crates.io
                                   "Ship it."
                                   Timeline: ~7 days
```

Total estimated timeline: ~10-12 weeks (working solo, focused)

---

## v0.1 — Core Engine

**Goal:** A working context graph that can store episodes, manually-tagged entities, and edges in SQLite. No extraction — user provides structured input. Prove the storage and query model works.

**What ships:**
- `ctxgraph init` creates `.ctxgraph/` directory with SQLite database
- `ctxgraph log` stores episodes with manual text
- `ctxgraph query` does basic FTS5 keyword search
- `ctxgraph entities list` shows all entities
- `ctxgraph decisions list` shows all decision traces
- `ctxgraph stats` shows counts

**Crates built:**
- `ctxgraph-core` — types, storage, basic query
- `ctxgraph-cli` — CLI binary

### Implementation Details

#### Step 1: Workspace Setup

```
ctxgraph/
├── Cargo.toml              # workspace
├── crates/
│   ├── ctxgraph-core/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── types.rs     # Episode, Entity, Edge, SearchResult
│   │       ├── graph.rs     # Graph struct — the main API
│   │       ├── storage.rs   # SQLite operations
│   │       ├── migrations.rs # schema creation
│   │       ├── query.rs     # FTS5 search
│   │       └── temporal.rs  # bi-temporal timestamp logic
│   └── ctxgraph-cli/
│       ├── Cargo.toml
│       └── src/
│           ├── main.rs
│           └── commands/
│               ├── mod.rs
│               ├── init.rs
│               ├── log.rs
│               ├── query.rs
│               ├── entities.rs
│               ├── decisions.rs
│               └── stats.rs
└── ctxgraph.toml.example
```

#### Step 2: Core Types (types.rs)

```rust
// These are the fundamental types everything else builds on

pub struct Episode {
    pub id: String,           // UUID v7 (time-sortable)
    pub content: String,
    pub source: Option<String>,
    pub recorded_at: DateTime<Utc>,
    pub metadata: Option<serde_json::Value>,
}

pub struct Entity {
    pub id: String,
    pub name: String,
    pub entity_type: String,  // "Person", "Component", etc.
    pub summary: Option<String>,
    pub created_at: DateTime<Utc>,
    pub metadata: Option<serde_json::Value>,
}

pub struct Edge {
    pub id: String,
    pub source_id: String,    // entity ID
    pub target_id: String,    // entity ID
    pub relation: String,     // "chose", "rejected", etc.
    pub fact: Option<String>, // human-readable fact
    pub valid_from: Option<DateTime<Utc>>,
    pub valid_until: Option<DateTime<Utc>>,  // None = currently true
    pub recorded_at: DateTime<Utc>,
    pub confidence: f64,
    pub episode_id: Option<String>,
}

pub struct SearchResult {
    pub episodes: Vec<Episode>,
    pub entities: Vec<Entity>,
    pub edges: Vec<Edge>,
    pub score: f64,
}

// Builder pattern for episodes
impl Episode {
    pub fn new(content: &str) -> EpisodeBuilder { ... }
}

pub struct EpisodeBuilder {
    content: String,
    source: Option<String>,
    metadata: Option<serde_json::Value>,
    tags: Vec<String>,
}

impl EpisodeBuilder {
    pub fn source(mut self, s: &str) -> Self { ... }
    pub fn tag(mut self, t: &str) -> Self { ... }
    pub fn meta(mut self, key: &str, val: &str) -> Self { ... }
    pub fn build(self) -> Episode { ... }
}
```

#### Step 3: SQLite Storage (storage.rs)

```rust
pub struct Storage {
    conn: rusqlite::Connection,
}

impl Storage {
    pub fn open(path: &Path) -> Result<Self> { ... }
    
    // Episodes
    pub fn insert_episode(&self, episode: &Episode) -> Result<()> { ... }
    pub fn get_episode(&self, id: &str) -> Result<Option<Episode>> { ... }
    pub fn list_episodes(&self, limit: usize, offset: usize) -> Result<Vec<Episode>> { ... }
    
    // Entities
    pub fn insert_entity(&self, entity: &Entity) -> Result<()> { ... }
    pub fn get_entity(&self, id: &str) -> Result<Option<Entity>> { ... }
    pub fn get_entity_by_name(&self, name: &str) -> Result<Option<Entity>> { ... }
    pub fn list_entities(&self, entity_type: Option<&str>) -> Result<Vec<Entity>> { ... }
    
    // Edges
    pub fn insert_edge(&self, edge: &Edge) -> Result<()> { ... }
    pub fn get_edges_for_entity(&self, entity_id: &str) -> Result<Vec<Edge>> { ... }
    pub fn invalidate_edge(&self, edge_id: &str, until: DateTime<Utc>) -> Result<()> { ... }
    
    // Episode-Entity links
    pub fn link_episode_entity(&self, episode_id: &str, entity_id: &str) -> Result<()> { ... }
    
    // FTS5 search
    pub fn search_episodes(&self, query: &str, limit: usize) -> Result<Vec<Episode>> { ... }
    pub fn search_entities(&self, query: &str, limit: usize) -> Result<Vec<Entity>> { ... }
    pub fn search_edges(&self, query: &str, limit: usize) -> Result<Vec<Edge>> { ... }
    
    // Stats
    pub fn stats(&self) -> Result<GraphStats> { ... }
}
```

#### Step 4: Graph API (graph.rs)

```rust
// This is the public API that all consumers use

pub struct Graph {
    storage: Storage,
}

impl Graph {
    pub async fn open(path: &str) -> Result<Self> { ... }
    pub async fn init(dir: &Path) -> Result<Self> { ... }
    
    // Core operations
    pub async fn add_episode(&self, episode: Episode) -> Result<EpisodeResult> { ... }
    pub async fn add_entity(&self, entity: Entity) -> Result<()> { ... }
    pub async fn add_edge(&self, edge: Edge) -> Result<()> { ... }
    
    // Search (basic FTS5 in v0.1)
    pub async fn search(&self, query: &str) -> Result<Vec<SearchResult>> { ... }
    
    // Traversal (basic in v0.1 — just immediate neighbors)
    pub async fn get_entity_context(&self, entity_id: &str) -> Result<EntityContext> { ... }
    
    // Stats
    pub async fn stats(&self) -> Result<GraphStats> { ... }
}
```

#### Step 5: CLI Commands

```bash
# init.rs — create .ctxgraph/ directory
ctxgraph init --name "my-project"
# Creates:
#   .ctxgraph/
#   .ctxgraph/graph.db       (SQLite)
#   .ctxgraph/config.toml    (copied from example)

# log.rs — store an episode (no extraction yet in v0.1)
ctxgraph log "Chose Postgres over SQLite for billing. Reason: concurrent writes."
ctxgraph log --source slack "Priya approved the discount for Reliance"
ctxgraph log --tags "architecture,database" "Switched from REST to gRPC"

# query.rs — basic FTS5 search
ctxgraph query "Postgres"
ctxgraph query "discount" --limit 5

# entities.rs — list/show entities  
# (In v0.1, entities are only created if user manually adds them
#  via the Rust API. Auto-extraction comes in v0.2.)
ctxgraph entities list
ctxgraph entities list --type Person

# decisions.rs — list decisions
ctxgraph decisions list
ctxgraph decisions list --after 2026-03-01

# stats.rs
ctxgraph stats
# Output:
# ctxgraph stats
# ──────────────────
# Episodes:  47
# Entities:  0  (auto-extraction coming in v0.2)
# Edges:     0
# Sources:   manual (35), slack (12)
# DB size:   124 KB
```

#### v0.1 Dependencies (Cargo.toml)

```toml
[workspace]
members = ["crates/*"]

# ctxgraph-core
[dependencies]
rusqlite = { version = "0.31", features = ["bundled", "vtab"] }
uuid = { version = "1", features = ["v7"] }
chrono = { version = "0.4", features = ["serde"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
thiserror = "1"
toml = "0.8"

# ctxgraph-cli
[dependencies]
ctxgraph-core = { path = "../ctxgraph-core" }
clap = { version = "4", features = ["derive"] }
colored = "2"
tokio = { version = "1", features = ["full"] }
```

#### v0.1 Tests

```
tests/
├── storage_test.rs         # CRUD for episodes, entities, edges
├── fts_test.rs             # FTS5 search returns correct results
├── temporal_test.rs        # bi-temporal invalidation logic
├── graph_api_test.rs       # Graph struct methods
└── cli_integration_test.rs # init, log, query end-to-end
```

Key test cases:
- Insert episode → retrieve by ID → content matches
- Insert 10 episodes → FTS5 search "Postgres" → returns correct subset
- Insert edge with valid_from → invalidate → query current → not returned
- Query with `--after` date filter → only newer episodes returned
- `ctxgraph init` creates directory structure
- `ctxgraph log` then `ctxgraph query` roundtrip works

#### v0.1 Definition of Done
- [ ] `ctxgraph init` creates `.ctxgraph/` with SQLite DB
- [ ] `ctxgraph log` stores episodes
- [ ] `ctxgraph query` returns episodes via FTS5
- [ ] `ctxgraph stats` shows correct counts
- [ ] All types are pub and documented
- [ ] 15+ tests passing
- [ ] README with install instructions and quick start
- [ ] Published to crates.io as `ctxgraph-core` and `ctxgraph-cli`

---

## v0.2 — Tier 1a: GLiNER2 Entity Extraction

**Goal:** When an episode is added, entities are automatically extracted using GLiNER2 (ONNX). No more manual entity creation. This is the first major feature that differentiates ctxgraph.

**What ships:**
- GLiNER2 ONNX model integration
- Automatic entity extraction on `ctxgraph log`
- Schema definition in `ctxgraph.toml`
- `ctxgraph models download` for model management
- Entity extraction is visible in `ctxgraph entities list`

**New crate:**
- `ctxgraph-extract` — extraction pipeline

### Implementation Details

#### ONNX Integration Architecture

```
Episode text
    │
    ▼
┌─────────────────────────────┐
│ Tokenizer (HF tokenizers)   │
│ Rust crate: tokenizers      │
│ Loads tokenizer.json from   │
│ model directory              │
└──────────┬──────────────────┘
           │ token IDs + attention masks
           ▼
┌─────────────────────────────┐
│ GLiNER2 ONNX Model          │
│ Rust crate: ort              │
│ File: gliner2-large-q8.onnx │
│ Size: ~200MB (INT8 quantized)│
│                              │
│ Inputs:                      │
│   - input_ids: [batch, seq]  │
│   - attention_mask: [b, seq] │
│   - entity_type_ids: [b, n]  │
│   - entity_type_mask: [b, n] │
│                              │
│ Output:                      │
│   - span_scores: [b, s, s, n]│
│   (start, end, entity_type)  │
└──────────┬──────────────────┘
           │ raw span predictions
           ▼
┌─────────────────────────────┐
│ Post-processing              │
│ - Threshold filtering (0.5)  │
│ - Span merging (overlaps)    │
│ - Character offset mapping   │
│ - Entity dedup within episode│
└──────────┬──────────────────┘
           │ Vec<ExtractedEntity>
           ▼
┌─────────────────────────────┐
│ Storage                      │
│ - Create Entity rows         │
│ - Link episode ↔ entity      │
└─────────────────────────────┘
```

#### Model Management

```rust
// ctxgraph-extract/src/models.rs

pub struct ModelManager {
    cache_dir: PathBuf,  // ~/.ctxgraph/models/
}

impl ModelManager {
    /// Download GLiNER2 ONNX model if not cached
    pub async fn ensure_gliner2(&self) -> Result<PathBuf> {
        let model_path = self.cache_dir.join("gliner2-large-q8.onnx");
        if model_path.exists() {
            return Ok(model_path);
        }
        
        // Download from HuggingFace
        let url = "https://huggingface.co/urchade/gliner_large-v2.1/resolve/main/onnx/model_optimized_quantized.onnx";
        download_with_progress(url, &model_path).await?;
        
        // Verify checksum
        verify_sha256(&model_path, GLINER2_SHA256)?;
        
        Ok(model_path)
    }
    
    /// Download tokenizer files
    pub async fn ensure_tokenizer(&self) -> Result<PathBuf> {
        let tok_path = self.cache_dir.join("gliner2-tokenizer.json");
        if tok_path.exists() {
            return Ok(tok_path);
        }
        
        let url = "https://huggingface.co/urchade/gliner_large-v2.1/resolve/main/tokenizer.json";
        download_with_progress(url, &tok_path).await?;
        Ok(tok_path)
    }
}
```

#### GLiNER2 Wrapper

```rust
// ctxgraph-extract/src/tier1/gliner.rs

pub struct GlinerExtractor {
    session: ort::Session,
    tokenizer: tokenizers::Tokenizer,
    threshold: f32,
}

impl GlinerExtractor {
    pub fn load(model_path: &Path, tokenizer_path: &Path) -> Result<Self> {
        let session = ort::Session::builder()?
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;
        
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)?;
        
        Ok(Self { session, tokenizer, threshold: 0.5 })
    }
    
    pub fn extract(
        &self, 
        text: &str, 
        labels: &[&str],  // ["Person", "Component", "Reason", ...]
    ) -> Result<Vec<ExtractedEntity>> {
        // 1. Tokenize text and labels
        let encoding = self.tokenizer.encode(text, false)?;
        
        // 2. Build input tensors
        //    GLiNER2 takes: text tokens + label tokens in a specific format
        //    Format: [CLS] label1 [SEP] label2 [SEP] ... [SEP] text tokens [SEP]
        let input_ids = self.build_input_ids(&encoding, labels)?;
        let attention_mask = self.build_attention_mask(&input_ids)?;
        
        // 3. Run inference
        let outputs = self.session.run(ort::inputs![
            "input_ids" => input_ids,
            "attention_mask" => attention_mask,
        ]?)?;
        
        // 4. Extract span scores
        let span_scores = outputs[0].try_extract_tensor::<f32>()?;
        
        // 5. Post-process: find spans above threshold
        let entities = self.decode_spans(
            &span_scores, 
            text, 
            labels, 
            &encoding,
        )?;
        
        Ok(entities)
    }
    
    fn decode_spans(
        &self,
        scores: &ndarray::ArrayViewD<f32>,
        text: &str,
        labels: &[&str],
        encoding: &tokenizers::Encoding,
    ) -> Result<Vec<ExtractedEntity>> {
        let mut entities = Vec::new();
        
        // scores shape: [1, num_tokens, num_tokens, num_labels]
        // For each (start, end, label) combination above threshold:
        for start in 0..scores.shape()[1] {
            for end in start..scores.shape()[2] {
                for (label_idx, label) in labels.iter().enumerate() {
                    let score = scores[[0, start, end, label_idx]];
                    if score > self.threshold {
                        // Map token offsets back to character offsets
                        let char_start = encoding.token_to_chars(start)?.0;
                        let char_end = encoding.token_to_chars(end)?.1;
                        let span_text = &text[char_start..char_end];
                        
                        entities.push(ExtractedEntity {
                            text: span_text.to_string(),
                            entity_type: label.to_string(),
                            confidence: score as f64,
                            span_start: char_start,
                            span_end: char_end,
                        });
                    }
                }
            }
        }
        
        // Remove overlapping spans (keep highest confidence)
        self.resolve_overlaps(&mut entities);
        
        Ok(entities)
    }
}

pub struct ExtractedEntity {
    pub text: String,
    pub entity_type: String,
    pub confidence: f64,
    pub span_start: usize,
    pub span_end: usize,
}
```

#### Schema Loading

```toml
# ctxgraph.toml
[schema]
name = "default"

[schema.entities]
Person = "A person involved in a decision"
Component = "A software component, tool, or technology"
Service = "A service or application"
Decision = "An explicit choice that was made"
Reason = "The justification behind a decision"
Alternative = "An option considered but not chosen"
Policy = "A rule or guideline referenced"
Amount = "A monetary value or metric"
Constraint = "A limitation or requirement"
```

```rust
// ctxgraph-extract/src/schema.rs

pub struct ExtractionSchema {
    pub entity_labels: Vec<EntityLabel>,
}

pub struct EntityLabel {
    pub name: String,        // "Person"
    pub description: String, // "A person involved in a decision"
}

impl ExtractionSchema {
    pub fn from_toml(path: &Path) -> Result<Self> { ... }
    pub fn label_names(&self) -> Vec<&str> { ... }
}
```

#### Integration into Graph.add_episode()

```rust
// graph.rs — updated for v0.2

impl Graph {
    pub async fn add_episode(&self, episode: Episode) -> Result<EpisodeResult> {
        // 1. Store the episode
        self.storage.insert_episode(&episode)?;
        
        // 2. Extract entities (NEW in v0.2)
        let labels = self.schema.label_names();
        let extracted = self.extractor.extract(&episode.content, &labels)?;
        
        // 3. Store entities and link to episode
        let mut entity_ids = Vec::new();
        for ext in &extracted {
            let entity = Entity {
                id: uuid::Uuid::now_v7().to_string(),
                name: ext.text.clone(),
                entity_type: ext.entity_type.clone(),
                ..Default::default()
            };
            self.storage.insert_entity(&entity)?;
            self.storage.link_episode_entity(&episode.id, &entity.id)?;
            entity_ids.push(entity.id);
        }
        
        Ok(EpisodeResult {
            episode_id: episode.id,
            entities_extracted: extracted.len(),
            entity_ids,
            edges_created: 0,  // v0.3
        })
    }
}
```

#### v0.2 CLI Changes

```bash
# Now shows auto-extracted entities
ctxgraph log "Chose Postgres over SQLite for billing service"
# ✓ Episode stored
# ✓ Extracted 3 entities:
#   Component: "Postgres" (0.94)
#   Alternative: "SQLite" (0.91)
#   Service: "billing service" (0.88)

# Entities are populated automatically
ctxgraph entities list
# ID          Type        Name              Episodes
# ent_a1b2    Component   Postgres          3
# ent_c3d4    Alternative SQLite            1
# ent_e5f6    Service     billing service   1
# ent_g7h8    Person      Priya             2

# First run downloads models
ctxgraph models download
# Downloading GLiNER2-large (INT8)... ━━━━━━━━━━ 198MB
# Downloading tokenizer... ━━━━━━━━━━ 2MB
# Models cached at ~/.ctxgraph/models/

ctxgraph models list
# Model              Size    Status
# gliner2-large-q8   198MB   ✓ cached
# tokenizer          2MB     ✓ cached
```

#### v0.2 Dependencies Added

```toml
# ctxgraph-extract/Cargo.toml
[dependencies]
ctxgraph-core = { path = "../ctxgraph-core" }
ort = { version = "2", features = ["download-binaries"] }
tokenizers = { version = "0.19" }
ndarray = "0.15"
reqwest = { version = "0.12", features = ["stream"] }
indicatif = "0.17"  # progress bars for downloads
sha2 = "0.10"       # checksum verification
```

#### v0.2 Tests

- GLiNER2 model loads successfully from ONNX
- Extract entities from "Chose Postgres for billing" → finds Component, Service
- Extract entities from "Priya approved 30% discount" → finds Person, Amount
- Schema loading from TOML works
- Custom schema with domain-specific labels works
- Empty text → zero entities (no crash)
- Very long text (>512 tokens) → handles truncation gracefully
- Model download + cache works
- Re-download skipped when cached

#### v0.2 Definition of Done
- [ ] GLiNER2 ONNX model loads and runs in Rust via `ort`
- [ ] `ctxgraph log` auto-extracts entities
- [ ] Schema configurable via `ctxgraph.toml`
- [ ] `ctxgraph models download` works
- [ ] `ctxgraph entities list` shows extracted entities
- [ ] 10+ new tests for extraction pipeline
- [ ] Extraction latency < 50ms on CPU

---

## v0.3 — Tier 1b: Relationship Extraction + Temporal

**Goal:** Entities are now connected by edges. The graph has structure. Temporal heuristics parse dates from text.

**What ships:**
- GLiREL ONNX relationship extraction
- Temporal heuristic parser (regex + chrono)
- Edges auto-created between entities
- `ctxgraph decisions show` displays full decision traces
- Graph traversal (1-hop neighbors)

### Implementation Details

#### GLiREL Integration

```rust
// ctxgraph-extract/src/tier1/glirel.rs

pub struct GlirelExtractor {
    session: ort::Session,
    tokenizer: tokenizers::Tokenizer,
    threshold: f32,
}

impl GlirelExtractor {
    pub fn extract_relations(
        &self,
        text: &str,
        entities: &[ExtractedEntity],
        relation_types: &[&str],  // ["chose", "rejected", "approved", ...]
    ) -> Result<Vec<ExtractedRelation>> {
        // GLiREL takes:
        //   - The original text
        //   - Entity spans (start, end, type)  
        //   - Candidate relation types
        // Returns:
        //   - (head_entity, relation, tail_entity, score) tuples
        
        // Build input with entity markers
        // GLiREL expects entities marked in text:
        //   "[E1] Postgres [/E1] was chosen over [E2] SQLite [/E2]"
        let marked_text = self.mark_entities(text, entities);
        
        // Run inference for each entity pair
        let mut relations = Vec::new();
        for (i, head) in entities.iter().enumerate() {
            for (j, tail) in entities.iter().enumerate() {
                if i == j { continue; }
                
                let pair_scores = self.score_pair(
                    &marked_text, head, tail, relation_types
                )?;
                
                for (rel_idx, score) in pair_scores.iter().enumerate() {
                    if *score > self.threshold {
                        relations.push(ExtractedRelation {
                            head: head.clone(),
                            tail: tail.clone(),
                            relation: relation_types[rel_idx].to_string(),
                            confidence: *score as f64,
                        });
                    }
                }
            }
        }
        
        Ok(relations)
    }
}

pub struct ExtractedRelation {
    pub head: ExtractedEntity,
    pub tail: ExtractedEntity,
    pub relation: String,
    pub confidence: f64,
}
```

#### Temporal Heuristics

```rust
// ctxgraph-extract/src/tier1/temporal.rs

pub struct TemporalParser;

impl TemporalParser {
    /// Parse temporal expressions from text, relative to reference_time
    pub fn extract(text: &str, reference_time: DateTime<Utc>) -> Vec<TemporalExtraction> {
        let mut results = Vec::new();
        
        // Layer 1: ISO-8601 dates
        // "2026-03-11", "2026-03-11T10:30:00Z"
        results.extend(Self::parse_iso8601(text));
        
        // Layer 2: Written dates
        // "March 11, 2026", "11th March 2026", "Mar 11"
        results.extend(Self::parse_written_dates(text));
        
        // Layer 3: Relative dates
        // "yesterday", "last week", "3 days ago", "next Monday"
        results.extend(Self::parse_relative(text, reference_time));
        
        // Layer 4: Fiscal/quarter dates
        // "Q3 2025" → 2025-07-01, "FY26" → 2025-04-01 (India fiscal year)
        results.extend(Self::parse_fiscal(text));
        
        // Layer 5: Duration expressions
        // "for 3 months", "over 2 years"
        results.extend(Self::parse_durations(text));
        
        results
    }
    
    fn parse_relative(text: &str, ref_time: DateTime<Utc>) -> Vec<TemporalExtraction> {
        let patterns = vec![
            // "yesterday" / "today" / "tomorrow"
            (r"(?i)\byesterday\b", -1, ChronoUnit::Day),
            (r"(?i)\btoday\b", 0, ChronoUnit::Day),
            (r"(?i)\btomorrow\b", 1, ChronoUnit::Day),
            
            // "last/next week/month/year"
            (r"(?i)\blast\s+week\b", -7, ChronoUnit::Day),
            (r"(?i)\bnext\s+week\b", 7, ChronoUnit::Day),
            (r"(?i)\blast\s+month\b", -1, ChronoUnit::Month),
            (r"(?i)\bnext\s+month\b", 1, ChronoUnit::Month),
            
            // "N days/weeks/months ago"
            (r"(?i)(\d+)\s+days?\s+ago", -1, ChronoUnit::Day),
            (r"(?i)(\d+)\s+weeks?\s+ago", -7, ChronoUnit::Day),
            (r"(?i)(\d+)\s+months?\s+ago", -1, ChronoUnit::Month),
            
            // "last Tuesday", "next Friday"
            // (more complex — resolve to nearest matching weekday)
        ];
        
        // Apply each pattern...
        todo!()
    }
}

pub struct TemporalExtraction {
    pub text: String,          // "last week"
    pub resolved: DateTime<Utc>, // 2026-03-04
    pub confidence: f64,
    pub span_start: usize,
    pub span_end: usize,
}
```

#### Updated add_episode Pipeline

```rust
// graph.rs — v0.3

impl Graph {
    pub async fn add_episode(&self, episode: Episode) -> Result<EpisodeResult> {
        // 1. Store episode
        self.storage.insert_episode(&episode)?;
        
        // 2. Extract entities (v0.2)
        let entities = self.extractor.extract_entities(&episode.content, &self.schema)?;
        
        // 3. Extract relations (NEW v0.3)
        let relations = self.extractor.extract_relations(
            &episode.content, &entities, &self.schema
        )?;
        
        // 4. Extract temporal info (NEW v0.3)
        let temporals = TemporalParser::extract(
            &episode.content, episode.recorded_at
        );
        
        // 5. Store entities
        let entity_map = self.store_entities(&entities, &episode.id)?;
        
        // 6. Store edges from relations (NEW v0.3)
        let mut edge_count = 0;
        for rel in &relations {
            let head_id = entity_map.get(&rel.head.text);
            let tail_id = entity_map.get(&rel.tail.text);
            
            if let (Some(h), Some(t)) = (head_id, tail_id) {
                let edge = Edge {
                    id: uuid::Uuid::now_v7().to_string(),
                    source_id: h.clone(),
                    target_id: t.clone(),
                    relation: rel.relation.clone(),
                    fact: Some(format!("{} {} {}", 
                        rel.head.text, rel.relation, rel.tail.text)),
                    valid_from: temporals.first().map(|t| t.resolved),
                    valid_until: None,
                    recorded_at: Utc::now(),
                    confidence: rel.confidence,
                    episode_id: Some(episode.id.clone()),
                };
                self.storage.insert_edge(&edge)?;
                edge_count += 1;
            }
        }
        
        Ok(EpisodeResult {
            episode_id: episode.id,
            entities_extracted: entities.len(),
            edges_created: edge_count,
            ..Default::default()
        })
    }
}
```

#### v0.3 CLI Changes

```bash
ctxgraph log "Chose Postgres over SQLite for billing. Reason: concurrent writes."
# ✓ Episode stored
# ✓ Extracted 4 entities
# ✓ Created 3 relationships:
#   "Postgres" ──[chose_for]──→ "billing"
#   "SQLite" ──[rejected_for]──→ "billing"  
#   "concurrent writes" ──[reason_for]──→ Decision

# Full decision trace
ctxgraph decisions show dec_a1b2
# ┌─────────────────────────────────────┐
# │ DECISION dec_a1b2                   │
# │ 2026-03-11 10:30 UTC                │
# │                                     │
# │ "Chose Postgres over SQLite for     │
# │  billing service"                   │
# │                                     │
# │ Entities:                           │
# │   rohan (Person) ──[decided]──→     │
# │   Postgres (Component) ──[chosen]── │
# │   SQLite (Alternative) ──[rejected] │
# │   billing (Service) ──[target]──    │
# │                                     │
# │ Reason: concurrent writes           │
# │ Source: manual                      │
# │ Confidence: 0.91                    │
# └─────────────────────────────────────┘
```

#### v0.3 Tests
- GLiREL extracts relations between known entities
- Temporal parser: "yesterday" → correct date
- Temporal parser: "Q3 2025" → 2025-07-01
- Temporal parser: "3 weeks ago" → correct date
- Full pipeline: episode → entities → edges → stored correctly
- Decision trace reconstruction from edges
- 1-hop graph traversal returns correct neighbors

#### v0.3 Definition of Done
- [ ] GLiREL ONNX model runs in Rust
- [ ] Relations auto-extracted between entities
- [ ] Temporal heuristics parse 80%+ of common date patterns
- [ ] `ctxgraph decisions show` displays full trace
- [ ] 1-hop neighbor traversal works
- [ ] 12+ new tests

---

## v0.4 — Tier 2: Enhanced Local Extraction

**Goal:** Quality improvement without any LLM. Coreference resolution, fuzzy entity dedup, and context-aware temporal parsing.

**What ships:**
- Pronoun resolution ("she" → nearest Person entity)
- Fuzzy entity dedup via Jaro-Winkler (merge "Priya Sharma" and "P. Sharma")
- Context-aware temporal resolution (relative-to-event dates)
- User-defined alias groups in config

### Implementation Details

#### Coreference Resolution

```rust
// ctxgraph-extract/src/tier2/coreference.rs

pub struct CoreferenceResolver;

impl CoreferenceResolver {
    /// Resolve pronouns and definite references to their antecedents
    pub fn resolve(
        text: &str,
        entities: &mut Vec<ExtractedEntity>,
    ) -> Vec<CoreferenceLink> {
        let mut links = Vec::new();
        
        let pronouns = Self::find_pronouns(text);
        
        for pronoun in &pronouns {
            // Find nearest entity of matching type
            let antecedent = Self::find_antecedent(
                pronoun, entities, text
            );
            
            if let Some(ant) = antecedent {
                links.push(CoreferenceLink {
                    pronoun: pronoun.clone(),
                    resolved_to: ant.clone(),
                    confidence: Self::score_link(pronoun, &ant, text),
                });
            }
        }
        
        links
    }
    
    fn find_antecedent(
        pronoun: &PronounSpan,
        entities: &[ExtractedEntity],
        text: &str,
    ) -> Option<ExtractedEntity> {
        // Filter entities by compatible type
        let compatible: Vec<_> = entities.iter()
            .filter(|e| Self::type_compatible(&pronoun.pronoun_type, &e.entity_type))
            .filter(|e| e.span_end < pronoun.span_start)  // must appear before
            .collect();
        
        // Return closest compatible entity (by character distance)
        compatible.into_iter()
            .min_by_key(|e| pronoun.span_start - e.span_end)
            .cloned()
    }
    
    fn type_compatible(pronoun_type: &PronounType, entity_type: &str) -> bool {
        match pronoun_type {
            PronounType::PersonMale | PronounType::PersonFemale 
                | PronounType::PersonNeutral => entity_type == "Person",
            PronounType::Organization => {
                entity_type == "Component" || entity_type == "Service" 
                || entity_type == "Organization"
            }
            PronounType::Thing => {
                entity_type != "Person"  // match any non-person
            }
        }
    }
}
```

#### Fuzzy Entity Deduplication

```rust
// ctxgraph-extract/src/tier2/dedup.rs

pub struct EntityDeduplicator {
    threshold: f64,          // 0.85 default
    same_source_threshold: f64,  // 0.75 
    aliases: HashMap<String, Vec<String>>,
}

impl EntityDeduplicator {
    /// Check if a new entity matches any existing entity in storage
    pub fn find_match(
        &self,
        new_entity: &ExtractedEntity,
        existing: &[Entity],
    ) -> Option<DeduplicationMatch> {
        // 1. Check exact alias match first
        if let Some(canonical) = self.check_aliases(&new_entity.name) {
            if let Some(existing_entity) = existing.iter()
                .find(|e| e.name == canonical) 
            {
                return Some(DeduplicationMatch {
                    existing_id: existing_entity.id.clone(),
                    similarity: 1.0,
                    method: MatchMethod::Alias,
                });
            }
        }
        
        // 2. Fuzzy matching via Jaro-Winkler
        let mut best_match: Option<(f64, &Entity)> = None;
        
        for entity in existing {
            // Only compare same type
            if entity.entity_type != new_entity.entity_type {
                continue;
            }
            
            let sim = strsim::jaro_winkler(
                &new_entity.name.to_lowercase(),
                &entity.name.to_lowercase(),
            );
            
            if sim >= self.threshold {
                if best_match.is_none() || sim > best_match.unwrap().0 {
                    best_match = Some((sim, entity));
                }
            }
        }
        
        best_match.map(|(sim, entity)| DeduplicationMatch {
            existing_id: entity.id.clone(),
            similarity: sim,
            method: MatchMethod::JaroWinkler,
        })
    }
}
```

#### v0.4 Config Additions

```toml
# ctxgraph.toml

[tier2]
enabled = true

[tier2.coreference]
enabled = true
max_distance = 500  # max character distance for antecedent search

[tier2.dedup]
threshold = 0.85
same_source_threshold = 0.75

[tier2.dedup.aliases]
"Priya" = ["Priya Sharma", "P. Sharma", "PS"]
"Postgres" = ["PostgreSQL", "PG", "psql"]
"K8s" = ["Kubernetes", "kube"]

[tier2.temporal]
fiscal_year_start = "april"   # India: April. US: January or October.
timezone = "Asia/Kolkata"
```

#### v0.4 Tests
- "She approved it" after "Priya reviewed the doc" → She resolves to Priya
- "The company" after "Reliance Industries reported" → resolves to Reliance
- "P. Sharma" dedup matches "Priya Sharma" (Jaro-Winkler > 0.85)
- "PostgreSQL" dedup matches alias "Postgres"
- "three weeks after the migration" → resolves relative to migration event
- Alias groups from config are loaded and applied
- Dedup only compares same entity types

#### v0.4 Definition of Done
- [ ] Coreference resolves pronouns to nearest entity
- [ ] Jaro-Winkler dedup merges similar entity names
- [ ] Alias groups configurable in TOML
- [ ] Context-aware temporal resolves relative-to-event dates
- [ ] 12+ new tests
- [ ] Quality benchmark: 88-90% on semi-structured text corpus

---

## v0.5 — Search: FTS5 + Embeddings + Graph Traversal + RRF

**Goal:** Queries return genuinely useful results. Three search modes fused together.

**What ships:**
- Local embedding model (all-MiniLM-L6-v2, ONNX, ~80MB)
- Semantic search via cosine similarity
- Multi-hop graph traversal via recursive CTEs
- Reciprocal Rank Fusion merging all three modes
- `ctxgraph query` returns rich, ranked results

**New crate:**
- `ctxgraph-embed` — local embedding generation

### Implementation Details

#### Embedding Generation

```rust
// ctxgraph-embed/src/model.rs

pub struct Embedder {
    session: ort::Session,
    tokenizer: tokenizers::Tokenizer,
}

impl Embedder {
    pub fn load(model_path: &Path, tokenizer_path: &Path) -> Result<Self> { ... }
    
    /// Generate 384-dim embedding vector for text
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let encoding = self.tokenizer.encode(text, true)?;
        let input_ids = encoding.get_ids().to_vec();
        let attention_mask = encoding.get_attention_mask().to_vec();
        
        let outputs = self.session.run(ort::inputs![
            "input_ids" => ndarray::Array2::from_shape_vec(
                (1, input_ids.len()), input_ids.iter().map(|x| *x as i64).collect()
            )?,
            "attention_mask" => ndarray::Array2::from_shape_vec(
                (1, attention_mask.len()), attention_mask.iter().map(|x| *x as i64).collect()
            )?,
        ]?)?;
        
        // Mean pooling over token embeddings
        let embeddings = outputs[0].try_extract_tensor::<f32>()?;
        let pooled = self.mean_pool(&embeddings, &attention_mask);
        
        Ok(pooled)
    }
}
```

#### Recursive CTE Graph Traversal

```rust
// ctxgraph-core/src/query/traverse.rs

impl Storage {
    /// Multi-hop graph traversal from a starting entity
    pub fn traverse(
        &self,
        start_entity_id: &str,
        max_depth: usize,
        valid_at: Option<DateTime<Utc>>,  // time-travel query
    ) -> Result<TraversalResult> {
        let valid_clause = match valid_at {
            Some(at) => format!(
                "AND (e.valid_from IS NULL OR e.valid_from <= '{}') 
                 AND (e.valid_until IS NULL OR e.valid_until > '{}')",
                at.to_rfc3339(), at.to_rfc3339()
            ),
            None => "AND e.valid_until IS NULL".to_string(), // only current
        };
        
        let sql = format!(r#"
            WITH RECURSIVE traversal(entity_id, depth, path, edges_path) AS (
                SELECT ?, 0, json_array(?), json_array()
                
                UNION ALL
                
                SELECT 
                    CASE 
                        WHEN e.source_id = t.entity_id THEN e.target_id
                        ELSE e.source_id 
                    END,
                    t.depth + 1,
                    json_insert(t.path, '$[#]', 
                        CASE WHEN e.source_id = t.entity_id 
                             THEN e.target_id ELSE e.source_id END),
                    json_insert(t.edges_path, '$[#]', e.id)
                FROM traversal t
                JOIN edges e ON (e.source_id = t.entity_id 
                                 OR e.target_id = t.entity_id)
                WHERE t.depth < ?
                  {valid_clause}
                  AND NOT json_each.value = 
                      CASE WHEN e.source_id = t.entity_id 
                           THEN e.target_id ELSE e.source_id END
            )
            SELECT DISTINCT 
                ent.*, 
                t.depth,
                t.path,
                t.edges_path
            FROM traversal t
            JOIN entities ent ON ent.id = t.entity_id
            ORDER BY t.depth
        "#);
        
        // Execute and build TraversalResult
        todo!()
    }
}
```

#### Reciprocal Rank Fusion

```rust
// ctxgraph-core/src/query/rrf.rs

pub fn reciprocal_rank_fusion(
    fts_results: &[ScoredResult],
    semantic_results: &[ScoredResult],
    graph_results: &[ScoredResult],
    k: f64,  // constant, typically 60.0
) -> Vec<FusedResult> {
    let mut scores: HashMap<String, f64> = HashMap::new();
    
    // FTS5 scores
    for (rank, result) in fts_results.iter().enumerate() {
        *scores.entry(result.id.clone()).or_default() 
            += 1.0 / (k + rank as f64 + 1.0);
    }
    
    // Semantic scores
    for (rank, result) in semantic_results.iter().enumerate() {
        *scores.entry(result.id.clone()).or_default() 
            += 1.0 / (k + rank as f64 + 1.0);
    }
    
    // Graph traversal scores (weighted by inverse depth)
    for (rank, result) in graph_results.iter().enumerate() {
        *scores.entry(result.id.clone()).or_default() 
            += 1.0 / (k + rank as f64 + 1.0);
    }
    
    // Sort by fused score descending
    let mut fused: Vec<_> = scores.into_iter()
        .map(|(id, score)| FusedResult { id, score })
        .collect();
    fused.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    
    fused
}
```

#### v0.5 Tests
- FTS5 search returns keyword-matching episodes
- Semantic search returns semantically similar episodes (not just keyword)
- Graph traversal from entity returns N-hop neighbors
- RRF fusion properly merges and ranks results from all 3 modes
- Time-travel query returns facts valid at a past date
- Embedding generation produces 384-dim vectors
- Cosine similarity correctly ranks similar vs dissimilar texts

#### v0.5 Definition of Done
- [ ] Local embedding model loads and generates vectors
- [ ] Semantic search returns relevant results beyond keyword match
- [ ] Multi-hop graph traversal via recursive CTEs works
- [ ] RRF fusion merges all three search modes
- [ ] `ctxgraph query` returns rich ranked results
- [ ] Query latency < 100ms for graphs under 10K episodes
- [ ] 15+ new tests

---

## v0.6 — MCP Server

**Goal:** AI assistants (Claude, Cursor) can use ctxgraph as a memory tool.

**What ships:**
- MCP server exposing 5 tools
- Works with Claude Desktop, Cursor, Claude Code
- `ctxgraph mcp start` command

**New crate:**
- `ctxgraph-mcp`

### Implementation Details

#### MCP Tool Definitions

```rust
// ctxgraph-mcp/src/tools.rs

pub fn tool_definitions() -> Vec<Tool> {
    vec![
        Tool {
            name: "ctxgraph_add_episode",
            description: "Record a new decision, event, or piece of context",
            parameters: json!({
                "type": "object",
                "properties": {
                    "content": { "type": "string", "description": "What happened" },
                    "source": { "type": "string", "description": "Where this came from" },
                    "tags": { "type": "array", "items": { "type": "string" } }
                },
                "required": ["content"]
            }),
        },
        Tool {
            name: "ctxgraph_search",
            description: "Search for relevant decisions, precedents, and context",
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "max_results": { "type": "number", "default": 5 },
                    "after": { "type": "string", "description": "ISO date filter" },
                    "source": { "type": "string" }
                },
                "required": ["query"]
            }),
        },
        Tool {
            name: "ctxgraph_get_decision",
            description: "Get the full context trace for a specific decision",
            parameters: json!({
                "type": "object",
                "properties": {
                    "decision_id": { "type": "string" }
                },
                "required": ["decision_id"]
            }),
        },
        Tool {
            name: "ctxgraph_traverse",
            description: "Explore the graph starting from an entity",
            parameters: json!({
                "type": "object",
                "properties": {
                    "entity_name": { "type": "string" },
                    "max_depth": { "type": "number", "default": 3 }
                },
                "required": ["entity_name"]
            }),
        },
        Tool {
            name: "ctxgraph_find_precedents",
            description: "Find similar past decisions for a current scenario",
            parameters: json!({
                "type": "object",
                "properties": {
                    "scenario": { "type": "string" },
                    "max_results": { "type": "number", "default": 3 }
                },
                "required": ["scenario"]
            }),
        },
    ]
}
```

#### MCP Transport

```rust
// ctxgraph-mcp/src/server.rs — stdio transport for MCP

pub async fn run_stdio_server(graph: Graph) -> Result<()> {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();
    
    let mut reader = BufReader::new(stdin);
    let mut writer = BufWriter::new(stdout);
    
    loop {
        // Read JSON-RPC message from stdin
        let mut line = String::new();
        reader.read_line(&mut line).await?;
        
        let request: JsonRpcRequest = serde_json::from_str(&line)?;
        
        let response = match request.method.as_str() {
            "initialize" => handle_initialize(),
            "tools/list" => handle_tools_list(),
            "tools/call" => handle_tool_call(&graph, &request.params).await,
            _ => error_response("method not found"),
        };
        
        let response_str = serde_json::to_string(&response)?;
        writer.write_all(response_str.as_bytes()).await?;
        writer.write_all(b"\n").await?;
        writer.flush().await?;
    }
}
```

#### Claude Desktop Configuration

```json
{
  "mcpServers": {
    "ctxgraph": {
      "command": "ctxgraph",
      "args": ["mcp", "start", "--db", "/path/to/project/.ctxgraph/graph.db"]
    }
  }
}
```

#### v0.6 Tests
- MCP initialize handshake works
- tools/list returns all 5 tools
- ctxgraph_add_episode stores and extracts
- ctxgraph_search returns ranked results
- ctxgraph_traverse returns graph neighbors
- Invalid tool call returns proper error

#### v0.6 Definition of Done
- [ ] `ctxgraph mcp start` runs MCP server on stdio
- [ ] All 5 tools functional
- [ ] Tested with Claude Desktop
- [ ] Tested with Cursor
- [ ] Error handling for invalid inputs
- [ ] 8+ new tests

---

## v0.7 — Tier 3: LLM Enhancement

**Goal:** Optional LLM integration for contradiction detection, summarization, complex temporal reasoning, and unstructured text extraction.

**What ships:**
- Ollama integration (local, free)
- OpenAI-compatible API integration (remote, paid)
- Contradiction detection on conflicting edges
- Community summarization
- Auto-escalation when Tier 1 misses entities
- `ctxgraph config set llm.enabled true`

### Implementation Details

#### LLM Provider Abstraction

```rust
// ctxgraph-extract/src/tier3/provider.rs

#[async_trait]
pub trait LlmProvider: Send + Sync {
    async fn complete(&self, prompt: &str, system: &str) -> Result<String>;
    async fn complete_json<T: DeserializeOwned>(
        &self, prompt: &str, system: &str
    ) -> Result<T>;
}

pub struct OllamaProvider {
    base_url: String,  // http://localhost:11434
    model: String,     // "llama3.2:8b"
}

pub struct OpenAiProvider {
    api_key: String,
    base_url: String,  // https://api.openai.com/v1 or compatible
    model: String,     // "gpt-4o-mini"
}
```

#### Contradiction Detection

```rust
// ctxgraph-extract/src/tier3/contradiction.rs

pub struct ContradictionDetector {
    provider: Box<dyn LlmProvider>,
}

impl ContradictionDetector {
    pub async fn check(
        &self,
        new_edge: &Edge,
        existing_edges: &[Edge],
    ) -> Result<Vec<Contradiction>> {
        if existing_edges.is_empty() {
            return Ok(vec![]);
        }
        
        let prompt = format!(
            "Existing facts:\n{}\n\nNew fact:\n{}\n\n\
             For each existing fact, does the new fact contradict it? \
             Respond as JSON array: \
             [{{\"existing_id\": \"...\", \"contradicts\": true/false, \
             \"explanation\": \"...\"}}]",
            existing_edges.iter()
                .map(|e| format!("- [{}] {}", e.id, e.fact.as_deref().unwrap_or("")))
                .collect::<Vec<_>>().join("\n"),
            new_edge.fact.as_deref().unwrap_or(""),
        );
        
        let result: Vec<ContradictionResult> = self.provider
            .complete_json(&prompt, SYSTEM_PROMPT)
            .await?;
        
        Ok(result.into_iter()
            .filter(|r| r.contradicts)
            .map(|r| Contradiction {
                existing_edge_id: r.existing_id,
                explanation: r.explanation,
            })
            .collect())
    }
}
```

#### Auto-Escalation Logic

```rust
// ctxgraph-extract/src/lib.rs — ExtractorPipeline

impl ExtractorPipeline {
    pub async fn extract(&self, episode: &Episode) -> Result<ExtractionResult> {
        // Tier 1: always run
        let mut entities = self.tier1.extract_entities(&episode.content)?;
        let mut relations = self.tier1.extract_relations(&episode.content, &entities)?;
        let mut temporals = self.tier1.extract_temporal(&episode.content, episode.recorded_at);
        
        // Tier 2: run if enabled
        if self.config.tier2.enabled {
            let coref_links = CoreferenceResolver::resolve(&episode.content, &mut entities);
            // Apply dedup against existing entities in storage...
            // Enhance temporal with context...
        }
        
        // Tier 3: conditional escalation
        if self.config.tier3.enabled {
            // Auto-escalation: did Tier 1 miss too many entities?
            let word_count = episode.content.split_whitespace().count();
            let expected_density = word_count as f64 / 15.0;
            let actual = entities.len() as f64;
            
            if actual < expected_density * 0.4 {
                // Tier 1 likely missed a lot — use LLM
                let llm_result = self.tier3.full_extraction(&episode.content).await?;
                entities = llm_result.entities;
                relations = llm_result.relations;
            }
            
            // Contradiction detection (always if Tier 3 enabled)
            // Only fires when overlapping edges exist
            
            // Community summarization 
            // Only when cluster grows past threshold
        }
        
        Ok(ExtractionResult { entities, relations, temporals })
    }
}
```

#### v0.7 Definition of Done
- [ ] Ollama provider connects and completes prompts
- [ ] OpenAI-compatible provider works
- [ ] Contradiction detection invalidates old edges
- [ ] Community summarization generates readable summaries
- [ ] Auto-escalation fires when Tier 1 extraction is sparse
- [ ] LLM calls are logged in audit trail (prompt metadata, not content)
- [ ] 10+ new tests (including mock LLM provider)

---

## v0.8 — Bulk Ingest

**Goal:** Ingest thousands of episodes efficiently from files, stdin, and webhooks.

**What ships:**
- `ctxgraph ingest --file data.jsonl`
- `ctxgraph ingest --stdin` for piping
- `ctxgraph ingest --csv data.csv --content-column text`
- Batch processing with progress bars
- Parallel extraction (multi-threaded)

### Implementation Details

#### JSONL Format

```jsonl
{"content": "Chose Postgres for billing", "source": "manual", "tags": ["db"]}
{"content": "Priya approved discount", "source": "slack", "timestamp": "2026-03-11T10:00:00Z"}
{"content": "Migrated from S3 to R2", "source": "github-pr", "metadata": {"pr": 156}}
```

#### Batch Processing

```rust
// ctxgraph-cli/src/commands/ingest.rs

pub async fn ingest_jsonl(
    graph: &Graph,
    path: &Path,
    batch_size: usize,  // default 50
    parallel: usize,    // default 4
) -> Result<IngestStats> {
    let file = BufReader::new(File::open(path)?);
    let pb = ProgressBar::new_spinner();
    
    let mut total = 0;
    let mut entities = 0;
    let mut edges = 0;
    let mut errors = 0;
    
    let mut batch = Vec::with_capacity(batch_size);
    
    for line in file.lines() {
        let line = line?;
        let episode: EpisodeInput = serde_json::from_str(&line)?;
        batch.push(episode);
        
        if batch.len() >= batch_size {
            let results = graph.add_episodes_batch(&batch, parallel).await?;
            for r in &results {
                total += 1;
                entities += r.entities_extracted;
                edges += r.edges_created;
                if r.error.is_some() { errors += 1; }
            }
            pb.set_message(format!("Processed {} episodes...", total));
            batch.clear();
        }
    }
    
    // Process remaining
    if !batch.is_empty() {
        let results = graph.add_episodes_batch(&batch, parallel).await?;
        // ...
    }
    
    Ok(IngestStats { total, entities, edges, errors })
}
```

#### v0.8 Definition of Done
- [ ] `ctxgraph ingest --file` processes JSONL
- [ ] `ctxgraph ingest --csv` processes CSV with column mapping
- [ ] `cat file | ctxgraph ingest --stdin` works
- [ ] Parallel extraction for batch processing
- [ ] Progress bar shows status
- [ ] Error handling: skip bad lines, report count
- [ ] Benchmark: 100 episodes/second on CPU (Tier 1)
- [ ] 8+ new tests

---

## v0.9 — Schemas + Export

**Goal:** Community-contributed schemas and export to other formats.

**What ships:**
- Built-in schemas: default, developer, support, finance
- `ctxgraph init --schema developer`
- `ctxgraph export --format json`
- `ctxgraph export --format csv`
- `ctxgraph export --format neo4j-cypher` (for migration to Graphiti)
- `ctxgraph schema validate` — check custom schema

### Implementation Details

#### Built-in Schemas

```toml
# schemas/developer.toml
[schema]
name = "developer"
description = "Software architecture and engineering decisions"

[schema.entities]
Person = "A developer, engineer, or team member"
Component = "A software library, framework, tool, or technology"
Service = "A microservice, application, or system"
Decision = "An explicit architecture or design choice"
Reason = "The justification behind a technical decision"
Alternative = "A technology or approach considered but not chosen"
Pattern = "A design pattern or architectural pattern"
Constraint = "A technical limitation, requirement, or SLA"
Bug = "A bug, vulnerability, or incident"
Metric = "A performance metric, benchmark, or KPI"

[schema.relations]
chose = { description = "selected a component or approach" }
rejected = { description = "decided against an alternative" }
depends_on = { description = "requires or relies on" }
caused = { description = "led to or caused" }
fixed = { description = "resolved or patched" }
replaced = { description = "superseded or migrated from" }
benchmarked_at = { description = "measured performance of" }
introduced = { description = "added or created" }
deprecated = { description = "marked for removal" }
```

```toml
# schemas/support.toml
[schema]
name = "support"
description = "Customer support decisions and escalations"

[schema.entities]
Customer = "A customer or account"
Agent = "A support agent or representative"
Manager = "A manager or supervisor who approves exceptions"
Ticket = "A support ticket or case"
Policy = "A support policy or guideline"
Product = "A product or service offering"
Amount = "A monetary value, refund, or credit"
Issue = "A problem, complaint, or defect"

[schema.relations]
filed = { description = "submitted or reported" }
assigned_to = { description = "routed or assigned to" }
escalated_to = { description = "elevated to higher authority" }
approved = { description = "authorized or signed off on" }
denied = { description = "rejected or declined" }
referenced = { description = "cited or pointed to" }
resolved_with = { description = "fixed or settled using" }
waived = { description = "exception granted for" }
```

#### Neo4j Cypher Export

```rust
// ctxgraph-cli/src/commands/export.rs

pub fn export_neo4j_cypher(graph: &Graph, writer: &mut impl Write) -> Result<()> {
    // Export entities as nodes
    let entities = graph.storage.list_entities(None)?;
    for entity in &entities {
        writeln!(writer, 
            "CREATE (n:{} {{id: '{}', name: '{}', created_at: '{}'}})",
            entity.entity_type, entity.id, 
            entity.name.replace("'", "\\'"),
            entity.created_at.to_rfc3339()
        )?;
    }
    
    // Export edges as relationships
    let edges = graph.storage.list_all_edges()?;
    for edge in &edges {
        writeln!(writer,
            "MATCH (a {{id: '{}'}}), (b {{id: '{}'}}) \
             CREATE (a)-[:{}  {{fact: '{}', valid_from: '{}', confidence: {}}}]->(b)",
            edge.source_id, edge.target_id,
            edge.relation.to_uppercase(),
            edge.fact.as_deref().unwrap_or("").replace("'", "\\'"),
            edge.valid_from.map(|t| t.to_rfc3339()).unwrap_or_default(),
            edge.confidence,
        )?;
    }
    
    Ok(())
}
```

#### v0.9 Definition of Done
- [ ] 4 built-in schemas ship with binary
- [ ] `ctxgraph init --schema developer` works
- [ ] `ctxgraph export --format json` exports full graph
- [ ] `ctxgraph export --format csv` exports entities + edges
- [ ] `ctxgraph export --format neo4j-cypher` generates valid Cypher
- [ ] `ctxgraph schema validate` checks custom TOML schemas
- [ ] 8+ new tests

---

## v1.0 — Production Ready

**Goal:** Ship it. Stable API, comprehensive docs, benchmarks, crates.io publication.

**What ships:**
- API stability guarantee (no breaking changes until v2.0)
- Comprehensive documentation (README, guide, API docs)
- Performance benchmarks published
- All crates published to crates.io
- GitHub Actions CI/CD
- Logo, website (ctxgraph.dev or similar)

### Deliverables

#### Benchmarks

```
Benchmark: 1,000 semi-structured episodes (avg 50 words each)
Hardware: M1 MacBook Air (baseline)

Ingestion (Tier 1 only):
  Total time:     12.3 seconds
  Per episode:    12.3ms
  Entities/sec:   ~300
  Edges/sec:      ~200

Ingestion (Tier 1 + Tier 2):
  Total time:     18.7 seconds
  Per episode:    18.7ms

Search (FTS5 only):
  p50 latency:    2ms
  p99 latency:    8ms

Search (FTS5 + semantic + graph):
  p50 latency:    15ms
  p99 latency:    45ms

Graph traversal (3-hop, 10K nodes):
  p50 latency:    5ms
  p99 latency:    20ms

Database size:
  1K episodes:    ~2MB
  10K episodes:   ~18MB
  100K episodes:  ~180MB

Model sizes:
  GLiNER2 (INT8):     198MB
  GLiREL:             147MB
  MiniLM embeddings:  80MB
  Total:              425MB (one-time download)
```

#### Documentation

```
docs/
├── README.md                # Quick start, install, basic usage
├── ARCHITECTURE.md          # Technical deep dive
├── GUIDE.md                 # Step-by-step tutorial
├── API.md                   # Rust API reference
├── MCP.md                   # MCP server setup guide
├── SCHEMAS.md               # Schema authoring guide
├── BENCHMARKS.md            # Performance numbers
├── MIGRATION.md             # Migrating from/to Graphiti
├── FAQ.md                   # Common questions
└── CHANGELOG.md             # Version history
```

#### crates.io Publication

```
ctxgraph-core     — types, storage, query (no ML deps)
ctxgraph-extract  — extraction pipeline (depends on ort)
ctxgraph-embed    — local embeddings (depends on ort)
ctxgraph-mcp      — MCP server
ctxgraph-sdk      — meta-crate re-exporting everything
ctxgraph-cli      — binary crate (cargo install ctxgraph)
```

#### CI/CD

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo test --workspace
      - run: cargo clippy --workspace -- -D warnings
      - run: cargo fmt --check
  
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo bench
      # Upload results to GitHub Pages
  
  release:
    if: startsWith(github.ref, 'refs/tags/')
    needs: test
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo build --release
      # Upload binaries to GitHub Release
```

#### v1.0 Definition of Done
- [ ] All previous version DoDs met
- [ ] 80+ total tests passing
- [ ] Zero clippy warnings
- [ ] All pub items documented with /// doc comments
- [ ] README with install, quick start, and examples
- [ ] ARCHITECTURE.md explaining the tier system
- [ ] Benchmarks published
- [ ] All 6 crates published to crates.io
- [ ] GitHub Release with pre-built binaries (Linux, macOS, Windows)
- [ ] CI/CD green on all platforms
- [ ] Blog post: "Introducing ctxgraph: SQLite for Context Graphs"

---

## Post-v1.0 Ideas

- **v1.1:** `ctxgraph watch --git` (auto-capture from git — the DevTrace foundation)
- **v1.2:** Team sync via SQLite replication (Litestream or custom)
- **v1.3:** CloakPipe integration (privacy-safe context graphs)
- **v1.4:** Web dashboard for graph visualization
- **v1.5:** Plugin system for custom extractors