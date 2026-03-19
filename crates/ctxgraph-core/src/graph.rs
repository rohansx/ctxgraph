use std::fs;
use std::path::{Path, PathBuf};

use crate::error::{CtxGraphError, Result};
use crate::storage::Storage;
use crate::types::*;

#[cfg(feature = "extract")]
use ctxgraph_extract::pipeline::ExtractionPipeline;
#[cfg(feature = "extract")]
use ctxgraph_extract::schema::ExtractionSchema;

pub struct Graph {
    storage: Storage,
    #[allow(dead_code)]
    db_path: PathBuf,
    #[cfg(feature = "extract")]
    pipeline: Option<ExtractionPipeline>,
}

impl Graph {
    /// Open an existing ctxgraph database.
    pub fn open(db_path: &Path) -> Result<Self> {
        if !db_path.exists() {
            return Err(CtxGraphError::NotFound(format!(
                "database not found at {}. Run `ctxgraph init` first.",
                db_path.display()
            )));
        }
        let storage = Storage::open(db_path)?;
        Ok(Self {
            storage,
            db_path: db_path.to_path_buf(),
            #[cfg(feature = "extract")]
            pipeline: None,
        })
    }

    /// Initialize a new ctxgraph project in the given directory.
    /// Creates `.ctxgraph/` directory with a fresh database.
    pub fn init(dir: &Path) -> Result<Self> {
        let ctxgraph_dir = dir.join(".ctxgraph");
        let db_path = ctxgraph_dir.join("graph.db");

        if db_path.exists() {
            return Err(CtxGraphError::AlreadyExists(format!(
                "ctxgraph already initialized at {}",
                ctxgraph_dir.display()
            )));
        }

        fs::create_dir_all(&ctxgraph_dir)?;

        let storage = Storage::open(&db_path)?;
        Ok(Self {
            storage,
            db_path,
            #[cfg(feature = "extract")]
            pipeline: None,
        })
    }

    /// Open in-memory database (for testing).
    pub fn in_memory() -> Result<Self> {
        let storage = Storage::open_in_memory()?;
        Ok(Self {
            storage,
            db_path: PathBuf::from(":memory:"),
            #[cfg(feature = "extract")]
            pipeline: None,
        })
    }

    /// Load the extraction pipeline from models in the given directory.
    ///
    /// Once loaded, `add_episode()` will automatically extract entities and relations.
    /// Call this after `open()` or `init()` to enable extraction.
    #[cfg(feature = "extract")]
    pub fn load_extraction_pipeline(
        &mut self,
        models_dir: &Path,
    ) -> Result<()> {
        let pipeline = ExtractionPipeline::with_defaults(models_dir)
            .map_err(|e| CtxGraphError::Extraction(e.to_string()))?;
        self.pipeline = Some(pipeline);
        Ok(())
    }

    /// Load the extraction pipeline with a custom schema.
    #[cfg(feature = "extract")]
    pub fn load_extraction_pipeline_with_schema(
        &mut self,
        models_dir: &Path,
        schema: ExtractionSchema,
        confidence_threshold: f64,
    ) -> Result<()> {
        let pipeline = ExtractionPipeline::new(schema, models_dir, confidence_threshold)
            .map_err(|e| CtxGraphError::Extraction(e.to_string()))?;
        self.pipeline = Some(pipeline);
        Ok(())
    }

    /// Check if the extraction pipeline is loaded.
    #[cfg(feature = "extract")]
    pub fn has_extraction_pipeline(&self) -> bool {
        self.pipeline.is_some()
    }

    // ── Core Operations ──

    /// Add an episode to the graph. Returns the episode ID and extraction results.
    ///
    /// If an extraction pipeline is loaded, entities and relations are automatically
    /// extracted from the episode content and stored in the graph.
    pub fn add_episode(&self, episode: Episode) -> Result<EpisodeResult> {
        self.storage.insert_episode(&episode)?;

        #[cfg(feature = "extract")]
        if let Some(ref pipeline) = self.pipeline {
            return self.add_episode_with_extraction(&episode, pipeline);
        }

        Ok(EpisodeResult {
            episode_id: episode.id,
            entities_extracted: 0,
            edges_created: 0,
        })
    }

    /// Internal: extract entities/relations and store them.
    #[cfg(feature = "extract")]
    fn add_episode_with_extraction(
        &self,
        episode: &Episode,
        pipeline: &ExtractionPipeline,
    ) -> Result<EpisodeResult> {
        let result = pipeline
            .extract(&episode.content, episode.recorded_at)
            .map_err(|e| CtxGraphError::Extraction(e.to_string()))?;

        let mut entities_extracted = 0;
        let mut edges_created = 0;

        // Map extracted entity text → entity ID for edge creation
        let mut entity_id_map: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();

        // Step 1: Create or reuse entities (with fuzzy dedup at 0.85 threshold)
        for extracted in &result.entities {
            let entity = Entity::new(&extracted.text, &extracted.entity_type);
            let entity_id = match self.add_entity_deduped(entity, 0.85)? {
                (id, false) => {
                    entities_extracted += 1;
                    id
                }
                (id, true) => id, // merged into existing entity
            };

            entity_id_map.insert(extracted.text.clone(), entity_id.clone());

            // Link episode ↔ entity
            let _ = self.storage.link_episode_entity(
                &episode.id,
                &entity_id,
                Some(extracted.span_start),
                Some(extracted.span_end),
            );
        }

        // Step 2: Create edges from relations
        for rel in &result.relations {
            let source_id = match entity_id_map.get(&rel.head) {
                Some(id) => id,
                None => continue, // head entity not found
            };
            let target_id = match entity_id_map.get(&rel.tail) {
                Some(id) => id,
                None => continue, // tail entity not found
            };

            let mut edge = Edge::new(source_id, target_id, &rel.relation);
            edge.confidence = rel.confidence;
            edge.episode_id = Some(episode.id.clone());
            edge.fact = Some(format!("{} {} {}", rel.head, rel.relation, rel.tail));

            self.storage.insert_edge(&edge)?;
            edges_created += 1;
        }

        Ok(EpisodeResult {
            episode_id: episode.id.clone(),
            entities_extracted,
            edges_created,
        })
    }

    /// Get an episode by ID.
    pub fn get_episode(&self, id: &str) -> Result<Option<Episode>> {
        self.storage.get_episode(id)
    }

    /// List episodes with pagination.
    pub fn list_episodes(&self, limit: usize, offset: usize) -> Result<Vec<Episode>> {
        self.storage.list_episodes(limit, offset)
    }

    /// Add an entity to the graph.
    pub fn add_entity(&self, entity: Entity) -> Result<()> {
        self.storage.insert_entity(&entity)
    }

    /// Add an entity with fuzzy deduplication against existing entities of the same type.
    ///
    /// If an existing entity with Jaro-Winkler similarity >= threshold exists,
    /// returns that entity's ID and stores the new name as an alias.
    /// Otherwise creates a new entity.
    ///
    /// Returns (entity_id, was_merged: bool).
    pub fn add_entity_deduped(
        &self,
        entity: Entity,
        threshold: f64,
    ) -> Result<(String, bool)> {
        // 1. Check alias table first (exact alias match)
        if let Some(canonical_id) = self.storage.find_by_alias(&entity.name)? {
            return Ok((canonical_id, true));
        }

        // 2. Get all existing entities of same type
        let existing = self.storage.get_entity_names_by_type(&entity.entity_type)?;

        // 3. Compute Jaro-Winkler similarity to each
        let name_lower = entity.name.to_lowercase();
        let mut best: Option<(String, f64)> = None;
        for (existing_id, existing_name) in &existing {
            let sim = strsim::jaro_winkler(&name_lower, &existing_name.to_lowercase());
            if sim >= threshold {
                if best.as_ref().map_or(true, |(_, best_sim)| sim > *best_sim) {
                    best = Some((existing_id.clone(), sim));
                }
            }
        }

        // 4. If match found: add alias and return existing id
        if let Some((canonical_id, sim)) = best {
            self.storage.add_alias(&canonical_id, &entity.name, sim)?;
            return Ok((canonical_id, true));
        }

        // 5. Otherwise: insert new entity
        let id = entity.id.clone();
        self.storage.insert_entity(&entity)?;
        Ok((id, false))
    }

    /// Check if any episode with source='git' has the given commit hash in metadata.
    pub fn has_episode_by_git_hash(&self, hash: &str) -> Result<bool> {
        self.storage.episode_exists_by_git_hash(hash)
    }

    /// Get an entity by ID.
    pub fn get_entity(&self, id: &str) -> Result<Option<Entity>> {
        self.storage.get_entity(id)
    }

    /// Get an entity by name.
    pub fn get_entity_by_name(&self, name: &str) -> Result<Option<Entity>> {
        self.storage.get_entity_by_name(name)
    }

    /// List entities, optionally filtered by type.
    pub fn list_entities(&self, entity_type: Option<&str>, limit: usize) -> Result<Vec<Entity>> {
        self.storage.list_entities(entity_type, limit)
    }

    /// Add an edge between two entities.
    pub fn add_edge(&self, edge: Edge) -> Result<()> {
        self.storage.insert_edge(&edge)
    }

    /// Get all edges for an entity (both as source and target).
    pub fn get_edges_for_entity(&self, entity_id: &str) -> Result<Vec<Edge>> {
        self.storage.get_edges_for_entity(entity_id)
    }

    /// Invalidate an edge (set valid_until to now).
    pub fn invalidate_edge(&self, edge_id: &str) -> Result<()> {
        self.storage.invalidate_edge(edge_id, chrono::Utc::now())
    }

    /// Link an episode to an entity.
    pub fn link_episode_entity(
        &self,
        episode_id: &str,
        entity_id: &str,
        span_start: Option<usize>,
        span_end: Option<usize>,
    ) -> Result<()> {
        self.storage
            .link_episode_entity(episode_id, entity_id, span_start, span_end)
    }

    // ── Embeddings ──

    /// Store an embedding for an episode. The embedding is serialized as
    /// little-endian f32 bytes.
    pub fn store_embedding(&self, episode_id: &str, embedding: &[f32]) -> Result<()> {
        let bytes: Vec<u8> = embedding
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        self.storage.store_episode_embedding(episode_id, &bytes)
    }

    /// Store an embedding for an entity.
    pub fn store_entity_embedding(&self, entity_id: &str, embedding: &[f32]) -> Result<()> {
        let bytes: Vec<u8> = embedding
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        self.storage.store_entity_embedding(entity_id, &bytes)
    }

    /// Load all episode embeddings as (episode_id, Vec<f32>) pairs.
    pub fn get_embeddings(&self) -> Result<Vec<(String, Vec<f32>)>> {
        let raw = self.storage.get_all_episode_embeddings()?;
        let result = raw
            .into_iter()
            .map(|(id, bytes)| {
                let floats: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                (id, floats)
            })
            .collect();
        Ok(result)
    }

    /// Fused search using Reciprocal Rank Fusion (RRF) over FTS5 + semantic results.
    ///
    /// `query_embedding` should be the pre-computed embedding for `query`.
    /// Returns episodes ranked by combined RRF score.
    pub fn search_fused(
        &self,
        query: &str,
        query_embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<FusedEpisodeResult>> {
        const K: f64 = 60.0;

        // Accumulate RRF scores per episode id
        let mut scores: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
        let mut episodes_map: std::collections::HashMap<String, Episode> =
            std::collections::HashMap::new();

        // --- FTS5 ranked list ---
        // Fetch a generous pool for RRF (up to 10x limit or 200)
        let fts_pool = (limit * 10).max(200);
        let fts_results = self.storage.search_episodes(query, fts_pool);
        if let Ok(fts) = fts_results {
            for (rank, (episode, _)) in fts.into_iter().enumerate() {
                let rrf = 1.0 / (K + rank as f64 + 1.0);
                *scores.entry(episode.id.clone()).or_insert(0.0) += rrf;
                episodes_map.insert(episode.id.clone(), episode);
            }
        }

        // --- Semantic (cosine similarity) ranked list ---
        let all_embeddings = self.get_embeddings()?;
        if !all_embeddings.is_empty() && !query_embedding.is_empty() {
            // Compute cosine similarities
            let mut semantic: Vec<(String, f32)> = all_embeddings
                .into_iter()
                .map(|(id, vec)| {
                    let sim = cosine_similarity(query_embedding, &vec);
                    (id, sim)
                })
                .collect();
            // Sort descending by similarity
            semantic.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            for (rank, (ep_id, _sim)) in semantic.into_iter().enumerate() {
                let rrf = 1.0 / (K + rank as f64 + 1.0);
                *scores.entry(ep_id.clone()).or_insert(0.0) += rrf;
                // Fetch episode if not already cached
                if !episodes_map.contains_key(&ep_id) {
                    if let Ok(Some(ep)) = self.storage.get_episode(&ep_id) {
                        episodes_map.insert(ep_id, ep);
                    }
                }
            }
        }

        // Sort by total RRF score descending
        let mut fused: Vec<(String, f64)> = scores.into_iter().collect();
        fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let results = fused
            .into_iter()
            .take(limit)
            .filter_map(|(id, score)| {
                episodes_map.remove(&id).map(|episode| FusedEpisodeResult { episode, score })
            })
            .collect();

        Ok(results)
    }

    // ── Search ──

    /// Search episodes via FTS5 full-text search.
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<(Episode, f64)>> {
        self.storage.search_episodes(query, limit)
    }

    /// Search entities via FTS5.
    pub fn search_entities(&self, query: &str, limit: usize) -> Result<Vec<(Entity, f64)>> {
        self.storage.search_entities(query, limit)
    }

    // ── Traversal ──

    /// Get context around an entity — its neighbors and connecting edges.
    pub fn get_entity_context(&self, entity_id: &str) -> Result<EntityContext> {
        let entity = self
            .storage
            .get_entity(entity_id)?
            .ok_or_else(|| CtxGraphError::NotFound(format!("entity {entity_id}")))?;

        let edges = self.storage.get_current_edges_for_entity(entity_id)?;

        // Collect neighbor IDs
        let mut neighbor_ids: Vec<String> = Vec::new();
        for edge in &edges {
            if edge.source_id == entity_id {
                neighbor_ids.push(edge.target_id.clone());
            } else {
                neighbor_ids.push(edge.source_id.clone());
            }
        }

        let mut neighbors = Vec::new();
        for nid in &neighbor_ids {
            if let Some(n) = self.storage.get_entity(nid)? {
                neighbors.push(n);
            }
        }

        Ok(EntityContext {
            entity,
            edges,
            neighbors,
        })
    }

    /// Multi-hop graph traversal from a starting entity.
    pub fn traverse(
        &self,
        start_entity_id: &str,
        max_depth: usize,
    ) -> Result<(Vec<Entity>, Vec<Edge>)> {
        self.storage.traverse(start_entity_id, max_depth, true)
    }

    // ── Stats ──

    /// Get graph-wide statistics.
    pub fn stats(&self) -> Result<GraphStats> {
        self.storage.stats()
    }
}

/// Compute cosine similarity between two f32 vectors.
/// Returns 0.0 if either vector has zero magnitude.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag_a == 0.0 || mag_b == 0.0 {
        0.0
    } else {
        dot / (mag_a * mag_b)
    }
}
