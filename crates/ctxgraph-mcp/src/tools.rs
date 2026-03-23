use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use ctxgraph::{Episode, Graph};
use ctxgraph_embed::EmbedEngine;
use serde_json::{Value, json};

pub struct ToolContext {
    pub graph: Arc<Mutex<Graph>>,
    pub embed: Option<Arc<EmbedEngine>>,
    /// In-memory embedding cache: episode_id → 384-dim vector.
    ///
    /// Populated lazily on the first `find_precedents` call, then kept warm.
    /// Invalidated (new entry appended) when `add_episode` stores a new embedding
    /// so subsequent searches never re-hit SQLite for already-loaded episodes.
    embedding_cache: Mutex<Option<HashMap<String, Vec<f32>>>>,
}

impl ToolContext {
    pub fn new(graph: Graph, embed: Option<EmbedEngine>) -> Self {
        Self {
            graph: Arc::new(Mutex::new(graph)),
            embed: embed.map(Arc::new),
            embedding_cache: Mutex::new(None),
        }
    }

    /// Populate the in-memory embedding cache from SQLite if it hasn't been loaded yet.
    /// Subsequent calls return immediately — the Option acts as a once-flag.
    fn warm_cache(&self) -> Result<(), String> {
        let mut cache = self.embedding_cache.lock().map_err(|e| e.to_string())?;
        if cache.is_none() {
            let graph = self.graph.lock().map_err(|e| e.to_string())?;
            let rows = graph.get_embeddings().map_err(|e| e.to_string())?;
            *cache = Some(rows.into_iter().collect());
        }
        Ok(())
    }

    /// Tool: add_episode
    /// Adds a new episode to the graph, computes and stores its embedding.
    pub async fn add_episode(&self, args: Value) -> Result<Value, String> {
        let text = args["text"]
            .as_str()
            .ok_or("missing required field: text")?
            .to_string();

        let source = args["source"].as_str().map(|s| s.to_string());

        let mut builder = Episode::builder(&text);
        if let Some(ref src) = source {
            builder = builder.source(src);
        }
        if let Some(tags) = args["tags"].as_array() {
            for tag in tags {
                if let Some(t) = tag.as_str() {
                    builder = builder.tag(t);
                }
            }
        }
        let episode = builder.build();
        let episode_id = episode.id.clone();

        // Store episode
        let result = {
            let graph = self.graph.lock().map_err(|e| e.to_string())?;
            graph.add_episode(episode).map_err(|e| e.to_string())?
        };

        // Compute embedding and persist to SQLite (skipped if embed engine is unavailable)
        if let Some(ref embed) = self.embed {
            let embedding = embed.embed(&text).map_err(|e| e.to_string())?;
            {
                let graph = self.graph.lock().map_err(|e| e.to_string())?;
                graph
                    .store_embedding(&episode_id, &embedding)
                    .map_err(|e| e.to_string())?;
            }

            // Insert into in-memory cache so find_precedents sees it immediately
            // without another SQLite round-trip.
            if let Ok(mut cache) = self.embedding_cache.lock()
                && let Some(ref mut map) = *cache
            {
                map.insert(episode_id.clone(), embedding);
                // If cache is None (never warmed), leave it — it will be populated
                // from SQLite (including this episode) on the first find_precedents call.
            }
        }

        Ok(json!({
            "episode_id": result.episode_id,
            "entities_found": result.entities_extracted,
            "edges_created": result.edges_created,
        }))
    }

    /// Tool: search
    /// Fused FTS5 + semantic search via RRF. Falls back to FTS5-only if embed is unavailable.
    pub async fn search(&self, args: Value) -> Result<Value, String> {
        let query = args["query"]
            .as_str()
            .ok_or("missing required field: query")?
            .to_string();
        let limit = args["limit"].as_u64().unwrap_or(10) as usize;

        let items: Vec<Value> = if let Some(ref embed) = self.embed {
            // Fused FTS5 + semantic search via RRF
            let query_embedding = embed.embed(&query).map_err(|e| e.to_string())?;
            let results = {
                let graph = self.graph.lock().map_err(|e| e.to_string())?;
                graph
                    .search_fused(&query, &query_embedding, limit)
                    .map_err(|e| e.to_string())?
            };
            results
                .into_iter()
                .map(|r| {
                    json!({
                        "id": r.episode.id,
                        "content": r.episode.content,
                        "score": r.score,
                        "source": r.episode.source,
                        "recorded_at": r.episode.recorded_at.to_rfc3339(),
                    })
                })
                .collect()
        } else {
            // FTS5-only search (embedding not available)
            let results = {
                let graph = self.graph.lock().map_err(|e| e.to_string())?;
                graph.search(&query, limit).map_err(|e| e.to_string())?
            };
            results
                .into_iter()
                .map(|(episode, score)| {
                    json!({
                        "id": episode.id,
                        "content": episode.content,
                        "score": score,
                        "source": episode.source,
                        "recorded_at": episode.recorded_at.to_rfc3339(),
                    })
                })
                .collect()
        };

        Ok(json!(items))
    }

    /// Tool: get_decision
    /// Retrieve a specific episode by ID with full context.
    pub async fn get_decision(&self, args: Value) -> Result<Value, String> {
        let id = args["id"]
            .as_str()
            .ok_or("missing required field: id")?
            .to_string();

        let graph = self.graph.lock().map_err(|e| e.to_string())?;

        let episode = graph
            .get_episode(&id)
            .map_err(|e| e.to_string())?
            .ok_or_else(|| format!("episode not found: {id}"))?;

        // Serialize episode to JSON
        let episode_json = serde_json::to_value(&episode).map_err(|e| e.to_string())?;

        Ok(json!({
            "episode": episode_json,
        }))
    }

    /// Tool: traverse
    /// Traverse the knowledge graph from an entity.
    pub async fn traverse(&self, args: Value) -> Result<Value, String> {
        let entity_name = args["entity_name"]
            .as_str()
            .ok_or("missing required field: entity_name")?
            .to_string();
        let max_depth = (args["max_depth"].as_u64().unwrap_or(2) as usize).min(5);

        let graph = self.graph.lock().map_err(|e| e.to_string())?;

        let entity = graph
            .get_entity_by_name(&entity_name)
            .map_err(|e| e.to_string())?
            .ok_or_else(|| format!("entity not found: {entity_name}"))?;

        let (entities, edges) = graph
            .traverse(&entity.id, max_depth)
            .map_err(|e| e.to_string())?;

        let entities_json: Vec<Value> = entities
            .into_iter()
            .map(|e| serde_json::to_value(e).unwrap_or(Value::Null))
            .collect();

        let edges_json: Vec<Value> = edges
            .into_iter()
            .map(|e| serde_json::to_value(e).unwrap_or(Value::Null))
            .collect();

        Ok(json!({
            "entities": entities_json,
            "edges": edges_json,
        }))
    }

    /// Tool: find_precedents
    /// Find past episodes most semantically similar to a given context.
    ///
    /// Uses an in-memory embedding cache so the full SQLite embedding table is
    /// only read once per process lifetime. Subsequent calls score entirely in
    /// RAM — O(n) dot-products, no I/O.
    pub async fn find_precedents(&self, args: Value) -> Result<Value, String> {
        let context = args["context"]
            .as_str()
            .ok_or("missing required field: context")?
            .to_string();
        let limit = args["limit"].as_u64().unwrap_or(5) as usize;

        let embed = match self.embed {
            Some(ref e) => e,
            None => {
                return Ok(
                    json!({"error": "embedding not available, start with CTXGRAPH_NO_EMBED unset"}),
                );
            }
        };

        // Embed the query (always ~20-50ms CPU inference, unavoidable)
        let context_embedding = embed.embed(&context).map_err(|e| e.to_string())?;

        // Ensure the cache is warm (no-op after first call)
        self.warm_cache()?;

        // Score entirely in RAM — no SQLite I/O
        let mut scored: Vec<(String, f32)> = {
            let cache = self.embedding_cache.lock().map_err(|e| e.to_string())?;
            cache
                .as_ref()
                .expect("cache must be Some after warm_cache")
                .iter()
                .map(|(id, vec)| {
                    let sim = EmbedEngine::cosine_similarity(&context_embedding, vec);
                    (id.clone(), sim)
                })
                .collect()
        };

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top: Vec<(String, f32)> = scored.into_iter().take(limit).collect();

        let mut results = Vec::new();
        let graph = self.graph.lock().map_err(|e| e.to_string())?;
        for (id, sim) in top {
            if let Ok(Some(ep)) = graph.get_episode(&id) {
                results.push(json!({
                    "id": ep.id,
                    "content": ep.content,
                    "source": ep.source,
                    "recorded_at": ep.recorded_at.to_rfc3339(),
                    "similarity": sim,
                }));
            }
        }

        Ok(json!(results))
    }

    /// Tool: traverse_batch
    /// Traverse multiple entities in one call, returning a merged result.
    ///
    /// Replaces N sequential `traverse` calls with a single round-trip.
    /// Entities not found in the graph are silently skipped (same behaviour as
    /// individual `traverse` returning an error for unknown entities).
    pub async fn traverse_batch(&self, args: Value) -> Result<Value, String> {
        let entity_names: Vec<String> = args["entity_names"]
            .as_array()
            .ok_or("missing required field: entity_names")?
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect();

        if entity_names.is_empty() {
            return Ok(json!({"entities": [], "edges": []}));
        }

        let max_depth = (args["max_depth"].as_u64().unwrap_or(2) as usize).min(5);

        let graph = self.graph.lock().map_err(|e| e.to_string())?;

        let mut all_entities: Vec<Value> = Vec::new();
        let mut all_edges: Vec<Value> = Vec::new();
        // Dedup by ID across multiple traversals
        let mut seen_entities = std::collections::HashSet::new();
        let mut seen_edges = std::collections::HashSet::new();

        for name in &entity_names {
            let entity = match graph.get_entity_by_name(name) {
                Ok(Some(e)) => e,
                Ok(None) | Err(_) => continue, // skip unknown entities
            };

            let (entities, edges) = match graph.traverse(&entity.id, max_depth) {
                Ok(r) => r,
                Err(_) => continue,
            };

            for e in entities {
                if seen_entities.insert(e.id.clone()) {
                    all_entities.push(serde_json::to_value(e).unwrap_or(Value::Null));
                }
            }
            for e in edges {
                if seen_edges.insert(e.id.clone()) {
                    all_edges.push(serde_json::to_value(e).unwrap_or(Value::Null));
                }
            }
        }

        Ok(json!({
            "entities": all_entities,
            "edges": all_edges,
        }))
    }
}

/// Wrap a tool result into an MCP content array.
pub fn tool_result(result: Result<Value, String>) -> Value {
    let text = match result {
        Ok(val) => serde_json::to_string_pretty(&val).unwrap_or_else(|_| val.to_string()),
        Err(msg) => serde_json::to_string_pretty(&json!({"error": msg}))
            .unwrap_or_else(|_| format!(r#"{{"error": "{msg}"}}"#)),
    };
    json!({
        "content": [{"type": "text", "text": text}]
    })
}

/// Returns the static tools list payload.
pub fn tools_list() -> Value {
    json!({
        "tools": [
            {
                "name": "add_episode",
                "description": "Add a new episode (decision, observation, or event) to the context graph. Extracts entities and relations automatically.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "The episode content"},
                        "source": {"type": "string", "description": "Source label (e.g. 'slack', 'meeting', 'code-review')"},
                        "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional tags"}
                    },
                    "required": ["text"]
                }
            },
            {
                "name": "search",
                "description": "Search the context graph using semantic + keyword fusion (RRF). Returns ranked episodes.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "limit": {"type": "integer", "description": "Max results (default 10)"},
                        "source": {"type": "string", "description": "Filter by source"}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "get_decision",
                "description": "Retrieve a specific episode by ID with full entity and relation context.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "Episode UUID"}
                    },
                    "required": ["id"]
                }
            },
            {
                "name": "traverse",
                "description": "Traverse the knowledge graph from an entity, returning connected entities and edges.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "entity_name": {"type": "string", "description": "Entity name to start from"},
                        "max_depth": {"type": "integer", "description": "Traversal depth (default 2, max 5)"}
                    },
                    "required": ["entity_name"]
                }
            },
            {
                "name": "traverse_batch",
                "description": "Traverse multiple entities in one call. Equivalent to N sequential traverse calls but with a single round-trip. Deduplicates entities and edges across all traversals.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "entity_names": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of entity names to traverse from"
                        },
                        "max_depth": {"type": "integer", "description": "Traversal depth per entity (default 2, max 5)"}
                    },
                    "required": ["entity_names"]
                }
            },
            {
                "name": "find_precedents",
                "description": "Find past decisions or context most semantically similar to a given situation.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "context": {"type": "string", "description": "Description of current situation"},
                        "limit": {"type": "integer", "description": "Max results (default 5)"}
                    },
                    "required": ["context"]
                }
            }
        ]
    })
}
