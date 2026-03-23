use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// An episode is the fundamental unit of information.
/// It represents "something happened" — a decision, conversation, event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    pub id: String,
    pub content: String,
    pub source: Option<String>,
    pub recorded_at: DateTime<Utc>,
    pub metadata: Option<serde_json::Value>,
}

/// Builder for constructing episodes with a fluent API.
pub struct EpisodeBuilder {
    content: String,
    source: Option<String>,
    metadata: serde_json::Map<String, serde_json::Value>,
    tags: Vec<String>,
}

impl EpisodeBuilder {
    pub fn source(mut self, s: &str) -> Self {
        self.source = Some(s.to_string());
        self
    }

    pub fn tag(mut self, t: &str) -> Self {
        self.tags.push(t.to_string());
        self
    }

    pub fn meta(mut self, key: &str, val: impl Into<serde_json::Value>) -> Self {
        self.metadata.insert(key.to_string(), val.into());
        self
    }

    pub fn build(mut self) -> Episode {
        if !self.tags.is_empty() {
            let tags: Vec<serde_json::Value> = self
                .tags
                .into_iter()
                .map(serde_json::Value::String)
                .collect();
            self.metadata
                .insert("tags".to_string(), serde_json::Value::Array(tags));
        }

        let metadata = if self.metadata.is_empty() {
            None
        } else {
            Some(serde_json::Value::Object(self.metadata))
        };

        Episode {
            id: uuid::Uuid::now_v7().to_string(),
            content: self.content,
            source: self.source,
            recorded_at: Utc::now(),
            metadata,
        }
    }
}

impl Episode {
    pub fn builder(content: &str) -> EpisodeBuilder {
        EpisodeBuilder {
            content: content.to_string(),
            source: None,
            metadata: serde_json::Map::new(),
            tags: Vec::new(),
        }
    }
}

/// An entity is a thing mentioned in episodes — people, components, decisions, etc.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub id: String,
    pub name: String,
    pub entity_type: String,
    pub summary: Option<String>,
    pub created_at: DateTime<Utc>,
    pub metadata: Option<serde_json::Value>,
}

impl Entity {
    pub fn new(name: &str, entity_type: &str) -> Self {
        Self {
            id: uuid::Uuid::now_v7().to_string(),
            name: name.to_string(),
            entity_type: entity_type.to_string(),
            summary: None,
            created_at: Utc::now(),
            metadata: None,
        }
    }
}

/// An edge is a relationship between two entities.
/// Edges are bi-temporal: valid_from/valid_until (real-world) + recorded_at (system).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub id: String,
    pub source_id: String,
    pub target_id: String,
    pub relation: String,
    pub fact: Option<String>,
    pub valid_from: Option<DateTime<Utc>>,
    pub valid_until: Option<DateTime<Utc>>,
    pub recorded_at: DateTime<Utc>,
    pub confidence: f64,
    pub episode_id: Option<String>,
    pub metadata: Option<serde_json::Value>,
}

impl Edge {
    pub fn new(source_id: &str, target_id: &str, relation: &str) -> Self {
        Self {
            id: uuid::Uuid::now_v7().to_string(),
            source_id: source_id.to_string(),
            target_id: target_id.to_string(),
            relation: relation.to_string(),
            fact: None,
            valid_from: None,
            valid_until: None,
            recorded_at: Utc::now(),
            confidence: 1.0,
            episode_id: None,
            metadata: None,
        }
    }

    /// Check if this edge is currently valid (not invalidated).
    pub fn is_current(&self) -> bool {
        self.valid_until.is_none()
    }

    /// Check if this edge was valid at a specific point in time.
    pub fn is_valid_at(&self, at: DateTime<Utc>) -> bool {
        let after_start = self.valid_from.is_none_or(|vf| vf <= at);
        let before_end = self.valid_until.is_none_or(|vu| vu > at);
        after_start && before_end
    }
}

/// Result from adding an episode to the graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeResult {
    pub episode_id: String,
    pub entities_extracted: usize,
    pub edges_created: usize,
}

/// Unified search result combining episodes, entities, and edges.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub episodes: Vec<Episode>,
    pub entities: Vec<Entity>,
    pub edges: Vec<Edge>,
    pub score: f64,
}

/// Per-episode result from fused (RRF) search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedEpisodeResult {
    pub episode: Episode,
    pub score: f64,
}

/// Graph-wide statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStats {
    pub episode_count: usize,
    pub entity_count: usize,
    pub edge_count: usize,
    pub sources: Vec<(String, usize)>,
    pub db_size_bytes: u64,
}

/// Context around an entity — its immediate neighbors and edges.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityContext {
    pub entity: Entity,
    pub edges: Vec<Edge>,
    pub neighbors: Vec<Entity>,
}

/// Options for search filtering.
#[derive(Debug, Clone, Default)]
pub struct SearchFilter {
    pub after: Option<DateTime<Utc>>,
    pub before: Option<DateTime<Utc>>,
    pub source: Option<String>,
    pub entity_type: Option<String>,
    pub limit: Option<usize>,
}
