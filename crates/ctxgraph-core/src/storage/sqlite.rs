use std::path::Path;

use chrono::{DateTime, Utc};
use rusqlite::{Connection, params};

use crate::error::{CtxGraphError, Result};
use crate::storage::migrations::run_migrations;
use crate::types::*;

pub struct Storage {
    conn: Connection,
}

impl Storage {
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path)?;
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;
             PRAGMA foreign_keys = ON;",
        )?;
        run_migrations(&conn)?;
        Ok(Self { conn })
    }

    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch("PRAGMA foreign_keys = ON;")?;
        run_migrations(&conn)?;
        Ok(Self { conn })
    }

    // ── Episodes ──

    pub fn insert_episode(&self, episode: &Episode) -> Result<()> {
        self.conn.execute(
            "INSERT INTO episodes (id, content, source, recorded_at, metadata)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                episode.id,
                episode.content,
                episode.source,
                episode.recorded_at.to_rfc3339(),
                episode.metadata.as_ref().map(|m| m.to_string()),
            ],
        )?;
        Ok(())
    }

    pub fn get_episode(&self, id: &str) -> Result<Option<Episode>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, content, source, recorded_at, metadata FROM episodes WHERE id = ?1",
        )?;

        let result = stmt
            .query_row(params![id], |row| {
                Ok(Episode {
                    id: row.get(0)?,
                    content: row.get(1)?,
                    source: row.get(2)?,
                    recorded_at: parse_datetime(&row.get::<_, String>(3)?),
                    metadata: row
                        .get::<_, Option<String>>(4)?
                        .and_then(|s| serde_json::from_str(&s).ok()),
                })
            })
            .optional()?;

        Ok(result)
    }

    pub fn list_episodes(&self, limit: usize, offset: usize) -> Result<Vec<Episode>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, content, source, recorded_at, metadata
             FROM episodes ORDER BY recorded_at DESC LIMIT ?1 OFFSET ?2",
        )?;

        let episodes = stmt
            .query_map(params![limit as i64, offset as i64], |row| {
                Ok(Episode {
                    id: row.get(0)?,
                    content: row.get(1)?,
                    source: row.get(2)?,
                    recorded_at: parse_datetime(&row.get::<_, String>(3)?),
                    metadata: row
                        .get::<_, Option<String>>(4)?
                        .and_then(|s| serde_json::from_str(&s).ok()),
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(episodes)
    }

    // ── Entities ──

    pub fn insert_entity(&self, entity: &Entity) -> Result<()> {
        self.conn.execute(
            "INSERT INTO entities (id, name, entity_type, summary, created_at, metadata)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                entity.id,
                entity.name,
                entity.entity_type,
                entity.summary,
                entity.created_at.to_rfc3339(),
                entity.metadata.as_ref().map(|m| m.to_string()),
            ],
        )?;
        Ok(())
    }

    pub fn get_entity(&self, id: &str) -> Result<Option<Entity>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, entity_type, summary, created_at, metadata
             FROM entities WHERE id = ?1",
        )?;

        let result = stmt
            .query_row(params![id], |row| {
                Ok(Entity {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    entity_type: row.get(2)?,
                    summary: row.get(3)?,
                    created_at: parse_datetime(&row.get::<_, String>(4)?),
                    metadata: row
                        .get::<_, Option<String>>(5)?
                        .and_then(|s| serde_json::from_str(&s).ok()),
                })
            })
            .optional()?;

        Ok(result)
    }

    pub fn get_entity_by_name(&self, name: &str) -> Result<Option<Entity>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, entity_type, summary, created_at, metadata
             FROM entities WHERE name = ?1",
        )?;

        let result = stmt
            .query_row(params![name], |row| {
                Ok(Entity {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    entity_type: row.get(2)?,
                    summary: row.get(3)?,
                    created_at: parse_datetime(&row.get::<_, String>(4)?),
                    metadata: row
                        .get::<_, Option<String>>(5)?
                        .and_then(|s| serde_json::from_str(&s).ok()),
                })
            })
            .optional()?;

        Ok(result)
    }

    pub fn list_entities(
        &self,
        entity_type: Option<&str>,
        limit: usize,
    ) -> Result<Vec<Entity>> {
        let (sql, type_param);
        if let Some(et) = entity_type {
            sql = "SELECT id, name, entity_type, summary, created_at, metadata
                   FROM entities WHERE entity_type = ?1 ORDER BY created_at DESC LIMIT ?2";
            type_param = Some(et.to_string());
        } else {
            sql = "SELECT id, name, entity_type, summary, created_at, metadata
                   FROM entities ORDER BY created_at DESC LIMIT ?2";
            type_param = None;
        }

        let mut stmt = self.conn.prepare(sql)?;

        let rows = if let Some(ref tp) = type_param {
            stmt.query_map(params![tp, limit as i64], map_entity_row)?
        } else {
            // For the no-filter case, use a placeholder for ?1 that matches nothing
            // Actually, we need different SQL. Let's handle this properly.
            drop(stmt);
            let mut stmt2 = self.conn.prepare(
                "SELECT id, name, entity_type, summary, created_at, metadata
                 FROM entities ORDER BY created_at DESC LIMIT ?1",
            )?;
            let entities = stmt2
                .query_map(params![limit as i64], map_entity_row)?
                .collect::<std::result::Result<Vec<_>, _>>()?;
            return Ok(entities);
        };

        let entities = rows.collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(entities)
    }

    // ── Edges ──

    pub fn insert_edge(&self, edge: &Edge) -> Result<()> {
        self.conn.execute(
            "INSERT INTO edges (id, source_id, target_id, relation, fact,
             valid_from, valid_until, recorded_at, confidence, episode_id, metadata)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                edge.id,
                edge.source_id,
                edge.target_id,
                edge.relation,
                edge.fact,
                edge.valid_from.map(|d| d.to_rfc3339()),
                edge.valid_until.map(|d| d.to_rfc3339()),
                edge.recorded_at.to_rfc3339(),
                edge.confidence,
                edge.episode_id,
                edge.metadata.as_ref().map(|m| m.to_string()),
            ],
        )?;
        Ok(())
    }

    pub fn get_edge(&self, id: &str) -> Result<Option<Edge>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, source_id, target_id, relation, fact,
                    valid_from, valid_until, recorded_at, confidence, episode_id, metadata
             FROM edges WHERE id = ?1",
        )?;

        let result = stmt
            .query_row(params![id], map_edge_row)
            .optional()?;

        Ok(result)
    }

    pub fn get_edges_for_entity(&self, entity_id: &str) -> Result<Vec<Edge>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, source_id, target_id, relation, fact,
                    valid_from, valid_until, recorded_at, confidence, episode_id, metadata
             FROM edges WHERE source_id = ?1 OR target_id = ?1
             ORDER BY recorded_at DESC",
        )?;

        let edges = stmt
            .query_map(params![entity_id], map_edge_row)?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(edges)
    }

    pub fn get_current_edges_for_entity(&self, entity_id: &str) -> Result<Vec<Edge>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, source_id, target_id, relation, fact,
                    valid_from, valid_until, recorded_at, confidence, episode_id, metadata
             FROM edges
             WHERE (source_id = ?1 OR target_id = ?1) AND valid_until IS NULL
             ORDER BY recorded_at DESC",
        )?;

        let edges = stmt
            .query_map(params![entity_id], map_edge_row)?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(edges)
    }

    pub fn invalidate_edge(&self, edge_id: &str, until: DateTime<Utc>) -> Result<()> {
        let changed = self.conn.execute(
            "UPDATE edges SET valid_until = ?1 WHERE id = ?2 AND valid_until IS NULL",
            params![until.to_rfc3339(), edge_id],
        )?;

        if changed == 0 {
            return Err(CtxGraphError::NotFound(format!(
                "edge {edge_id} not found or already invalidated"
            )));
        }

        Ok(())
    }

    // ── Episode-Entity links ──

    pub fn link_episode_entity(
        &self,
        episode_id: &str,
        entity_id: &str,
        span_start: Option<usize>,
        span_end: Option<usize>,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT OR IGNORE INTO episode_entities (episode_id, entity_id, span_start, span_end)
             VALUES (?1, ?2, ?3, ?4)",
            params![
                episode_id,
                entity_id,
                span_start.map(|s| s as i64),
                span_end.map(|s| s as i64),
            ],
        )?;
        Ok(())
    }

    // ── FTS5 Search ──

    pub fn search_episodes(&self, query: &str, limit: usize) -> Result<Vec<(Episode, f64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT e.id, e.content, e.source, e.recorded_at, e.metadata,
                    rank
             FROM episodes_fts fts
             JOIN episodes e ON e.rowid = fts.rowid
             WHERE episodes_fts MATCH ?1
             ORDER BY rank
             LIMIT ?2",
        )?;

        let results = stmt
            .query_map(params![query, limit as i64], |row| {
                let episode = Episode {
                    id: row.get(0)?,
                    content: row.get(1)?,
                    source: row.get(2)?,
                    recorded_at: parse_datetime(&row.get::<_, String>(3)?),
                    metadata: row
                        .get::<_, Option<String>>(4)?
                        .and_then(|s| serde_json::from_str(&s).ok()),
                };
                let rank: f64 = row.get(5)?;
                Ok((episode, -rank)) // FTS5 rank is negative (lower = better)
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(results)
    }

    pub fn search_entities(&self, query: &str, limit: usize) -> Result<Vec<(Entity, f64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT e.id, e.name, e.entity_type, e.summary, e.created_at, e.metadata,
                    rank
             FROM entities_fts fts
             JOIN entities e ON e.rowid = fts.rowid
             WHERE entities_fts MATCH ?1
             ORDER BY rank
             LIMIT ?2",
        )?;

        let results = stmt
            .query_map(params![query, limit as i64], |row| {
                let entity = Entity {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    entity_type: row.get(2)?,
                    summary: row.get(3)?,
                    created_at: parse_datetime(&row.get::<_, String>(4)?),
                    metadata: row
                        .get::<_, Option<String>>(5)?
                        .and_then(|s| serde_json::from_str(&s).ok()),
                };
                let rank: f64 = row.get(6)?;
                Ok((entity, -rank))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(results)
    }

    // ── Stats ──

    pub fn stats(&self) -> Result<GraphStats> {
        let episode_count: usize = self
            .conn
            .query_row("SELECT COUNT(*) FROM episodes", [], |row| row.get(0))?;
        let entity_count: usize = self
            .conn
            .query_row("SELECT COUNT(*) FROM entities", [], |row| row.get(0))?;
        let edge_count: usize = self
            .conn
            .query_row("SELECT COUNT(*) FROM edges", [], |row| row.get(0))?;

        let mut stmt = self.conn.prepare(
            "SELECT COALESCE(source, 'unknown'), COUNT(*)
             FROM episodes GROUP BY source ORDER BY COUNT(*) DESC",
        )?;
        let sources = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, usize>(1)?))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        let db_size_bytes: u64 = self.conn.query_row(
            "SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()",
            [],
            |row| row.get(0),
        )?;

        Ok(GraphStats {
            episode_count,
            entity_count,
            edge_count,
            sources,
            db_size_bytes,
        })
    }

    // ── Graph Traversal ──

    pub fn traverse(
        &self,
        start_entity_id: &str,
        max_depth: usize,
        current_only: bool,
    ) -> Result<(Vec<Entity>, Vec<Edge>)> {
        let valid_clause = if current_only {
            "AND e.valid_until IS NULL"
        } else {
            ""
        };

        let sql = format!(
            r#"
            WITH RECURSIVE traversal(entity_id, depth) AS (
                SELECT ?1, 0

                UNION

                SELECT
                    CASE WHEN e.source_id = t.entity_id THEN e.target_id
                         ELSE e.source_id END,
                    t.depth + 1
                FROM traversal t
                JOIN edges e ON (e.source_id = t.entity_id OR e.target_id = t.entity_id)
                WHERE t.depth < ?2
                  {valid_clause}
            )
            SELECT DISTINCT ent.id, ent.name, ent.entity_type, ent.summary,
                            ent.created_at, ent.metadata, t.depth
            FROM traversal t
            JOIN entities ent ON ent.id = t.entity_id
            ORDER BY t.depth
            "#
        );

        let mut stmt = self.conn.prepare(&sql)?;
        let entities = stmt
            .query_map(params![start_entity_id, max_depth as i64], |row| {
                Ok(Entity {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    entity_type: row.get(2)?,
                    summary: row.get(3)?,
                    created_at: parse_datetime(&row.get::<_, String>(4)?),
                    metadata: row
                        .get::<_, Option<String>>(5)?
                        .and_then(|s| serde_json::from_str(&s).ok()),
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        // Collect all edges between traversed entities
        let entity_ids: Vec<String> = entities.iter().map(|e| e.id.clone()).collect();
        let edges = self.get_edges_between(&entity_ids, current_only)?;

        Ok((entities, edges))
    }

    fn get_edges_between(&self, entity_ids: &[String], current_only: bool) -> Result<Vec<Edge>> {
        if entity_ids.is_empty() {
            return Ok(Vec::new());
        }

        let placeholders: Vec<String> = (1..=entity_ids.len()).map(|i| format!("?{i}")).collect();
        let in_clause = placeholders.join(", ");
        let valid_clause = if current_only {
            "AND valid_until IS NULL"
        } else {
            ""
        };

        let sql = format!(
            "SELECT id, source_id, target_id, relation, fact,
                    valid_from, valid_until, recorded_at, confidence, episode_id, metadata
             FROM edges
             WHERE source_id IN ({in_clause}) AND target_id IN ({in_clause})
             {valid_clause}
             ORDER BY recorded_at DESC"
        );

        let mut stmt = self.conn.prepare(&sql)?;

        // Bind entity_ids twice (once for source_id IN, once for target_id IN)
        let mut all_params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
        for id in entity_ids {
            all_params.push(Box::new(id.clone()));
        }
        for id in entity_ids {
            all_params.push(Box::new(id.clone()));
        }

        let param_refs: Vec<&dyn rusqlite::types::ToSql> = all_params.iter().map(|p| p.as_ref()).collect();

        let edges = stmt
            .query_map(&*param_refs, map_edge_row)?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(edges)
    }
}

// ── Helper functions ──

fn parse_datetime(s: &str) -> DateTime<Utc> {
    DateTime::parse_from_rfc3339(s)
        .map(|dt| dt.with_timezone(&Utc))
        .unwrap_or_else(|_| Utc::now())
}

fn map_entity_row(row: &rusqlite::Row) -> rusqlite::Result<Entity> {
    Ok(Entity {
        id: row.get(0)?,
        name: row.get(1)?,
        entity_type: row.get(2)?,
        summary: row.get(3)?,
        created_at: parse_datetime(&row.get::<_, String>(4)?),
        metadata: row
            .get::<_, Option<String>>(5)?
            .and_then(|s| serde_json::from_str(&s).ok()),
    })
}

fn map_edge_row(row: &rusqlite::Row) -> rusqlite::Result<Edge> {
    Ok(Edge {
        id: row.get(0)?,
        source_id: row.get(1)?,
        target_id: row.get(2)?,
        relation: row.get(3)?,
        fact: row.get(4)?,
        valid_from: row
            .get::<_, Option<String>>(5)?
            .map(|s| parse_datetime(&s)),
        valid_until: row
            .get::<_, Option<String>>(6)?
            .map(|s| parse_datetime(&s)),
        recorded_at: parse_datetime(&row.get::<_, String>(7)?),
        confidence: row.get(8)?,
        episode_id: row.get(9)?,
        metadata: row
            .get::<_, Option<String>>(10)?
            .and_then(|s| serde_json::from_str(&s).ok()),
    })
}

/// rusqlite optional helper
trait OptionalExt<T> {
    fn optional(self) -> std::result::Result<Option<T>, rusqlite::Error>;
}

impl<T> OptionalExt<T> for std::result::Result<T, rusqlite::Error> {
    fn optional(self) -> std::result::Result<Option<T>, rusqlite::Error> {
        match self {
            Ok(val) => Ok(Some(val)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e),
        }
    }
}
