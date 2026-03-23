use rusqlite::Connection;

use crate::error::Result;

const MIGRATIONS: &[(&str, &str)] = &[
    (
        "001_initial",
        r#"
    -- Episodes: raw events
    CREATE TABLE IF NOT EXISTS episodes (
        id          TEXT PRIMARY KEY,
        content     TEXT NOT NULL,
        source      TEXT,
        recorded_at TEXT NOT NULL,
        metadata    TEXT,
        embedding   BLOB
    );

    -- Entities: extracted nodes
    CREATE TABLE IF NOT EXISTS entities (
        id          TEXT PRIMARY KEY,
        name        TEXT NOT NULL,
        entity_type TEXT NOT NULL,
        summary     TEXT,
        created_at  TEXT NOT NULL,
        metadata    TEXT
    );

    -- Edges: relationships between entities
    CREATE TABLE IF NOT EXISTS edges (
        id          TEXT PRIMARY KEY,
        source_id   TEXT NOT NULL REFERENCES entities(id),
        target_id   TEXT NOT NULL REFERENCES entities(id),
        relation    TEXT NOT NULL,
        fact        TEXT,
        valid_from  TEXT,
        valid_until TEXT,
        recorded_at TEXT NOT NULL,
        confidence  REAL DEFAULT 1.0,
        episode_id  TEXT REFERENCES episodes(id),
        metadata    TEXT
    );

    -- Episode-Entity junction table
    CREATE TABLE IF NOT EXISTS episode_entities (
        episode_id  TEXT REFERENCES episodes(id),
        entity_id   TEXT REFERENCES entities(id),
        span_start  INTEGER,
        span_end    INTEGER,
        PRIMARY KEY (episode_id, entity_id)
    );

    -- Entity aliases for deduplication
    CREATE TABLE IF NOT EXISTS aliases (
        canonical_id TEXT REFERENCES entities(id),
        alias_name   TEXT NOT NULL,
        similarity   REAL,
        UNIQUE(canonical_id, alias_name)
    );

    -- Community clusters
    CREATE TABLE IF NOT EXISTS communities (
        id          TEXT PRIMARY KEY,
        summary     TEXT,
        entity_ids  TEXT,
        created_at  TEXT NOT NULL,
        updated_at  TEXT
    );

    -- FTS5 indexes
    CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
        content, source, metadata,
        content=episodes, content_rowid=rowid
    );

    CREATE VIRTUAL TABLE IF NOT EXISTS entities_fts USING fts5(
        name, entity_type, summary,
        content=entities, content_rowid=rowid
    );

    CREATE VIRTUAL TABLE IF NOT EXISTS edges_fts USING fts5(
        fact, relation,
        content=edges, content_rowid=rowid
    );

    -- Performance indexes
    CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
    CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
    CREATE INDEX IF NOT EXISTS idx_edges_relation ON edges(relation);
    CREATE INDEX IF NOT EXISTS idx_edges_valid ON edges(valid_from, valid_until);
    CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
    CREATE INDEX IF NOT EXISTS idx_episode_entities ON episode_entities(entity_id);
    CREATE INDEX IF NOT EXISTS idx_episodes_source ON episodes(source);
    CREATE INDEX IF NOT EXISTS idx_episodes_recorded ON episodes(recorded_at);

    -- FTS5 triggers: keep indexes in sync
    CREATE TRIGGER IF NOT EXISTS episodes_ai AFTER INSERT ON episodes BEGIN
        INSERT INTO episodes_fts(rowid, content, source, metadata)
        VALUES (new.rowid, new.content, new.source, new.metadata);
    END;

    CREATE TRIGGER IF NOT EXISTS episodes_ad AFTER DELETE ON episodes BEGIN
        INSERT INTO episodes_fts(episodes_fts, rowid, content, source, metadata)
        VALUES ('delete', old.rowid, old.content, old.source, old.metadata);
    END;

    CREATE TRIGGER IF NOT EXISTS episodes_au AFTER UPDATE ON episodes BEGIN
        INSERT INTO episodes_fts(episodes_fts, rowid, content, source, metadata)
        VALUES ('delete', old.rowid, old.content, old.source, old.metadata);
        INSERT INTO episodes_fts(rowid, content, source, metadata)
        VALUES (new.rowid, new.content, new.source, new.metadata);
    END;

    CREATE TRIGGER IF NOT EXISTS entities_ai AFTER INSERT ON entities BEGIN
        INSERT INTO entities_fts(rowid, name, entity_type, summary)
        VALUES (new.rowid, new.name, new.entity_type, new.summary);
    END;

    CREATE TRIGGER IF NOT EXISTS entities_ad AFTER DELETE ON entities BEGIN
        INSERT INTO entities_fts(entities_fts, rowid, name, entity_type, summary)
        VALUES ('delete', old.rowid, old.name, old.entity_type, old.summary);
    END;

    CREATE TRIGGER IF NOT EXISTS entities_au AFTER UPDATE ON entities BEGIN
        INSERT INTO entities_fts(entities_fts, rowid, name, entity_type, summary)
        VALUES ('delete', old.rowid, old.name, old.entity_type, old.summary);
        INSERT INTO entities_fts(rowid, name, entity_type, summary)
        VALUES (new.rowid, new.name, new.entity_type, new.summary);
    END;

    CREATE TRIGGER IF NOT EXISTS edges_ai AFTER INSERT ON edges BEGIN
        INSERT INTO edges_fts(rowid, fact, relation)
        VALUES (new.rowid, new.fact, new.relation);
    END;

    CREATE TRIGGER IF NOT EXISTS edges_ad AFTER DELETE ON edges BEGIN
        INSERT INTO edges_fts(edges_fts, rowid, fact, relation)
        VALUES ('delete', old.rowid, old.fact, old.relation);
    END;

    CREATE TRIGGER IF NOT EXISTS edges_au AFTER UPDATE ON edges BEGIN
        INSERT INTO edges_fts(edges_fts, rowid, fact, relation)
        VALUES ('delete', old.rowid, old.fact, old.relation);
        INSERT INTO edges_fts(rowid, fact, relation)
        VALUES (new.rowid, new.fact, new.relation);
    END;
    "#,
    ),
    (
        "002_entity_embeddings",
        r#"
    -- Add embedding column to entities table (episodes already has it from 001)
    -- We use a Rust-side check since SQLite ALTER TABLE ADD COLUMN is not idempotent
    "#,
    ),
];

pub fn run_migrations(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS _migrations (
            version TEXT PRIMARY KEY,
            applied_at TEXT NOT NULL
        );",
    )?;

    for (version, sql) in MIGRATIONS {
        let already_applied: bool = conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM _migrations WHERE version = ?1)",
            [version],
            |row| row.get(0),
        )?;

        if !already_applied {
            // Migration 002: add embedding column to entities if not present
            if *version == "002_entity_embeddings" {
                let has_col: bool = {
                    let mut col_stmt = conn.prepare(
                        "SELECT COUNT(*) FROM pragma_table_info('entities') WHERE name = 'embedding'",
                    )?;
                    col_stmt
                        .query_row([], |row| row.get::<_, i64>(0))
                        .map(|n| n > 0)?
                };
                if !has_col {
                    conn.execute_batch("ALTER TABLE entities ADD COLUMN embedding BLOB;")?;
                }
            } else {
                conn.execute_batch(sql)?;
            }
            conn.execute(
                "INSERT INTO _migrations (version, applied_at) VALUES (?1, ?2)",
                rusqlite::params![version, chrono::Utc::now().to_rfc3339()],
            )?;
        }
    }

    Ok(())
}
