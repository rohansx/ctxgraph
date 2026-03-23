use chrono::Utc;
use ctxgraph::*;

fn test_graph() -> Graph {
    Graph::in_memory().expect("failed to create in-memory graph")
}

// ── Episode CRUD ──

#[test]
fn test_episode_insert_and_retrieve() {
    let graph = test_graph();
    let episode = Episode::builder("Chose Postgres over SQLite for billing").build();
    let id = episode.id.clone();

    let result = graph.add_episode(episode).unwrap();
    assert_eq!(result.episode_id, id);

    let retrieved = graph.get_episode(&id).unwrap().unwrap();
    assert_eq!(retrieved.content, "Chose Postgres over SQLite for billing");
}

#[test]
fn test_episode_with_source_and_tags() {
    let graph = test_graph();
    let episode = Episode::builder("Priya approved the discount")
        .source("slack")
        .tag("finance")
        .tag("approval")
        .build();
    let id = episode.id.clone();

    graph.add_episode(episode).unwrap();

    let retrieved = graph.get_episode(&id).unwrap().unwrap();
    assert_eq!(retrieved.source.as_deref(), Some("slack"));
    assert!(retrieved.metadata.is_some());

    let meta = retrieved.metadata.unwrap();
    let tags = meta.get("tags").unwrap().as_array().unwrap();
    assert_eq!(tags.len(), 2);
    assert_eq!(tags[0].as_str().unwrap(), "finance");
}

#[test]
fn test_episode_with_metadata() {
    let graph = test_graph();
    let episode = Episode::builder("Budget approved for Q3")
        .meta("author", "rohan")
        .meta("confidence", serde_json::json!(0.95))
        .build();
    let id = episode.id.clone();

    graph.add_episode(episode).unwrap();

    let retrieved = graph.get_episode(&id).unwrap().unwrap();
    let meta = retrieved.metadata.unwrap();
    assert_eq!(meta.get("author").unwrap().as_str().unwrap(), "rohan");
}

#[test]
fn test_list_episodes() {
    let graph = test_graph();

    for i in 0..5 {
        let ep = Episode::builder(&format!("Decision {i}")).build();
        graph.add_episode(ep).unwrap();
    }

    let episodes = graph.list_episodes(3, 0).unwrap();
    assert_eq!(episodes.len(), 3);

    let all = graph.list_episodes(100, 0).unwrap();
    assert_eq!(all.len(), 5);

    let offset = graph.list_episodes(100, 3).unwrap();
    assert_eq!(offset.len(), 2);
}

#[test]
fn test_episode_not_found() {
    let graph = test_graph();
    let result = graph.get_episode("nonexistent-id").unwrap();
    assert!(result.is_none());
}

// ── Entity CRUD ──

#[test]
fn test_entity_insert_and_retrieve() {
    let graph = test_graph();
    let entity = Entity::new("Postgres", "Component");
    let id = entity.id.clone();

    graph.add_entity(entity).unwrap();

    let retrieved = graph.get_entity(&id).unwrap().unwrap();
    assert_eq!(retrieved.name, "Postgres");
    assert_eq!(retrieved.entity_type, "Component");
}

#[test]
fn test_entity_by_name() {
    let graph = test_graph();
    let entity = Entity::new("Priya Sharma", "Person");
    graph.add_entity(entity).unwrap();

    let found = graph.get_entity_by_name("Priya Sharma").unwrap().unwrap();
    assert_eq!(found.entity_type, "Person");

    let not_found = graph.get_entity_by_name("Nonexistent").unwrap();
    assert!(not_found.is_none());
}

#[test]
fn test_list_entities_with_type_filter() {
    let graph = test_graph();

    graph
        .add_entity(Entity::new("Postgres", "Component"))
        .unwrap();
    graph
        .add_entity(Entity::new("SQLite", "Component"))
        .unwrap();
    graph.add_entity(Entity::new("Priya", "Person")).unwrap();
    graph.add_entity(Entity::new("billing", "Service")).unwrap();

    let all = graph.list_entities(None, 100).unwrap();
    assert_eq!(all.len(), 4);

    let components = graph.list_entities(Some("Component"), 100).unwrap();
    assert_eq!(components.len(), 2);

    let people = graph.list_entities(Some("Person"), 100).unwrap();
    assert_eq!(people.len(), 1);
    assert_eq!(people[0].name, "Priya");
}

// ── Edge CRUD + Bi-temporal ──

#[test]
fn test_edge_insert_and_retrieve() {
    let graph = test_graph();

    let pg = Entity::new("Postgres", "Component");
    let billing = Entity::new("billing", "Service");
    let pg_id = pg.id.clone();
    let billing_id = billing.id.clone();
    graph.add_entity(pg).unwrap();
    graph.add_entity(billing).unwrap();

    let edge = Edge::new(&pg_id, &billing_id, "chosen_for");
    let edge_id = edge.id.clone();
    graph.add_edge(edge).unwrap();

    let edges = graph.get_edges_for_entity(&pg_id).unwrap();
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].id, edge_id);
    assert_eq!(edges[0].relation, "chosen_for");
}

#[test]
fn test_edge_is_current() {
    let edge = Edge::new("a", "b", "test");
    assert!(edge.is_current());
}

#[test]
fn test_edge_invalidation() {
    let graph = test_graph();

    let alice = Entity::new("Alice", "Person");
    let google = Entity::new("Google", "Organization");
    let alice_id = alice.id.clone();
    let google_id = google.id.clone();
    graph.add_entity(alice).unwrap();
    graph.add_entity(google).unwrap();

    let mut edge = Edge::new(&alice_id, &google_id, "works_at");
    edge.valid_from = Some(Utc::now());
    let edge_id = edge.id.clone();
    graph.add_edge(edge).unwrap();

    // Edge should be current
    let edges = graph.get_edges_for_entity(&alice_id).unwrap();
    assert_eq!(edges.len(), 1);
    assert!(edges[0].is_current());

    // Invalidate
    graph.invalidate_edge(&edge_id).unwrap();

    // Should still appear in all-edges query
    let all_edges = graph.get_edges_for_entity(&alice_id).unwrap();
    assert_eq!(all_edges.len(), 1);
    assert!(!all_edges[0].is_current());
}

#[test]
fn test_edge_valid_at() {
    let mut edge = Edge::new("a", "b", "test");
    let now = Utc::now();
    edge.valid_from = Some(now - chrono::Duration::days(30));
    edge.valid_until = Some(now - chrono::Duration::days(10));

    // 20 days ago: should be valid
    assert!(edge.is_valid_at(now - chrono::Duration::days(20)));

    // 5 days ago: should not be valid (after valid_until)
    assert!(!edge.is_valid_at(now - chrono::Duration::days(5)));

    // 40 days ago: should not be valid (before valid_from)
    assert!(!edge.is_valid_at(now - chrono::Duration::days(40)));
}

#[test]
fn test_invalidate_nonexistent_edge() {
    let graph = test_graph();
    let result = graph.invalidate_edge("nonexistent");
    assert!(result.is_err());
}

// ── Episode-Entity Links ──

#[test]
fn test_episode_entity_link() {
    let graph = test_graph();

    let episode = Episode::builder("Chose Postgres for billing").build();
    let ep_id = episode.id.clone();
    graph.add_episode(episode).unwrap();

    let entity = Entity::new("Postgres", "Component");
    let ent_id = entity.id.clone();
    graph.add_entity(entity).unwrap();

    graph
        .link_episode_entity(&ep_id, &ent_id, Some(6), Some(14))
        .unwrap();

    // Link should be idempotent (INSERT OR IGNORE)
    graph
        .link_episode_entity(&ep_id, &ent_id, Some(6), Some(14))
        .unwrap();
}

// ── FTS5 Search ──

#[test]
fn test_fts5_search_episodes() {
    let graph = test_graph();

    graph
        .add_episode(Episode::builder("Chose Postgres over SQLite for billing").build())
        .unwrap();
    graph
        .add_episode(Episode::builder("Switched from REST to gRPC for internal services").build())
        .unwrap();
    graph
        .add_episode(Episode::builder("Priya approved the discount for Reliance").build())
        .unwrap();

    let results = graph.search("Postgres", 10).unwrap();
    assert_eq!(results.len(), 1);
    assert!(results[0].0.content.contains("Postgres"));

    let results = graph.search("billing OR discount", 10).unwrap();
    assert_eq!(results.len(), 2);
}

#[test]
fn test_fts5_search_empty_results() {
    let graph = test_graph();
    graph
        .add_episode(Episode::builder("Chose Postgres").build())
        .unwrap();

    let results = graph.search("nonexistent_term_xyz", 10).unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_fts5_search_entities() {
    let graph = test_graph();

    graph
        .add_entity(Entity::new("Postgres", "Component"))
        .unwrap();
    graph
        .add_entity(Entity::new("SQLite", "Component"))
        .unwrap();
    graph.add_entity(Entity::new("Priya", "Person")).unwrap();

    let results = graph.search_entities("Postgres", 10).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0.name, "Postgres");

    let results = graph.search_entities("Component", 10).unwrap();
    assert_eq!(results.len(), 2);
}

// ── Entity Context ──

#[test]
fn test_entity_context() {
    let graph = test_graph();

    let pg = Entity::new("Postgres", "Component");
    let billing = Entity::new("billing", "Service");
    let rohan = Entity::new("rohan", "Person");
    let pg_id = pg.id.clone();
    let billing_id = billing.id.clone();
    let rohan_id = rohan.id.clone();

    graph.add_entity(pg).unwrap();
    graph.add_entity(billing).unwrap();
    graph.add_entity(rohan).unwrap();

    graph
        .add_edge(Edge::new(&pg_id, &billing_id, "chosen_for"))
        .unwrap();
    graph
        .add_edge(Edge::new(&rohan_id, &pg_id, "chose"))
        .unwrap();

    let context = graph.get_entity_context(&pg_id).unwrap();
    assert_eq!(context.entity.name, "Postgres");
    assert_eq!(context.edges.len(), 2);
    assert_eq!(context.neighbors.len(), 2);
}

// ── Stats ──

#[test]
fn test_stats() {
    let graph = test_graph();

    graph
        .add_episode(Episode::builder("Decision 1").source("manual").build())
        .unwrap();
    graph
        .add_episode(Episode::builder("Decision 2").source("manual").build())
        .unwrap();
    graph
        .add_episode(Episode::builder("Slack message").source("slack").build())
        .unwrap();

    let pg = Entity::new("Postgres", "Component");
    let pg_id = pg.id.clone();
    graph.add_entity(pg).unwrap();
    let billing = Entity::new("billing", "Service");
    let billing_id = billing.id.clone();
    graph.add_entity(billing).unwrap();

    graph
        .add_edge(Edge::new(&pg_id, &billing_id, "chosen_for"))
        .unwrap();

    let stats = graph.stats().unwrap();
    assert_eq!(stats.episode_count, 3);
    assert_eq!(stats.entity_count, 2);
    assert_eq!(stats.edge_count, 1);
    assert_eq!(stats.sources.len(), 2);
}

// ── Graph Init ──

#[test]
fn test_graph_init_and_open() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();

    // Init should succeed
    let _graph = Graph::init(dir).unwrap();

    // Init again should fail (already exists)
    let result = Graph::init(dir);
    assert!(result.is_err());

    // Open should succeed
    let db_path = dir.join(".ctxgraph").join("graph.db");
    let _graph = Graph::open(&db_path).unwrap();
}

#[test]
fn test_graph_open_nonexistent() {
    let result = Graph::open(std::path::Path::new("/tmp/nonexistent/graph.db"));
    assert!(result.is_err());
}

// ── Embedding Storage ──

#[test]
fn test_store_and_retrieve_embedding() {
    let graph = test_graph();
    let episode = Episode::builder("Embedding test episode").build();
    let ep_id = episode.id.clone();
    graph.add_episode(episode).unwrap();

    // Store a fake 384-dim embedding
    let embedding: Vec<f32> = (0..384).map(|i| i as f32 / 384.0).collect();
    graph.store_embedding(&ep_id, &embedding).unwrap();

    // Retrieve all embeddings — should include ours
    let all = graph.get_embeddings().unwrap();
    assert_eq!(all.len(), 1);
    assert_eq!(all[0].0, ep_id);
    assert_eq!(all[0].1.len(), 384);
    // Check round-trip fidelity for a few values
    for (i, &v) in all[0].1.iter().enumerate() {
        let expected = i as f32 / 384.0;
        assert!(
            (v - expected).abs() < 1e-6,
            "mismatch at index {i}: {v} vs {expected}"
        );
    }
}

#[test]
fn test_get_embeddings_empty() {
    let graph = test_graph();
    let embeddings = graph.get_embeddings().unwrap();
    assert!(embeddings.is_empty());
}

#[test]
fn test_search_fused_no_embeddings() {
    let graph = test_graph();

    graph
        .add_episode(Episode::builder("Chose Postgres for billing").build())
        .unwrap();
    graph
        .add_episode(Episode::builder("Switched from REST to gRPC").build())
        .unwrap();

    // Fused search with a dummy query embedding — FTS5 results only
    let query_embedding = vec![0.0f32; 384];
    let results = graph
        .search_fused("Postgres", &query_embedding, 10)
        .unwrap();

    // Should still return FTS5 hits even with zero-magnitude query embedding
    assert!(!results.is_empty());
    assert!(results[0].episode.content.contains("Postgres"));
}

#[test]
fn test_search_fused_with_embeddings() {
    let graph = test_graph();

    let ep1 = Episode::builder("Chose Postgres for billing").build();
    let ep2 = Episode::builder("Switched from REST to gRPC").build();
    let id1 = ep1.id.clone();
    let id2 = ep2.id.clone();
    graph.add_episode(ep1).unwrap();
    graph.add_episode(ep2).unwrap();

    // Synthetic embeddings: ep1 in direction [1, 0, ...], ep2 in direction [0, 1, ...]
    let mut emb1 = vec![0.0f32; 384];
    emb1[0] = 1.0;
    let mut emb2 = vec![0.0f32; 384];
    emb2[1] = 1.0;

    graph.store_embedding(&id1, &emb1).unwrap();
    graph.store_embedding(&id2, &emb2).unwrap();

    // Query in direction of ep1
    let query_embedding = emb1.clone();
    let results = graph
        .search_fused("Postgres", &query_embedding, 10)
        .unwrap();

    // ep1 should rank first (matches both FTS5 and semantic)
    assert!(!results.is_empty());
    assert_eq!(results[0].episode.id, id1);
}

// ── UUID v7 Ordering ──

#[test]
fn test_uuid_v7_is_time_sortable() {
    let id1 = uuid::Uuid::now_v7().to_string();
    std::thread::sleep(std::time::Duration::from_millis(2));
    let id2 = uuid::Uuid::now_v7().to_string();

    assert!(
        id1 < id2,
        "UUID v7 should be lexicographically time-sortable"
    );
}

// ── Migrations Idempotent ──

#[test]
fn test_migrations_idempotent() {
    let tmp = tempfile::tempdir().unwrap();
    let db_path = tmp.path().join("test.db");

    // Open twice — migrations should not fail on second open
    let _storage = ctxgraph::storage::Storage::open(&db_path).unwrap();
    drop(_storage);
    let _storage = ctxgraph::storage::Storage::open(&db_path).unwrap();
}

// ── Entity Deduplication ──

#[test]
fn test_entity_dedup_merges_similar() {
    let graph = test_graph();

    // Add "PostgreSQL" entity
    let pg = Entity::new("PostgreSQL", "Component");
    let (pg_id, merged) = graph.add_entity_deduped(pg, 0.85).unwrap();
    assert!(!merged, "First insert should not be merged");

    // Add "Postgres" entity with dedup threshold 0.85 — should merge
    let postgres = Entity::new("Postgres", "Component");
    let (deduped_id, was_merged) = graph.add_entity_deduped(postgres, 0.85).unwrap();
    assert!(was_merged, "Postgres should be merged into PostgreSQL");
    assert_eq!(
        deduped_id, pg_id,
        "Should return canonical PostgreSQL entity id"
    );

    // Only one entity should exist
    let all = graph.list_entities(Some("Component"), 100).unwrap();
    assert_eq!(
        all.len(),
        1,
        "Only one Component entity should exist after merge"
    );
    assert_eq!(all[0].name, "PostgreSQL");
}

#[test]
fn test_entity_dedup_preserves_different() {
    let graph = test_graph();

    let pg = Entity::new("PostgreSQL", "Component");
    graph.add_entity_deduped(pg, 0.85).unwrap();

    // "Redis" has very low similarity to "PostgreSQL"
    let redis = Entity::new("Redis", "Component");
    let (_, was_merged) = graph.add_entity_deduped(redis, 0.85).unwrap();
    assert!(!was_merged, "Redis should not be merged with PostgreSQL");

    let all = graph.list_entities(Some("Component"), 100).unwrap();
    assert_eq!(
        all.len(),
        2,
        "Both PostgreSQL and Redis should exist as separate entities"
    );
}

#[test]
fn test_entity_dedup_alias_lookup() {
    let graph = test_graph();

    // Add canonical entity
    let pg = Entity::new("PostgreSQL", "Component");
    let (pg_id, _) = graph.add_entity_deduped(pg, 0.85).unwrap();

    // Add alias variant
    let postgres = Entity::new("Postgres", "Component");
    let (merged_id, was_merged) = graph.add_entity_deduped(postgres, 0.85).unwrap();
    assert!(was_merged);
    assert_eq!(merged_id, pg_id);

    // Adding "Postgres" again should hit alias table (exact alias match)
    let postgres2 = Entity::new("Postgres", "Component");
    let (alias_id, alias_merged) = graph.add_entity_deduped(postgres2, 0.85).unwrap();
    assert!(alias_merged, "Second 'Postgres' should hit alias table");
    assert_eq!(alias_id, pg_id, "Alias lookup should return canonical id");
}

// ── Empty Database ──

#[test]
fn test_empty_database_operations() {
    let graph = test_graph();

    // All operations should succeed on empty db
    assert!(graph.list_episodes(10, 0).unwrap().is_empty());
    assert!(graph.list_entities(None, 10).unwrap().is_empty());
    assert!(graph.search("anything", 10).unwrap().is_empty());

    let stats = graph.stats().unwrap();
    assert_eq!(stats.episode_count, 0);
    assert_eq!(stats.entity_count, 0);
    assert_eq!(stats.edge_count, 0);
}
