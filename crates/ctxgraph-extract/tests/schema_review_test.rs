//! Piece 5 Layer B integration tests.

use std::path::PathBuf;

use ctxgraph_extract::schema::ExtractionSchema;
use ctxgraph_extract::schema_review::{
    LoggedSuggestion, PromotionRules, ReviewReport, SuggestionPayload,
};

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn universal_schema() -> ExtractionSchema {
    let path = workspace_root().join("schemas").join("universal.toml");
    ExtractionSchema::load_universal(&path).unwrap()
}

#[test]
fn no_log_data_returns_empty_report() {
    let schema = universal_schema();
    let rules = PromotionRules::default();
    let report = ReviewReport::from_entries(vec![], &schema, rules).unwrap();
    assert_eq!(report.n_log_entries, 0);
    assert_eq!(report.candidates.len(), 0);
    assert_eq!(report.near_misses.len(), 0);
}

#[test]
fn suggestion_matching_existing_type_is_ignored() {
    let schema = universal_schema();
    let rules = PromotionRules::default();

    // 'Person' is already in the universal schema → must be skipped.
    let entries = vec![
        log_entry("entity_type", "Person", Some("journal"), Some(1.0)),
        log_entry("entity_type", "Person", Some("recipe"), Some(1.0)),
        log_entry("entity_type", "Person", Some("travel"), Some(1.0)),
        log_entry("entity_type", "Person", Some("meeting"), Some(1.0)),
        log_entry("entity_type", "Person", Some("book"), Some(1.0)),
    ];
    let report = ReviewReport::from_entries(entries, &schema, rules).unwrap();
    assert_eq!(
        report.candidates.len(),
        0,
        "existing types should be filtered"
    );
}

#[test]
fn promotes_a_clear_winner() {
    let schema = universal_schema();
    let rules = PromotionRules {
        min_episodes: 5,
        min_domains: 3,
        min_avg_confidence: 0.7,
        max_cos_to_existing: 0.85,
    };

    // 'Ingredient' appears in 6 episodes across 4 distinct domains
    // (recipe, restaurant, journal, blog) with high confidence.
    let entries = vec![
        log_entry("entity_type", "Ingredient", Some("recipe"), Some(0.9)),
        log_entry("entity_type", "Ingredient", Some("recipe"), Some(0.85)),
        log_entry("entity_type", "Ingredient", Some("restaurant"), Some(0.95)),
        log_entry("entity_type", "Ingredient", Some("journal"), Some(0.8)),
        log_entry("entity_type", "Ingredient", Some("blog"), Some(0.85)),
        log_entry("entity_type", "Ingredient", Some("blog"), Some(0.9)),
    ];
    let report = ReviewReport::from_entries(entries, &schema, rules).unwrap();
    assert_eq!(report.candidates.len(), 1);
    let c = &report.candidates[0];
    assert_eq!(c.name, "Ingredient");
    assert_eq!(c.n_episodes, 6);
    assert_eq!(c.n_domains, 4);
    assert!(c.avg_confidence > 0.85);
    assert!(c.eval.passes_all);
}

#[test]
fn fails_below_episode_threshold() {
    let schema = universal_schema();
    let rules = PromotionRules::default();

    // Only 3 episodes — below the default min_episodes=5
    let entries = vec![
        log_entry("entity_type", "Ingredient", Some("recipe"), Some(0.9)),
        log_entry("entity_type", "Ingredient", Some("restaurant"), Some(0.95)),
        log_entry("entity_type", "Ingredient", Some("blog"), Some(0.8)),
    ];
    let report = ReviewReport::from_entries(entries, &schema, rules).unwrap();
    assert_eq!(report.candidates.len(), 0);
    assert_eq!(report.near_misses.len(), 1);
    assert!(!report.near_misses[0].eval.episodes_ok);
}

#[test]
fn fails_below_domain_threshold() {
    let schema = universal_schema();
    let rules = PromotionRules::default();

    // 6 episodes but all in the same domain
    let entries = vec![
        log_entry("entity_type", "Ingredient", Some("recipe"), Some(0.9)),
        log_entry("entity_type", "Ingredient", Some("recipe"), Some(0.85)),
        log_entry("entity_type", "Ingredient", Some("recipe"), Some(0.95)),
        log_entry("entity_type", "Ingredient", Some("recipe"), Some(0.8)),
        log_entry("entity_type", "Ingredient", Some("recipe"), Some(0.85)),
        log_entry("entity_type", "Ingredient", Some("recipe"), Some(0.9)),
    ];
    let report = ReviewReport::from_entries(entries, &schema, rules).unwrap();
    assert_eq!(report.candidates.len(), 0);
    assert_eq!(report.near_misses.len(), 1);
    assert!(!report.near_misses[0].eval.domains_ok);
}

#[test]
fn filters_garbage_entity_id_suggestions() {
    // LLM occasionally mislabels "e2", "e3" etc. as suggestion names.
    // The filter should drop them silently.
    let schema = universal_schema();
    let entries = vec![
        log_entry("entity_type", "e2", Some("a"), Some(0.9)),
        log_entry("entity_type", "e10", Some("b"), Some(0.9)),
        log_entry("entity_type", "", Some("c"), Some(0.9)),
        // A real suggestion should survive
        log_entry("entity_type", "Ingredient", Some("recipe"), Some(0.9)),
    ];
    let report = ReviewReport::from_entries(entries, &schema, PromotionRules::default()).unwrap();
    let names: Vec<&str> = report
        .candidates
        .iter()
        .chain(report.near_misses.iter())
        .map(|c| c.name.as_str())
        .collect();
    assert!(
        !names
            .iter()
            .any(|n| n.starts_with('e') && n[1..].chars().all(|c| c.is_ascii_digit()))
    );
    assert!(!names.iter().any(|n| n.is_empty()));
}

#[test]
fn reads_log_with_missing_optional_fields() {
    // The Rust pipeline writes entries without `episode_domain` or
    // `extraction_confidence`. The Python harness writes them. Both
    // must parse.
    let schema = universal_schema();
    let json = r#"[
      {"suggestion": {"kind": "entity_type", "name": "Foo", "supporting_entity_ids": [], "rationale": ""}},
      {"suggestion": {"kind": "entity_type", "name": "Foo", "supporting_entity_ids": [], "rationale": ""}, "episode_domain": "x", "extraction_confidence": 0.9}
    ]"#;
    let entries: Vec<LoggedSuggestion> = serde_json::from_str(json).unwrap();
    assert_eq!(entries.len(), 2);
    let report = ReviewReport::from_entries(entries, &schema, PromotionRules::default()).unwrap();
    // 2 episodes total → not enough → goes to near-misses
    assert_eq!(report.candidates.len(), 0);
    assert_eq!(report.near_misses.len(), 1);
}

#[test]
#[ignore = "needs /tmp/schema_suggestions.json from running the Python smoke test"]
fn live_run_against_collected_suggestions() {
    let schema = universal_schema();
    let rules = PromotionRules::default();
    let log = PathBuf::from("/tmp/schema_suggestions.json");
    if !log.exists() {
        eprintln!("skipping — run scripts/test_5_pieces.py first");
        return;
    }
    let report = ReviewReport::from_log(&log, &schema, rules).expect("review");
    report.print();
    // Sanity: we should at least have parsed something
    assert!(report.n_log_entries > 0, "no entries in log");
}

// ── Helpers ────────────────────────────────────────────────────────

fn log_entry(
    kind: &str,
    name: &str,
    domain: Option<&str>,
    confidence: Option<f32>,
) -> LoggedSuggestion {
    LoggedSuggestion {
        suggestion: SuggestionPayload {
            kind: kind.to_string(),
            name: name.to_string(),
            supporting_entity_ids: vec![],
            rationale: format!("test rationale for {name}"),
        },
        episode_idx: None,
        episode_domain: domain.map(String::from),
        backend: None,
        model: None,
        extraction_confidence: confidence,
        logged_at: None,
    }
}
