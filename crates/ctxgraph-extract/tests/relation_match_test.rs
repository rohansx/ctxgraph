//! Piece 3 — Rust integration test for `RelationMatcher`.
//!
//! Mirrors the Python prototype `scripts/proto_relation_match.py` so we know
//! the Rust + Python implementations produce comparable results on the same
//! schema descriptions and verb test set.
//!
//! Run with: `cargo test --test relation_match_test -- --nocapture`

use std::path::PathBuf;

use ctxgraph_extract::relation_match::RelationMatcher;
use ctxgraph_extract::schema::ExtractionSchema;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn load_universal_schema() -> ExtractionSchema {
    let path = workspace_root().join("schemas").join("universal.toml");
    ExtractionSchema::load_universal(&path)
        .unwrap_or_else(|e| panic!("failed to load universal.toml: {e}"))
}

#[test]
fn schema_parses_with_9_entities_and_10_relations() {
    let schema = load_universal_schema();
    assert_eq!(schema.name, "universal");
    assert_eq!(schema.entity_types.len(), 9, "expected 9 entity types");
    assert_eq!(
        schema.relation_types.len(),
        10,
        "expected 10 relation types"
    );

    // Spot-check a few known names
    for required in [
        "Person",
        "Place",
        "Organization",
        "Concept",
        "Artifact",
        "Event",
        "Time",
        "Idea",
        "Fact",
    ] {
        assert!(
            schema.entity_types.contains_key(required),
            "entity type {required} missing",
        );
    }
    for required in [
        "mentions",
        "located_at",
        "related_to",
        "caused",
        "preceded",
        "references",
        "owned_by",
        "part_of",
        "depends_on",
        "participated_in",
    ] {
        assert!(
            schema.relation_types.contains_key(required),
            "relation type {required} missing",
        );
    }
}

#[test]
#[ignore = "downloads ~80MB all-MiniLM-L6-v2 on first run; run with --ignored locally"]
fn relation_matcher_resolves_verb_variations() {
    let schema = load_universal_schema();
    let matcher = RelationMatcher::build_from_schema(&schema).expect("relation matcher init");

    // Same test set as `scripts/proto_relation_match.py::_selftest`.
    // Python prototype achieves 90% (18/20) — Rust should land in the same band.
    let cases: &[(&str, &str)] = &[
        ("depends on", "depends_on"),
        ("relies on", "depends_on"),
        ("needs", "depends_on"),
        ("requires", "depends_on"),
        ("is part of", "part_of"),
        ("belongs to", "part_of"),
        ("is owned by", "owned_by"),
        ("controlled by", "owned_by"),
        ("located at", "located_at"),
        ("happens in", "located_at"),
        ("caused", "caused"),
        ("led to", "caused"),
        ("resulted in", "caused"),
        ("happened before", "preceded"),
        ("cites", "references"),
        ("builds on", "references"),
        ("links to", "references"),
        ("participated in", "participated_in"),
        ("attended", "participated_in"),
        ("mentioned", "mentions"),
    ];

    let mut correct = 0;
    let mut failures: Vec<String> = Vec::new();
    for (verb, expected) in cases {
        let m = matcher.resolve(verb).expect("resolve failed");
        let ok = m.relation == *expected;
        if ok {
            correct += 1;
        } else {
            failures.push(format!(
                "{verb:?} → {} (expected {expected}); runner_up={} ({:.3})",
                m.relation, m.runner_up, m.runner_up_score
            ));
        }
        println!(
            "  {} {:>22?} → {:<15} ({:.3}) runner_up={} ({:.3})",
            if ok { "ok" } else { "FAIL" },
            verb,
            m.relation,
            m.score,
            m.runner_up,
            m.runner_up_score
        );
    }
    let n = cases.len();
    let pct = correct as f32 / n as f32;
    println!();
    println!("  accuracy: {correct}/{n} = {:.1}%", pct * 100.0);

    // Threshold: at least 80% (parity with Python prototype).
    // Tighter assert because we want regressions to fail CI.
    assert!(
        pct >= 0.80,
        "relation matcher below 80% threshold ({:.1}%). Failures:\n  {}",
        pct * 100.0,
        failures.join("\n  ")
    );
}

#[test]
#[ignore = "downloads model; run with --ignored locally"]
fn resolve_returns_distinct_runner_up() {
    let schema = load_universal_schema();
    let matcher = RelationMatcher::build_from_schema(&schema).unwrap();
    let m = matcher.resolve("depends on").unwrap();
    assert_eq!(m.relation, "depends_on");
    assert_ne!(m.relation, m.runner_up);
    assert!(m.score >= m.runner_up_score);
}

#[test]
#[ignore = "downloads model"]
fn empty_input_errors() {
    let schema = load_universal_schema();
    let matcher = RelationMatcher::build_from_schema(&schema).unwrap();
    assert!(matcher.resolve("").is_err());
    assert!(matcher.resolve("   ").is_err());
}
