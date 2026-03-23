use serde::Deserialize;
use std::collections::HashSet;

#[derive(Debug, Deserialize)]
struct BenchmarkEpisode {
    text: String,
    expected_entities: Vec<ExpectedEntity>,
    expected_relations: Vec<ExpectedRelation>,
}

#[derive(Debug, Deserialize)]
struct ExpectedEntity {
    name: String,
    entity_type: String,
    span_start: usize,
    span_end: usize,
}

#[derive(Debug, Deserialize)]
struct ExpectedRelation {
    head: String,
    relation: String,
    tail: String,
}

/// Compute precision, recall, and F1 score given predicted and expected string sets.
///
/// Returns `(precision, recall, f1)`. If both sets are empty, returns `(1.0, 1.0, 1.0)`.
fn compute_f1(predicted: &[String], expected: &[String]) -> (f64, f64, f64) {
    if predicted.is_empty() && expected.is_empty() {
        return (1.0, 1.0, 1.0);
    }

    let predicted_set: HashSet<&String> = predicted.iter().collect();
    let expected_set: HashSet<&String> = expected.iter().collect();

    let true_positives = predicted_set.intersection(&expected_set).count() as f64;

    let precision = if predicted_set.is_empty() {
        0.0
    } else {
        true_positives / predicted_set.len() as f64
    };

    let recall = if expected_set.is_empty() {
        0.0
    } else {
        true_positives / expected_set.len() as f64
    };

    let f1 = if (precision + recall) == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };

    (precision, recall, f1)
}

fn load_episodes() -> Vec<BenchmarkEpisode> {
    let fixture_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/benchmark_episodes.json"
    );
    let data =
        std::fs::read_to_string(fixture_path).expect("Failed to read benchmark_episodes.json");
    serde_json::from_str(&data).expect("Failed to deserialize benchmark episodes")
}

#[test]
fn test_fixture_loads_and_has_50_episodes() {
    let episodes = load_episodes();
    assert_eq!(
        episodes.len(),
        50,
        "Expected exactly 50 benchmark episodes, got {}",
        episodes.len()
    );
}

#[test]
fn test_all_entity_types_covered() {
    let episodes = load_episodes();
    let required: HashSet<&str> = [
        "Person",
        "Component",
        "Service",
        "Language",
        "Database",
        "Infrastructure",
        "Decision",
        "Constraint",
        "Metric",
        "Pattern",
    ]
    .into_iter()
    .collect();

    let found: HashSet<&str> = episodes
        .iter()
        .flat_map(|ep| ep.expected_entities.iter())
        .map(|e| e.entity_type.as_str())
        .collect();

    let missing: HashSet<&&str> = required.iter().filter(|t| !found.contains(**t)).collect();
    assert!(
        missing.is_empty(),
        "Missing entity types in fixture: {:?}",
        missing
    );
}

#[test]
fn test_all_relation_types_covered() {
    let episodes = load_episodes();
    let required: HashSet<&str> = [
        "chose",
        "rejected",
        "replaced",
        "depends_on",
        "fixed",
        "introduced",
        "deprecated",
        "caused",
        "constrained_by",
    ]
    .into_iter()
    .collect();

    let found: HashSet<&str> = episodes
        .iter()
        .flat_map(|ep| ep.expected_relations.iter())
        .map(|r| r.relation.as_str())
        .collect();

    let missing: HashSet<&&str> = required.iter().filter(|t| !found.contains(**t)).collect();
    assert!(
        missing.is_empty(),
        "Missing relation types in fixture: {:?}",
        missing
    );
}

#[test]
fn test_span_offsets_are_valid() {
    let episodes = load_episodes();
    for (i, ep) in episodes.iter().enumerate() {
        for ent in &ep.expected_entities {
            assert!(
                ent.span_start < ent.span_end,
                "Episode {}: entity '{}' has span_start ({}) >= span_end ({})",
                i,
                ent.name,
                ent.span_start,
                ent.span_end
            );
            assert!(
                ent.span_end <= ep.text.len(),
                "Episode {}: entity '{}' has span_end ({}) > text.len() ({})",
                i,
                ent.name,
                ent.span_end,
                ep.text.len()
            );
            let extracted = &ep.text[ent.span_start..ent.span_end];
            assert_eq!(
                extracted, ent.name,
                "Episode {}: span [{}, {}) extracts '{}' but expected '{}'",
                i, ent.span_start, ent.span_end, extracted, ent.name
            );
        }
    }
}

#[test]
fn test_episode_entity_count_bounds() {
    let episodes = load_episodes();
    for (i, ep) in episodes.iter().enumerate() {
        let n_ent = ep.expected_entities.len();
        let n_rel = ep.expected_relations.len();
        assert!(
            (2..=6).contains(&n_ent),
            "Episode {}: expected 2-6 entities, got {}",
            i,
            n_ent
        );
        assert!(
            (1..=4).contains(&n_rel),
            "Episode {}: expected 1-4 relations, got {}",
            i,
            n_rel
        );
    }
}

#[test]
fn test_f1_perfect_match() {
    let predicted = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let expected = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let (p, r, f1) = compute_f1(&predicted, &expected);
    assert!((p - 1.0).abs() < f64::EPSILON);
    assert!((r - 1.0).abs() < f64::EPSILON);
    assert!((f1 - 1.0).abs() < f64::EPSILON);
}

#[test]
fn test_f1_no_overlap() {
    let predicted = vec!["a".to_string(), "b".to_string()];
    let expected = vec!["c".to_string(), "d".to_string()];
    let (p, r, f1) = compute_f1(&predicted, &expected);
    assert!((p - 0.0).abs() < f64::EPSILON);
    assert!((r - 0.0).abs() < f64::EPSILON);
    assert!((f1 - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_f1_partial_overlap() {
    // predicted: {a, b, c}, expected: {a, b, d}
    // TP=2, FP=1, FN=1 => P=2/3, R=2/3, F1=2/3
    let predicted = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let expected = vec!["a".to_string(), "b".to_string(), "d".to_string()];
    let (p, r, f1) = compute_f1(&predicted, &expected);
    let expected_val = 2.0 / 3.0;
    assert!((p - expected_val).abs() < 1e-9);
    assert!((r - expected_val).abs() < 1e-9);
    assert!((f1 - expected_val).abs() < 1e-9);
}

#[test]
fn test_f1_empty_inputs() {
    let (p, r, f1) = compute_f1(&[], &[]);
    assert!((p - 1.0).abs() < f64::EPSILON);
    assert!((r - 1.0).abs() < f64::EPSILON);
    assert!((f1 - 1.0).abs() < f64::EPSILON);
}

#[test]
fn test_f1_predicted_empty_expected_nonempty() {
    let expected = vec!["a".to_string()];
    let (p, r, f1) = compute_f1(&[], &expected);
    assert!((p - 0.0).abs() < f64::EPSILON);
    assert!((r - 0.0).abs() < f64::EPSILON);
    assert!((f1 - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_f1_high_precision_low_recall() {
    // predicted: {a}, expected: {a, b, c, d}
    // TP=1, FP=0, FN=3 => P=1.0, R=0.25, F1=0.4
    let predicted = vec!["a".to_string()];
    let expected = vec![
        "a".to_string(),
        "b".to_string(),
        "c".to_string(),
        "d".to_string(),
    ];
    let (p, r, f1) = compute_f1(&predicted, &expected);
    assert!((p - 1.0).abs() < 1e-9);
    assert!((r - 0.25).abs() < 1e-9);
    assert!((f1 - 0.4).abs() < 1e-9);
}

// ---------------------------------------------------------------------------
// Pipeline integration benchmark (requires ONNX models)
// ---------------------------------------------------------------------------

/// Run the extraction pipeline against all 50 benchmark episodes and compute F1.
///
/// This test is ignored by default because it requires ONNX model files.
/// To run: download models, then `cargo test --test benchmark_test -- --ignored`
///
/// Set `CTXGRAPH_MODELS_DIR` to point to your models directory.
#[test]
#[ignore]
fn test_extraction_f1_against_benchmark() {
    use chrono::Utc;
    use ctxgraph_extract::pipeline::ExtractionPipeline;
    use ctxgraph_extract::schema::ExtractionSchema;

    let models_dir = std::env::var("CTXGRAPH_MODELS_DIR").unwrap_or_else(|_| {
        let home = dirs::cache_dir().expect("no cache dir");
        home.join("ctxgraph").join("models").display().to_string()
    });

    let pipeline = ExtractionPipeline::new(
        ExtractionSchema::default(),
        std::path::Path::new(&models_dir),
        0.2, // low threshold maximises recall; filter at evaluation time if needed
    )
    .expect("Failed to create pipeline. Are models downloaded?");

    let episodes = load_episodes();
    let mut total_entity_f1 = 0.0;
    let mut total_text_only_f1 = 0.0;
    let mut total_relation_f1 = 0.0;

    for (i, ep) in episodes.iter().enumerate() {
        let result = pipeline
            .extract(&ep.text, Utc::now())
            .unwrap_or_else(|e| panic!("Extraction failed on episode {i}: {e}"));

        // Compare entities (name:type format for strict matching)
        let predicted_entities: Vec<String> = result
            .entities
            .iter()
            .map(|e| format!("{}:{}", e.text, e.entity_type))
            .collect();
        let expected_entities: Vec<String> = ep
            .expected_entities
            .iter()
            .map(|e| format!("{}:{}", e.name, e.entity_type))
            .collect();
        let (ep_p, ep_r, ep_f1) = compute_f1(&predicted_entities, &expected_entities);

        // Text-only entity F1 (ignores type — measures raw mention detection)
        let predicted_texts: Vec<String> = result
            .entities
            .iter()
            .map(|e| e.text.to_lowercase())
            .collect();
        let expected_texts: Vec<String> = ep
            .expected_entities
            .iter()
            .map(|e| e.name.to_lowercase())
            .collect();
        let (_, _, text_only_f1) = compute_f1(&predicted_texts, &expected_texts);

        // Compare relations (head:relation:tail format)
        let predicted_relations: Vec<String> = result
            .relations
            .iter()
            .map(|r| format!("{}:{}:{}", r.head, r.relation, r.tail))
            .collect();
        let expected_relations: Vec<String> = ep
            .expected_relations
            .iter()
            .map(|r| format!("{}:{}:{}", r.head, r.relation, r.tail))
            .collect();
        let (rp_p, rp_r, rp_f1) = compute_f1(&predicted_relations, &expected_relations);

        eprintln!(
            "Episode {i:2}: entities F1={ep_f1:.3} (P={ep_p:.3} R={ep_r:.3}) text-only={text_only_f1:.3} | relations F1={rp_f1:.3} (P={rp_p:.3} R={rp_r:.3})"
        );

        // Show missed and spurious relations for debugging
        if rp_f1 < 1.0 {
            let pred_set: std::collections::HashSet<&String> = predicted_relations.iter().collect();
            let exp_set: std::collections::HashSet<&String> = expected_relations.iter().collect();
            let missed: Vec<&&String> = exp_set.difference(&pred_set).collect();
            let spurious: Vec<&&String> = pred_set.difference(&exp_set).collect();
            if !missed.is_empty() {
                eprintln!("  MISSED: {:?}", missed);
            }
            if !spurious.is_empty() {
                eprintln!("  SPURIOUS: {:?}", spurious);
            }
        }

        total_entity_f1 += ep_f1;
        total_text_only_f1 += text_only_f1;
        total_relation_f1 += rp_f1;
    }

    let n = episodes.len() as f64;
    let avg_entity_f1 = total_entity_f1 / n;
    let avg_text_only_f1 = total_text_only_f1 / n;
    let avg_relation_f1 = total_relation_f1 / n;
    let combined_f1 = (avg_entity_f1 + avg_relation_f1) / 2.0;

    eprintln!();
    eprintln!("=== BENCHMARK RESULTS ===");
    eprintln!("Average entity F1 (name+type): {avg_entity_f1:.3}");
    eprintln!("Average entity F1 (name only):  {avg_text_only_f1:.3}");
    eprintln!("Average relation F1:            {avg_relation_f1:.3}");
    eprintln!("Combined F1 (strict):           {combined_f1:.3}");
    eprintln!("Target:                         0.800");
    eprintln!("=========================");

    assert!(
        combined_f1 >= 0.80,
        "Combined F1 {combined_f1:.3} is below 0.80 target"
    );
}

/// Debug test: show extracted entities and relations for specific episodes.
#[test]
#[ignore]
fn debug_extraction_for_episodes() {
    use chrono::Utc;
    use ctxgraph_extract::pipeline::ExtractionPipeline;
    use ctxgraph_extract::schema::ExtractionSchema;

    let models_dir = std::env::var("CTXGRAPH_MODELS_DIR").unwrap_or_else(|_| {
        let home = dirs::cache_dir().expect("no cache dir");
        home.join("ctxgraph").join("models").display().to_string()
    });

    let pipeline = ExtractionPipeline::new(
        ExtractionSchema::default(),
        std::path::Path::new(&models_dir),
        0.2,
    )
    .expect("Failed to create pipeline");

    let episodes = load_episodes();
    let debug_indices: Vec<usize> = std::env::var("DEBUG_EPISODES")
        .unwrap_or_else(|_| "1,5,7,10,13,14,15,19,24,26,29,49".to_string())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    for &i in &debug_indices {
        if i >= episodes.len() {
            continue;
        }
        let ep = &episodes[i];
        let result = pipeline.extract(&ep.text, Utc::now()).unwrap();

        eprintln!("=== Episode {i} ===");
        eprintln!("Text: {}", &ep.text[..ep.text.len().min(120)]);
        eprintln!(
            "Entities: {:?}",
            result
                .entities
                .iter()
                .map(|e| format!("{}:{}", e.text, e.entity_type))
                .collect::<Vec<_>>()
        );
        eprintln!("Relations:");
        for r in &result.relations {
            eprintln!(
                "  {}:{}:{} (conf={:.3})",
                r.head, r.relation, r.tail, r.confidence
            );
        }
        eprintln!("Expected relations:");
        for r in &ep.expected_relations {
            eprintln!("  {}:{}:{}", r.head, r.relation, r.tail);
        }
        eprintln!();
    }
}
