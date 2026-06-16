//! LLM cross-domain extraction harness — exercises `LlmExtractor` (the
//! OpenAI-compatible / Ollama path) instead of the local GLiNER+GLiREL ONNX
//! pipeline used by `cross_domain_test.rs`.
//!
//! This is how you smoke- and hard-test an arbitrary local model (e.g. a GGUF
//! served by Ollama) against ctxgraph's entity + relation extraction contract.
//!
//! Point it at any OpenAI-compatible model via env vars, e.g. a local Ollama:
//!
//! ```bash
//! CTXGRAPH_LLM_KEY=ollama \
//! CTXGRAPH_LLM_URL=http://localhost:11434/v1/chat/completions \
//! CTXGRAPH_LLM_MODEL='hf.co/yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF:Q4_K_M' \
//! CTXGRAPH_LLM_TIMEOUT=600 \
//!   cargo test --package ctxgraph-extract --test llm_cross_domain_test -- --ignored --nocapture
//! ```
//!
//! Both tests are `#[ignore]` because they need a reachable model. They never
//! panic on extraction failure — a model that returns garbage is a *result*,
//! not a test crash, so failures are recorded and reported.

use serde::Deserialize;
use std::collections::HashSet;

use ctxgraph_extract::llm_extract::LlmExtractor;
use ctxgraph_extract::schema::ExtractionSchema;

#[derive(Debug, Deserialize)]
struct CrossDomainEpisode {
    domain: String,
    text: String,
    expected_entities: Vec<ExpectedEntity>,
    expected_relations: Vec<ExpectedRelation>,
}

#[derive(Debug, Deserialize)]
struct ExpectedEntity {
    name: String,
    entity_type: String,
}

#[derive(Debug, Deserialize)]
struct ExpectedRelation {
    head: String,
    relation: String,
    tail: String,
}

/// Fuzzy string match — equal, or one contains the other (case-insensitive).
/// Mirrors the matcher in `cross_domain_test.rs` so scores are comparable.
fn fuzzy_contains(a: &str, b: &str) -> bool {
    let al = a.to_lowercase();
    let bl = b.to_lowercase();
    al == bl || al.contains(&bl) || bl.contains(&al)
}

/// Precision / recall / F1 with greedy fuzzy one-to-one matching.
fn compute_f1(predicted: &[String], expected: &[String]) -> (f64, f64, f64) {
    if predicted.is_empty() && expected.is_empty() {
        return (1.0, 1.0, 1.0);
    }

    let mut matched_expected = vec![false; expected.len()];
    let mut true_positives = 0.0;
    for pred in predicted {
        for (i, exp) in expected.iter().enumerate() {
            if !matched_expected[i] && fuzzy_contains(pred, exp) {
                true_positives += 1.0;
                matched_expected[i] = true;
                break;
            }
        }
    }

    let precision = if predicted.is_empty() {
        0.0
    } else {
        true_positives / predicted.len() as f64
    };
    let recall = if expected.is_empty() {
        0.0
    } else {
        true_positives / expected.len() as f64
    };
    let f1 = if (precision + recall) == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };
    (precision, recall, f1)
}

fn load_cross_domain_episodes() -> Vec<CrossDomainEpisode> {
    let fixture_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/cross_domain_episodes.json"
    );
    let data =
        std::fs::read_to_string(fixture_path).expect("Failed to read cross_domain_episodes.json");
    serde_json::from_str(&data).expect("Failed to deserialize cross-domain episodes")
}

/// Build an extractor from the environment, or print a skip notice and return
/// `None`. Also echoes which model/url it resolved to so the log is unambiguous
/// about *what* was actually tested.
fn extractor_or_skip(test_name: &str) -> Option<LlmExtractor> {
    match LlmExtractor::from_env() {
        Some(ex) => Some(ex),
        None => {
            eprintln!(
                "SKIP {test_name}: no LLM backend. Set CTXGRAPH_LLM_KEY/_URL/_MODEL \
                 (or run Ollama) and re-run with --ignored --nocapture."
            );
            None
        }
    }
}

/// SMOKE TEST — does the model produce *valid, non-empty* extraction JSON on a
/// handful of fresh, unlabeled real-world snippets that are NOT in any fixture?
/// This answers "does the model work at all through the pipeline" before we
/// bother scoring it.
#[test]
#[ignore = "needs a reachable LLM; run with --ignored"]
fn llm_smoke_test() {
    let Some(extractor) = extractor_or_skip("llm_smoke_test") else {
        return;
    };
    let schema = ExtractionSchema::default();

    // Random real-world text spanning a few domains, intentionally off-fixture.
    let samples: &[(&str, &str)] = &[
        (
            "git/devops",
            "Bumped tokio from 1.35 to 1.40 to fix the CVE in the multipart parser. \
             The Redis-backed session store now depends on the new connection pool, \
             and CI on GitHub Actions passes after the Postgres migration.",
        ),
        (
            "business/news",
            "Acme Corp acquired Beta Inc for $2.3B. The deal closes in Q4 2026 and \
             Jane Doe will lead the merged cloud division out of the Austin office.",
        ),
        (
            "ops/incident",
            "Postgres replica lag spiked to 12s during the deploy, so the platform team \
             rolled back the Kafka consumer change and paged the on-call SRE. \
             The 99.9% uptime SLA was nearly breached.",
        ),
    ];

    let mut any_nonempty = false;
    let mut failures = 0usize;
    for (label, text) in samples {
        eprintln!("\n=== smoke [{label}] ===\n{text}");
        match extractor.extract(text, &schema) {
            Ok(result) => {
                eprintln!(
                    "  -> {} entities, {} relations",
                    result.entities.len(),
                    result.relations.len()
                );
                for e in &result.entities {
                    eprintln!("     entity: {:?} :: {}", e.text, e.entity_type);
                }
                for r in &result.relations {
                    eprintln!("     rel:    {} --{}--> {}", r.head, r.relation, r.tail);
                }
                if !result.entities.is_empty() {
                    any_nonempty = true;
                }
            }
            Err(e) => {
                failures += 1;
                eprintln!("  -> EXTRACTION ERROR: {e}");
            }
        }
    }

    eprintln!("\nsmoke summary: {failures}/{} samples errored", samples.len());
    assert!(
        any_nonempty,
        "model returned zero entities on every smoke sample (or every call errored) — \
         the model does not usably produce extraction JSON through this pipeline"
    );
}

/// HARD TEST — per-domain entity & relation F1 across finance, healthcare,
/// legal, manufacturing, education, and government. Exploratory: no threshold
/// is asserted, every miss is logged so you can see exactly where the model
/// breaks down. Extraction errors are counted as zero-score episodes rather
/// than aborting the run.
#[test]
#[ignore = "needs a reachable LLM; run with --ignored"]
fn llm_cross_domain_hard_test() {
    let Some(extractor) = extractor_or_skip("llm_cross_domain_hard_test") else {
        return;
    };
    let schema = ExtractionSchema::default();
    let episodes = load_cross_domain_episodes();

    // domain -> Vec<(entity_f1, relation_f1)>
    let mut domain_scores: std::collections::BTreeMap<String, Vec<(f64, f64)>> =
        std::collections::BTreeMap::new();

    let mut total_entity_f1 = 0.0;
    let mut total_entity_strict_f1 = 0.0;
    let mut total_relation_f1 = 0.0;
    let mut error_episodes = 0usize;

    for (i, ep) in episodes.iter().enumerate() {
        let (predicted_entities, predicted_strict, predicted_relations) =
            match extractor.extract(&ep.text, &schema) {
                Ok(result) => {
                    let names: Vec<String> =
                        result.entities.iter().map(|e| e.text.to_lowercase()).collect();
                    let strict: Vec<String> = result
                        .entities
                        .iter()
                        .map(|e| format!("{}:{}", e.text.to_lowercase(), e.entity_type))
                        .collect();
                    let rels: Vec<String> = result
                        .relations
                        .iter()
                        .map(|r| {
                            format!(
                                "{}:{}:{}",
                                r.head.to_lowercase(),
                                r.relation.to_lowercase(),
                                r.tail.to_lowercase()
                            )
                        })
                        .collect();
                    (names, strict, rels)
                }
                Err(e) => {
                    error_episodes += 1;
                    eprintln!("[{:>15}] ep{i:2}: EXTRACTION ERROR: {e}", ep.domain);
                    (Vec::new(), Vec::new(), Vec::new())
                }
            };

        let expected_entities: Vec<String> =
            ep.expected_entities.iter().map(|e| e.name.to_lowercase()).collect();
        let expected_strict: Vec<String> = ep
            .expected_entities
            .iter()
            .map(|e| format!("{}:{}", e.name.to_lowercase(), e.entity_type))
            .collect();
        let expected_relations: Vec<String> = ep
            .expected_relations
            .iter()
            .map(|r| {
                format!(
                    "{}:{}:{}",
                    r.head.to_lowercase(),
                    r.relation.to_lowercase(),
                    r.tail.to_lowercase()
                )
            })
            .collect();

        let (ep_p, ep_r, entity_f1) = compute_f1(&predicted_entities, &expected_entities);
        let (_, _, entity_strict_f1) = compute_f1(&predicted_strict, &expected_strict);
        let (rp_p, rp_r, relation_f1) = compute_f1(&predicted_relations, &expected_relations);

        eprintln!(
            "[{:>15}] ep{i:2}: entity={entity_f1:.3} (P={ep_p:.3} R={ep_r:.3}) \
             strict={entity_strict_f1:.3} | rel={relation_f1:.3} (P={rp_p:.3} R={rp_r:.3})",
            ep.domain,
        );
        eprintln!("  found:    {predicted_entities:?}");
        eprintln!("  expected: {expected_entities:?}");
        if relation_f1 < 1.0 {
            let pred_set: HashSet<&String> = predicted_relations.iter().collect();
            let exp_set: HashSet<&String> = expected_relations.iter().collect();
            let missed: Vec<&&String> = exp_set.difference(&pred_set).collect();
            let spurious: Vec<&&String> = pred_set.difference(&exp_set).collect();
            if !missed.is_empty() {
                eprintln!("  MISSED rels:   {missed:?}");
            }
            if !spurious.is_empty() {
                eprintln!("  SPURIOUS rels: {spurious:?}");
            }
        }
        eprintln!();

        total_entity_f1 += entity_f1;
        total_entity_strict_f1 += entity_strict_f1;
        total_relation_f1 += relation_f1;
        domain_scores
            .entry(ep.domain.clone())
            .or_default()
            .push((entity_f1, relation_f1));
    }

    let n = episodes.len() as f64;
    let avg_entity_f1 = total_entity_f1 / n;
    let avg_entity_strict_f1 = total_entity_strict_f1 / n;
    let avg_relation_f1 = total_relation_f1 / n;
    let combined_f1 = (avg_entity_f1 + avg_relation_f1) / 2.0;

    eprintln!("=== LLM CROSS-DOMAIN RESULTS ===");
    for (domain, scores) in &domain_scores {
        let dn = scores.len() as f64;
        let avg_e: f64 = scores.iter().map(|(e, _)| e).sum::<f64>() / dn;
        let avg_r: f64 = scores.iter().map(|(_, r)| r).sum::<f64>() / dn;
        eprintln!(
            "  {domain:>15}: entity={avg_e:.3}  relation={avg_r:.3}  combined={:.3}  (n={dn:.0})",
            (avg_e + avg_r) / 2.0
        );
    }
    eprintln!();
    eprintln!("Overall avg entity F1 (name):   {avg_entity_f1:.3}");
    eprintln!("Overall avg entity F1 (strict): {avg_entity_strict_f1:.3}");
    eprintln!("Overall avg relation F1:        {avg_relation_f1:.3}");
    eprintln!("Overall combined F1:            {combined_f1:.3}");
    eprintln!("Episodes with extraction errors: {error_episodes}/{}", episodes.len());
    eprintln!("================================");
}
