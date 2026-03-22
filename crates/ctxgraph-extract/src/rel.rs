use std::path::Path;

use composable::Composable;
use gliner::model::input::relation::schema::RelationSchema;
use gliner::model::input::text::TextInput;
use gliner::model::output::decoded::SpanOutput;
use gliner::model::output::relation::RelationOutput;
use gliner::model::params::Parameters;
use gliner::model::pipeline::relation::RelationPipeline;
use gliner::model::pipeline::token::TokenPipeline;
use orp::model::Model;
use orp::params::RuntimeParameters;
use orp::pipeline::Pipeline;

use crate::ner::ExtractedEntity;
use crate::relclf::RelationClassifier;
use crate::relex::RelexEngine;
use crate::schema::ExtractionSchema;

/// A relation extracted between two entities.
#[derive(Debug, Clone)]
pub struct ExtractedRelation {
    pub head: String,
    pub relation: String,
    pub tail: String,
    pub confidence: f64,
}

/// Relation extraction engine — fully local, no external API calls.
///
/// Extraction tiers:
/// - **Model-based**: GLiNER multitask ONNX model for typed relation extraction.
/// - **Heuristic**: Pattern-based keyword + proximity extraction (~0.51 F1).
///
/// Experimental (opt-in via `CTXGRAPH_RELEX=1`):
/// - **Relex**: gliner-relex ONNX model (joint NER + relation extraction).
pub enum RelEngine {
    /// Has both multitask model and optionally relex model.
    ModelBased(ModelBasedRelEngine),
    Heuristic,
}

/// Cached relex engine (loaded once per process).
static RELEX_ENGINE: std::sync::OnceLock<Option<RelexEngine>> = std::sync::OnceLock::new();

/// Cached relation classifier (loaded once per process).
static RELCLF_ENGINE: std::sync::OnceLock<Option<RelationClassifier>> = std::sync::OnceLock::new();

/// Model-based relation extraction using gline-rs.
///
/// Requires `gliner-multitask-large-v0.5` ONNX model.
pub struct ModelBasedRelEngine {
    model: Model,
    params: Parameters,
    tokenizer_path: String,
}

impl ModelBasedRelEngine {
    pub fn new(model_path: &Path, tokenizer_path: &Path) -> Result<Self, RelError> {
        let runtime_params = RuntimeParameters::default();
        let model = Model::new(
            model_path
                .to_str()
                .ok_or(RelError::InvalidPath(model_path.display().to_string()))?,
            runtime_params,
        )
        .map_err(|e| RelError::ModelLoad(e.to_string()))?;

        Ok(Self {
            model,
            params: Parameters::default(),
            tokenizer_path: tokenizer_path
                .to_str()
                .ok_or(RelError::InvalidPath(
                    tokenizer_path.display().to_string(),
                ))?
                .to_string(),
        })
    }

    pub fn extract(
        &self,
        text: &str,
        labels: &[&str],
        schema: &ExtractionSchema,
    ) -> Result<(Vec<ExtractedEntity>, Vec<ExtractedRelation>), RelError> {
        // Build relation schema from extraction schema
        let mut relation_schema = RelationSchema::new();
        for (rel_name, spec) in &schema.relation_types {
            let heads: Vec<&str> = spec.head.iter().map(|s| s.as_str()).collect();
            let tails: Vec<&str> = spec.tail.iter().map(|s| s.as_str()).collect();
            relation_schema.push_with_allowed_labels(rel_name, &heads, &tails);
        }

        let input = TextInput::from_str(&[text], labels)
            .map_err(|e| RelError::Inference(e.to_string()))?;

        // Step 1: Run NER via TokenPipeline
        let ner_pipeline = TokenPipeline::new(&self.tokenizer_path)
            .map_err(|e| RelError::Inference(e.to_string()))?;
        let ner_composable = ner_pipeline.to_composable(&self.model, &self.params);
        let ner_output: SpanOutput = ner_composable
            .apply(input)
            .map_err(|e| RelError::Inference(e.to_string()))?;

        // Collect entities from NER output using span character offsets directly
        let mut entities = Vec::new();
        for sequence_spans in &ner_output.spans {
            for span in sequence_spans {
                let (start, end) = span.offsets();
                entities.push(ExtractedEntity {
                    text: span.text().to_string(),
                    entity_type: span.class().to_string(),
                    span_start: start,
                    span_end: end,
                    confidence: span.probability() as f64,
                });
            }
        }

        // Step 2: Run relation extraction on top of NER output
        let rel_pipeline =
            RelationPipeline::default(&self.tokenizer_path, &relation_schema)
                .map_err(|e| RelError::Inference(e.to_string()))?;
        let rel_composable = rel_pipeline.to_composable(&self.model, &self.params);
        let rel_output: RelationOutput = rel_composable
            .apply(ner_output)
            .map_err(|e| RelError::Inference(e.to_string()))?;

        // Collect relations
        let mut relations = Vec::new();
        for sequence_rels in &rel_output.relations {
            for rel in sequence_rels {
                relations.push(ExtractedRelation {
                    head: rel.subject().to_string(),
                    relation: rel.class().to_string(),
                    tail: rel.object().to_string(),
                    confidence: rel.probability() as f64,
                });
            }
        }

        Ok((entities, relations))
    }
}

impl RelEngine {
    /// Create a model-based engine if the multitask model is available,
    /// otherwise fall back to heuristic mode.
    pub fn new(model_path: Option<&Path>, tokenizer_path: Option<&Path>) -> Result<Self, RelError> {
        match (model_path, tokenizer_path) {
            (Some(mp), Some(tp)) if mp.exists() && tp.exists() => {
                let engine = ModelBasedRelEngine::new(mp, tp)?;
                Ok(Self::ModelBased(engine))
            }
            _ => Ok(Self::Heuristic),
        }
    }

    /// Extract relations between entities — fully local, no API calls.
    ///
    /// Uses heuristic keyword + proximity extraction (~0.51 F1).
    ///
    /// Environment variables:
    /// - `CTXGRAPH_RELEX=1`: Enable experimental relex ONNX tier
    pub fn extract(
        &self,
        text: &str,
        entities: &[ExtractedEntity],
        schema: &ExtractionSchema,
    ) -> Result<Vec<ExtractedRelation>, RelError> {
        let mut relations = heuristic_relations(text, entities, schema);

        // Relex ONNX model (experimental — opt-in via CTXGRAPH_RELEX=1)
        if std::env::var("CTXGRAPH_RELEX").is_ok() {
            let relex = RELEX_ENGINE.get_or_init(|| {
                let mgr = crate::model_manager::ModelManager::new().ok()?;
                let (model_path, tok_path) = mgr.find_relex_model()?;
                RelexEngine::new(&model_path, &tok_path).ok()
            });

            if let Some(engine) = relex {
                let entity_labels: Vec<&str> = schema.entity_labels();
                let relation_labels: Vec<&str> = schema.relation_labels();
                if let Ok(result) = engine.extract(text, &entity_labels, &relation_labels, 0.5, 0.5, schema) {
                    let existing: std::collections::HashSet<(String, String)> = relations
                        .iter()
                        .map(|r| (r.head.clone(), r.tail.clone()))
                        .collect();

                    for rel in &result.relations {
                        if rel.confidence < 0.80 {
                            continue;
                        }

                        let mapped_head = map_span_to_entity(&rel.head, entities);
                        let mapped_tail = map_span_to_entity(&rel.tail, entities);

                        if let (Some(head), Some(tail)) = (mapped_head, mapped_tail) {
                            if head != tail
                                && !existing.contains(&(head.clone(), tail.clone()))
                                && !existing.contains(&(tail.clone(), head.clone()))
                            {
                                relations.push(ExtractedRelation {
                                    head,
                                    relation: rel.relation.clone(),
                                    tail,
                                    confidence: rel.confidence,
                                });
                            }
                        }
                    }
                }
            }
        }

        // Relation classifier (embedding-based) — auto-enabled if model found.
        let relclf = RELCLF_ENGINE.get_or_init(|| {
            let mgr = crate::model_manager::ModelManager::new().ok()?;
            let model_path = mgr.find_relation_classifier()?;
            RelationClassifier::new(&model_path).ok()
        });

        if let Some(classifier) = relclf {
            // Use fastembed for generating embeddings (same model as ctxgraph-embed).
            static EMBED_ENGINE: std::sync::OnceLock<
                Option<fastembed::TextEmbedding>,
            > = std::sync::OnceLock::new();
            let embed = EMBED_ENGINE.get_or_init(|| {
                fastembed::TextEmbedding::try_new(
                    fastembed::InitOptions::new(fastembed::EmbeddingModel::AllMiniLML6V2),
                )
                .ok()
            });

            if let Some(embed_model) = embed {
                let embed_fn = |text: &str| -> Result<Vec<f32>, RelError> {
                    let mut vecs = embed_model
                        .embed(vec![text], None)
                        .map_err(|e| RelError::Inference(e.to_string()))?;
                    vecs.pop()
                        .ok_or_else(|| RelError::Inference("empty embedding".into()))
                };

                if let Ok(clf_relations) =
                    classifier.classify_batch(text, entities, &embed_fn)
                {
                    let existing: std::collections::HashSet<(String, String)> = relations
                        .iter()
                        .map(|r| (r.head.clone(), r.tail.clone()))
                        .collect();

                    for rel in clf_relations {
                        if rel.confidence < 0.76 {
                            continue;
                        }
                        // Only add relations for pairs not already covered
                        if !existing.contains(&(rel.head.clone(), rel.tail.clone()))
                            && !existing.contains(&(rel.tail.clone(), rel.head.clone()))
                        {
                            relations.push(rel);
                        }
                    }
                }
            }
        }

        Ok(relations)
    }
}

/// Heuristic relation extraction using sentence-level co-occurrence with
/// expanded keyword coverage and relaxed type constraints.
///
/// Splits text on sentence boundaries, then matches relation keywords against
/// entity pairs within ±1 sentence window. Type constraints from the schema
/// are used to prefer valid pairs but relaxed matching is used as fallback
/// when the schema check fails (since GLiNER may misassign types).
fn heuristic_relations(
    text: &str,
    entities: &[ExtractedEntity],
    schema: &ExtractionSchema,
) -> Vec<ExtractedRelation> {
    // Expanded keyword sets with verb base forms for broader matching.
    // Each keyword is checked via `sent_lower.contains(kw)` so "use" matches
    // "to use", "we use", "using" etc.
    // Keywords use verb stems where safe (domain-specific, unlikely to cause
    // false positives) and full words/phrases for common verbs.
    // Each keyword is checked via `sent_lower.contains(kw)`.
    let patterns: &[(&str, &[&str])] = &[
        ("chose", &[
            "chose", "choose", "select", "picked", "went with", "adopt",
            "decided to use", "decided to add", "opted for", "settled on",
            "standardiz", "switched to",
            "decided that",
        ]),
        ("rejected", &[
            "reject", "ruled out", "decided against", "dropped",
            "abandon", "discard", "veto",
        ]),
        ("replaced", &[
            // Core replacement verbs (broad keywords like "migrat", "switched",
            // "transition" removed — they cause 5+ spurious replaced relations
            // and real from/to migrations are caught by detect_from_to_pattern).
            "replac", "swapped",
            "in favor of", "instead of", "in place of",
            "over the legacy", "over the old",
        ]),
        ("depends_on", &[
            "depend", "relies on", "rely on", "built on",
            " uses ", " use ", "using ",
            "connect", "backed by", "powered by",
            "running on", "runs on", "deployed on", "hosted on",
            "integrat", "communicat",
            "proxied by", "proxies", "in front of",
            "managed by", "orchestrat",
            "reads from", "writes to", "persist",
            "publish", "subscrib", "consum",
            "sends to", "receives from",
            "queries", "fetches from",
            "leverag", "switched to",
            "scraped", "scrapes",
            "flow through", "flows through",
            "target", "written in", "implemented in",
            "-based", "caching layer",
            "counter in", "stored in", "cache to",
            "stitched", "backed by",
            "local cache", "sync when",
            // Containment / composition patterns
            "used by", "via ",
            "package manager",
            "is down",
            // Language/runtime patterns
            "goroutine",
            // Monitoring dependencies
            "scraped by", "dashboards",
        ]),
        ("fixed", &[
            "fixed", "fixing", "resolv", "patched", "repaired",
            "debugged", "addressed", "correct ",
            "eliminat", "mitigat", "diagnos", "root-caus",
            "identified", "found ",
            "traced", "investigated",
            "patch ",
        ]),
        ("introduced", &[
            "introduc", " add ", "added", "implement", "created", "built",
            "set up", "deploy", "enabl", "integrat", "install",
            "configur", "establish", "rolled out", "launched",
            "onboard", "provision", "stood up", "spun up",
            "upgrad", "extract",
            // Selection/adoption that introduces something
            "chosen to enforc", "chosen to implement",
        ]),
        ("deprecated", &[
            "deprecat", "removed", "removing", "phased out", "phase out",
            "sunset", "decommission", "retired", "killed", "shut down",
            "tore down", "ripped out", "turned off",
        ]),
        ("caused", &[
            "caused", "causing", "resulted in", "led to", "trigger",
            "contributed to",
            "dropped to", "reduced to", "improved to", "decreased to",
            "reduced by", "improved by", "increased by", "decreased by",
            "spiked", "spike",
        ]),
        ("constrained_by", &[
            "constrain", "blocked by", "due to",
            "required to", "has to", "have to",
            "cannot exceed", "comply", "enforc",
            "subject to", "bound by", "governed by",
            "mandated", "driven by", "must comply",
            "guarantee", "accepted",
            "rate limit", "forbidden", "must not",
            "exceed", "scoped", "capped at", "cap at",
            "broke", "break ", "breaking",
            "zero-trust", "least privilege",
            "cannot handle",
            // Quality / constraint patterns
            "exactly-once",
            " sla ",
            "memory safety",
        ]),
    ];

    let sentence_ranges = split_sentences(text);

    let mut relations = Vec::new();
    let mut seen = std::collections::HashSet::<(String, String, String)>::new();

    // Pre-pass: detect explicit "from X to Y" migration patterns for "replaced" relations.
    // This catches patterns that proximity scoring might miss (e.g., when a third
    // entity is closer to the keyword).
    detect_from_to_pattern(text, entities, schema, &mut relations, &mut seen);

    for (sent_idx, &(sent_start, sent_end)) in sentence_ranges.iter().enumerate() {
        let sent_text = &text[sent_start..sent_end];
        let sent_lower = sent_text.to_lowercase();

        // Entities in this sentence
        let sent_entities: Vec<&ExtractedEntity> = entities
            .iter()
            .filter(|e| e.span_start >= sent_start && e.span_start < sent_end)
            .collect();

        if sent_entities.len() < 2 {
            // Need at least 2 entities in the window for a relation
            // Check the expanded window
        }

        // Expanded window: this sentence ± 1 adjacent sentence
        let window_start = if sent_idx > 0 { sentence_ranges[sent_idx - 1].0 } else { sent_start };
        let window_end = if sent_idx + 1 < sentence_ranges.len() {
            sentence_ranges[sent_idx + 1].1
        } else {
            sent_end
        };

        let window_entities: Vec<&ExtractedEntity> = entities
            .iter()
            .filter(|e| e.span_start >= window_start && e.span_start < window_end)
            .collect();

        for (relation, keywords) in patterns {
            if !keywords.iter().any(|kw| sent_lower.contains(kw)) {
                continue;
            }

            // Try schema-valid pairs first, then relaxed matching
            let rel_spec = schema.relation_types.get(*relation);

            // Find keyword position in sentence for proximity scoring
            let kw_pos = keywords.iter()
                .filter_map(|kw| sent_lower.find(kw).map(|p| p + kw.len() / 2))
                .min()
                .unwrap_or(sent_lower.len() / 2);
            let kw_abs_pos = sent_start + kw_pos;

            // Collect schema-valid candidate pairs with proximity scores
            let mut candidates: Vec<(f64, &ExtractedEntity, &ExtractedEntity)> = Vec::new();

            for &head in &sent_entities {
                for &tail in &window_entities {
                    if std::ptr::eq(head, tail) || head.text == tail.text {
                        continue;
                    }

                    // Skip reference-like entities (ADR-001, PR, Issue #N) as
                    // they're document labels, not domain entities for relations
                    if is_reference_entity(&head.text)
                        || is_reference_entity(&tail.text)
                    {
                        continue;
                    }

                    // Require strict schema validity
                    let schema_valid = rel_spec
                        .map(|spec| {
                            spec.head.contains(&head.entity_type)
                                && spec.tail.contains(&tail.entity_type)
                        })
                        .unwrap_or(false);

                    if !schema_valid {
                        continue;
                    }

                    // Skip Person→Person for most relations
                    let both_person = head.entity_type == "Person" && tail.entity_type == "Person";
                    if both_person && *relation != "chose" && *relation != "rejected" {
                        continue;
                    }

                    let in_same_sentence =
                        tail.span_start >= sent_start && tail.span_start < sent_end;

                    // Proximity: closer entities to keyword get higher score
                    let head_dist = (head.span_start as f64 - kw_abs_pos as f64).abs();
                    let tail_dist = (tail.span_start as f64 - kw_abs_pos as f64).abs();
                    let avg_dist = (head_dist + tail_dist) / 2.0;
                    let proximity = 1.0 / (1.0 + avg_dist / 50.0);

                    let base = if in_same_sentence { 0.65 } else { 0.45 };
                    let confidence = base * proximity;

                    candidates.push((confidence, head, tail));
                }
            }

            // Sort by confidence descending, take top 1 pair per relation per sentence
            // to maximize precision — second candidates are usually false positives.
            candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

            for (confidence, head, tail) in candidates.into_iter().take(1) {
                    let (actual_head, actual_tail) = determine_direction(
                        relation, head, tail, rel_spec, &sent_lower, sent_start,
                    );

                    let key = (actual_head.clone(), relation.to_string(), actual_tail.clone());
                    if seen.insert(key) {
                        relations.push(ExtractedRelation {
                            head: actual_head.clone(),
                            relation: relation.to_string(),
                            tail: actual_tail.clone(),
                            confidence,
                        });
                    }
            }
        }

        // "chose X over Y" → also emit rejected:Y
        // Detect "over" separating a chosen entity from a rejected one.
        if let Some(over_pos) = sent_lower.find(" over ") {
            let rejected_spec = schema.relation_types.get("rejected");
            // Find the entity appearing right after "over" (the rejected one)
            // and the Person entity who did the choosing (for head).
            let over_abs = sent_start + over_pos;
            let mut rejected_entity: Option<&ExtractedEntity> = None;
            let mut chooser: Option<&ExtractedEntity> = None;

            for ent in &sent_entities {
                if is_reference_entity(&ent.text) {
                    continue;
                }
                // Entity after "over " is the rejected one
                if ent.span_start > over_abs && ent.span_start <= over_abs + 15 {
                    rejected_entity = Some(ent);
                }
                // Person entity is the chooser
                if ent.entity_type == "Person" {
                    chooser = Some(ent);
                }
            }

            if let (Some(rej), Some(ch)) = (rejected_entity, chooser) {
                if rej.text != ch.text {
                    // Check schema validity for rejected relation
                    let schema_valid = rejected_spec
                        .map(|spec| {
                            spec.head.contains(&ch.entity_type)
                                && spec.tail.contains(&rej.entity_type)
                        })
                        .unwrap_or(true);

                    if schema_valid {
                        let key = (ch.text.clone(), "rejected".to_string(), rej.text.clone());
                        if seen.insert(key) {
                            relations.push(ExtractedRelation {
                                head: ch.text.clone(),
                                relation: "rejected".to_string(),
                                tail: rej.text.clone(),
                                confidence: 0.60,
                            });
                        }
                    }
                }
            }
        }
    }

    // Post-processing: filter out common false positive patterns.
    relations.retain(|r| {
        // Remove version-to-version "replaced" (e.g., "Java 21:replaced:Java 11").
        // Version upgrades are not replacements of different technologies.
        if r.relation == "replaced" {
            let h = r.head.to_lowercase();
            let t = r.tail.to_lowercase();
            // Same base name with version numbers → version upgrade, not replacement
            let h_base = h.trim_end_matches(|c: char| c.is_ascii_digit() || c == '.' || c == ' ');
            let t_base = t.trim_end_matches(|c: char| c.is_ascii_digit() || c == '.' || c == ' ');
            if !h_base.is_empty() && h_base == t_base {
                return false;
            }
        }
        true
    });

    // Post-processing: resolve conflicting relation types for the same entity pair.
    // E.g., if both "chose" and "rejected" match for (X, Y), keep only one based
    // on which is more specific.
    let conflicts: &[(&str, &str)] = &[
        ("chose", "rejected"),        // Can't both choose and reject the same thing
        ("introduced", "deprecated"), // Can't introduce and deprecate same thing
        ("replaced", "depends_on"),   // Replacement is more specific than dependency
        ("introduced", "depends_on"), // "introduced" is more specific for same pair
        ("introduced", "replaced"),   // Can't both introduce and replace same pair
        ("chose", "depends_on"),      // Choice is more specific than dependency
        ("caused", "fixed"),          // Prefer "fixed" over "caused" for same pair
        ("introduced", "caused"),     // "introduced" is more specific than "caused"
    ];

    let mut to_remove = std::collections::HashSet::new();
    for (i, r1) in relations.iter().enumerate() {
        for (j, r2) in relations.iter().enumerate() {
            if i >= j { continue; }
            // Same head-tail pair with conflicting relations
            let same_pair = (r1.head == r2.head && r1.tail == r2.tail)
                || (r1.head == r2.tail && r1.tail == r2.head);
            if !same_pair { continue; }

            for &(a, b) in conflicts {
                if (r1.relation == a && r2.relation == b)
                    || (r1.relation == b && r2.relation == a)
                {
                    // Keep the higher confidence one, remove the lower
                    if r1.confidence >= r2.confidence {
                        to_remove.insert(j);
                    } else {
                        to_remove.insert(i);
                    }
                }
            }
        }
    }

    if !to_remove.is_empty() {
        let mut idx = 0;
        relations.retain(|_| {
            let keep = !to_remove.contains(&idx);
            idx += 1;
            keep
        });
    }

    relations
}

/// Map a relex entity span text to the nearest known entity by substring matching.
///
/// Returns the entity name if a match is found with sufficient overlap.
fn map_span_to_entity(span_text: &str, entities: &[ExtractedEntity]) -> Option<String> {
    let span_lower = span_text.to_lowercase();

    // Exact match first
    for ent in entities {
        if ent.text.to_lowercase() == span_lower {
            return Some(ent.text.clone());
        }
    }

    // Substring match (either direction)
    let mut best: Option<(&ExtractedEntity, f64)> = None;
    for ent in entities {
        let ent_lower = ent.text.to_lowercase();
        if ent_lower.contains(&span_lower) || span_lower.contains(&ent_lower) {
            let score = span_lower.len().min(ent_lower.len()) as f64
                / span_lower.len().max(ent_lower.len()) as f64;
            if score >= 0.3 {
                if best.as_ref().is_none_or(|(_, s)| score > *s) {
                    best = Some((ent, score));
                }
            }
        }
    }

    best.map(|(ent, _)| ent.text.clone())
}

/// Detect explicit "from X to Y" migration patterns and emit replaced relations.
///
/// Scans text for "from <entity> to <entity>" patterns which strongly indicate
/// that Y replaced X. This runs before the general keyword matching to ensure
/// the correct entity pair is selected.
fn detect_from_to_pattern(
    text: &str,
    entities: &[ExtractedEntity],
    schema: &ExtractionSchema,
    relations: &mut Vec<ExtractedRelation>,
    seen: &mut std::collections::HashSet<(String, String, String)>,
) {
    let text_lower = text.to_lowercase();
    let rel_spec = schema.relation_types.get("replaced");

    // Find "from" positions
    let mut search_start = 0;
    while let Some(from_pos) = text_lower[search_start..].find("from ") {
        let abs_from = search_start + from_pos;

        // Find entities near "from" (the OLD entity, appearing after "from")
        for old_ent in entities.iter() {
            if old_ent.span_start < abs_from + 4 || old_ent.span_start > abs_from + 40 {
                continue;
            }

            // Look for "to" after the old entity
            let after_old = old_ent.span_end;
            if let Some(to_rel_pos) = text_lower[after_old..].find(" to ") {
                let abs_to = after_old + to_rel_pos;

                // Find entity after "to" (the NEW entity)
                for new_ent in entities.iter() {
                    if new_ent.span_start < abs_to + 3 || new_ent.span_start > abs_to + 40 {
                        continue;
                    }
                    if new_ent.text == old_ent.text {
                        continue;
                    }
                    if is_reference_entity(&new_ent.text) || is_reference_entity(&old_ent.text) {
                        continue;
                    }

                    // Check schema validity
                    let schema_valid = rel_spec
                        .map(|spec| {
                            (spec.head.contains(&new_ent.entity_type)
                                && spec.tail.contains(&old_ent.entity_type))
                            || (spec.head.contains(&old_ent.entity_type)
                                && spec.tail.contains(&new_ent.entity_type))
                        })
                        .unwrap_or(true);

                    if !schema_valid {
                        continue;
                    }

                    // NEW:replaced:OLD
                    let key = (new_ent.text.clone(), "replaced".to_string(), old_ent.text.clone());
                    if seen.insert(key) {
                        relations.push(ExtractedRelation {
                            head: new_ent.text.clone(),
                            relation: "replaced".to_string(),
                            tail: old_ent.text.clone(),
                            confidence: 0.75,
                        });
                    }
                }
            }
        }

        search_start = abs_from + 5;
    }
}

/// Returns true for entities that are document references (ADR-001, PR, Issue #N)
/// rather than real domain entities. These generate spurious relations.
fn is_reference_entity(text: &str) -> bool {
    // ADR-NNN, PR, Issue #N
    if text.starts_with("ADR-") || text.starts_with("ADR ") {
        return true;
    }
    if text == "PR" || text.starts_with("PR #") || text.starts_with("PR:") {
        return true;
    }
    if text.starts_with("Issue #") || text.starts_with("Issue:") {
        return true;
    }
    false
}

/// Determine head/tail direction for a relation using text cues and schema.
///
/// For "replaced" relations: convention is `NEW:replaced:OLD`.
/// Detects "from X to Y" → Y is new; "Replace X with Y" → Y is new;
/// "X replaced Y" → X is new; "in favor of X" → X is new.
///
/// For "depends_on": Service/Component depends on Infrastructure/Database/Tool.
///
/// Falls back to schema role matching, then text order.
fn determine_direction(
    relation: &str,
    head: &ExtractedEntity,
    tail: &ExtractedEntity,
    rel_spec: Option<&crate::schema::RelationSpec>,
    sent_lower: &str,
    _sent_start: usize,
) -> (String, String) {
    let (first, second) = if head.span_start <= tail.span_start {
        (head, tail)
    } else {
        (tail, head)
    };

    // For "replaced": detect text cues for NEW:replaced:OLD direction
    if relation == "replaced" {
        let first_lower = first.text.to_lowercase();
        let second_lower = second.text.to_lowercase();

        // "from X to Y" / "migrate from X to Y" → Y:replaced:X (second is new)
        let from_to = sent_lower.contains(&format!("from {}", first_lower))
            && (sent_lower.contains(&format!("to {}", second_lower))
                || sent_lower.contains(&format!("to the {}", second_lower)));
        if from_to {
            return (second.text.clone(), first.text.clone());
        }

        // "from Y to X" (reversed order in text)
        let rev_from_to = sent_lower.contains(&format!("from {}", second_lower))
            && (sent_lower.contains(&format!("to {}", first_lower))
                || sent_lower.contains(&format!("to the {}", first_lower)));
        if rev_from_to {
            return (first.text.clone(), second.text.clone());
        }

        // "fallback using Y" / "fallback to Y" → Y is the NEW replacement
        if sent_lower.contains("fallback") {
            if let Some(fb_pos) = sent_lower.find("fallback") {
                let after_fb = &sent_lower[fb_pos..];
                let first_in_fb = after_fb.find(&first_lower);
                let second_in_fb = after_fb.find(&second_lower);
                match (first_in_fb, second_in_fb) {
                    (Some(_), None) => {
                        // first entity near "fallback" → first is new replacement
                        return (first.text.clone(), second.text.clone());
                    }
                    (None, Some(_)) => {
                        // second entity near "fallback" → second is new replacement
                        return (second.text.clone(), first.text.clone());
                    }
                    (Some(fp), Some(sp)) => {
                        // Entity closer to "fallback using/to" is the new thing
                        if fp < sp {
                            return (first.text.clone(), second.text.clone());
                        } else {
                            return (second.text.clone(), first.text.clone());
                        }
                    }
                    _ => {}
                }
            }
        }

        // "replace X with Y" / "replaced X with Y" / "replaced it with Y" → Y:replaced:X
        if let Some(pos) = sent_lower.find("replac") {
            let after_replace = &sent_lower[pos..];
            // Handle "replaced it with Y" — "it" is a pronoun for the OLD thing;
            // only one entity will appear after "replac", and it's the NEW thing.
            if after_replace.contains(" it ") && after_replace.contains(" with ") {
                let first_in_replace = after_replace.find(&first_lower);
                let second_in_replace = after_replace.find(&second_lower);
                match (first_in_replace, second_in_replace) {
                    (Some(_), None) => {
                        // first entity appears after "replaced it with" → first is NEW
                        return (first.text.clone(), second.text.clone());
                    }
                    (None, Some(_)) => {
                        // second entity appears after "replaced it with" → second is NEW
                        return (second.text.clone(), first.text.clone());
                    }
                    _ => {} // Both found, fall through to normal logic
                }
            }

            let first_in_replace = after_replace.find(&first_lower);
            let second_in_replace = after_replace.find(&second_lower);
            if let (Some(fp), Some(sp)) = (first_in_replace, second_in_replace) {
                // First mentioned after "replace" is the OLD, second is NEW
                if fp < sp {
                    return (second.text.clone(), first.text.clone());
                } else {
                    return (first.text.clone(), second.text.clone());
                }
            }
        }

        // "in favor of X" → X is the new thing
        if let Some(pos) = sent_lower.find("in favor of") {
            let favor_text = &sent_lower[pos..];
            if favor_text.contains(&second_lower) {
                return (second.text.clone(), first.text.clone());
            }
            if favor_text.contains(&first_lower) {
                return (first.text.clone(), second.text.clone());
            }
        }

        // "X replaced Y" (natural English subject-verb-object) → X is new
        // But "replaced X with Y" was handled above. Here X appears before "replaced"
        if let Some(rp) = sent_lower.find("replac") {
            let first_pos = sent_lower.find(&first_lower);
            let second_pos = sent_lower.find(&second_lower);
            if let (Some(fp), Some(sp)) = (first_pos, second_pos) {
                if fp < rp && sp > rp {
                    // "first ... replaced ... second" → first:replaced:second
                    return (first.text.clone(), second.text.clone());
                }
                if sp < rp && fp > rp {
                    return (second.text.clone(), first.text.clone());
                }
            }
        }
    }

    // For "deprecated": detect passive voice "{entity} is/are deprecated" → entity is TAIL
    if relation == "deprecated" {
        let first_lower = first.text.to_lowercase();
        let second_lower = second.text.to_lowercase();

        // "{entity} is deprecated" / "{entity} are deprecated" → entity is the TAIL (old thing)
        let first_passive = sent_lower.contains(&format!("{} is deprecated", first_lower))
            || sent_lower.contains(&format!("{} are deprecated", first_lower))
            || sent_lower.contains(&format!("{}-based", first_lower))
                && sent_lower.contains("deprecated");
        let second_passive = sent_lower.contains(&format!("{} is deprecated", second_lower))
            || sent_lower.contains(&format!("{} are deprecated", second_lower))
            || sent_lower.contains(&format!("{}-based", second_lower))
                && sent_lower.contains("deprecated");

        if first_passive && !second_passive {
            // first entity is the deprecated thing (tail), second is head
            return (second.text.clone(), first.text.clone());
        }
        if second_passive && !first_passive {
            // second entity is the deprecated thing (tail), first is head
            return (first.text.clone(), second.text.clone());
        }
    }

    // For "depends_on": use entity type semantics and text cues.
    if relation == "depends_on" {
        let first_lower = first.text.to_lowercase();
        let second_lower = second.text.to_lowercase();

        // Passive voice: "X <verb> by Y" → Y:depends_on:X
        // "scraped by Grafana" → Grafana:depends_on:Prometheus
        // "managed by Kubernetes" → Envoy:depends_on:Kubernetes
        let passive_markers = ["used by", "scraped by", "managed by", "orchestrated by", "handled by"];
        for marker in passive_markers {
            if let Some(pos) = sent_lower.find(marker) {
                let first_pos = sent_lower.find(&first_lower);
                let second_pos = sent_lower.find(&second_lower);
                if let (Some(fp), Some(sp)) = (first_pos, second_pos) {
                    if fp < pos && sp > pos {
                        // first is the provider, second is the consumer (after "by")
                        return (second.text.clone(), first.text.clone());
                    }
                    if sp < pos && fp > pos {
                        return (first.text.clone(), second.text.clone());
                    }
                }
            }
        }

        // "in front of X" → thing-in-front depends_on X (reverse intuition:
        // the proxy in front depends on what it proxies)
        // Actually: "gateway in front of PaymentService" → PaymentService:constrained_by or
        // more commonly the service behind depends on the gateway.
        // Keep default text-order for this case.

        // "X via Y" → X:depends_on:Y
        if let Some(pos) = sent_lower.find(" via ") {
            let first_pos = sent_lower.find(&first_lower);
            let second_pos = sent_lower.find(&second_lower);
            if let (Some(fp), Some(sp)) = (first_pos, second_pos) {
                if fp < pos && sp > pos {
                    return (first.text.clone(), second.text.clone());
                }
                if sp < pos && fp > pos {
                    return (second.text.clone(), first.text.clone());
                }
            }
        }

        // "X-based Y" → Y:depends_on:X (e.g., "SQLite-based local cache" → cache:depends_on:SQLite)
        if sent_lower.contains("-based") {
            let first_pos = sent_lower.find(&first_lower);
            let second_pos = sent_lower.find(&second_lower);
            if let Some(based_pos) = sent_lower.find("-based") {
                if let (Some(fp), Some(sp)) = (first_pos, second_pos) {
                    // Entity whose name appears right before "-based" is the provider
                    if fp < based_pos && based_pos <= fp + first_lower.len() + 1 {
                        // first is the technology (provider), second depends on it
                        return (second.text.clone(), first.text.clone());
                    }
                    if sp < based_pos && based_pos <= sp + second_lower.len() + 1 {
                        return (first.text.clone(), second.text.clone());
                    }
                }
            }
        }

        let consumer_types = ["Service", "Component"];
        let provider_types = ["Database", "Infrastructure", "Language"];

        let first_is_consumer = consumer_types.contains(&first.entity_type.as_str());
        let second_is_consumer = consumer_types.contains(&second.entity_type.as_str());
        let first_is_provider = provider_types.contains(&first.entity_type.as_str());
        let second_is_provider = provider_types.contains(&second.entity_type.as_str());

        // Consumer depends_on Provider (Service/Component → Database/Infrastructure)
        if first_is_consumer && second_is_provider {
            return (first.text.clone(), second.text.clone());
        }
        if second_is_consumer && first_is_provider {
            return (second.text.clone(), first.text.clone());
        }

        // When both are same category, use naming convention:
        // *Service names are consumers (they depend on tools/libraries)
        let first_is_service = first.entity_type == "Service"
            || first.text.ends_with("Service");
        let second_is_service = second.entity_type == "Service"
            || second.text.ends_with("Service");

        if first_is_service && !second_is_service {
            return (first.text.clone(), second.text.clone());
        }
        if second_is_service && !first_is_service {
            return (second.text.clone(), first.text.clone());
        }
    }

    // Schema role matching
    if let Some(spec) = rel_spec {
        let fwd_valid = spec.head.contains(&head.entity_type)
            && spec.tail.contains(&tail.entity_type);
        let rev_valid = spec.head.contains(&tail.entity_type)
            && spec.tail.contains(&head.entity_type);
        match (fwd_valid, rev_valid) {
            (true, false) => return (head.text.clone(), tail.text.clone()),
            (false, true) => return (tail.text.clone(), head.text.clone()),
            _ => {}
        }
    }

    // Default: text order
    (first.text.clone(), second.text.clone())
}

/// Split text into sentence ranges: `(start_byte, end_byte)`.
fn split_sentences(text: &str) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let bytes = text.as_bytes();
    let len = text.len();
    let mut seg_start = 0usize;
    let mut i = 0usize;

    while i < len {
        let boundary = if i + 1 < len
            && (bytes[i] == b'.' || bytes[i] == b'!' || bytes[i] == b'?')
            && bytes[i + 1] == b' '
        {
            Some(i + 1)
        } else if i + 1 < len && bytes[i] == b'\n' && bytes[i + 1] == b'\n' {
            Some(i)
        } else {
            None
        };

        if let Some(end) = boundary {
            ranges.push((seg_start, end));
            seg_start = end + 1;
            i = seg_start;
            continue;
        }
        i += 1;
    }
    if seg_start < len {
        ranges.push((seg_start, len));
    }
    if ranges.is_empty() {
        ranges.push((0, len));
    }
    ranges
}

#[derive(Debug, thiserror::Error)]
pub enum RelError {
    #[error("invalid path: {0}")]
    InvalidPath(String),

    #[error("failed to load model: {0}")]
    ModelLoad(String),

    #[error("inference error: {0}")]
    Inference(String),
}
