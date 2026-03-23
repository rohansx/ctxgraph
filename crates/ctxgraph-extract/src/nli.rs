//! Zero-shot relation classification via NLI cross-encoder.
//!
//! Uses a DeBERTa-v3-xsmall NLI model to score (premise, hypothesis) pairs.
//! For relation extraction, the premise is the source text and the hypothesis
//! is a natural language statement like "X depends on Y".
//!
//! Output labels: index 0 = contradiction, index 1 = entailment, index 2 = neutral.

use std::path::Path;

use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;

use crate::ner::ExtractedEntity;
use crate::rel::ExtractedRelation;
use crate::schema::ExtractionSchema;

/// Hypothesis templates for each relation type.
/// Multiple templates per relation improve recall.
const HYPOTHESIS_TEMPLATES: &[(&str, &[&str])] = &[
    ("chose", &["{head} chose {tail}", "{head} selected {tail}"]),
    (
        "rejected",
        &["{head} rejected {tail}", "{head} decided against {tail}"],
    ),
    (
        "replaced",
        &["{head} replaced {tail}", "{tail} was replaced by {head}"],
    ),
    (
        "depends_on",
        &["{head} depends on {tail}", "{head} uses {tail}"],
    ),
    ("fixed", &["{head} fixed {tail}", "{head} resolved {tail}"]),
    (
        "introduced",
        &["{head} introduced {tail}", "{head} added {tail}"],
    ),
    (
        "deprecated",
        &["{head} deprecated {tail}", "{head} removed {tail}"],
    ),
    ("caused", &["{head} caused {tail}", "{head} led to {tail}"]),
    (
        "constrained_by",
        &[
            "{head} is constrained by {tail}",
            "{head} must comply with {tail}",
        ],
    ),
];

/// NLI-based relation extraction engine.
pub struct NliEngine {
    session: Session,
    tokenizer: Tokenizer,
}

/// Index of the "entailment" label in model output.
const ENTAILMENT_IDX: usize = 1;

impl NliEngine {
    /// Load the NLI ONNX model and tokenizer.
    pub fn new(model_path: &Path, tokenizer_path: &Path) -> Result<Self, NliError> {
        let session = Session::builder()
            .and_then(|b| b.with_intra_threads(1))
            .and_then(|b| b.commit_from_file(model_path))
            .map_err(|e| NliError::ModelLoad(e.to_string()))?;

        let tokenizer =
            Tokenizer::from_file(tokenizer_path).map_err(|e| NliError::ModelLoad(e.to_string()))?;

        Ok(Self { session, tokenizer })
    }

    /// Score a single (premise, hypothesis) pair.
    /// Returns softmax probabilities [contradiction, entailment, neutral].
    fn score(&self, premise: &str, hypothesis: &str) -> Result<[f32; 3], NliError> {
        let encoding = self
            .tokenizer
            .encode((premise, hypothesis), true)
            .map_err(|e| NliError::Inference(e.to_string()))?;

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| m as i64)
            .collect();

        let seq_len = input_ids.len();

        let ids_tensor = Tensor::from_array(([1, seq_len], input_ids))
            .map_err(|e| NliError::Inference(e.to_string()))?;
        let mask_tensor = Tensor::from_array(([1, seq_len], attention_mask))
            .map_err(|e| NliError::Inference(e.to_string()))?;

        let inputs = ort::inputs![ids_tensor, mask_tensor]
            .map_err(|e| NliError::Inference(e.to_string()))?;

        let outputs = self
            .session
            .run(inputs)
            .map_err(|e| NliError::Inference(e.to_string()))?;

        // Output shape: [1, 3] — logits for [contradiction, entailment, neutral]
        let logits_view = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| NliError::Inference(e.to_string()))?;

        let logits = logits_view
            .as_slice()
            .ok_or_else(|| NliError::Inference("non-contiguous logits".into()))?;

        if logits.len() < 3 {
            return Err(NliError::Inference(format!(
                "expected 3 logits, got {}",
                logits.len()
            )));
        }

        // Softmax
        let max_logit = logits[0].max(logits[1]).max(logits[2]);
        let exp: Vec<f32> = logits[..3].iter().map(|&l| (l - max_logit).exp()).collect();
        let sum: f32 = exp.iter().sum();
        Ok([exp[0] / sum, exp[1] / sum, exp[2] / sum])
    }

    /// Extract relations using NLI entailment scoring.
    ///
    /// For each entity pair in the text, tests hypothesis templates for each
    /// relation type. Returns relations where entailment score exceeds the threshold.
    pub fn extract(
        &self,
        text: &str,
        entities: &[ExtractedEntity],
        schema: &ExtractionSchema,
        threshold: f32,
    ) -> Result<Vec<ExtractedRelation>, NliError> {
        let mut relations = Vec::new();
        let mut seen = std::collections::HashSet::<(String, String, String)>::new();

        // Split text into sentences for focused premises
        let sentences = split_into_sentences(text);

        for (sent_start, sent_end) in &sentences {
            let premise = &text[*sent_start..*sent_end];

            // Find entities in this sentence (+ adjacent sentence window)
            let sent_entities: Vec<&ExtractedEntity> = entities
                .iter()
                .filter(|e| e.span_start >= *sent_start && e.span_start < *sent_end)
                .collect();

            if sent_entities.len() < 2 {
                continue;
            }

            // Test all entity pairs
            for (i, head) in sent_entities.iter().enumerate() {
                for tail in sent_entities.iter().skip(i + 1) {
                    if head.text == tail.text {
                        continue;
                    }

                    // Test both directions for each relation
                    for &(rel_name, templates) in HYPOTHESIS_TEMPLATES {
                        // Check schema validity
                        let schema_valid = schema
                            .relation_types
                            .get(rel_name)
                            .map(|spec| {
                                (spec.head.contains(&head.entity_type)
                                    && spec.tail.contains(&tail.entity_type))
                                    || (spec.head.contains(&tail.entity_type)
                                        && spec.tail.contains(&head.entity_type))
                            })
                            .unwrap_or(false);

                        if !schema_valid {
                            continue;
                        }

                        // Try head→tail direction
                        let mut best_score_fwd: f32 = 0.0;
                        for template in templates {
                            let hypothesis = template
                                .replace("{head}", &head.text)
                                .replace("{tail}", &tail.text);
                            if let Ok(probs) = self.score(premise, &hypothesis) {
                                best_score_fwd = best_score_fwd.max(probs[ENTAILMENT_IDX]);
                            }
                        }

                        // Try tail→head direction
                        let mut best_score_rev: f32 = 0.0;
                        for template in templates {
                            let hypothesis = template
                                .replace("{head}", &tail.text)
                                .replace("{tail}", &head.text);
                            if let Ok(probs) = self.score(premise, &hypothesis) {
                                best_score_rev = best_score_rev.max(probs[ENTAILMENT_IDX]);
                            }
                        }

                        // Pick the best direction
                        let (actual_head, actual_tail, score) = if best_score_fwd >= best_score_rev
                        {
                            (&head.text, &tail.text, best_score_fwd)
                        } else {
                            (&tail.text, &head.text, best_score_rev)
                        };

                        if score >= threshold {
                            let key = (
                                actual_head.clone(),
                                rel_name.to_string(),
                                actual_tail.clone(),
                            );
                            if seen.insert(key) {
                                relations.push(ExtractedRelation {
                                    head: actual_head.clone(),
                                    relation: rel_name.to_string(),
                                    tail: actual_tail.clone(),
                                    confidence: score as f64,
                                });
                            }
                        }
                    }
                }
            }
        }

        // Keep only top relation per entity pair (highest confidence)
        deduplicate_by_pair(&mut relations);

        Ok(relations)
    }
}

/// Keep only the highest-confidence relation per (head, tail) pair.
fn deduplicate_by_pair(relations: &mut Vec<ExtractedRelation>) {
    let mut best: std::collections::HashMap<(String, String), usize> =
        std::collections::HashMap::new();

    for (i, rel) in relations.iter().enumerate() {
        let key = (rel.head.clone(), rel.tail.clone());
        let rev_key = (rel.tail.clone(), rel.head.clone());
        let existing_key = if best.contains_key(&key) {
            Some(key.clone())
        } else if best.contains_key(&rev_key) {
            Some(rev_key)
        } else {
            None
        };

        if let Some(k) = existing_key {
            let prev_idx = best[&k];
            if rel.confidence > relations[prev_idx].confidence {
                best.insert(k, i);
            }
        } else {
            best.insert(key, i);
        }
    }

    let keep: std::collections::HashSet<usize> = best.values().copied().collect();
    let mut idx = 0;
    relations.retain(|_| {
        let k = keep.contains(&idx);
        idx += 1;
        k
    });
}

/// Simple sentence splitting (byte-level).
fn split_into_sentences(text: &str) -> Vec<(usize, usize)> {
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
pub enum NliError {
    #[error("failed to load NLI model: {0}")]
    ModelLoad(String),

    #[error("NLI inference error: {0}")]
    Inference(String),
}
