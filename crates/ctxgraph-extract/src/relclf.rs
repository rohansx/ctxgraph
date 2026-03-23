//! Relation classifier using sentence embeddings + logistic regression.
//!
//! Takes a 384-dim embedding (from all-MiniLM-L6-v2) as input and outputs
//! one of 10 relation classes. The ONNX model is a tiny linear classifier
//! (~15 KB) that runs a single matmul + bias.
//!
//! Entity markers `[E1]`/`[/E1]` and `[E2]`/`[/E2]` are inserted around
//! entities in the text before embedding.

use std::collections::HashMap;
use std::path::Path;

use ort::session::Session;
use ort::value::Tensor;

use crate::ner::ExtractedEntity;
use crate::rel::{ExtractedRelation, RelError};

/// Default confidence threshold for accepting a classification.
const DEFAULT_THRESHOLD: f32 = 0.55;

/// Number of output classes (including "none").
#[allow(dead_code)]
const NUM_CLASSES: usize = 10;

/// The label map: class index → relation name.
const LABEL_MAP: &[&str] = &[
    "chose",          // 0
    "rejected",       // 1
    "replaced",       // 2
    "depends_on",     // 3
    "fixed",          // 4
    "introduced",     // 5
    "deprecated",     // 6
    "caused",         // 7
    "constrained_by", // 8
    "none",           // 9
];

/// Index of the "none" class in the label map.
const NONE_CLASS_IDX: usize = 9;

/// Embedding-based relation classifier (logistic regression on MiniLM embeddings).
pub struct RelationClassifier {
    session: Session,
    label_map: Vec<String>,
    threshold: f32,
}

impl RelationClassifier {
    /// Load the ONNX model from disk.
    ///
    /// The model expects a single input "embedding" of shape [1, 384]
    /// and produces "logits" of shape [1, 10].
    pub fn new(model_path: &Path) -> Result<Self, RelError> {
        let session = Session::builder()
            .and_then(|b| b.with_intra_threads(1))
            .and_then(|b| b.commit_from_file(model_path))
            .map_err(|e| RelError::ModelLoad(e.to_string()))?;

        // Try loading label_map.json from model directory
        let label_map = if let Some(parent) = model_path.parent() {
            let label_map_path = parent.join("label_map.json");
            load_label_map(&label_map_path).unwrap_or_else(default_label_map)
        } else {
            default_label_map()
        };

        Ok(Self {
            session,
            label_map,
            threshold: DEFAULT_THRESHOLD,
        })
    }

    /// Set the confidence threshold for accepting a classification.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Classify a relation from a pre-computed 384-dim embedding.
    ///
    /// Returns `Some((relation_type, confidence))` if a relation is detected,
    /// or `None` if the "none" class wins or confidence is below threshold.
    pub fn classify_embedding(&self, embedding: &[f32]) -> Result<Option<(String, f32)>, RelError> {
        let dim = embedding.len();
        let tensor = Tensor::from_array(([1, dim], embedding.to_vec()))
            .map_err(|e| RelError::Inference(e.to_string()))?;

        let inputs = ort::inputs![
            "embedding" => tensor,
        ]
        .map_err(|e| RelError::Inference(e.to_string()))?;

        let outputs = self
            .session
            .run(inputs)
            .map_err(|e| RelError::Inference(e.to_string()))?;

        let logits_view = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| RelError::Inference(e.to_string()))?;

        let logits = logits_view
            .as_slice()
            .ok_or_else(|| RelError::Inference("non-contiguous logits".into()))?;

        let num_classes = logits.len().min(self.label_map.len());
        if num_classes == 0 {
            return Err(RelError::Inference("empty logits output".into()));
        }

        // Softmax
        let probs = softmax(&logits[..num_classes]);

        // Find the top class
        let (best_idx, best_prob) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap(); // safe: num_classes > 0

        // Return None if "none" class wins or confidence is below threshold
        if best_idx == NONE_CLASS_IDX || best_idx >= self.label_map.len() {
            return Ok(None);
        }
        if *best_prob < self.threshold {
            return Ok(None);
        }

        Ok(Some((self.label_map[best_idx].clone(), *best_prob)))
    }

    /// Classify all unique entity pairs using single-embedding approach.
    ///
    /// For each pair, inserts [E1]/[E2] markers into the text and embeds the
    /// full marked sentence as a single 384-dim vector for classification.
    /// `embed_fn` takes text and returns a 384-dim vector.
    pub fn classify_batch<F>(
        &self,
        text: &str,
        entities: &[ExtractedEntity],
        embed_fn: &F,
    ) -> Result<Vec<ExtractedRelation>, RelError>
    where
        F: Fn(&str) -> Result<Vec<f32>, RelError>,
    {
        let mut relations = Vec::new();
        let mut seen = std::collections::HashSet::<(String, String)>::new();

        for (i, head_ent) in entities.iter().enumerate() {
            for tail_ent in entities.iter().skip(i + 1) {
                if head_ent.text == tail_ent.text {
                    continue;
                }

                let pair_key = if head_ent.text < tail_ent.text {
                    (head_ent.text.clone(), tail_ent.text.clone())
                } else {
                    (tail_ent.text.clone(), head_ent.text.clone())
                };

                if !seen.insert(pair_key) {
                    continue;
                }

                // Try head→tail
                if let Some((relation, confidence)) =
                    self.classify_single(text, &head_ent.text, &tail_ent.text, embed_fn)?
                {
                    relations.push(ExtractedRelation {
                        head: head_ent.text.clone(),
                        relation,
                        tail: tail_ent.text.clone(),
                        confidence: confidence as f64,
                    });
                    continue;
                }

                // Try tail→head
                if let Some((relation, confidence)) =
                    self.classify_single(text, &tail_ent.text, &head_ent.text, embed_fn)?
                {
                    relations.push(ExtractedRelation {
                        head: tail_ent.text.clone(),
                        relation,
                        tail: head_ent.text.clone(),
                        confidence: confidence as f64,
                    });
                }
            }
        }

        // Post-process: correct direction using text heuristics
        for rel in &mut relations {
            let (new_head, new_tail) = correct_direction(text, &rel.head, &rel.tail, &rel.relation);
            rel.head = new_head;
            rel.tail = new_tail;
        }

        Ok(relations)
    }

    /// Check whether a given entity pair is likely "none" (no relation).
    ///
    /// Returns the probability of the "none" class for this entity pair.
    /// Used to filter out spurious heuristic relations.
    pub fn none_probability<F>(
        &self,
        text: &str,
        head: &str,
        tail: &str,
        embed_fn: &F,
    ) -> Result<f32, RelError>
    where
        F: Fn(&str) -> Result<Vec<f32>, RelError>,
    {
        let marked_text = insert_single_marker(
            &insert_single_marker(text, head, "[E1]", "[/E1]"),
            tail,
            "[E2]",
            "[/E2]",
        );
        let embedding = embed_fn(&marked_text)?;
        self.none_probability_from_embedding(&embedding)
    }

    /// Get the "none" class probability from a pre-computed embedding.
    fn none_probability_from_embedding(&self, embedding: &[f32]) -> Result<f32, RelError> {
        let dim = embedding.len();
        let tensor = Tensor::from_array(([1, dim], embedding.to_vec()))
            .map_err(|e| RelError::Inference(e.to_string()))?;
        let inputs =
            ort::inputs!["embedding" => tensor].map_err(|e| RelError::Inference(e.to_string()))?;
        let outputs = self
            .session
            .run(inputs)
            .map_err(|e| RelError::Inference(e.to_string()))?;
        let logits_view = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| RelError::Inference(e.to_string()))?;
        let logits = logits_view
            .as_slice()
            .ok_or_else(|| RelError::Inference("non-contiguous logits".into()))?;
        let num_classes = logits.len().min(self.label_map.len());
        let probs = softmax(&logits[..num_classes]);
        Ok(if NONE_CLASS_IDX < probs.len() {
            probs[NONE_CLASS_IDX]
        } else {
            0.0
        })
    }

    /// Classify a single entity pair using a single 384-dim embedding.
    ///
    /// Inserts [E1]/[/E1] around head and [E2]/[/E2] around tail in the text,
    /// then embeds the full marked text as one vector.
    fn classify_single<F>(
        &self,
        text: &str,
        head: &str,
        tail: &str,
        embed_fn: &F,
    ) -> Result<Option<(String, f32)>, RelError>
    where
        F: Fn(&str) -> Result<Vec<f32>, RelError>,
    {
        let marked_text = insert_single_marker(
            &insert_single_marker(text, head, "[E1]", "[/E1]"),
            tail,
            "[E2]",
            "[/E2]",
        );
        let embedding = embed_fn(&marked_text)?;
        self.classify_embedding(&embedding)
    }
}

/// Correct relation direction using text heuristics.
///
/// The classifier sometimes gets direction wrong because embeddings don't
/// capture directionality well. This applies simple text-pattern rules to
/// swap head/tail when the text clearly indicates the opposite direction.
fn correct_direction(text: &str, head: &str, tail: &str, relation: &str) -> (String, String) {
    let text_lower = text.to_lowercase();
    let head_lower = head.to_lowercase();
    let tail_lower = tail.to_lowercase();

    match relation {
        "depends_on" => {
            // If text says "tail uses/relies on head", swap
            let patterns = [
                format!("{} uses {}", tail_lower, head_lower),
                format!("{} relies on {}", tail_lower, head_lower),
                format!("{} depends on {}", tail_lower, head_lower),
                format!("{} requires {}", tail_lower, head_lower),
                format!("{} connects to {}", tail_lower, head_lower),
                format!("{} calls {}", tail_lower, head_lower),
            ];
            for pat in &patterns {
                if text_lower.contains(pat.as_str()) {
                    return (tail.to_string(), head.to_string());
                }
            }
        }
        "replaced" => {
            // "from head to tail" means tail is the new thing (tail replaced head)
            let from_to = format!("from {} to {}", head_lower, tail_lower);
            if text_lower.contains(&from_to) {
                return (tail.to_string(), head.to_string());
            }
            // "migrated to tail from head" means tail replaced head — already correct
            // "tail replaced head" — already correct if head is the replacer
            // But if text says "head was replaced by tail", tail is the new thing
            let passive = format!("{} was replaced by {}", head_lower, tail_lower);
            if text_lower.contains(&passive) {
                return (tail.to_string(), head.to_string());
            }
        }
        "introduced" => {
            // If text says "tail introduced/added head", swap
            let patterns = [
                format!("{} introduced {}", tail_lower, head_lower),
                format!("{} added {}", tail_lower, head_lower),
                format!("{} brought in {}", tail_lower, head_lower),
                format!("{} set up {}", tail_lower, head_lower),
            ];
            for pat in &patterns {
                if text_lower.contains(pat.as_str()) {
                    return (tail.to_string(), head.to_string());
                }
            }
        }
        "constrained_by" => {
            // The constraint/policy/SLA should be tail. If head looks like a
            // constraint (contains SLA/QPS/compliance/policy/ms/etc.), swap.
            let constraint_indicators = [
                "sla",
                "qps",
                "compliance",
                "policy",
                "latency",
                "budget",
                "limit",
                "quota",
                "regulation",
            ];
            let head_is_constraint = constraint_indicators
                .iter()
                .any(|ind| head_lower.contains(ind));
            let tail_is_constraint = constraint_indicators
                .iter()
                .any(|ind| tail_lower.contains(ind));
            if head_is_constraint && !tail_is_constraint {
                return (tail.to_string(), head.to_string());
            }
        }
        _ => {}
    }

    (head.to_string(), tail.to_string())
}

/// Insert `[E1]`/`[/E1]` around the head entity and `[E2]`/`[/E2]` around
/// the tail entity in the text.
#[allow(dead_code)]
fn insert_entity_markers(text: &str, head: &str, tail: &str) -> String {
    let text_lower = text.to_lowercase();
    let head_lower = head.to_lowercase();
    let tail_lower = tail.to_lowercase();

    let head_pos = text_lower.find(&head_lower);
    let tail_pos = text_lower.find(&tail_lower);

    match (head_pos, tail_pos) {
        (Some(hp), Some(tp)) if hp <= tp => {
            let head_end = hp + head.len();
            // Find tail after head to avoid overlap
            let tail_in_rest = text[head_end..].to_lowercase().find(&tail_lower);
            if let Some(rel_tp) = tail_in_rest {
                let abs_tp = head_end + rel_tp;
                let tail_end = abs_tp + tail.len();
                format!(
                    "{}[E1]{}[/E1]{}[E2]{}[/E2]{}",
                    &text[..hp],
                    &text[hp..head_end],
                    &text[head_end..abs_tp],
                    &text[abs_tp..tail_end],
                    &text[tail_end..]
                )
            } else {
                // Tail overlaps with head or not found after — prepend tail marker
                format!(
                    "[E2]{}[/E2] {}[E1]{}[/E1]{}",
                    tail,
                    &text[..hp],
                    &text[hp..head_end],
                    &text[head_end..]
                )
            }
        }
        (Some(hp), Some(tp)) => {
            // Tail before head
            let tail_end = tp + tail.len();
            let head_end = hp + head.len();
            if tail_end <= hp {
                format!(
                    "{}[E2]{}[/E2]{}[E1]{}[/E1]{}",
                    &text[..tp],
                    &text[tp..tail_end],
                    &text[tail_end..hp],
                    &text[hp..head_end],
                    &text[head_end..]
                )
            } else {
                // Overlapping — prepend head marker
                format!(
                    "[E1]{}[/E1] {}[E2]{}[/E2]{}",
                    head,
                    &text[..tp],
                    &text[tp..tail_end],
                    &text[tail_end..]
                )
            }
        }
        (Some(hp), None) => {
            let head_end = hp + head.len();
            format!(
                "[E2]{}[/E2] {}[E1]{}[/E1]{}",
                tail,
                &text[..hp],
                &text[hp..head_end],
                &text[head_end..]
            )
        }
        (None, Some(tp)) => {
            let tail_end = tp + tail.len();
            format!(
                "[E1]{}[/E1] {}[E2]{}[/E2]{}",
                head,
                &text[..tp],
                &text[tp..tail_end],
                &text[tail_end..]
            )
        }
        (None, None) => {
            format!("[E1]{}[/E1] [E2]{}[/E2] {}", head, tail, text)
        }
    }
}

/// Insert a single entity marker around the first occurrence of `entity` in `text`.
fn insert_single_marker(text: &str, entity: &str, open: &str, close: &str) -> String {
    let text_lower = text.to_lowercase();
    let entity_lower = entity.to_lowercase();

    if let Some(pos) = text_lower.find(&entity_lower) {
        let end = pos + entity.len();
        format!(
            "{}{}{}{}{}",
            &text[..pos],
            open,
            &text[pos..end],
            close,
            &text[end..]
        )
    } else {
        format!("{}{}{} {}", open, entity, close, text)
    }
}

/// Compute softmax over a slice of f32 values.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

/// Load label_map.json from disk: `{"0": "chose", "1": "rejected", ...}`.
fn load_label_map(path: &Path) -> Option<Vec<String>> {
    let content = std::fs::read_to_string(path).ok()?;
    let map: HashMap<String, String> = serde_json::from_str(&content).ok()?;

    let max_idx = map.keys().filter_map(|k| k.parse::<usize>().ok()).max()?;
    let mut labels = vec!["none".to_string(); max_idx + 1];
    for (k, v) in &map {
        if let Ok(idx) = k.parse::<usize>() {
            labels[idx] = v.clone();
        }
    }
    Some(labels)
}

/// Return the built-in label map as a Vec<String>.
fn default_label_map() -> Vec<String> {
    LABEL_MAP.iter().map(|&s| s.to_string()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_markers_both_found() {
        let text = "The team chose PostgreSQL over MySQL for the project.";
        let result = insert_entity_markers(text, "PostgreSQL", "MySQL");
        assert!(result.contains("[E1]PostgreSQL[/E1]"));
        assert!(result.contains("[E2]MySQL[/E2]"));
    }

    #[test]
    fn test_insert_markers_tail_before_head() {
        let text = "MySQL was replaced by PostgreSQL.";
        let result = insert_entity_markers(text, "PostgreSQL", "MySQL");
        assert!(result.contains("[E1]PostgreSQL[/E1]"));
        assert!(result.contains("[E2]MySQL[/E2]"));
    }

    #[test]
    fn test_insert_markers_neither_found() {
        let text = "Some unrelated text about databases.";
        let result = insert_entity_markers(text, "Redis", "Kafka");
        assert!(result.contains("[E1]Redis[/E1]"));
        assert!(result.contains("[E2]Kafka[/E2]"));
        assert!(result.contains("databases"));
    }

    #[test]
    fn test_softmax() {
        let probs = softmax(&[1.0, 2.0, 3.0]);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_default_label_map() {
        let map = default_label_map();
        assert_eq!(map.len(), NUM_CLASSES);
        assert_eq!(map[0], "chose");
        assert_eq!(map[9], "none");
    }
}
