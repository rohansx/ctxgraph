//! Custom ONNX pipeline for `gliner-relex-large-v0.5`.
//!
//! This model performs joint NER + relation extraction in a single forward pass
//! using a DeBERTa-v3-large backbone with a GCN-based relation layer.
//!
//! Architecture: UniEncoderSpanRelex (span_mode=markerV0)
//!
//! ONNX inputs (6 tensors):
//!   input_ids, attention_mask, words_mask, text_lengths, span_idx, span_mask
//!
//! ONNX outputs (4 tensors):
//!   logits         — entity span scores [batch, num_words, max_width, num_classes]
//!   rel_idx        — entity pair indices [batch, num_pairs, 2]
//!   rel_logits     — relation type scores [batch, num_pairs, num_rel_classes]
//!   rel_mask       — valid pair mask [batch, num_pairs]

use std::path::Path;

use ndarray::{Array2, Array3, ArrayD};
use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;

use crate::ner::ExtractedEntity;
use crate::rel::ExtractedRelation;
use crate::schema::ExtractionSchema;

/// Raw tensor outputs from ONNX inference, fully owned.
struct InferenceOutputs {
    logits: ArrayD<f32>,
    rel_idx: Option<ArrayD<i64>>,
    rel_logits: Option<ArrayD<f32>>,
    rel_mask: Option<ArrayD<f32>>,
    word_spans: Vec<(usize, usize, String)>,
    num_words: usize,
}

/// Maximum span width (number of words). Matches model config `max_width: 12`.
const MAX_WIDTH: usize = 12;

/// Special token IDs from the relex tokenizer.
const ENT_TOKEN: &str = "<<ENT>>";
const SEP_TOKEN: &str = "<<SEP>>";
const REL_TOKEN: &str = "<<REL>>";

/// The relex ONNX inference engine.
pub struct RelexEngine {
    session: Session,
    tokenizer: Tokenizer,
}

/// Result from a single relex inference pass.
pub struct RelexResult {
    pub entities: Vec<ExtractedEntity>,
    pub relations: Vec<ExtractedRelation>,
}

impl RelexEngine {
    /// Load the relex ONNX model and tokenizer.
    pub fn new(model_path: &Path, tokenizer_path: &Path) -> Result<Self, RelexError> {
        let session = Session::builder()
            .map_err(|e| RelexError::ModelLoad(e.to_string()))?
            .with_intra_threads(4)
            .map_err(|e| RelexError::ModelLoad(e.to_string()))?
            .commit_from_file(model_path)
            .map_err(|e| RelexError::ModelLoad(format!("{}: {}", model_path.display(), e)))?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| RelexError::ModelLoad(format!("tokenizer: {}", e)))?;

        Ok(Self { session, tokenizer })
    }

    /// Run joint NER + relation extraction on text.
    ///
    /// - `entity_labels`: entity type names (e.g., ["Person", "Database", "Service"])
    /// - `relation_labels`: relation type names (e.g., ["chose", "replaced", "depends_on"])
    /// - `entity_threshold`: minimum sigmoid score for entity spans (0.0-1.0)
    /// - `relation_threshold`: minimum sigmoid score for relations (0.0-1.0)
    pub fn extract(
        &self,
        text: &str,
        entity_labels: &[&str],
        relation_labels: &[&str],
        entity_threshold: f32,
        relation_threshold: f32,
        schema: &ExtractionSchema,
    ) -> Result<RelexResult, RelexError> {
        let out = self.run_inference(text, entity_labels, relation_labels)?;

        let entities = decode_entities(
            &out.logits.view(),
            &out.word_spans,
            out.num_words,
            text,
            entity_labels,
            entity_threshold,
        );

        let relations = match (out.rel_idx, out.rel_logits, out.rel_mask) {
            (Some(ri), Some(rl), Some(rm)) => decode_relations(
                &ri.view(),
                &rl.view(),
                &rm.view(),
                &entities,
                relation_labels,
                relation_threshold,
                schema,
            ),
            _ => Vec::new(),
        };

        Ok(RelexResult {
            entities,
            relations,
        })
    }

    /// Build the 6 ONNX input tensors and run inference.
    ///
    /// Returns (outputs, word_spans, num_words).
    fn run_inference(
        &self,
        text: &str,
        entity_labels: &[&str],
        relation_labels: &[&str],
    ) -> Result<InferenceOutputs, RelexError> {
        // Split text into words (simple whitespace split with char offsets)
        let words: Vec<(usize, usize, &str)> = split_words(text);
        let num_words = words.len();

        // Build the prompt string:
        // <<ENT>> Person <<ENT>> Database ... <<SEP>> <<REL>> chose <<REL>> replaced ... <<SEP>> text
        let mut prompt_parts: Vec<String> = Vec::new();
        for label in entity_labels {
            prompt_parts.push(format!("{} {}", ENT_TOKEN, label));
        }
        prompt_parts.push(SEP_TOKEN.to_string());
        for label in relation_labels {
            prompt_parts.push(format!("{} {}", REL_TOKEN, label));
        }
        prompt_parts.push(SEP_TOKEN.to_string());

        let prompt_prefix = prompt_parts.join(" ");
        let full_text = format!("{} {}", prompt_prefix, text);

        // Tokenize
        let encoding = self
            .tokenizer
            .encode(full_text.as_str(), true)
            .map_err(|e| RelexError::Inference(format!("tokenize: {}", e)))?;

        let ids = encoding.get_ids();
        let attention = encoding.get_attention_mask();
        let seq_len = ids.len();

        // Build input_ids and attention_mask
        let input_ids: Vec<i64> = ids.iter().map(|&id| id as i64).collect();
        let attention_mask: Vec<i64> = attention.iter().map(|&a| a as i64).collect();

        // Build words_mask: maps each token to its word index in the text portion.
        //
        // The sentencepiece tokenizer produces offsets that include a leading `▁`
        // (space) character as part of the token, so token offsets don't align
        // exactly with word start positions. We use overlap-based matching: a
        // token maps to a word if the token's character range overlaps the word's
        // range. Only the first sub-token of each word receives the 1-based word
        // index; continuation sub-tokens remain 0 (matching GLiNER's
        // `prepare_word_mask` which uses `word_ids()` from HuggingFace tokenizers).
        let mut words_mask = vec![0i64; seq_len];

        let offsets = encoding.get_offsets();
        let prompt_char_len = prompt_prefix.len() + 1; // +1 for the space between prompt and text

        let mut prev_word_idx: Option<usize> = None;
        for (tok_idx, &(tok_start, tok_end)) in offsets.iter().enumerate() {
            if tok_idx == 0 || (tok_start == 0 && tok_end == 0) {
                continue; // skip [CLS], [SEP], and padding
            }
            if tok_start < prompt_char_len {
                continue; // skip prompt tokens
            }

            // Convert token char range to text-relative offsets
            let t_start = tok_start - prompt_char_len;
            let t_end = tok_end - prompt_char_len;

            // Find word whose range overlaps this token (first sub-token only)
            for (word_idx, &(w_start, w_end, _)) in words.iter().enumerate() {
                if t_start < w_end && t_end > w_start {
                    if prev_word_idx != Some(word_idx) {
                        words_mask[tok_idx] = (word_idx + 1) as i64; // 1-based
                        prev_word_idx = Some(word_idx);
                    }
                    break;
                }
            }
        }

        // Build span_idx and span_mask
        // Model expects exactly num_words * MAX_WIDTH spans (reshape requirement)
        let num_spans = num_words * MAX_WIDTH;
        let mut span_indices: Vec<i64> = Vec::with_capacity(num_spans * 2);
        let mut span_mask: Vec<bool> = Vec::with_capacity(num_spans);
        for start in 0..num_words {
            for width in 0..MAX_WIDTH {
                let end = start + width;
                if end < num_words {
                    span_indices.push(start as i64);
                    span_indices.push(end as i64);
                    span_mask.push(true);
                } else {
                    // Padding span (invalid)
                    span_indices.push(0);
                    span_indices.push(0);
                    span_mask.push(false);
                }
            }
        }

        // Build ndarray tensors
        let input_ids_arr = Array2::from_shape_vec((1, seq_len), input_ids)
            .map_err(|e| RelexError::Inference(e.to_string()))?;
        let attention_mask_arr = Array2::from_shape_vec((1, seq_len), attention_mask)
            .map_err(|e| RelexError::Inference(e.to_string()))?;
        let words_mask_arr = Array2::from_shape_vec((1, seq_len), words_mask)
            .map_err(|e| RelexError::Inference(e.to_string()))?;
        let text_lengths_arr = Array2::from_shape_vec((1, 1), vec![num_words as i64])
            .map_err(|e| RelexError::Inference(e.to_string()))?;
        let span_idx_arr = Array3::from_shape_vec((1, num_spans, 2), span_indices)
            .map_err(|e| RelexError::Inference(e.to_string()))?;
        let span_mask_arr = Array2::from_shape_vec((1, num_spans), span_mask)
            .map_err(|e| RelexError::Inference(e.to_string()))?;

        // Convert to ort Values
        let v_ids =
            Tensor::from_array(input_ids_arr).map_err(|e| RelexError::Inference(e.to_string()))?;
        let v_attn = Tensor::from_array(attention_mask_arr)
            .map_err(|e| RelexError::Inference(e.to_string()))?;
        let v_wmask =
            Tensor::from_array(words_mask_arr).map_err(|e| RelexError::Inference(e.to_string()))?;
        let v_tlen = Tensor::from_array(text_lengths_arr)
            .map_err(|e| RelexError::Inference(e.to_string()))?;
        let v_sidx =
            Tensor::from_array(span_idx_arr).map_err(|e| RelexError::Inference(e.to_string()))?;
        let v_smask =
            Tensor::from_array(span_mask_arr).map_err(|e| RelexError::Inference(e.to_string()))?;

        let inputs = ort::inputs![
            "input_ids" => v_ids,
            "attention_mask" => v_attn,
            "words_mask" => v_wmask,
            "text_lengths" => v_tlen,
            "span_idx" => v_sidx,
            "span_mask" => v_smask,
        ]
        .map_err(|e| RelexError::Inference(e.to_string()))?;

        let outputs = self
            .session
            .run(inputs)
            .map_err(|e| RelexError::Inference(e.to_string()))?;

        // Extract tensors into owned arrays before SessionOutputs is dropped
        let logits: ArrayD<f32> = outputs
            .get("logits")
            .ok_or_else(|| RelexError::Inference("missing 'logits' output".into()))?
            .try_extract_tensor::<f32>()
            .map_err(|e| RelexError::Inference(format!("logits tensor: {}", e)))?
            .into_owned();

        let rel_idx = outputs
            .get("rel_idx")
            .and_then(|v| v.try_extract_tensor::<i64>().ok())
            .map(|t| t.into_owned());
        let rel_logits = outputs
            .get("rel_logits")
            .and_then(|v| v.try_extract_tensor::<f32>().ok())
            .map(|t| t.into_owned());
        // rel_mask is output as bool by the ONNX model; convert to f32 for decode.
        let rel_mask = outputs.get("rel_mask").and_then(|v| {
            // Try bool first (matches ONNX export), fall back to f32
            if let Ok(t) = v.try_extract_tensor::<bool>() {
                let converted: ArrayD<f32> = t.mapv(|b| if b { 1.0f32 } else { 0.0 });
                Some(converted)
            } else {
                v.try_extract_tensor::<f32>().ok().map(|t| t.into_owned())
            }
        });

        let owned_words: Vec<(usize, usize, String)> = words
            .iter()
            .map(|&(s, e, w)| (s, e, w.to_string()))
            .collect();

        Ok(InferenceOutputs {
            logits,
            rel_idx,
            rel_logits,
            rel_mask,
            word_spans: owned_words,
            num_words,
        })
    }
}

/// Split text into words with byte offsets: (start, end, word_str).
fn split_words(text: &str) -> Vec<(usize, usize, &str)> {
    let mut words = Vec::new();
    let mut start = None;

    for (i, c) in text.char_indices() {
        if c.is_whitespace() {
            if let Some(s) = start {
                words.push((s, i, &text[s..i]));
                start = None;
            }
        } else if start.is_none() {
            start = Some(i);
        }
    }
    if let Some(s) = start {
        words.push((s, text.len(), &text[s..]));
    }
    words
}

/// Decode entity spans from the logits tensor.
///
/// logits shape: [batch=1, num_words, max_width, num_entity_classes]
/// Each value is the score for span (word_i, word_i+width) being entity class c.
fn decode_entities(
    logits: &ndarray::ArrayViewD<f32>,
    word_spans: &[(usize, usize, String)],
    num_words: usize,
    text: &str,
    entity_labels: &[&str],
    threshold: f32,
) -> Vec<ExtractedEntity> {
    let shape = logits.shape();
    // shape: [1, num_words, max_width, num_classes]
    if shape.len() != 4 {
        return Vec::new();
    }

    let _batch = shape[0];
    let n_words = shape[1];
    let max_w = shape[2];
    let n_classes = shape[3];

    let mut entities = Vec::new();

    for word_start in 0..n_words.min(num_words) {
        for width in 0..max_w.min(num_words - word_start) {
            let word_end = word_start + width;

            for class_idx in 0..n_classes.min(entity_labels.len()) {
                let score = logits[[0, word_start, width, class_idx]];
                let prob = sigmoid(score);

                if prob >= threshold {
                    // Convert word indices to character offsets
                    if word_start < word_spans.len() && word_end < word_spans.len() {
                        let char_start = word_spans[word_start].0;
                        let char_end = word_spans[word_end].1;

                        if char_end <= text.len() {
                            let span_text = text[char_start..char_end].to_string();
                            entities.push(ExtractedEntity {
                                text: span_text,
                                entity_type: entity_labels[class_idx].to_string(),
                                span_start: char_start,
                                span_end: char_end,
                                confidence: prob as f64,
                            });
                        }
                    }
                }
            }
        }
    }

    // Greedy dedup: for overlapping spans, keep highest confidence
    entities.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
    let mut used_ranges: Vec<(usize, usize)> = Vec::new();
    entities.retain(|e| {
        let overlaps = used_ranges
            .iter()
            .any(|&(s, end)| e.span_start < end && e.span_end > s);
        if !overlaps {
            used_ranges.push((e.span_start, e.span_end));
            true
        } else {
            false
        }
    });

    entities
}

/// Decode relations from model outputs.
///
/// rel_idx shape:    [batch=1, num_pairs, 2]     — indices into entity list
/// rel_logits shape: [batch=1, num_pairs, num_rel_classes] — scores per relation type
/// rel_mask shape:   [batch=1, num_pairs]         — valid pair indicator
fn decode_relations(
    rel_idx: &ndarray::ArrayViewD<i64>,
    rel_logits: &ndarray::ArrayViewD<f32>,
    rel_mask: &ndarray::ArrayViewD<f32>,
    entities: &[ExtractedEntity],
    relation_labels: &[&str],
    threshold: f32,
    schema: &ExtractionSchema,
) -> Vec<ExtractedRelation> {
    let shape = rel_logits.shape();
    if shape.len() != 3 {
        return Vec::new();
    }

    let num_pairs = shape[1];
    let num_rel_classes = shape[2];

    let mut relations = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for pair_idx in 0..num_pairs {
        // Check mask
        let mask_val = rel_mask[[0, pair_idx]];
        if mask_val < 0.5 {
            continue;
        }

        let head_idx = rel_idx[[0, pair_idx, 0]] as usize;
        let tail_idx = rel_idx[[0, pair_idx, 1]] as usize;

        if head_idx >= entities.len() || tail_idx >= entities.len() {
            continue;
        }

        let head_entity = &entities[head_idx];
        let tail_entity = &entities[tail_idx];

        if head_entity.text == tail_entity.text {
            continue;
        }

        // Find best relation type for this pair
        for rel_idx_inner in 0..num_rel_classes.min(relation_labels.len()) {
            let score = rel_logits[[0, pair_idx, rel_idx_inner]];
            let prob = sigmoid(score);

            if prob >= threshold {
                let relation = relation_labels[rel_idx_inner];

                // Validate against schema
                if let Some(spec) = schema.relation_types.get(relation) {
                    let valid = spec.head.contains(&head_entity.entity_type)
                        && spec.tail.contains(&tail_entity.entity_type);
                    // Also check reverse direction
                    let valid_rev = spec.head.contains(&tail_entity.entity_type)
                        && spec.tail.contains(&head_entity.entity_type);

                    if !valid && !valid_rev {
                        continue;
                    }

                    let (h, t) = if valid {
                        (head_entity.text.clone(), tail_entity.text.clone())
                    } else {
                        (tail_entity.text.clone(), head_entity.text.clone())
                    };

                    let key = (h.clone(), relation.to_string(), t.clone());
                    if seen.insert(key) {
                        relations.push(ExtractedRelation {
                            head: h,
                            relation: relation.to_string(),
                            tail: t,
                            confidence: prob as f64,
                        });
                    }
                }
            }
        }
    }

    relations
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[derive(Debug, thiserror::Error)]
pub enum RelexError {
    #[error("failed to load relex model: {0}")]
    ModelLoad(String),

    #[error("relex inference error: {0}")]
    Inference(String),
}
