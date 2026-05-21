use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};

use crate::coref::CorefResolver;
use crate::ner::{ExtractedEntity, NerEngine, NerError};
use crate::rel::{ExtractedRelation, RelEngine, RelError};
use crate::remap;
use crate::schema::{ExtractionSchema, SchemaError};
use crate::temporal::{self, TemporalResult};

/// Complete result of running the extraction pipeline on a piece of text.
#[derive(Debug, Clone)]
pub struct ExtractionResult {
    pub entities: Vec<ExtractedEntity>,
    pub relations: Vec<ExtractedRelation>,
    pub temporal: Vec<TemporalResult>,
}

/// The extraction pipeline orchestrates NER, relation extraction, and temporal parsing.
///
/// Created once and reused across multiple episodes. Model loading happens at construction
/// time (~100-500ms), but subsequent inference calls are fast (<15ms).
pub struct ExtractionPipeline {
    schema: ExtractionSchema,
    ner: NerEngine,
    rel: RelEngine,
    confidence_threshold: f64,
    /// NuNER Zero requires lowercase labels; GLiNER uses PascalCase.
    ner_needs_lowercase: bool,
    /// Optional LLM extractor (enabled when `OPENROUTER_API_KEY` is set).
    llm: Option<crate::llm_extract::LlmExtractor>,
}

impl ExtractionPipeline {
    /// Create a new extraction pipeline.
    ///
    /// - `schema`: Entity/relation type definitions.
    /// - `models_dir`: Directory containing ONNX model files.
    /// - `confidence_threshold`: Minimum confidence to keep an extraction (default: 0.5).
    pub fn new(
        schema: ExtractionSchema,
        models_dir: &Path,
        confidence_threshold: f64,
    ) -> Result<Self, PipelineError> {
        // Locate NER model files (NuNER Zero-span preferred, GLiNER v2.1 fallback)
        let ner_model = find_ner_model(models_dir)?;
        let ner_tokenizer = find_tokenizer(models_dir, "gliner")?;

        // Use the pipeline's confidence_threshold as the GLiNER model-level threshold too.
        // Parameters::default() hardcodes 0.5 which silently drops low-confidence spans
        // before we ever see them; pass our threshold so callers control the cutoff.
        let ner = NerEngine::new(&ner_model, &ner_tokenizer, confidence_threshold as f32)
            .map_err(PipelineError::Ner)?;

        // NuNER Zero requires lowercase labels — detect by model path.
        let ner_needs_lowercase = ner_model.display().to_string().contains("NuNER");

        // Locate relation model files (multitask GLiNER) — optional
        let rel_model = find_rel_model(models_dir);
        let rel_tokenizer = find_tokenizer(models_dir, "multitask").ok();

        let rel = RelEngine::new(rel_model.as_deref(), rel_tokenizer.as_deref())
            .map_err(PipelineError::Rel)?;

        // LLM fallback — auto-enabled when env vars or config are set.
        let llm = crate::llm_extract::LlmExtractor::from_env();

        Ok(Self {
            schema,
            ner,
            rel,
            confidence_threshold,
            ner_needs_lowercase,
            llm,
        })
    }

    /// Create a pipeline with explicit LLM config (from ctxgraph.toml).
    pub fn with_llm_config(
        schema: ExtractionSchema,
        models_dir: &Path,
        confidence_threshold: f64,
        llm_config: &crate::llm_extract::LlmConfig,
    ) -> Result<Self, PipelineError> {
        let mut pipeline = Self::new(schema, models_dir, confidence_threshold)?;

        // Override LLM from config (takes precedence over env vars)
        if !llm_config.provider.is_empty() || !llm_config.api_key.is_empty() {
            pipeline.llm = crate::llm_extract::LlmExtractor::from_config(llm_config);
        }

        Ok(pipeline)
    }

    /// Create a pipeline with default settings.
    ///
    /// Uses `ExtractionSchema::default()` and 0.5 confidence threshold.
    pub fn with_defaults(models_dir: &Path) -> Result<Self, PipelineError> {
        Self::new(ExtractionSchema::default(), models_dir, 0.5)
    }

    /// Extract entities, relations, and temporal expressions from text.
    pub fn extract(
        &self,
        text: &str,
        reference_time: DateTime<Utc>,
    ) -> Result<ExtractionResult, PipelineError> {
        // Step 1: NER — extract entities.
        // NuNER Zero requires lowercase labels; GLiNER v2.1 uses PascalCase keys.
        // When using NuNER, we lowercase labels and build a reverse mapping to
        // recover canonical entity type names (e.g. "person" → "Person").
        let canonical_labels: Vec<&str> = self.schema.entity_labels();
        let labels_owned;
        let label_map;
        let labels: Vec<&str>;
        let label_to_type: Option<std::collections::HashMap<&str, &str>>;

        if self.ner_needs_lowercase {
            labels_owned = canonical_labels
                .iter()
                .map(|l| l.to_lowercase())
                .collect::<Vec<_>>();
            labels = labels_owned.iter().map(|s| s.as_str()).collect();
            label_map = labels_owned
                .iter()
                .zip(canonical_labels.iter())
                .map(|(lower, &canon)| (lower.as_str(), canon))
                .collect::<std::collections::HashMap<&str, &str>>();
            label_to_type = Some(label_map);
        } else {
            labels = canonical_labels;
            label_to_type = None;
        }

        let mut entities = self
            .ner
            .extract(text, &labels, label_to_type.as_ref())
            .map_err(PipelineError::Ner)?;

        // Filter by confidence
        entities.retain(|e| e.confidence >= self.confidence_threshold);

        // Step 1b: Coreference resolution — resolve pronouns to preceding entities
        let coref_entities = CorefResolver::resolve(text, &entities);
        entities.extend(coref_entities);

        // Step 1c: Supplement entities — dictionary-based detection for known names
        // that GLiNER missed, boosting recall from ~0.59 toward ~0.75+
        remap::supplement_entities(text, &mut entities);

        // Step 1d: Entity type remapping — fix common GLiNER misclassifications
        // using domain knowledge lookup tables (Database, Infrastructure, Pattern, etc.)
        remap::remap_entity_types(&mut entities);

        // Step 1e: Canonicalize entity names — strip generic suffixes like
        // " pattern", " framework", " modules" when the base name is a known
        // entity or a proper noun (e.g., "CQRS pattern" → "CQRS").
        remap::canonicalize_entities(&mut entities);

        // Step 1f: Deduplicate overlapping spans — when the model extracts both
        // "CQRS" and "CQRS pattern" at overlapping positions, keep the best one.
        remap::deduplicate_overlapping(&mut entities);

        // Step 1f2: Strip leading articles ("the", "a", "an") from all entity names.
        // GLiNER sometimes extracts "the plant manager" instead of "plant manager".
        for ent in &mut entities {
            for prefix in ["the ", "The ", "a ", "A ", "an ", "An "] {
                if ent.text.starts_with(prefix) && ent.text.len() > prefix.len() + 1 {
                    ent.span_start += prefix.len();
                    ent.text = ent.text[prefix.len()..].to_string();
                    break;
                }
            }
        }

        // Step 1g: LLM entity cleanup — DISABLED.
        // Tested both local Ollama (qwen2.5:3b) and GPT-4.1-mini for cleaning
        // up GLiNER entity names. Both hurt full benchmark F1:
        //   - Ollama: entity 0.845→0.787, combined 0.678→0.630
        //   - GPT:    entity 0.845→0.811, combined 0.700→0.659
        // Root cause: most entities are already correct; cleanup removes/renames
        // valid multi-word entities like "saga pattern", "Cloudflare Workers".
        // The approach helps ~5 hard cases but hurts ~15 others.
        // Future: try full LLM entity extraction (Graphiti-style) instead of
        // cleanup, or only apply cleanup to entities with low confidence.

        // Step 1h: LLM fallback — when GLiNER confidence is low, call LLM for
        // BOTH entities AND relations. The LLM understands cross-domain language
        // that local models miss. One call handles both.
        let mut llm_relations: Vec<ExtractedRelation> = Vec::new();
        if let Some(llm) = &self.llm {
            // Smart gate: call LLM when GLiNER output looks weak.
            // Tech text: GLiNER finds 4-6 entities with >0.5 confidence → skip LLM
            // Cross-domain: fewer entities, lower confidence, more unknown types → call LLM
            let avg_confidence = if entities.is_empty() {
                0.0
            } else {
                entities.iter().map(|e| e.confidence).sum::<f64>() / entities.len() as f64
            };
            let word_count = text.split_whitespace().count();
            let entity_density = entities.len() as f64 / (word_count as f64 / 10.0);

            // Count how many entities the schema recognizes as valid types
            let valid_type_count = entities
                .iter()
                .filter(|e| self.schema.entity_types.contains_key(&e.entity_type))
                .count();
            let valid_ratio = if entities.is_empty() {
                0.0
            } else {
                valid_type_count as f64 / entities.len() as f64
            };

            // Escalate when ANY of:
            // - Few entities for text length (density < 1.5 per 10 words)
            // - Low average confidence (GLiNER unsure)
            // - Low valid type ratio (GLiNER assigning wrong types)
            // - Text is complex (contains version numbers, URLs, stack traces)
            //   which suggests messy real-world data where GLiNER struggles
            let unique_entity_count = {
                let mut names: Vec<String> =
                    entities.iter().map(|e| e.text.to_lowercase()).collect();
                names.sort();
                names.dedup();
                names.len()
            };
            let text_lower = text.to_lowercase();
            let looks_complex = text_lower.contains('@')
                || text_lower.contains("v2")
                || text_lower.contains("v3")
                || text_lower.contains("#")
                || text_lower.contains("::")
                || text_lower.contains("->")
                || text_lower.contains("stack")
                || text_lower.contains("error:")
                || text_lower.contains("broke")
                || text_lower.contains("outage")
                || text_lower.contains("incident");
            let very_sparse = word_count > 25 && unique_entity_count < 5;

            let should_escalate = entity_density < 1.5
                || avg_confidence < 0.4
                || valid_ratio < 0.6
                || very_sparse
                || looks_complex;

            if should_escalate {
                // LLM extracts BOTH entities and relations in one call.
                //
                // When the `cloakpipe` feature is enabled, PII is stripped before
                // sending text to the LLM, then entity names are mapped back.
                #[cfg(feature = "cloakpipe")]
                let (llm_text, pii_mappings) = crate::llm_extract::pseudonymize_for_llm(text);
                #[cfg(not(feature = "cloakpipe"))]
                let llm_text = text;

                let llm_result_try = llm.extract(llm_text, &self.schema);
                if let Err(ref e) = llm_result_try {
                    eprintln!("[ctxgraph] LLM escalation failed: {e}");
                }
                if let Ok(llm_result) = llm_result_try {
                    // Rehydrate PII in entity names when CloakPipe is active
                    #[cfg(feature = "cloakpipe")]
                    {
                        llm_result.entities = crate::llm_extract::rehydrate_entities(
                            llm_result.entities,
                            &pii_mappings,
                        );
                        // Also rehydrate relation head/tail names
                        for rel in &mut llm_result.relations {
                            if let Some(orig) = pii_mappings.get(&rel.head) {
                                rel.head = orig.clone();
                            }
                            if let Some(orig) = pii_mappings.get(&rel.tail) {
                                rel.tail = orig.clone();
                            }
                        }
                    }
                    // Merge LLM entities not already found locally
                    let existing_names: std::collections::HashSet<String> =
                        entities.iter().map(|e| e.text.to_lowercase()).collect();
                    for ent in llm_result.entities {
                        if !existing_names.contains(&ent.text.to_lowercase()) {
                            entities.push(ent);
                        }
                    }

                    // Store LLM relations — will merge with GLiREL after step 2
                    llm_relations = llm_result.relations;
                }
            }
        }

        // Step 2: Relation extraction (GLiREL — always local, domain-agnostic)
        let mut relations = self
            .rel
            .extract(text, &entities, &self.schema)
            .map_err(PipelineError::Rel)?;

        // Filter by confidence
        relations.retain(|r| r.confidence >= self.confidence_threshold);

        // Step 2b: Merge LLM relations (if gate fired) — LLM relations supplement
        // GLiREL, adding relations that zero-shot scoring missed.
        if !llm_relations.is_empty() {
            let existing_rels: std::collections::HashSet<(String, String)> = relations
                .iter()
                .map(|r| (r.head.to_lowercase(), r.tail.to_lowercase()))
                .collect();
            for rel in llm_relations {
                let key = (rel.head.to_lowercase(), rel.tail.to_lowercase());
                let key_rev = (rel.tail.to_lowercase(), rel.head.to_lowercase());
                if !existing_rels.contains(&key) && !existing_rels.contains(&key_rev) {
                    relations.push(rel);
                }
            }
        }

        // Step 3: Temporal parsing
        let temporal = temporal::parse_temporal(text, reference_time);

        Ok(ExtractionResult {
            entities,
            relations,
            temporal,
        })
    }

    /// Get the schema used by this pipeline.
    pub fn schema(&self) -> &ExtractionSchema {
        &self.schema
    }

    /// Get the confidence threshold.
    pub fn confidence_threshold(&self) -> f64 {
        self.confidence_threshold
    }
}

/// Find the NER ONNX model file in the models directory.
///
/// GLiNER v2.1 is the default (best on tech benchmark). NuNER Zero-span is
/// supported as an alternative — better on cross-domain (+40% entity F1) but
/// slightly lower on tech. Users can switch by downloading NuNER and removing
/// the GLiNER model directory.
fn find_ner_model(models_dir: &Path) -> Result<PathBuf, PipelineError> {
    let candidates = [
        // GLiNER v2.1 (default — best on tech benchmark)
        models_dir.join("gliner_large-v2.1/onnx/model_int8.onnx"),
        models_dir.join("gliner_large-v2.1/onnx/model.onnx"),
        // NuNER Zero-span (alternative — better cross-domain, same architecture)
        models_dir.join("NuNER_Zero-span/onnx/model_int8.onnx"),
        models_dir.join("NuNER_Zero-span/onnx/model.onnx"),
        models_dir.join("gliner2-large-q8.onnx"),
    ];

    for c in &candidates {
        if c.exists() {
            return Ok(c.clone());
        }
    }

    Err(PipelineError::ModelNotFound {
        model: "GLiNER v2.1 NER".into(),
        searched: candidates.iter().map(|p| p.display().to_string()).collect(),
    })
}

/// Find the relation extraction model (token-level multitask GLiNER).
///
/// NOTE: gline-rs RelationPipeline requires a **token-level** model (4 inputs:
/// input_ids, attention_mask, words_mask, text_lengths). Span-level models like
/// `gliner_multi-v2.1` are NOT compatible and must not be listed here.
///
/// Compatible model: `knowledgator/gliner-multitask-large-v0.5` (token_level mode).
/// Pre-converted ONNX available from `onnx-community/gliner-multitask-large-v0.5`.
fn find_rel_model(models_dir: &Path) -> Option<PathBuf> {
    let candidates = [
        // INT8 quantized (from onnx-community, downloaded by ModelManager)
        models_dir.join("gliner-multitask-large-v0.5/onnx/model_int8.onnx"),
        // Full precision (from manual conversion via scripts/convert_model.py)
        models_dir.join("gliner-multitask-large-v0.5/onnx/model.onnx"),
        // Legacy flat layout
        models_dir.join("gliner-multitask-large.onnx"),
    ];

    candidates.into_iter().find(|c| c.exists())
}

/// Find a tokenizer.json file associated with a model.
fn find_tokenizer(models_dir: &Path, prefix: &str) -> Result<PathBuf, PipelineError> {
    let candidates = if prefix == "gliner" {
        vec![
            models_dir.join("NuNER_Zero-span/tokenizer.json"),
            models_dir.join("gliner_large-v2.1/tokenizer.json"),
            models_dir.join("tokenizer.json"),
        ]
    } else if prefix == "multitask" {
        vec![
            models_dir.join("gliner-multitask-large-v0.5/tokenizer.json"),
            models_dir.join("tokenizer.json"),
        ]
    } else {
        vec![models_dir.join("tokenizer.json")]
    };

    for c in &candidates {
        if c.exists() {
            return Ok(c.clone());
        }
    }

    Err(PipelineError::ModelNotFound {
        model: format!("{prefix} tokenizer"),
        searched: candidates.iter().map(|p| p.display().to_string()).collect(),
    })
}

#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("NER error: {0}")]
    Ner(#[from] NerError),

    #[error("relation extraction error: {0}")]
    Rel(#[from] RelError),

    #[error("schema error: {0}")]
    Schema(#[from] SchemaError),

    #[error("model not found: {model}. Searched: {searched:?}")]
    ModelNotFound {
        model: String,
        searched: Vec<String>,
    },
}
