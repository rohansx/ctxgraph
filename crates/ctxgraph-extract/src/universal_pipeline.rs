//! Universal pipeline — the v0.3 LLM-first extraction path.
//!
//! Implements Pieces 1 + 2 from `docs/CLARITY.md`:
//!   1. Loads `schemas/universal.toml` (9 entity types, 10 relations)
//!   2. Loads `prompts/extract.txt` (the single-call JSON-contract prompt)
//!   3. Calls the configured LLM via [`LlmExtractor::chat_completion`]
//!   4. Parses the universal JSON envelope (entities + relations +
//!      invalidates + suggestions + confidence)
//!   5. Persists `suggestions` to a side-table file (Piece 5 Layer A)
//!
//! This is the LLM-first path. The legacy GLiNER+GLiREL+tech-schema pipeline
//! in `pipeline.rs` is left untouched for backward compatibility.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::llm_extract::{LlmError, LlmExtractor};
use crate::schema::{ExtractionSchema, SchemaError};

#[derive(Debug, thiserror::Error)]
pub enum UniversalPipelineError {
    #[error("schema error: {0}")]
    Schema(#[from] SchemaError),

    #[error("LLM error: {0}")]
    Llm(#[from] LlmError),

    #[error("prompt file error: {path}: {source}")]
    PromptIo {
        path: String,
        source: std::io::Error,
    },

    #[error("response parse error: {0}\nRaw output: {1}")]
    Parse(String, String),

    #[error("suggestion log io error: {0}")]
    SuggestionLogIo(#[from] std::io::Error),
}

// ── Universal JSON envelope (matches `prompts/extract.txt`) ─────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalEntity {
    pub id: String,
    pub name: String,
    #[serde(rename = "type")]
    pub entity_type: String,
    #[serde(default)]
    pub attributes: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalRelation {
    pub head: String,
    pub relation: String,
    pub tail: String,
    #[serde(default = "default_confidence")]
    pub confidence: f32,
    #[serde(default)]
    pub valid_from: Option<String>,
    #[serde(default)]
    pub valid_to: Option<String>,
}

fn default_confidence() -> f32 {
    1.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalSuggestion {
    pub kind: String,
    pub name: String,
    #[serde(default)]
    pub supporting_entity_ids: Vec<String>,
    #[serde(default)]
    pub rationale: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalExtractionResult {
    #[serde(default)]
    pub entities: Vec<UniversalEntity>,
    #[serde(default)]
    pub relations: Vec<UniversalRelation>,
    #[serde(default)]
    pub invalidates: Vec<String>,
    #[serde(default)]
    pub suggestions: Vec<UniversalSuggestion>,
    #[serde(default = "default_confidence")]
    pub confidence: f32,
}

impl UniversalExtractionResult {
    /// Match each `invalidates` description against a candidate edge using
    /// substring matching on `head + relation + tail` and `fact`. Returns
    /// edge IDs the caller can pass to `Graph::invalidate_edge`. This is the
    /// description-based path (CLARITY § 5.1 long-form variant); the edge-ID
    /// variant requires a current-facts pre-pass and is wired in
    /// `Graph::invalidate_from_universal_extraction`.
    pub fn invalidate_candidates(
        &self,
        candidate_edges: &[(String, String, String, String, Option<String>)],
        // (edge_id, head_name, relation, tail_name, fact)
    ) -> Vec<String> {
        const STOP: &[&str] = &[
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "no", "not", "longer",
            "anymore", "now", "any", "this", "that", "of", "to", "in", "on", "at", "by", "for",
            "with", "and", "or", "has", "have", "had", "do", "does", "did",
        ];

        let mut out: Vec<String> = Vec::new();
        for descr in &self.invalidates {
            let dl = descr.to_lowercase();
            // Drop stopwords + 1-char tokens; only "substantive" words count.
            let words: Vec<String> = dl
                .split_whitespace()
                .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
                .filter(|w| w.len() >= 2 && !STOP.contains(&w.as_str()))
                .collect();
            if words.len() < 2 {
                continue;
            }

            // Score each candidate; pick the best match above threshold.
            let mut best_id: Option<String> = None;
            let mut best_score = 0.0f32;
            for (id, h, r, t, f) in candidate_edges {
                let blob = format!(
                    "{} {} {} {}",
                    h.to_lowercase(),
                    r.to_lowercase().replace('_', " "),
                    t.to_lowercase(),
                    f.clone().unwrap_or_default().to_lowercase()
                );
                let hits = words.iter().filter(|w| blob.contains(w.as_str())).count();
                let score = hits as f32 / words.len() as f32;
                if score > best_score {
                    best_score = score;
                    best_id = Some(id.clone());
                }
            }
            if best_score >= 0.5
                && let Some(id) = best_id {
                    out.push(id);
                }
        }
        out
    }

    /// Resolve relation head/tail from entity IDs to entity names.
    /// Returns `(name_resolved_relations, unresolved_count)`.
    /// Unresolved means head/tail is an ID that wasn't in the entities list
    /// (caller can decide whether to drop or keep).
    pub fn resolved_relations(&self) -> (Vec<UniversalRelation>, usize) {
        use std::collections::HashMap;
        let id_to_name: HashMap<&str, &str> = self
            .entities
            .iter()
            .map(|e| (e.id.as_str(), e.name.as_str()))
            .collect();

        let mut unresolved = 0usize;
        let resolved = self
            .relations
            .iter()
            .map(|r| {
                let mut new_r = r.clone();
                if let Some(n) = id_to_name.get(r.head.as_str()) {
                    new_r.head = (*n).to_string();
                } else if r.head.starts_with('e') && r.head[1..].chars().all(|c| c.is_ascii_digit())
                {
                    // Looks like an unresolved ID
                    unresolved += 1;
                }
                if let Some(n) = id_to_name.get(r.tail.as_str()) {
                    new_r.tail = (*n).to_string();
                } else if r.tail.starts_with('e') && r.tail[1..].chars().all(|c| c.is_ascii_digit())
                {
                    unresolved += 1;
                }
                new_r
            })
            .collect();
        (resolved, unresolved)
    }
}

// ── Pipeline ───────────────────────────────────────────────────────

pub struct UniversalPipeline {
    pub schema: ExtractionSchema,
    pub prompt_template: String,
    pub llm: LlmExtractor,
    pub suggestions_log_path: Option<PathBuf>,
}

impl UniversalPipeline {
    /// Build from filesystem paths.
    pub fn new(
        schema_path: &Path,
        prompt_path: &Path,
        llm: LlmExtractor,
    ) -> Result<Self, UniversalPipelineError> {
        let schema = ExtractionSchema::load_universal(schema_path)?;
        let prompt_template =
            fs::read_to_string(prompt_path).map_err(|e| UniversalPipelineError::PromptIo {
                path: prompt_path.display().to_string(),
                source: e,
            })?;
        Ok(Self {
            schema,
            prompt_template,
            llm,
            suggestions_log_path: None,
        })
    }

    /// Build from in-memory schema + prompt (useful for tests).
    pub fn from_strings(
        schema: ExtractionSchema,
        prompt_template: String,
        llm: LlmExtractor,
    ) -> Self {
        Self {
            schema,
            prompt_template,
            llm,
            suggestions_log_path: None,
        }
    }

    /// Enable Piece 5 Layer A — append suggestions to a JSON file after each
    /// extraction. The file is created if absent. Format: a JSON array of
    /// per-call objects.
    pub fn with_suggestions_log(mut self, path: PathBuf) -> Self {
        self.suggestions_log_path = Some(path);
        self
    }

    /// Run the universal extraction on a single episode of text.
    pub fn extract(
        &self,
        episode_text: &str,
    ) -> Result<UniversalExtractionResult, UniversalPipelineError> {
        // The prompt template ends with `{episode_text}` — we just append the
        // user text in a separate user message so the system prompt stays
        // byte-for-byte stable (prefix-cache friendly per CLARITY § 6).
        let system_prompt = self
            .prompt_template
            .replace("{episode_text}", "")
            .trim_end()
            .to_string();
        let user_prompt = format!("Episode text:\n{episode_text}");

        let raw = self.llm.chat_completion(&system_prompt, &user_prompt)?;
        let json_text = extract_json_object(&raw);
        let parsed: UniversalExtractionResult = serde_json::from_str(json_text).map_err(|e| {
            UniversalPipelineError::Parse(e.to_string(), raw.chars().take(400).collect())
        })?;

        if let Some(path) = &self.suggestions_log_path {
            self.append_suggestions(path, &parsed)?;
        }

        Ok(parsed)
    }

    fn append_suggestions(
        &self,
        path: &Path,
        parsed: &UniversalExtractionResult,
    ) -> Result<(), UniversalPipelineError> {
        if parsed.suggestions.is_empty() {
            return Ok(());
        }

        let mut log: Vec<serde_json::Value> = if path.exists() {
            let content = fs::read_to_string(path)?;
            serde_json::from_str(&content).unwrap_or_default()
        } else {
            Vec::new()
        };
        let now = chrono::Utc::now().to_rfc3339();
        for sug in &parsed.suggestions {
            log.push(serde_json::json!({
                "logged_at": now,
                "suggestion": sug,
                "extraction_confidence": parsed.confidence,
            }));
        }
        fs::write(path, serde_json::to_string_pretty(&log).unwrap())?;
        Ok(())
    }
}

// ── JSON extraction helper (mirrors Python harness) ─────────────────

/// Pull the first JSON object out of a possibly-fence-wrapped LLM response.
fn extract_json_object(raw: &str) -> &str {
    // Strip <think>...</think> first
    let after_think = if let Some(end) = raw.find("</think>") {
        &raw[end + "</think>".len()..]
    } else {
        raw
    };

    // Strip markdown fences if present
    let s = if let Some(start) = after_think.find("```json") {
        let rest = &after_think[start + "```json".len()..];
        if let Some(end) = rest.find("```") {
            &rest[..end]
        } else {
            rest
        }
    } else if let Some(start) = after_think.find("```") {
        let rest = &after_think[start + "```".len()..];
        if let Some(end) = rest.find("```") {
            &rest[..end]
        } else {
            rest
        }
    } else {
        after_think
    };

    // Trim to first '{' / last '}'
    let trimmed = s.trim();
    if let (Some(start), Some(end)) = (trimmed.find('{'), trimmed.rfind('}'))
        && end > start {
            return &trimmed[start..=end];
        }
    trimmed
}
