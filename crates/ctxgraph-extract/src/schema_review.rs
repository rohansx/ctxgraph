//! Piece 5 Layer B — schema suggestion review and promotion.
//!
//! Reads `schema_suggestions.json` (the side-table populated by
//! `UniversalPipeline` during extraction) and produces a list of promotion
//! candidates per the CLARITY § 3 / Piece 5 thresholds:
//!
//!   - appears in ≥ K distinct episodes (default K = 5)
//!   - appears across ≥ M distinct domains/sources (default M = 3)
//!   - average confidence > 0.7
//!   - not semantically near an existing type (cosine sim < 0.85)
//!
//! Two output modes:
//!   - `ReviewReport::candidates()` — promotion-ready, all rules pass
//!   - `ReviewReport::near_misses()` — failed one rule, useful for human review

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::schema::ExtractionSchema;

#[derive(Debug, thiserror::Error)]
pub enum SchemaReviewError {
    #[error("io error reading {path}: {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },
    #[error("parse error: {0}")]
    Parse(String),
}

/// One row in the suggestion log. Permissive shape — accepts both the Python
/// harness format (`episode_domain`, `backend`) and the Rust pipeline format
/// (`logged_at`, `extraction_confidence`).
#[derive(Debug, Clone, Deserialize)]
pub struct LoggedSuggestion {
    /// Required: nested suggestion record.
    pub suggestion: SuggestionPayload,
    /// Optional metadata that we use for promotion rules when present.
    #[serde(default)]
    pub episode_idx: Option<u64>,
    #[serde(default)]
    pub episode_domain: Option<String>,
    #[serde(default)]
    pub backend: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub extraction_confidence: Option<f32>,
    #[serde(default)]
    pub logged_at: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SuggestionPayload {
    pub kind: String, // "entity_type" | "relation_type" | "relation"
    pub name: String,
    #[serde(default)]
    pub supporting_entity_ids: Vec<String>,
    #[serde(default)]
    pub rationale: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct PromotionRules {
    pub min_episodes: usize,
    pub min_domains: usize,
    pub min_avg_confidence: f32,
    pub max_cos_to_existing: f32,
}

impl Default for PromotionRules {
    fn default() -> Self {
        Self {
            min_episodes: 5,
            min_domains: 3,
            min_avg_confidence: 0.7,
            max_cos_to_existing: 0.85,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct CandidateRow {
    pub kind: String,
    pub name: String,
    pub n_episodes: usize,
    pub n_domains: usize,
    pub avg_confidence: f32,
    pub rationales: Vec<String>,
    /// All metadata reasons this candidate did or didn't pass.
    pub eval: CandidateEval,
}

#[derive(Debug, Clone, Serialize)]
pub struct CandidateEval {
    pub episodes_ok: bool,
    pub domains_ok: bool,
    pub confidence_ok: bool,
    pub passes_all: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReviewReport {
    pub rules: PromotionRules,
    pub n_log_entries: usize,
    pub n_unique_suggestions: usize,
    pub candidates: Vec<CandidateRow>,
    pub near_misses: Vec<CandidateRow>,
}

impl ReviewReport {
    /// Load `schema_suggestions.json`, apply promotion rules, return report.
    /// Note: this version does NOT use embedding similarity yet — the
    /// `max_cos_to_existing` rule is enforced post-hoc by a separate caller
    /// that has access to the embedding model (avoids forcing this module to
    /// depend on fastembed init in the cargo-test path).
    pub fn from_log<P: AsRef<Path>>(
        log_path: P,
        existing_schema: &ExtractionSchema,
        rules: PromotionRules,
    ) -> Result<Self, SchemaReviewError> {
        let path_str = log_path.as_ref().display().to_string();
        let content = fs::read_to_string(&log_path).map_err(|e| SchemaReviewError::Io {
            path: path_str.clone(),
            source: e,
        })?;
        let entries: Vec<LoggedSuggestion> = serde_json::from_str(&content)
            .map_err(|e| SchemaReviewError::Parse(format!("{e} (file: {path_str})")))?;

        Self::from_entries(entries, existing_schema, rules)
    }

    pub fn from_entries(
        entries: Vec<LoggedSuggestion>,
        existing_schema: &ExtractionSchema,
        rules: PromotionRules,
    ) -> Result<Self, SchemaReviewError> {
        let n_log_entries = entries.len();

        // Group by (kind, name) — case-insensitive on name.
        let mut groups: BTreeMap<(String, String), Vec<LoggedSuggestion>> = BTreeMap::new();
        for e in entries {
            let key = (e.suggestion.kind.clone(), e.suggestion.name.to_lowercase());
            groups.entry(key).or_default().push(e);
        }

        let mut candidates: Vec<CandidateRow> = Vec::new();
        let mut near_misses: Vec<CandidateRow> = Vec::new();

        // Existing type names (lowercased) for the "already in schema" check
        let existing_entity_types: BTreeSet<String> = existing_schema
            .entity_types
            .keys()
            .map(|k| k.to_lowercase())
            .collect();
        let existing_relation_types: BTreeSet<String> = existing_schema
            .relation_types
            .keys()
            .map(|k| k.to_lowercase())
            .collect();

        for ((kind, name_lower), rows) in groups {
            let display_name = rows
                .first()
                .map(|r| r.suggestion.name.clone())
                .unwrap_or_else(|| name_lower.clone());

            // Filter garbage: skip suggestions whose name is just an entity ID
            // like "e1", "e23" — the LLM occasionally mislabels an ID as a
            // suggestion name. Also skip empty/whitespace names.
            let trimmed = name_lower.trim();
            if trimmed.is_empty() {
                continue;
            }
            if trimmed.len() <= 4
                && trimmed.starts_with('e')
                && trimmed[1..].chars().all(|c| c.is_ascii_digit())
            {
                continue;
            }

            // Skip if the suggestion is identical to an existing schema entry
            let already_exists = match kind.as_str() {
                "entity_type" => existing_entity_types.contains(&name_lower),
                "relation_type" | "relation" => existing_relation_types.contains(&name_lower),
                _ => false,
            };
            if already_exists {
                continue;
            }

            let n_episodes = rows.len();
            let n_domains = rows
                .iter()
                .filter_map(|r| r.episode_domain.as_ref())
                .map(|s| s.to_lowercase())
                .collect::<BTreeSet<_>>()
                .len();

            let confidences: Vec<f32> = rows
                .iter()
                .filter_map(|r| r.extraction_confidence)
                .collect();
            let avg_confidence = if confidences.is_empty() {
                // No confidence data → assume default 1.0 so we don't unfairly fail
                1.0
            } else {
                confidences.iter().sum::<f32>() / confidences.len() as f32
            };

            let rationales: Vec<String> = rows
                .iter()
                .map(|r| r.suggestion.rationale.clone())
                .filter(|s| !s.is_empty())
                .take(3)
                .collect();

            let eval = CandidateEval {
                episodes_ok: n_episodes >= rules.min_episodes,
                domains_ok: n_domains >= rules.min_domains
                    // If no domains in the data, defer to the episode rule
                    || rows.iter().all(|r| r.episode_domain.is_none()),
                confidence_ok: avg_confidence >= rules.min_avg_confidence,
                passes_all: false,
            };
            let mut eval = eval;
            eval.passes_all = eval.episodes_ok && eval.domains_ok && eval.confidence_ok;

            let row = CandidateRow {
                kind,
                name: display_name,
                n_episodes,
                n_domains,
                avg_confidence,
                rationales,
                eval,
            };

            if row.eval.passes_all {
                candidates.push(row);
            } else if row.n_episodes >= 2 {
                // Near miss = appeared at least twice but failed some rule
                near_misses.push(row);
            }
        }

        // Sort by episode count desc for stable display
        candidates.sort_by_key(|c| std::cmp::Reverse(c.n_episodes));
        near_misses.sort_by_key(|c| std::cmp::Reverse(c.n_episodes));

        Ok(Self {
            rules,
            n_log_entries,
            n_unique_suggestions: candidates.len() + near_misses.len(),
            candidates,
            near_misses,
        })
    }

    /// Convenience pretty-print to stdout.
    pub fn print(&self) {
        println!("\nSchema review report");
        println!("  log entries: {}", self.n_log_entries);
        println!("  unique suggestions: {}", self.n_unique_suggestions);
        println!(
            "  rules: ≥{} eps, ≥{} domains, conf ≥{:.2}",
            self.rules.min_episodes, self.rules.min_domains, self.rules.min_avg_confidence
        );

        if self.candidates.is_empty() {
            println!("\n  Promotion candidates: (none yet)");
        } else {
            println!("\n  Promotion candidates ({}):", self.candidates.len());
            for c in &self.candidates {
                println!(
                    "    + {:<14} {:<30} eps={}  domains={}  avg_conf={:.2}",
                    format!("[{}]", c.kind),
                    c.name,
                    c.n_episodes,
                    c.n_domains,
                    c.avg_confidence
                );
            }
        }

        if !self.near_misses.is_empty() {
            println!("\n  Near-misses ({}):", self.near_misses.len());
            for c in self.near_misses.iter().take(10) {
                let why = [
                    if !c.eval.episodes_ok { "few-eps" } else { "" },
                    if !c.eval.domains_ok {
                        "few-domains"
                    } else {
                        ""
                    },
                    if !c.eval.confidence_ok {
                        "low-conf"
                    } else {
                        ""
                    },
                ]
                .iter()
                .filter(|s| !s.is_empty())
                .copied()
                .collect::<Vec<_>>()
                .join(",");
                println!(
                    "    ? {:<14} {:<30} eps={}  domains={}  avg_conf={:.2}  why={}",
                    format!("[{}]", c.kind),
                    c.name,
                    c.n_episodes,
                    c.n_domains,
                    c.avg_confidence,
                    why
                );
            }
        }
    }
}
