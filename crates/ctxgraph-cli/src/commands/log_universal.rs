//! `ctxgraph log --universal "text"` — log an episode using the v0.3 universal
//! pipeline (Pieces 1, 2, 5A from `docs/CLARITY.md`).
//!
//! Differs from `commands::log` in that it:
//!   * loads `schemas/universal.toml` instead of the legacy tech 10/9 schema
//!   * uses `prompts/extract.txt` as the system prompt
//!   * calls the LLM directly (no GLiNER local tier — universal schema is
//!     LLM-first per CLARITY § 4 Mode B/C)
//!   * persists `suggestions` to `.ctxgraph/schema_suggestions.json` for
//!     Piece 5 Layer B promotion review

use std::env;
use std::path::{Path, PathBuf};

use ctxgraph::{CtxGraphError, Edge, Entity, Episode};
use ctxgraph_extract::llm_extract::LlmExtractor;
use ctxgraph_extract::universal_pipeline::UniversalPipeline;

use super::open_graph_no_extraction;

pub fn run(text: String, source: Option<String>, tags: Option<String>) -> ctxgraph::Result<()> {
    // Find schema + prompt next to the binary's source tree, or via env vars.
    let schema_path = resolve_path(
        "CTXGRAPH_UNIVERSAL_SCHEMA",
        "crates/ctxgraph-extract/schemas/universal.toml",
    )?;
    let prompt_path = resolve_path(
        "CTXGRAPH_UNIVERSAL_PROMPT",
        "crates/ctxgraph-extract/prompts/extract.txt",
    )?;

    let llm = LlmExtractor::from_env().ok_or_else(|| {
        CtxGraphError::Extraction(
            "no LLM backend available. Set OPENROUTER_API_KEY, CEREBRAS_API_KEY, \
             OPENAI_API_KEY, or ANTHROPIC_API_KEY (or run Ollama locally)."
                .into(),
        )
    })?;

    let mut graph = open_graph_no_extraction()?;
    let db_path = graph.db_path().to_path_buf();
    let suggestions_log = db_path
        .parent()
        .map(|p| p.join("schema_suggestions.json"))
        .unwrap_or_else(|| PathBuf::from("schema_suggestions.json"));

    let pipeline = UniversalPipeline::new(&schema_path, &prompt_path, llm)
        .map_err(|e| CtxGraphError::Extraction(e.to_string()))?
        .with_suggestions_log(suggestions_log.clone());

    // Build + store the episode first (no auto-extraction since pipeline isn't loaded)
    let mut builder = Episode::builder(&text);
    if let Some(src) = &source {
        builder = builder.source(src);
    }
    if let Some(tags_str) = &tags {
        for tag in tags_str.split(',') {
            builder = builder.tag(tag.trim());
        }
    }
    let episode = builder.build();
    let episode_id = episode.id.clone();
    let result = graph.add_episode(episode)?;
    println!("Episode stored: {}", &result.episode_id[..8]);

    // Now run universal extraction
    let extraction = pipeline
        .extract(&text)
        .map_err(|e| CtxGraphError::Extraction(e.to_string()))?;

    // Persist entities (with fuzzy dedup) and remember the resolved ID per
    // universal entity ID (e1, e2, …) so we can wire relations.
    let mut id_map: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    let mut entities_added = 0usize;

    for ent in &extraction.entities {
        let new_entity = Entity::new(&ent.name, &ent.entity_type);
        let (db_id, merged) = graph.add_entity_deduped(new_entity, 0.85)?;
        if !merged {
            entities_added += 1;
        }
        id_map.insert(ent.id.clone(), db_id);
    }

    // Persist relations (resolving e1, e2, … to real DB IDs)
    let mut edges_added = 0usize;
    let mut skipped_unresolved = 0usize;
    for rel in &extraction.relations {
        let source_db = id_map.get(&rel.head);
        let target_db = id_map.get(&rel.tail);
        match (source_db, target_db) {
            (Some(s), Some(t)) => {
                let mut edge = Edge::new(s, t, &rel.relation);
                edge.confidence = rel.confidence as f64;
                edge.episode_id = Some(episode_id.clone());
                graph.add_edge(edge)?;
                edges_added += 1;
            }
            _ => skipped_unresolved += 1,
        }
    }

    // Item 2: Execute invalidates — match the LLM's natural-language
    // invalidation descriptions against current edges touching the entities
    // mentioned in this episode. Soft match: ≥50% word overlap on
    // head+relation+tail+fact.
    let mut invalidated_count = 0usize;
    if !extraction.invalidates.is_empty() {
        // Collect candidate edges: those touching any entity from this episode.
        // The entity IDs in id_map are the DB-resolved ones for this episode.
        use std::collections::HashSet;
        let touched_ids: HashSet<String> = id_map.values().cloned().collect();
        let mut candidates: Vec<(String, String, String, String, Option<String>)> = Vec::new();
        let mut seen_edges: HashSet<String> = HashSet::new();
        for tid in &touched_ids {
            let edges = graph.get_edges_for_entity(tid)?;
            for e in edges {
                if !e.is_current() {
                    continue;
                }
                if seen_edges.contains(&e.id) {
                    continue;
                }
                seen_edges.insert(e.id.clone());
                // Look up source/target entity names
                let src = graph
                    .get_entity(&e.source_id)?
                    .map(|x| x.name)
                    .unwrap_or_default();
                let tgt = graph
                    .get_entity(&e.target_id)?
                    .map(|x| x.name)
                    .unwrap_or_default();
                candidates.push((e.id, src, e.relation, tgt, e.fact));
            }
        }
        let to_invalidate = extraction.invalidate_candidates(&candidates);
        for edge_id in &to_invalidate {
            graph.invalidate_edge(edge_id)?;
            invalidated_count += 1;
        }
    }

    println!(
        "  Universal extraction: {} entities ({} new), {} relations, {} suggestions, confidence={:.2}",
        extraction.entities.len(),
        entities_added,
        edges_added,
        extraction.suggestions.len(),
        extraction.confidence,
    );
    if skipped_unresolved > 0 {
        println!(
            "  Note: {skipped_unresolved} relation(s) skipped (head/tail id not in entities list)"
        );
    }
    if !extraction.invalidates.is_empty() {
        println!(
            "  {} invalidation hint(s) emitted; {} edge(s) invalidated",
            extraction.invalidates.len(),
            invalidated_count,
        );
    }
    if !extraction.suggestions.is_empty() {
        println!(
            "  Schema suggestions logged → {}",
            suggestions_log.display()
        );
    }

    Ok(())
}

fn resolve_path(env_var: &str, fallback_rel: &str) -> ctxgraph::Result<PathBuf> {
    if let Ok(val) = env::var(env_var) {
        let p = PathBuf::from(val);
        if p.is_file() {
            return Ok(p);
        }
    }
    // Try repo-root relative (running from a checkout)
    if let Ok(cwd) = env::current_dir() {
        let candidates = [
            cwd.join(fallback_rel),
            cwd.parent()
                .map(|p| p.join(fallback_rel))
                .unwrap_or_default(),
            cwd.parent()
                .and_then(Path::parent)
                .map(|p| p.join(fallback_rel))
                .unwrap_or_default(),
        ];
        for c in &candidates {
            if c.is_file() {
                return Ok(c.clone());
            }
        }
    }
    Err(CtxGraphError::NotFound(format!(
        "{fallback_rel} not found. Set ${env_var} to its path."
    )))
}
