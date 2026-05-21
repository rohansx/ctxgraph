//! `ctxgraph query --universal "<question>"` — the v0.3 read-path (CLARITY § 5).
//!
//! Two-phase resolution:
//!   1. SIMPLE PATH (~90% target): regex-extract a verb + an entity hint from
//!      the query; cosine-match the verb to one of the 10 typed relations via
//!      `RelationMatcher`; run a deterministic SQL traversal from that entity
//!      filtered by the matched relation.
//!   2. FALLBACK: if no entity is found or verb confidence is below threshold,
//!      fall back to the existing fused FTS5 + semantic + graph search.
//!
//! The complex path (multi-hop / time filters / conjunctions via local
//! Qwen3-1.5B per CLARITY § 5 / Piece 4) is documented but not wired here —
//! requires a running Ollama with Qwen3-1.5B. It's a v0.3 W5 deliverable per
//! `docs/ROADMAP.md`.

use std::env;
use std::path::{Path, PathBuf};

use ctxgraph::{CtxGraphError, Entity};
use ctxgraph_extract::llm_extract::LlmExtractor;
use ctxgraph_extract::relation_match::RelationMatcher;
use ctxgraph_extract::schema::ExtractionSchema;
use serde::Deserialize;

use super::open_graph_no_extraction;

const VERB_CONFIDENCE_THRESHOLD: f32 = 0.15;
const COMPLEXITY_THRESHOLD: usize = 2;
// Time terms that mark a query as "complex" (need NL → graph-op parser)
const COMPLEX_MARKERS: &[&str] = &[
    "this week",
    "last week",
    "this month",
    "last month",
    "today",
    "yesterday",
    "before",
    "after",
    "since",
    "until",
    "and ",
    " or ",
    "compared to",
    "vs ",
    "between",
    "first",
    "earliest",
    "latest",
    "most",
    "all the",
    "every",
];

#[derive(Debug, Deserialize)]
struct GraphOp {
    op: String,
    #[serde(default)]
    head: Option<String>,
    #[serde(default)]
    relation: Option<String>,
    #[serde(default)]
    tail: Option<String>,
    #[serde(default)]
    filters: serde_json::Value,
}

pub fn run(query_text: String, limit: usize) -> ctxgraph::Result<()> {
    let schema_path = resolve_universal_schema()?;
    let schema = ExtractionSchema::load_universal(&schema_path)
        .map_err(|e| CtxGraphError::Extraction(format!("schema load: {e}")))?;
    let matcher = RelationMatcher::build_from_schema(&schema)
        .map_err(|e| CtxGraphError::Extraction(format!("relation matcher init: {e}")))?;

    let graph = open_graph_no_extraction()?;

    // ── Stage 1: parse the query ──────────────────────────────────
    let (verb, entity_hint) = split_verb_entity(&query_text, &graph)?;
    let m: Option<MatchView> = if !verb.is_empty() {
        let rm = matcher
            .resolve(&verb)
            .map_err(|e| CtxGraphError::Extraction(format!("verb match: {e}")))?;
        Some(rm.into())
    } else {
        None
    };

    println!("┌─ universal query path ─────────────────────────────────");
    println!("│ query : {query_text}");
    println!("│ verb  : {verb:?}");
    if let Some(rm) = &m {
        println!(
            "│ → relation: {} (score={:.3}, runner_up={} {:.3})",
            rm.relation, rm.score, rm.runner_up, rm.runner_up_score
        );
    }
    if let Some(e) = &entity_hint {
        println!(
            "│ entity hint: {} (type={}, id={})",
            e.name,
            e.entity_type,
            &e.id[..8]
        );
    }
    println!("└────────────────────────────────────────────────────────");

    // ── Stage 2: simple path ─ verb resolves + entity found ──────
    if let (Some(rel), Some(head)) = (m.as_ref(), entity_hint.as_ref())
        && rel.score >= VERB_CONFIDENCE_THRESHOLD {
            let (entities, edges) = graph.traverse(&head.id, 1)?;
            let connected: Vec<(String, Entity)> = edges
                .iter()
                .filter(|e| e.relation == rel.relation)
                .filter_map(|e| {
                    let other_id = if e.source_id == head.id {
                        e.target_id.as_str()
                    } else {
                        e.source_id.as_str()
                    };
                    entities
                        .iter()
                        .find(|x| x.id == other_id)
                        .cloned()
                        .map(|x| (e.relation.clone(), x))
                })
                .collect();

            println!(
                "\n  SIMPLE PATH results ({}, relation={})",
                connected.len(),
                rel.relation
            );
            for (rel_name, ent) in connected.iter().take(limit) {
                println!(
                    "    {} --{}--> {} [{}]",
                    head.name, rel_name, ent.name, ent.entity_type
                );
            }
            if !connected.is_empty() {
                println!();
                return Ok(());
            }
            println!(
                "  (no edges matched {} for {}; falling back to search)",
                rel.relation, head.name
            );
        }

    // ── Stage 2.5: complex path (only if simple path produced nothing and
    //     the query has structural markers — multi-hop, time filter, etc.).
    //     CLARITY § 5: ~10% of real queries hit this path. Uses the local
    //     query-parse prompt against any LLM backend (Cerebras/Ollama/OpenRouter).
    let is_complex = looks_complex(&query_text);
    if is_complex
        && let Some(parsed) = try_complex_path(&query_text) {
            println!("\n  COMPLEX PATH (NL → graph op)");
            println!(
                "    op={}, head={:?}, relation={:?}, tail={:?}",
                parsed.op, parsed.head, parsed.relation, parsed.tail
            );
            let hits = dispatch_op(&graph, &parsed, limit)?;
            if !hits.is_empty() {
                println!("\n  Results ({}):", hits.len());
                for (label, ent) in hits.iter().take(limit) {
                    println!("    {label}  {} [{}]", ent.name, ent.entity_type);
                }
                println!();
                return Ok(());
            }
            println!("    (no results for parsed op; falling back to search)");
        }

    // ── Stage 3: fallback fused search ───────────────────────────
    // FTS5 doesn't like punctuation. Strip everything except word chars.
    let safe_query: String = query_text
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == ' ' || c == '_' {
                c
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");

    println!("\n  FALLBACK fused search results:");
    let entity_hits = graph.search_entities(&safe_query, limit.min(10))?;
    if !entity_hits.is_empty() {
        println!("\n  Entities (top {}):", entity_hits.len());
        for (ent, score) in &entity_hits {
            println!("    {:.3}  {:<32} [{}]", score, ent.name, ent.entity_type);
        }
    }
    let episode_hits = graph.search(&safe_query, limit.min(5))?;
    if !episode_hits.is_empty() {
        println!("\n  Episodes (top {}):", episode_hits.len());
        for (ep, score) in &episode_hits {
            let preview: String = ep.content.chars().take(120).collect();
            println!("    {:.3}  [{}] {}", score, &ep.id[..8], preview);
        }
    }
    if entity_hits.is_empty() && episode_hits.is_empty() {
        println!("    (no results)");
    }
    Ok(())
}

// ── Helpers ───────────────────────────────────────────────────────

/// Hold the embedding match info without exposing the extract crate's struct
/// directly to keep this module's public surface narrow.
struct MatchView {
    relation: String,
    score: f32,
    runner_up: String,
    runner_up_score: f32,
}

impl From<ctxgraph_extract::relation_match::RelationMatch> for MatchView {
    fn from(m: ctxgraph_extract::relation_match::RelationMatch) -> Self {
        Self {
            relation: m.relation,
            score: m.score,
            runner_up: m.runner_up,
            runner_up_score: m.runner_up_score,
        }
    }
}

/// Split a natural-language query into a verb phrase and an optional entity.
/// Strategy:
///   1. Try every contiguous N-gram (longest first) against `get_entity_by_name`
///      and `search_entities`; the highest-scoring match becomes the entity.
///   2. Remove the matched span; the remainder is the verb phrase.
///   3. Strip common interrogatives + stopwords from the verb side.
fn split_verb_entity(
    query: &str,
    graph: &ctxgraph::Graph,
) -> ctxgraph::Result<(String, Option<Entity>)> {
    let cleaned = query
        .trim()
        .trim_end_matches('?')
        .trim_end_matches('.')
        .to_string();
    let tokens: Vec<&str> = cleaned.split_whitespace().collect();
    if tokens.is_empty() {
        return Ok((String::new(), None));
    }

    // Try N-grams from longest down to single token. Cap at 5 to keep it cheap.
    let max_n = tokens.len().min(5);
    let mut best_match: Option<(Entity, usize, usize)> = None; // (entity, start, end_inclusive)
    let mut best_score: f64 = 0.0;
    for n in (1..=max_n).rev() {
        for start in 0..=(tokens.len() - n) {
            let end = start + n;
            let span = tokens[start..end].join(" ");
            // Skip if the span is just stopwords / question words
            if is_pure_stopwords(&span) {
                continue;
            }
            // Exact name lookup (cheap)
            if let Ok(Some(e)) = exact_entity_lookup(graph, &span) {
                let score = (n as f64) * 2.0; // bias toward longest exact match
                if score > best_score {
                    best_score = score;
                    best_match = Some((e, start, end));
                }
            }
        }
        if best_match.is_some() {
            break; // longest n-gram match wins
        }
    }

    // Fallback: fuzzy search if no exact match found
    if best_match.is_none() {
        // Try the *whole* query against search_entities, take top hit
        if let Ok(hits) = graph.search_entities(&cleaned, 3)
            && let Some((e, score)) = hits.into_iter().next()
                && score > 0.0 {
                    best_match = Some((e, 0, tokens.len()));
                }
    }

    let (entity, verb) = match best_match {
        Some((e, start, end)) => {
            let mut verb_tokens: Vec<&str> = Vec::with_capacity(tokens.len());
            verb_tokens.extend_from_slice(&tokens[..start]);
            verb_tokens.extend_from_slice(&tokens[end..]);
            let verb_raw = verb_tokens.join(" ");
            (Some(e), strip_interrogatives(&verb_raw))
        }
        None => (None, strip_interrogatives(&cleaned)),
    };
    Ok((verb, entity))
}

fn exact_entity_lookup(graph: &ctxgraph::Graph, name: &str) -> ctxgraph::Result<Option<Entity>> {
    // Strip punctuation for FTS5 compatibility
    let safe: String = name
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == ' ' || c == '_' {
                c
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if safe.is_empty() {
        return Ok(None);
    }
    let hits = graph.search_entities(&safe, 5)?;
    let lower = name.to_lowercase();
    for (e, _) in hits {
        let en = e.name.to_lowercase();
        if en == lower || lower.contains(&en) || en.contains(&lower) {
            return Ok(Some(e));
        }
    }
    Ok(None)
}

fn is_pure_stopwords(s: &str) -> bool {
    const STOP: &[&str] = &[
        "what", "who", "where", "when", "why", "how", "the", "a", "an", "is", "are", "was", "were",
        "be", "do", "does", "did", "of", "to", "in", "on", "at", "by", "for", "with", "and", "or",
        "i", "you", "he", "she", "we", "they", "it",
    ];
    s.split_whitespace()
        .all(|w| STOP.contains(&w.to_lowercase().as_str()))
}

fn strip_interrogatives(s: &str) -> String {
    // Drop leading question words + obvious filler so the verb embedding has
    // a tight phrase to match against.
    let lower = s.to_lowercase();
    let mut out: Vec<&str> = lower.split_whitespace().collect();
    while let Some(&first) = out.first() {
        if matches!(
            first,
            "what"
                | "who"
                | "where"
                | "when"
                | "why"
                | "how"
                | "tell"
                | "me"
                | "show"
                | "list"
                | "find"
                | "does"
                | "do"
                | "did"
                | "is"
                | "are"
                | "was"
                | "were"
                | "the"
                | "a"
                | "an"
        ) {
            out.remove(0);
        } else {
            break;
        }
    }
    out.join(" ")
}

// ── Complex-path helpers ──────────────────────────────────────────

fn looks_complex(q: &str) -> bool {
    let lower = q.to_lowercase();
    if COMPLEX_MARKERS.iter().any(|m| lower.contains(m)) {
        return true;
    }
    // Heuristic 2: more than COMPLEXITY_THRESHOLD verbs/clauses
    let comma_count = lower.matches(',').count();
    if comma_count >= COMPLEXITY_THRESHOLD {
        return true;
    }
    false
}

fn try_complex_path(query_text: &str) -> Option<GraphOp> {
    // Need an LLM backend. If none configured, skip the complex path silently.
    let llm = LlmExtractor::from_env()?;

    // Load the few-shot prompt. Tolerate either bin-relative or repo-relative.
    let prompt = load_prompt_text(
        "CTXGRAPH_QUERY_PROMPT",
        "crates/ctxgraph-cli/prompts/query_parse.txt",
    )?;
    let today = chrono::Utc::now().format("%Y-%m-%d").to_string();
    let today_minus_7 = (chrono::Utc::now() - chrono::Duration::days(7))
        .format("%Y-%m-%d")
        .to_string();
    let system_prompt = prompt
        .replace("{user_query}", "")
        .replace("{today}", &today)
        .replace("{today_minus_7}", &today_minus_7);
    let user_prompt = format!("Query: {query_text}");

    let raw = llm.chat_completion(&system_prompt, &user_prompt).ok()?;
    // Extract first JSON object
    let json_text = extract_json_object(&raw);
    serde_json::from_str::<GraphOp>(json_text).ok()
}

fn extract_json_object(raw: &str) -> &str {
    let after_think = if let Some(end) = raw.find("</think>") {
        &raw[end + "</think>".len()..]
    } else {
        raw
    };
    let s = if let Some(start) = after_think.find("```json") {
        let rest = &after_think[start + "```json".len()..];
        rest.split("```").next().unwrap_or(rest)
    } else if let Some(start) = after_think.find("```") {
        let rest = &after_think[start + "```".len()..];
        rest.split("```").next().unwrap_or(rest)
    } else {
        after_think
    };
    let t = s.trim();
    if let (Some(a), Some(b)) = (t.find('{'), t.rfind('}'))
        && b > a {
            return &t[a..=b];
        }
    t
}

fn dispatch_op(
    graph: &ctxgraph::Graph,
    op: &GraphOp,
    limit: usize,
) -> ctxgraph::Result<Vec<(String, Entity)>> {
    let entity_type_filter = op
        .filters
        .get("entity_type")
        .and_then(|v| v.as_str())
        .map(String::from);

    match op.op.as_str() {
        "lookup" | "traverse" => {
            // Resolve head entity
            let head_name = match &op.head {
                Some(n) if !n.is_empty() => n.clone(),
                _ => return Ok(Vec::new()),
            };
            let head = exact_entity_lookup(graph, &head_name)?;
            let Some(head) = head else {
                return Ok(Vec::new());
            };

            let (entities, edges) = graph.traverse(&head.id, 1)?;
            let want_rel = op.relation.as_deref();
            let mut out: Vec<(String, Entity)> = Vec::new();
            for e in &edges {
                if let Some(r) = want_rel
                    && e.relation != r {
                        continue;
                    }
                let other_id = if e.source_id == head.id {
                    &e.target_id
                } else {
                    &e.source_id
                };
                if let Some(ent) = entities.iter().find(|x| &x.id == other_id) {
                    if let Some(t) = &entity_type_filter
                        && &ent.entity_type != t {
                            continue;
                        }
                    out.push((format!("{} -[{}]->", head.name, e.relation), ent.clone()));
                }
            }
            Ok(out)
        }
        "list" => {
            let entities = graph.list_entities(entity_type_filter.as_deref(), limit)?;
            Ok(entities
                .into_iter()
                .map(|e| (format!("[{}]", e.entity_type), e))
                .collect())
        }
        "filter" | "compare" => {
            // Fall back to type-filtered listing for now; full compare/filter
            // semantics deferred to v0.4 (per ROADMAP).
            let entities = graph.list_entities(entity_type_filter.as_deref(), limit)?;
            Ok(entities
                .into_iter()
                .map(|e| (format!("[{}]", e.entity_type), e))
                .collect())
        }
        _ => Ok(Vec::new()),
    }
}

fn load_prompt_text(env_var: &str, fallback_rel: &str) -> Option<String> {
    if let Ok(val) = env::var(env_var)
        && let Ok(s) = std::fs::read_to_string(&val) {
            return Some(s);
        }
    let cwd = env::current_dir().ok()?;
    let candidates = [
        cwd.join(fallback_rel),
        cwd.parent().map(|p| p.join(fallback_rel))?,
        cwd.parent()
            .and_then(Path::parent)
            .map(|p| p.join(fallback_rel))?,
    ];
    for c in &candidates {
        if let Ok(s) = std::fs::read_to_string(c) {
            return Some(s);
        }
    }
    None
}

fn resolve_universal_schema() -> ctxgraph::Result<PathBuf> {
    if let Ok(val) = env::var("CTXGRAPH_UNIVERSAL_SCHEMA") {
        let p = PathBuf::from(val);
        if p.is_file() {
            return Ok(p);
        }
    }
    if let Ok(cwd) = env::current_dir() {
        let rel = "crates/ctxgraph-extract/schemas/universal.toml";
        let candidates = [
            cwd.join(rel),
            cwd.parent().map(|p| p.join(rel)).unwrap_or_default(),
            cwd.parent()
                .and_then(Path::parent)
                .map(|p| p.join(rel))
                .unwrap_or_default(),
        ];
        for c in &candidates {
            if c.is_file() {
                return Ok(c.clone());
            }
        }
    }
    Err(CtxGraphError::NotFound(
        "universal.toml not found. Set $CTXGRAPH_UNIVERSAL_SCHEMA".into(),
    ))
}
