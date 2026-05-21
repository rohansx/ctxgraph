//! LLM-based entity and relation extraction via OpenRouter API.
//!
//! Opt-in fallback for cross-domain extraction when local models (GLiNER/NuNER)
//! produce low-confidence results. Activated by setting `OPENROUTER_API_KEY`.
//!
//! Default model: `google/gemini-2.0-flash-001` (~$0.0003/episode).
//! Override with `CTXGRAPH_LLM_MODEL`.

use serde::{Deserialize, Serialize};

use crate::ner::ExtractedEntity;
use crate::rel::ExtractedRelation;
use crate::schema::ExtractionSchema;

const DEFAULT_MODEL: &str = "gpt-4o-mini";
const DEFAULT_URL: &str = "https://api.openai.com/v1/chat/completions";

/// Preferred local models for Ollama auto-detection (tried in order).
/// These are small models that run on 6-8GB VRAM GPUs.
const OLLAMA_PREFERRED_MODELS: &[&str] = &[
    "gemma3n:e4b", // Gemma 3n E4B — best extraction quality at 6GB, native tool calling
    "gemma4:e4b",  // Gemma 4 E4B — newer but may need more VRAM
    "gemma3n:e2b", // Gemma 3n E2B — lighter fallback, ~3GB
    "gemma4:e2b",  // Gemma 4 E2B — lighter fallback, ~5GB
    "llama3.2:3b", // Llama 3.2 3B — widely available
];

/// LLM extraction engine — works with any OpenAI-compatible endpoint.
///
/// Supports: nvidia-litellm-router (free), OpenRouter, Ollama, OpenAI, Anthropic.
/// When the `cloakpipe` feature is enabled, PII is detected and stripped before
/// any text is sent to the LLM, then entity names are mapped back after extraction.
///
/// Configured via environment variables or ctxgraph.toml:
/// - `CTXGRAPH_LLM_URL`: API base URL (default: localhost:4000 for nvidia-litellm-router)
/// - `CTXGRAPH_LLM_KEY`: API key (or `OPENROUTER_API_KEY` for backward compat)
/// - `CTXGRAPH_LLM_MODEL`: Model name (default: nvidia-auto)
pub struct LlmExtractor {
    client: reqwest::blocking::Client,
    api_key: String,
    model: String,
    url: String,
}

/// Combined extraction result from the LLM.
pub struct LlmExtractionResult {
    pub entities: Vec<ExtractedEntity>,
    pub relations: Vec<ExtractedRelation>,
}

// --- JSON schema types for structured output ---

#[derive(Debug, Serialize, Deserialize)]
struct LlmResponse {
    entities: Vec<LlmEntity>,
    relations: Vec<LlmRelation>,
}

#[derive(Debug, Serialize, Deserialize)]
struct LlmEntity {
    name: String,
    entity_type: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct LlmRelation {
    head: String,
    relation: String,
    tail: String,
}

// --- OpenRouter API types ---

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormat>,
    temperature: f32,
    max_tokens: u32,
}

#[derive(Debug, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct ResponseFormat {
    r#type: String,
    json_schema: JsonSchemaWrapper,
}

#[derive(Debug, Serialize)]
struct JsonSchemaWrapper {
    name: String,
    strict: bool,
    schema: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatChoiceMessage,
}

#[derive(Debug, Deserialize)]
struct ChatChoiceMessage {
    content: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("API error: {0}")]
    Api(String),

    #[error("parse error: {0}")]
    Parse(String),
}

/// LLM configuration — can be loaded from TOML or environment variables.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LlmConfig {
    /// Provider name: "openrouter", "ollama", "openai", "anthropic", "nvidia", or "custom"
    #[serde(default)]
    pub provider: String,
    /// API key (or env var name to read from, if prefixed with "env:")
    #[serde(default)]
    pub api_key: String,
    /// API key environment variable name (alternative to api_key)
    #[serde(default)]
    pub api_key_env: String,
    /// Model name
    #[serde(default)]
    pub model: String,
    /// Base URL for the API (without /chat/completions)
    #[serde(default)]
    pub base_url: String,
    /// Whether to auto-escalate to LLM when local confidence is low
    #[serde(default = "default_true")]
    pub auto_escalate: bool,
}

fn default_true() -> bool {
    true
}

impl LlmExtractor {
    /// Create from explicit config (loaded from ctxgraph.toml).
    pub fn from_config(config: &LlmConfig) -> Option<Self> {
        if config.provider.is_empty() && config.api_key.is_empty() && config.api_key_env.is_empty()
        {
            return None;
        }

        let api_key = if !config.api_key.is_empty() {
            config.api_key.clone()
        } else if !config.api_key_env.is_empty() {
            std::env::var(&config.api_key_env).ok()?
        } else {
            // Infer from provider
            match config.provider.as_str() {
                "nvidia" => "sk-litellm-master".to_string(),
                "ollama" => "ollama".to_string(),
                "openrouter" => std::env::var("OPENROUTER_API_KEY").ok()?,
                "openai" => std::env::var("OPENAI_API_KEY").ok()?,
                "anthropic" => std::env::var("ANTHROPIC_API_KEY").ok()?,
                _ => return None,
            }
        };

        if api_key.is_empty() {
            return None;
        }

        let url = if !config.base_url.is_empty() {
            format!("{}/chat/completions", config.base_url.trim_end_matches('/'))
        } else {
            match config.provider.as_str() {
                "nvidia" => "http://localhost:4000/v1/chat/completions".to_string(),
                "ollama" => "http://localhost:11434/v1/chat/completions".to_string(),
                "openrouter" => "https://openrouter.ai/api/v1/chat/completions".to_string(),
                "openai" => "https://api.openai.com/v1/chat/completions".to_string(),
                "anthropic" => "https://api.anthropic.com/v1/chat/completions".to_string(),
                _ => DEFAULT_URL.to_string(),
            }
        };

        let model = if !config.model.is_empty() {
            config.model.clone()
        } else {
            match config.provider.as_str() {
                "nvidia" => "nvidia-auto".to_string(),
                "ollama" => "gemma3n:e4b".to_string(),
                "openrouter" => "openai/gpt-4o-mini".to_string(),
                "openai" => "gpt-4o-mini".to_string(),
                "anthropic" => "claude-3-5-haiku-20241022".to_string(),
                _ => DEFAULT_MODEL.to_string(),
            }
        };

        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .ok()?;

        Some(Self {
            client,
            api_key,
            model,
            url,
        })
    }

    /// Create a new LLM extractor from environment variables.
    ///
    /// Tiered auto-detection (local-first):
    /// 1. `CTXGRAPH_LLM_KEY` + `CTXGRAPH_LLM_URL` (explicit config — highest priority)
    /// 2. **Ollama local** (auto-detected if running, zero cost, fully private)
    /// 3. `OPENAI_API_KEY` (OpenAI direct)
    /// 4. `ANTHROPIC_API_KEY` (Anthropic direct)
    /// 5. `OPENROUTER_API_KEY` (OpenRouter — multi-provider)
    ///
    /// Set `CTXGRAPH_NO_LLM=1` to disable LLM entirely.
    /// Set `CTXGRAPH_NO_OLLAMA=1` to skip Ollama auto-detection.
    ///
    /// Returns `None` if no LLM backend is available.
    pub fn from_env() -> Option<Self> {
        // Respect explicit disable
        if std::env::var("CTXGRAPH_NO_LLM").is_ok_and(|v| v == "1" || v == "true") {
            return None;
        }

        // Tier 0: Explicit config (overrides everything)
        if let Ok(key) = std::env::var("CTXGRAPH_LLM_KEY")
            && !key.is_empty() {
                let url =
                    std::env::var("CTXGRAPH_LLM_URL").unwrap_or_else(|_| DEFAULT_URL.to_string());
                let model = std::env::var("CTXGRAPH_LLM_MODEL")
                    .unwrap_or_else(|_| DEFAULT_MODEL.to_string());
                let client = reqwest::blocking::Client::builder()
                    .timeout(std::time::Duration::from_secs(60))
                    .build()
                    .ok()?;
                return Some(Self {
                    client,
                    api_key: key,
                    model,
                    url,
                });
            }

        // Tier 1: Ollama local (auto-detect — free, private, no API key needed)
        if !std::env::var("CTXGRAPH_NO_OLLAMA").is_ok_and(|v| v == "1" || v == "true")
            && let Some(extractor) = Self::detect_ollama() {
                return Some(extractor);
            }

        // Tier 2: Cloud providers (need API keys)
        let (api_key, default_url) = if let Ok(key) = std::env::var("OPENAI_API_KEY") {
            if !key.is_empty() {
                (
                    key,
                    "https://api.openai.com/v1/chat/completions".to_string(),
                )
            } else {
                return None;
            }
        } else if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
            if !key.is_empty() {
                (
                    key,
                    "https://api.anthropic.com/v1/chat/completions".to_string(),
                )
            } else {
                return None;
            }
        } else if let Ok(key) = std::env::var("OPENROUTER_API_KEY") {
            if !key.is_empty() {
                (
                    key,
                    "https://openrouter.ai/api/v1/chat/completions".to_string(),
                )
            } else {
                return None;
            }
        } else {
            return None;
        };

        let url = std::env::var("CTXGRAPH_LLM_URL").unwrap_or(default_url);
        let model =
            std::env::var("CTXGRAPH_LLM_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());

        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .ok()?;

        Some(Self {
            client,
            api_key,
            model,
            url,
        })
    }

    /// Auto-detect Ollama running locally and find the best available model.
    ///
    /// Probes `http://localhost:11434/api/tags` to list installed models,
    /// then picks the best one from `OLLAMA_PREFERRED_MODELS` order.
    fn detect_ollama() -> Option<Self> {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(3))
            .build()
            .ok()?;

        // Probe Ollama API
        let resp = client.get("http://localhost:11434/api/tags").send().ok()?;

        if !resp.status().is_success() {
            return None;
        }

        // Parse available models
        let body: serde_json::Value = resp.json().ok()?;
        let models = body.get("models")?.as_array()?;
        let available: Vec<String> = models
            .iter()
            .filter_map(|m| m.get("name")?.as_str().map(String::from))
            .collect();

        if available.is_empty() {
            return None;
        }

        // Find the best model from our preferred list
        let chosen = OLLAMA_PREFERRED_MODELS
            .iter()
            .find(|&&pref| {
                available.iter().any(|a| {
                    a == pref
                        || a == &format!("{pref}:latest")
                        || a.starts_with(&format!("{pref}:"))
                })
            })
            .copied()
            // Fall back to any available model
            .unwrap_or({
                // Use the first available model (as &str via leak — acceptable for a one-time init)
                // We can't return a reference to `available` so use a known fallback
                "qwen2.5:3b"
            });

        // Check that chosen model is actually available
        let model_available = available.iter().any(|a| {
            a == chosen || a == &format!("{chosen}:latest") || a.starts_with(&format!("{chosen}:"))
        });

        if !model_available {
            // None of our preferred models installed, use whatever is first
            if let Some(first) = available.first() {
                let client = reqwest::blocking::Client::builder()
                    .timeout(std::time::Duration::from_secs(120))
                    .build()
                    .ok()?;
                eprintln!(
                    "[ctxgraph] Ollama detected with model '{}' (auto-selected)",
                    first
                );
                return Some(Self {
                    client,
                    api_key: "ollama".to_string(),
                    model: first.clone(),
                    url: "http://localhost:11434/v1/chat/completions".to_string(),
                });
            }
            return None;
        }

        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .ok()?;

        eprintln!(
            "[ctxgraph] Ollama detected with model '{}' (local, free)",
            chosen
        );

        Some(Self {
            client,
            api_key: "ollama".to_string(),
            model: chosen.to_string(),
            url: "http://localhost:11434/v1/chat/completions".to_string(),
        })
    }

    /// Extract entities only from text using the LLM.
    ///
    /// Used as fallback when GLiNER confidence is low. Relations are handled
    /// by GLiREL locally — the LLM only needs to find entity spans.
    pub fn extract_entities(
        &self,
        text: &str,
        schema: &ExtractionSchema,
    ) -> Result<Vec<ExtractedEntity>, LlmError> {
        let entity_types: Vec<String> = schema
            .entity_types
            .iter()
            .map(|(k, v)| format!("- {k}: {v}"))
            .collect();

        let system = format!(
            r#"Extract all named entities from the text. Return JSON only.

ENTITY TYPES:
{entity_list}

RULES:
- Use the SHORTEST canonical name for each entity. Examples:
  - "Cerner" not "Cerner EHR system"
  - "Blackboard" not "Blackboard LMS"
  - "FHIR R4" not "FHIR R4 APIs"
  - "Bloomberg Terminal" not "Bloomberg's API"
  - "blue-green deployment" not "zero-downtime blue-green deployment pattern"
- Strip articles (the, a, an) from the start of entity names.
- Entity names should be findable as substrings in the text.
- Each entity_type must be one of the types listed above.
- Extract ALL entities. Common things people miss:
  - Teams, departments, and roles count as Person (e.g., "radiology department", "compliance officer", "plant manager", "legal ops team")
  - Budget caps, SLAs, compliance requirements, certifications count as Constraint (e.g., "$50K budget cap", "HIPAA", "FedRAMP certification", "99.99% uptime SLA")
  - Incidents, errors, bottlenecks count as relevant entities (e.g., "ransomware incident", "T+3 settlement bottleneck")
  - Systems and platforms count as Component/Service even if not software (e.g., "trading engine", "clearing system", "predictive maintenance system")
- Do not invent entities not supported by the text."#,
            entity_list = entity_types.join("\n"),
        );

        let entity_type_enum: Vec<&str> = schema.entity_types.keys().map(|k| k.as_str()).collect();

        let json_schema = serde_json::json!({
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "entity_type": {"type": "string", "enum": entity_type_enum}
                        },
                        "required": ["name", "entity_type"],
                        "additionalProperties": false
                    }
                }
            },
            "required": ["entities"],
            "additionalProperties": false
        });

        let request = ChatRequest {
            model: self.model.clone(),
            messages: vec![
                ChatMessage {
                    role: "system".into(),
                    content: system,
                },
                ChatMessage {
                    role: "user".into(),
                    content: format!("Extract entities from:\n\n{text}"),
                },
            ],
            response_format: Some(ResponseFormat {
                r#type: "json_schema".into(),
                json_schema: JsonSchemaWrapper {
                    name: "entity_result".into(),
                    strict: true,
                    schema: json_schema,
                },
            }),
            temperature: 0.0,
            max_tokens: 1024,
        };

        let response = self
            .client
            .post(&self.url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().unwrap_or_default();
            return Err(LlmError::Api(format!("{status}: {body}")));
        }

        let chat_resp: ChatResponse = response.json()?;
        let content = chat_resp
            .choices
            .first()
            .and_then(|c| c.message.content.as_deref())
            .ok_or_else(|| LlmError::Parse("empty response".into()))?;

        let json_str = extract_json_from_response(content);

        #[derive(Deserialize)]
        struct EntityOnly {
            entities: Vec<LlmEntity>,
        }

        let parsed: EntityOnly = serde_json::from_str(json_str)
            .map_err(|e| LlmError::Parse(format!("JSON parse: {e}\nRaw: {content}")))?;

        Ok(parsed
            .entities
            .into_iter()
            .map(|e| {
                let name = canonicalize_llm_entity(&e.name, text);
                let span_start = text.to_lowercase().find(&name.to_lowercase()).unwrap_or(0);
                let span_end = if span_start > 0 {
                    span_start + name.len()
                } else {
                    // Fallback: try original name
                    text.find(&e.name).map(|s| s + e.name.len()).unwrap_or(0)
                };
                ExtractedEntity {
                    span_start,
                    span_end,
                    text: name,
                    entity_type: e.entity_type,
                    confidence: 1.0,
                }
            })
            .collect())
    }

    /// Extract entities and relations from text using the LLM (full extraction).
    ///
    /// Tries strict JSON schema first (works with large models like GPT-4o, Gemini).
    /// Falls back to prompt-based JSON (works with small models like Llama 3B, Qwen 3B).
    pub fn extract(
        &self,
        text: &str,
        schema: &ExtractionSchema,
    ) -> Result<LlmExtractionResult, LlmError> {
        // Try strict JSON schema first
        if let Ok(result) = self.extract_strict(text, schema) {
            return Ok(result);
        }
        // Fall back to prompt-based JSON for small models
        self.extract_prompt_json(text, schema)
    }

    fn extract_strict(
        &self,
        text: &str,
        schema: &ExtractionSchema,
    ) -> Result<LlmExtractionResult, LlmError> {
        let system_prompt = build_system_prompt(schema);
        let user_prompt = format!("Extract entities and relations from this text:\n\n{text}");

        let request = ChatRequest {
            model: self.model.clone(),
            messages: vec![
                ChatMessage {
                    role: "system".into(),
                    content: system_prompt,
                },
                ChatMessage {
                    role: "user".into(),
                    content: user_prompt,
                },
            ],
            response_format: Some(ResponseFormat {
                r#type: "json_schema".into(),
                json_schema: JsonSchemaWrapper {
                    name: "extraction_result".into(),
                    strict: true,
                    schema: extraction_json_schema(schema),
                },
            }),
            temperature: 0.0,
            max_tokens: 2048,
        };

        let response = self
            .client
            .post(&self.url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().unwrap_or_default();
            return Err(LlmError::Api(format!("{status}: {body}")));
        }

        let chat_resp: ChatResponse = response.json()?;
        let content = chat_resp
            .choices
            .first()
            .and_then(|c| c.message.content.as_deref())
            .ok_or_else(|| LlmError::Parse("empty response".into()))?;

        let parsed: LlmResponse = serde_json::from_str(content)
            .map_err(|e| LlmError::Parse(format!("JSON parse: {e}\nRaw: {content}")))?;

        // Map to internal types
        let entities = parsed
            .entities
            .into_iter()
            .map(|e| ExtractedEntity {
                span_start: text.find(&e.name).unwrap_or(0),
                span_end: text.find(&e.name).map(|s| s + e.name.len()).unwrap_or(0),
                text: e.name,
                entity_type: e.entity_type,
                confidence: 1.0, // LLM doesn't provide confidence
            })
            .collect();

        let relations = parsed
            .relations
            .into_iter()
            .map(|r| ExtractedRelation {
                head: r.head,
                relation: r.relation,
                tail: r.tail,
                confidence: 1.0,
            })
            .collect();

        Ok(LlmExtractionResult {
            entities,
            relations,
        })
    }

    /// Fallback: prompt-based JSON extraction for small models that don't support
    /// structured output / json_schema response format.
    fn extract_prompt_json(
        &self,
        text: &str,
        schema: &ExtractionSchema,
    ) -> Result<LlmExtractionResult, LlmError> {
        let entity_types: Vec<String> = schema
            .entity_types
            .iter()
            .map(|(k, v)| format!("  - {k}: {v}"))
            .collect();
        let relation_types: Vec<String> = schema
            .relation_types
            .iter()
            .map(|(k, v)| format!("  - {k}: {}", v.description))
            .collect();

        let prompt = format!(
            r#"Extract entities and relations. ONLY output JSON, nothing else. /no_think

Entity types:
{entities}

Relation types:
{relations}

Rules:
- SHORT canonical names ("Redis" not "Redis server", "Stripe" not "Stripe SDK v2")
- Teams/departments/roles are Person
- Relation head and tail MUST exactly match an entity name you extracted
- For EVERY pair of related entities, add a relation

Text: "{text}"

{{"entities": [{{"name": "...", "entity_type": "..."}}], "relations": [{{"head": "exact entity name", "relation": "type", "tail": "exact entity name"}}]}}"#,
            entities = entity_types.join("\n"),
            relations = relation_types.join("\n"),
        );

        let request = ChatRequest {
            model: self.model.clone(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: prompt,
            }],
            response_format: None,
            temperature: 0.0,
            max_tokens: 2048,
        };

        let response = self
            .client
            .post(&self.url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().unwrap_or_default();
            return Err(LlmError::Api(format!("{status}: {body}")));
        }

        let chat_resp: ChatResponse = response.json()?;
        let raw = chat_resp
            .choices
            .first()
            .and_then(|c| c.message.content.as_deref())
            .ok_or_else(|| LlmError::Parse("empty response".into()))?;

        // Extract JSON from the response — small models sometimes wrap it in markdown
        let json_str = extract_json_from_response(raw);

        let parsed: LlmResponse = serde_json::from_str(json_str)
            .map_err(|e| LlmError::Parse(format!("JSON parse: {e}\nRaw: {raw}")))?;

        let entities = parsed
            .entities
            .into_iter()
            .map(|e| {
                let name = canonicalize_llm_entity(&e.name, text);
                ExtractedEntity {
                    span_start: text.to_lowercase().find(&name.to_lowercase()).unwrap_or(0),
                    span_end: text
                        .to_lowercase()
                        .find(&name.to_lowercase())
                        .map(|s| s + name.len())
                        .unwrap_or(0),
                    text: name,
                    entity_type: e.entity_type,
                    confidence: 1.0,
                }
            })
            .collect();

        let relations = parsed
            .relations
            .into_iter()
            .map(|r| ExtractedRelation {
                head: r.head,
                relation: r.relation,
                tail: r.tail,
                confidence: 1.0,
            })
            .collect();

        Ok(LlmExtractionResult {
            entities,
            relations,
        })
    }

    /// Send a system+user prompt to the LLM and return the raw response content.
    ///
    /// This is the low-level escape hatch used by the universal v0.3 pipeline,
    /// which supplies its own prompt template (`prompts/extract.txt`) instead of
    /// going through `extract_strict` / `extract_prompt_json`.
    pub fn chat_completion(
        &self,
        system_prompt: &str,
        user_prompt: &str,
    ) -> Result<String, LlmError> {
        let request = ChatRequest {
            model: self.model.clone(),
            messages: vec![
                ChatMessage {
                    role: "system".into(),
                    content: system_prompt.to_string(),
                },
                ChatMessage {
                    role: "user".into(),
                    content: user_prompt.to_string(),
                },
            ],
            response_format: None,
            temperature: 0.0,
            max_tokens: 2048,
        };

        let response = self
            .client
            .post(&self.url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().unwrap_or_default();
            return Err(LlmError::Api(format!("{status}: {body}")));
        }

        let chat_resp: ChatResponse = response.json()?;
        chat_resp
            .choices
            .first()
            .and_then(|c| c.message.content.clone())
            .ok_or_else(|| LlmError::Parse("empty response".into()))
    }
}

/// Extract JSON object from LLM response that may contain markdown fences,
/// `<think>` tags, or other preamble.
fn extract_json_from_response(raw: &str) -> &str {
    // Strip <think>...</think> tags (Qwen, DeepSeek reasoning models)
    let stripped = if let Some(think_end) = raw.find("</think>") {
        raw[think_end + 8..].trim()
    } else {
        raw
    };

    // Try to find JSON between ```json ... ``` markers
    if let Some(start) = stripped.find("```json") {
        let after = &stripped[start + 7..];
        if let Some(end) = after.find("```") {
            return after[..end].trim();
        }
    }
    // Try to find JSON between ``` ... ``` markers
    if let Some(start) = stripped.find("```") {
        let after = &stripped[start + 3..];
        if let Some(end) = after.find("```") {
            return after[..end].trim();
        }
    }
    // Try to find { ... } directly
    if let Some(start) = stripped.find('{')
        && let Some(end) = stripped.rfind('}')
    {
        return &stripped[start..=end];
    }
    stripped.trim()
}

/// Build the system prompt with schema-aware entity and relation types.
fn build_system_prompt(schema: &ExtractionSchema) -> String {
    let entity_types: Vec<String> = schema
        .entity_types
        .iter()
        .map(|(k, v)| format!("- {k}: {v}"))
        .collect();

    let relation_types: Vec<String> = schema
        .relation_types
        .iter()
        .map(|(k, v)| format!("- {k}: {}", v.description))
        .collect();

    format!(
        r#"You are an entity and relation extraction system.

ENTITY TYPES:
{entity_list}

RELATION TYPES:
{relation_list}

RULES:
- Use the SHORTEST canonical name for entities ("Redis" not "Redis cache", "Stripe" not "Stripe SDK v2").
- Teams, departments, and roles are Person entities ("platform team", "treasury department").
- Constraints include: compliance requirements, SLAs, certifications, budget caps.
- Each relation head and tail MUST be the EXACT name string from your entities list.
- CRITICAL: First extract all entities, then for EVERY pair of related entities, add a relation.
- Prefer specific relation types: "replaced" over "depends_on" when migration is described.
- Do not invent entities or relations not supported by the text."#,
        entity_list = entity_types.join("\n"),
        relation_list = relation_types.join("\n"),
    )
}

/// Build the JSON schema for structured output.
fn extraction_json_schema(schema: &ExtractionSchema) -> serde_json::Value {
    let entity_type_enum: Vec<&str> = schema.entity_types.keys().map(|k| k.as_str()).collect();
    let relation_enum: Vec<&str> = schema.relation_types.keys().map(|k| k.as_str()).collect();

    serde_json::json!({
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Exact entity name from text"},
                        "entity_type": {"type": "string", "enum": entity_type_enum}
                    },
                    "required": ["name", "entity_type"],
                    "additionalProperties": false
                }
            },
            "relations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "head": {"type": "string", "description": "Head entity name"},
                        "relation": {"type": "string", "enum": relation_enum},
                        "tail": {"type": "string", "description": "Tail entity name"}
                    },
                    "required": ["head", "relation", "tail"],
                    "additionalProperties": false
                }
            }
        },
        "required": ["entities", "relations"],
        "additionalProperties": false
    })
}

/// Canonicalize an LLM-extracted entity name to match what appears in text.
///
/// LLMs tend to return verbose names ("Cerner EHR", "Blackboard LMS",
/// "FHIR R4 APIs"). This strips common suffixes, articles, and possessives
/// to produce shorter canonical names that match text substrings.
/// Strip PII from text before sending to LLM, then map entity names back.
///
/// Only active when the `cloakpipe` feature is enabled.
/// Returns (pseudonymized_text, token→original mappings).
#[cfg(feature = "cloakpipe")]
pub fn pseudonymize_for_llm(text: &str) -> (String, std::collections::HashMap<String, String>) {
    use cloakpipe_core::config::DetectionConfig;
    use cloakpipe_core::detector::Detector;
    use cloakpipe_core::replacer::Replacer;
    use cloakpipe_core::vault::Vault;

    // Use detection config with patterns + financial only (no NER to avoid GLiNER conflict)
    let config_toml = r#"
secrets = true
financial = true
dates = false
emails = true
phone_numbers = false
ip_addresses = false
urls_internal = false

[ner]
enabled = false

[custom]
[overrides]
[resolver]
"#;
    let config: DetectionConfig = toml::from_str(config_toml).unwrap_or_else(|_| {
        return DetectionConfig {
            secrets: true,
            financial: true,
            dates: false,
            emails: true,
            phone_numbers: false,
            ip_addresses: false,
            urls_internal: false,
            ner: Default::default(),
            custom: Default::default(),
            overrides: Default::default(),
            resolver: Default::default(),
        };
    });
    let detector = match Detector::from_config(&config) {
        Ok(d) => d,
        Err(_) => return (text.to_string(), std::collections::HashMap::new()),
    };

    let entities = match detector.detect(text) {
        Ok(e) => e,
        Err(_) => return (text.to_string(), std::collections::HashMap::new()),
    };

    if entities.is_empty() {
        return (text.to_string(), std::collections::HashMap::new());
    }

    let mut vault = Vault::ephemeral();
    match Replacer::pseudonymize(text, &entities, &mut vault) {
        Ok(result) => (result.text, result.mappings), // mappings: token → original
        Err(_) => (text.to_string(), std::collections::HashMap::new()),
    }
}

/// Map entity names back from pseudonymized text to originals.
#[cfg(feature = "cloakpipe")]
pub fn rehydrate_entities(
    entities: Vec<crate::ner::ExtractedEntity>,
    mappings: &std::collections::HashMap<String, String>,
) -> Vec<crate::ner::ExtractedEntity> {
    entities
        .into_iter()
        .map(|mut e| {
            // Check if entity text matches any pseudo-token (token → original)
            if let Some(original) = mappings.get(&e.text) {
                e.text = original.clone();
            }
            e
        })
        .collect()
}

fn canonicalize_llm_entity(name: &str, text: &str) -> String {
    let mut result = name.to_string();

    // Strip leading articles
    for prefix in ["the ", "The ", "a ", "A ", "an ", "An "] {
        if result.starts_with(prefix) {
            result = result[prefix.len()..].to_string();
        }
    }

    // Strip possessive 's (e.g., "Epic's MyChart" → "Epic MyChart")
    result = result.replace("'s ", " ");

    // Strip verbose suffixes ONLY when the trimmed version is a proper noun
    // or known entity. Avoid stripping meaningful parts like "report generation".
    let safe_suffixes = [
        " LMS",
        " EHR",
        " CRM",
        " APIs",
        " API",
        " system",
        " systems",
        " platform",
        " platforms",
    ];

    for suffix in &safe_suffixes {
        if result.ends_with(suffix) {
            let trimmed = &result[..result.len() - suffix.len()];
            // Only strip if trimmed is at least 3 chars and found in text
            if trimmed.len() >= 3 && text.to_lowercase().contains(&trimmed.to_lowercase()) {
                result = trimmed.to_string();
                break;
            }
        }
    }

    // If result doesn't appear in text but original does, keep original
    if !text.to_lowercase().contains(&result.to_lowercase())
        && text.to_lowercase().contains(&name.to_lowercase())
    {
        return name.to_string();
    }

    result
}
