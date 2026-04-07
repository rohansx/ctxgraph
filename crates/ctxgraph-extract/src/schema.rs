use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// Extraction schema defining which entity types and relation types to extract.
///
/// Loaded from a `ctxgraph.toml` file or constructed via `ExtractionSchema::default()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionSchema {
    pub name: String,
    pub entity_types: BTreeMap<String, String>,
    pub relation_types: BTreeMap<String, RelationSpec>,
}

/// Specification for a relation type — which entity types can be head/tail.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationSpec {
    pub head: Vec<String>,
    pub tail: Vec<String>,
    pub description: String,
}

/// Raw TOML structure for deserialization.
#[derive(Debug, Deserialize)]
struct SchemaToml {
    schema: SchemaSection,
}

#[derive(Debug, Deserialize)]
struct SchemaSection {
    name: String,
    entities: BTreeMap<String, String>,
    #[serde(default)]
    relations: BTreeMap<String, RelationSpecToml>,
}

#[derive(Debug, Deserialize)]
struct RelationSpecToml {
    head: Vec<String>,
    tail: Vec<String>,
    #[serde(default)]
    description: String,
}

impl ExtractionSchema {
    /// Load schema from a TOML file.
    pub fn load(path: &Path) -> Result<Self, SchemaError> {
        let content = std::fs::read_to_string(path).map_err(|e| SchemaError::Io {
            path: path.display().to_string(),
            source: e,
        })?;
        Self::from_toml(&content)
    }

    /// Parse schema from a TOML string.
    pub fn from_toml(content: &str) -> Result<Self, SchemaError> {
        let parsed: SchemaToml =
            toml::from_str(content).map_err(|e| SchemaError::Parse(e.to_string()))?;

        let relation_types = parsed
            .schema
            .relations
            .into_iter()
            .map(|(k, v)| {
                (
                    k,
                    RelationSpec {
                        head: v.head,
                        tail: v.tail,
                        description: v.description,
                    },
                )
            })
            .collect();

        Ok(Self {
            name: parsed.schema.name,
            entity_types: parsed.schema.entities,
            relation_types,
        })
    }

    /// Entity label strings for GLiNER input.
    ///
    /// Returns the type key names (e.g. "Person", "Database"). Suitable for
    /// models trained on those label conventions.
    pub fn entity_labels(&self) -> Vec<&str> {
        self.entity_types.keys().map(|s| s.as_str()).collect()
    }

    /// Entity descriptions for zero-shot GLiNER inference.
    ///
    /// Returns `(description, key)` pairs. Passing the description as the label
    /// to GLiNER improves zero-shot recall because the model uses the label text
    /// as a natural-language prompt. The key is the canonical type name used in
    /// `ExtractionSchema` and benchmark fixtures.
    pub fn entity_label_descriptions(&self) -> Vec<(&str, &str)> {
        self.entity_types
            .iter()
            .map(|(k, v)| (v.as_str(), k.as_str()))
            .collect()
    }

    /// Map a GLiNER class string back to the canonical entity type key.
    ///
    /// When descriptions are used as labels, GLiNER returns the description as
    /// the span class. This method reverses that lookup.
    pub fn entity_type_from_label<'a>(&'a self, label: &str) -> Option<&'a str> {
        // Check descriptions first (zero-shot mode)
        if let Some(key) = self
            .entity_types
            .iter()
            .find(|(_, v)| v.as_str() == label)
            .map(|(k, _)| k.as_str())
        {
            return Some(key);
        }
        // Fall back to direct key match (standard mode)
        if self.entity_types.contains_key(label) {
            return Some(self.entity_types.get_key_value(label).unwrap().0.as_str());
        }
        None
    }

    /// Relation label strings for GLiREL/relation extraction input.
    pub fn relation_labels(&self) -> Vec<&str> {
        self.relation_types.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for ExtractionSchema {
    fn default() -> Self {
        let mut entity_types = BTreeMap::new();
        // Descriptions are short (2-4 words) so they fit inside GLiNER's token
        // budget alongside the input text. They are used as the actual label
        // strings passed to the model for zero-shot extraction, and are more
        // semantically precise than the bare key names.
        entity_types.insert("Person".into(), "person, team, or role".into());
        entity_types.insert("Component".into(), "software, tool, or product".into());
        entity_types.insert("Service".into(), "service, platform, or API".into());
        entity_types.insert("Language".into(), "programming language".into());
        entity_types.insert("Database".into(), "database or data store".into());
        entity_types.insert(
            "Infrastructure".into(),
            "server, hardware, or cloud platform".into(),
        );
        entity_types.insert("Decision".into(), "decision or policy".into());
        entity_types.insert("Constraint".into(), "constraint or requirement".into());
        entity_types.insert("Metric".into(), "metric or measurement".into());
        entity_types.insert("Pattern".into(), "pattern or methodology".into());

        let mut relation_types = BTreeMap::new();
        relation_types.insert(
            "chose".into(),
            RelationSpec {
                head: vec!["Person".into(), "Service".into(), "Component".into()],
                tail: vec![
                    "Component".into(),
                    "Database".into(),
                    "Language".into(),
                    "Infrastructure".into(),
                    "Pattern".into(),
                ],
                description: "chose or adopted a technology".into(),
            },
        );
        relation_types.insert(
            "rejected".into(),
            RelationSpec {
                head: vec!["Person".into(), "Service".into(), "Component".into()],
                tail: vec![
                    "Component".into(),
                    "Database".into(),
                    "Language".into(),
                    "Infrastructure".into(),
                ],
                description: "rejected an alternative".into(),
            },
        );
        relation_types.insert(
            "replaced".into(),
            RelationSpec {
                head: vec![
                    "Component".into(),
                    "Database".into(),
                    "Infrastructure".into(),
                    "Service".into(),
                    "Pattern".into(),
                    "Language".into(),
                ],
                tail: vec![
                    "Component".into(),
                    "Database".into(),
                    "Infrastructure".into(),
                    "Pattern".into(),
                    "Language".into(),
                ],
                description: "one thing replaced another".into(),
            },
        );
        relation_types.insert(
            "depends_on".into(),
            RelationSpec {
                head: vec![
                    "Service".into(),
                    "Component".into(),
                    "Infrastructure".into(),
                    "Language".into(),
                    "Pattern".into(),
                    "Decision".into(),
                ],
                tail: vec![
                    "Service".into(),
                    "Component".into(),
                    "Database".into(),
                    "Infrastructure".into(),
                    "Pattern".into(),
                    "Language".into(),
                ],
                description: "dependency relationship".into(),
            },
        );
        relation_types.insert(
            "fixed".into(),
            RelationSpec {
                head: vec![
                    "Person".into(),
                    "Component".into(),
                    "Service".into(),
                    "Language".into(),
                    "Infrastructure".into(),
                ],
                tail: vec![
                    "Component".into(),
                    "Service".into(),
                    "Database".into(),
                    "Pattern".into(),
                    "Metric".into(),
                    "Constraint".into(),
                ],
                description: "something fixed an issue".into(),
            },
        );
        relation_types.insert(
            "introduced".into(),
            RelationSpec {
                head: vec![
                    "Person".into(),
                    "Service".into(),
                    "Infrastructure".into(),
                    "Component".into(),
                    "Language".into(),
                ],
                tail: vec![
                    "Component".into(),
                    "Pattern".into(),
                    "Infrastructure".into(),
                    "Database".into(),
                    "Language".into(),
                    "Metric".into(),
                ],
                description: "introduced or added a component".into(),
            },
        );
        relation_types.insert(
            "deprecated".into(),
            RelationSpec {
                head: vec![
                    "Person".into(),
                    "Decision".into(),
                    "Service".into(),
                    "Component".into(),
                    "Infrastructure".into(),
                    "Pattern".into(),
                ],
                tail: vec![
                    "Component".into(),
                    "Pattern".into(),
                    "Infrastructure".into(),
                    "Database".into(),
                    "Language".into(),
                ],
                description: "deprecation action".into(),
            },
        );
        relation_types.insert(
            "caused".into(),
            RelationSpec {
                head: vec![
                    "Component".into(),
                    "Decision".into(),
                    "Service".into(),
                    "Infrastructure".into(),
                    "Language".into(),
                    "Pattern".into(),
                    "Database".into(),
                ],
                tail: vec!["Metric".into(), "Constraint".into(), "Pattern".into()],
                description: "causal relationship".into(),
            },
        );
        relation_types.insert(
            "constrained_by".into(),
            RelationSpec {
                head: vec![
                    "Decision".into(),
                    "Component".into(),
                    "Service".into(),
                    "Infrastructure".into(),
                    "Database".into(),
                    "Pattern".into(),
                ],
                tail: vec![
                    "Constraint".into(),
                    "Pattern".into(),
                    "Infrastructure".into(),
                    "Metric".into(),
                ],
                description: "decision constrained by".into(),
            },
        );

        Self {
            name: "default".into(),
            entity_types,
            relation_types,
        }
    }
}

impl ExtractionSchema {
    /// Infer a schema from sample text using an LLM.
    ///
    /// Sends sample episodes to the LLM and asks it to identify the domain,
    /// relevant entity types, and relation types. Returns a schema ready for
    /// extraction. The LLM call uses the same `LlmExtractor` infrastructure.
    ///
    /// This is the "zero-config" path: users log episodes, and ctxgraph
    /// automatically figures out what entities and relations matter.
    pub fn infer_from_text(
        samples: &[&str],
        llm_url: &str,
        llm_key: &str,
        llm_model: &str,
    ) -> Result<Self, SchemaError> {
        let combined = samples
            .iter()
            .enumerate()
            .map(|(i, t)| format!("Text {}: {}", i + 1, t))
            .collect::<Vec<_>>()
            .join("\n\n");

        let prompt = format!(
            r#"Analyze these text samples and design a knowledge graph schema for this domain.

{combined}

Return a JSON schema with:
1. "name": short domain name (e.g. "healthcare", "finance", "devops")
2. "entity_types": object mapping type names to short descriptions (5-10 types)
3. "relation_types": object mapping relation names to descriptions (5-10 relations)

Rules:
- Entity type names should be PascalCase (Person, Organization, Technology)
- Relation names should be snake_case (depends_on, partnered_with)
- Descriptions should be 3-6 words
- Always include Person, Organization, and Metric as entity types
- Relations should be verb phrases that connect entity types

Return ONLY valid JSON:
{{"name": "...", "entity_types": {{"TypeName": "short description"}}, "relation_types": {{"relation_name": "short description"}}}}"#
        );

        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .map_err(|e| SchemaError::Parse(format!("HTTP client: {e}")))?;

        let payload = serde_json::json!({
            "model": llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 1024,
        });

        let resp = client
            .post(llm_url)
            .header("Authorization", format!("Bearer {llm_key}"))
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .map_err(|e| SchemaError::Parse(format!("LLM request failed: {e}")))?;

        if !resp.status().is_success() {
            let body = resp.text().unwrap_or_default();
            return Err(SchemaError::Parse(format!("LLM error: {body}")));
        }

        let chat: serde_json::Value = resp
            .json()
            .map_err(|e| SchemaError::Parse(format!("LLM response parse: {e}")))?;
        let content = chat["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| SchemaError::Parse("empty LLM response".into()))?;

        // Extract JSON from response (may have markdown fences)
        let json_str = extract_json_block(content);

        let raw: serde_json::Value = serde_json::from_str(json_str)
            .map_err(|e| SchemaError::Parse(format!("schema JSON parse: {e}\nRaw: {content}")))?;

        let name = raw["name"].as_str().unwrap_or("inferred").to_string();

        let mut entity_types = BTreeMap::new();
        if let Some(obj) = raw["entity_types"].as_object() {
            for (k, v) in obj {
                entity_types.insert(k.clone(), v.as_str().unwrap_or("").to_string());
            }
        }

        // Build relation types (simplified — no head/tail constraints for inferred schemas)
        let mut relation_types = BTreeMap::new();
        if let Some(obj) = raw["relation_types"].as_object() {
            let all_types: Vec<String> = entity_types.keys().cloned().collect();
            for (k, v) in obj {
                relation_types.insert(
                    k.clone(),
                    RelationSpec {
                        head: all_types.clone(),
                        tail: all_types.clone(),
                        description: v.as_str().unwrap_or("").to_string(),
                    },
                );
            }
        }

        if entity_types.is_empty() {
            return Err(SchemaError::Parse(
                "LLM returned empty entity types".into(),
            ));
        }

        eprintln!(
            "[ctxgraph] Schema inferred: domain='{}', {} entity types, {} relation types",
            name,
            entity_types.len(),
            relation_types.len()
        );

        Ok(Self {
            name,
            entity_types,
            relation_types,
        })
    }

    /// Save schema to a TOML file for future use.
    pub fn save(&self, path: &Path) -> Result<(), SchemaError> {
        let mut toml = String::new();
        toml.push_str(&format!("[schema]\nname = \"{}\"\n\n", self.name));

        toml.push_str("[schema.entities]\n");
        for (k, v) in &self.entity_types {
            toml.push_str(&format!("{k} = \"{v}\"\n"));
        }

        if !self.relation_types.is_empty() {
            toml.push_str("\n");
            for (k, v) in &self.relation_types {
                let head: Vec<String> = v.head.iter().map(|h| format!("\"{h}\"")).collect();
                let tail: Vec<String> = v.tail.iter().map(|t| format!("\"{t}\"")).collect();
                toml.push_str(&format!(
                    "[schema.relations.{k}]\nhead = [{}]\ntail = [{}]\ndescription = \"{}\"\n\n",
                    head.join(", "),
                    tail.join(", "),
                    v.description
                ));
            }
        }

        std::fs::write(path, &toml).map_err(|e| SchemaError::Io {
            path: path.display().to_string(),
            source: e,
        })
    }
}

/// Extract a JSON block from LLM output that may contain markdown fences.
fn extract_json_block(content: &str) -> &str {
    // Strip <think>...</think>
    let stripped = if let Some(end) = content.find("</think>") {
        &content[end + 8..]
    } else {
        content
    }
    .trim();

    // Try ```json ... ```
    if let Some(start) = stripped.find("```json") {
        let after = &stripped[start + 7..];
        if let Some(end) = after.find("```") {
            return after[..end].trim();
        }
    }
    // Try { ... }
    if let Some(start) = stripped.find('{')
        && let Some(end) = stripped.rfind('}')
    {
        return &stripped[start..=end];
    }
    stripped
}

#[derive(Debug, thiserror::Error)]
pub enum SchemaError {
    #[error("failed to read schema at {path}: {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },

    #[error("failed to parse schema: {0}")]
    Parse(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_schema_has_all_entity_types() {
        let schema = ExtractionSchema::default();
        let labels = schema.entity_labels();
        assert!(labels.contains(&"Person"));
        assert!(labels.contains(&"Component"));
        assert!(labels.contains(&"Service"));
        assert!(labels.contains(&"Language"));
        assert!(labels.contains(&"Database"));
        assert!(labels.contains(&"Infrastructure"));
        assert!(labels.contains(&"Decision"));
        assert!(labels.contains(&"Constraint"));
        assert!(labels.contains(&"Metric"));
        assert!(labels.contains(&"Pattern"));
        assert_eq!(labels.len(), 10);
    }

    #[test]
    fn default_schema_has_all_relation_types() {
        let schema = ExtractionSchema::default();
        let labels = schema.relation_labels();
        assert!(labels.contains(&"chose"));
        assert!(labels.contains(&"rejected"));
        assert!(labels.contains(&"replaced"));
        assert!(labels.contains(&"depends_on"));
        assert!(labels.contains(&"fixed"));
        assert!(labels.contains(&"introduced"));
        assert!(labels.contains(&"deprecated"));
        assert!(labels.contains(&"caused"));
        assert!(labels.contains(&"constrained_by"));
        assert_eq!(labels.len(), 9);
    }

    #[test]
    fn parse_toml_schema() {
        let toml = r#"
[schema]
name = "test"

[schema.entities]
Person = "A person"
Component = "A software component"

[schema.relations]
chose = { head = ["Person"], tail = ["Component"], description = "person chose" }
"#;
        let schema = ExtractionSchema::from_toml(toml).unwrap();
        assert_eq!(schema.name, "test");
        assert_eq!(schema.entity_types.len(), 2);
        assert_eq!(schema.relation_types.len(), 1);
        assert_eq!(schema.relation_types["chose"].head, vec!["Person"]);
    }

    #[test]
    fn parse_toml_schema_no_relations() {
        let toml = r#"
[schema]
name = "entities-only"

[schema.entities]
Person = "A person"
"#;
        let schema = ExtractionSchema::from_toml(toml).unwrap();
        assert_eq!(schema.relation_types.len(), 0);
    }
}
