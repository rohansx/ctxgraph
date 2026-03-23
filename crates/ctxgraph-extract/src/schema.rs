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
        entity_types.insert("Person".into(), "person or engineer".into());
        entity_types.insert("Component".into(), "software library or framework".into());
        entity_types.insert("Service".into(), "cloud service or API".into());
        entity_types.insert("Language".into(), "programming language".into());
        entity_types.insert("Database".into(), "database or data store".into());
        entity_types.insert("Infrastructure".into(), "server or cloud platform".into());
        entity_types.insert("Decision".into(), "architectural decision".into());
        entity_types.insert("Constraint".into(), "technical constraint".into());
        entity_types.insert("Metric".into(), "performance metric".into());
        entity_types.insert("Pattern".into(), "design pattern".into());

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
