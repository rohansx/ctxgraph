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
    pub fn entity_labels(&self) -> Vec<&str> {
        self.entity_types.keys().map(|s| s.as_str()).collect()
    }

    /// Relation label strings for GLiREL/relation extraction input.
    pub fn relation_labels(&self) -> Vec<&str> {
        self.relation_types.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for ExtractionSchema {
    fn default() -> Self {
        let mut entity_types = BTreeMap::new();
        entity_types.insert(
            "Person".into(),
            "A person who made or was involved in a decision".into(),
        );
        entity_types.insert(
            "Component".into(),
            "A software component, tool, library, framework, or technology".into(),
        );
        entity_types.insert(
            "Service".into(),
            "A service, system, or application".into(),
        );
        entity_types.insert("Language".into(), "A programming language".into());
        entity_types.insert("Database".into(), "A database system".into());
        entity_types.insert(
            "Infrastructure".into(),
            "Infrastructure or DevOps tooling".into(),
        );
        entity_types.insert(
            "Decision".into(),
            "An explicit choice or judgment that was made".into(),
        );
        entity_types.insert(
            "Constraint".into(),
            "A limitation, requirement, or condition".into(),
        );
        entity_types.insert(
            "Metric".into(),
            "A quantifiable performance or business metric".into(),
        );
        entity_types.insert(
            "Pattern".into(),
            "A design pattern or architectural pattern".into(),
        );

        let mut relation_types = BTreeMap::new();
        relation_types.insert(
            "chose".into(),
            RelationSpec {
                head: vec!["Person".into()],
                tail: vec!["Component".into(), "Database".into(), "Language".into()],
                description: "person chose a technology".into(),
            },
        );
        relation_types.insert(
            "rejected".into(),
            RelationSpec {
                head: vec!["Person".into()],
                tail: vec!["Component".into(), "Database".into()],
                description: "person rejected an alternative".into(),
            },
        );
        relation_types.insert(
            "replaced".into(),
            RelationSpec {
                head: vec!["Component".into(), "Database".into()],
                tail: vec!["Component".into(), "Database".into()],
                description: "one thing replaced another".into(),
            },
        );
        relation_types.insert(
            "depends_on".into(),
            RelationSpec {
                head: vec!["Service".into(), "Component".into()],
                tail: vec![
                    "Service".into(),
                    "Component".into(),
                    "Database".into(),
                ],
                description: "dependency relationship".into(),
            },
        );
        relation_types.insert(
            "fixed".into(),
            RelationSpec {
                head: vec!["Person".into(), "Component".into()],
                tail: vec!["Component".into(), "Service".into()],
                description: "something fixed an issue".into(),
            },
        );
        relation_types.insert(
            "introduced".into(),
            RelationSpec {
                head: vec!["Person".into()],
                tail: vec!["Component".into(), "Pattern".into()],
                description: "person introduced a component".into(),
            },
        );
        relation_types.insert(
            "deprecated".into(),
            RelationSpec {
                head: vec!["Person".into(), "Decision".into()],
                tail: vec!["Component".into(), "Pattern".into()],
                description: "deprecation action".into(),
            },
        );
        relation_types.insert(
            "caused".into(),
            RelationSpec {
                head: vec!["Component".into(), "Decision".into()],
                tail: vec!["Metric".into(), "Constraint".into()],
                description: "causal relationship".into(),
            },
        );
        relation_types.insert(
            "constrained_by".into(),
            RelationSpec {
                head: vec!["Decision".into(), "Component".into()],
                tail: vec!["Constraint".into()],
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
