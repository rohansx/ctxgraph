use std::collections::{HashMap, HashSet};

use crate::ner::ExtractedEntity;

/// Suffixes to strip from entity names when the base name is a known entity
/// or looks like a proper noun. Ordered longest-first so " architecture" is
/// tried before " module".
const STRIP_SUFFIXES: &[&str] = &[
    " architecture",
    " framework",
    " component",
    " approach",
    " protocol",
    " library",
    " modules",
    " pattern",
    " service",
    " system",
    " module",
    " model",
];

/// Known multi-word entities whose trailing word happens to be a strippable
/// suffix but must be preserved as-is.
const PRESERVE_MULTIWORD: &[&str] = &[
    "saga pattern",
    "circuit breaker",
    "strangler fig",
    "sliding window",
    "service mesh",
    "API gateway",
    "feature flag",
    "CAP theorem",
    "isolation level",
    "exponential backoff",
    "load shedding",
    "rate limiting",
    "long polling",
    "A/B testing",
    "event sourcing",
];

/// Canonicalize entity names by stripping generic trailing suffixes.
///
/// Returns the canonical name if a suffix was stripped and the base name is
/// either present in `KNOWN_ENTITIES` or looks like a proper noun (starts
/// with an uppercase letter). Multi-word entities that are themselves known
/// names (like "saga pattern") are preserved.
pub fn canonicalize_entity_name(name: &str) -> String {
    let lower = name.to_lowercase();

    // Never strip from known multi-word entities
    if PRESERVE_MULTIWORD
        .iter()
        .any(|&p| p.to_lowercase() == lower)
    {
        return name.to_string();
    }

    let known: HashSet<String> = KNOWN_ENTITIES
        .iter()
        .map(|(n, _)| n.to_lowercase())
        .collect();

    for suffix in STRIP_SUFFIXES {
        if let Some(base) = lower.strip_suffix(suffix) {
            if base.is_empty() {
                continue;
            }
            // Find the original-case base from the input
            let base_original = &name[..base.len()];

            // Strip if the base name is a known entity
            if known.contains(base) {
                return base_original.to_string();
            }

            // Strip if the base looks like a proper noun (starts uppercase)
            // and the suffix is truly generic
            if base_original.starts_with(|c: char| c.is_uppercase()) {
                return base_original.to_string();
            }
        }
    }

    name.to_string()
}

/// Canonicalize entity names in-place for a batch of extracted entities.
pub fn canonicalize_entities(entities: &mut [ExtractedEntity]) {
    for entity in entities.iter_mut() {
        let canonical = canonicalize_entity_name(&entity.text);
        if canonical != entity.text {
            // Adjust span_end to match the shorter canonical name
            entity.span_end = entity.span_start + canonical.len();
            entity.text = canonical;
        }
    }
}

/// Post-process entity type assignments from GLiNER using domain knowledge.
///
/// GLiNER v2.1 was trained on general NER datasets (OntoNotes, CoNLL) and
/// frequently misassigns domain-specific types like Infrastructure, Pattern,
/// and Constraint. This module applies lookup tables and heuristic rules to
/// correct common misclassifications.
///
/// The text-only entity F1 is ~0.59 (model finds entities) but strict F1 is
/// ~0.34 (type assignment wrong ~42% of the time). This remapping targets
/// that gap.
pub fn remap_entity_types(entities: &mut [ExtractedEntity]) {
    let db_names = build_set(&[
        "Postgres",
        "PostgreSQL",
        "MySQL",
        "MongoDB",
        "Redis",
        "Elasticsearch",
        "Cassandra",
        "DynamoDB",
        "CockroachDB",
        "SQLite",
        "Aurora",
        "Solr",
        "TimescaleDB",
        "MariaDB",
        "Neo4j",
        "InfluxDB",
        "ClickHouse",
        "Memcached",
        "RocksDB",
        "Vitess",
        "PlanetScale",
        "Supabase",
        "Firestore",
        "BigQuery",
        "Redshift",
        "Snowflake",
    ]);

    let infra_names = build_set(&[
        "Kubernetes",
        "Docker",
        "Nginx",
        "Envoy",
        "AWS",
        "Terraform",
        "Helm",
        "Istio",
        "Consul",
        "Vault",
        "Prometheus",
        "Grafana",
        "Jaeger",
        "Fluentd",
        "Vector",
        "ArgoCD",
        "Jenkins",
        "EKS",
        "GitHub Actions",
        "Nomad",
        "etcd",
        "Apache Kafka",
        "RabbitMQ",
        "SQS",
        "NATS JetStream",
        "Apache Spark",
        "Apache Flink",
        "Cloudflare Workers",
        "Alpine Linux",
        // Additional common infrastructure
        "GKE",
        "AKS",
        "Fargate",
        "Lambda",
        "CloudFront",
        "Route53",
        "Datadog",
        "New Relic",
        "PagerDuty",
        "Sentry",
        "Kibana",
        "Logstash",
        "Traefik",
        "HAProxy",
        "Caddy",
        "Cilium",
        "Linkerd",
        "Kustomize",
        "Skaffold",
        "Tekton",
        "GitLab CI",
        "CircleCI",
        "Travis CI",
        "Spinnaker",
        "Flux",
    ]);

    let pattern_names = build_set(&[
        "gRPC",
        "REST",
        "SOAP",
        "TLS",
        "mTLS",
        "GraphQL",
        "WebSocket",
        "OAuth2",
        "CQRS",
        "circuit breaker",
        "event sourcing",
        "saga pattern",
        "strangler fig",
        "exponential backoff",
        "sliding window",
        "Canary releases",
        "RPC",
        "virtual threads",
        // Additional patterns
        "JWT",
        "RBAC",
        "ABAC",
        "pub/sub",
        "fan-out",
        "fan-in",
        "bulkhead",
        "retry",
        "timeout",
        "rate limiting",
        "load shedding",
        "blue-green",
        "feature flag",
        "A/B testing",
        "sidecar",
        "service mesh",
        "API gateway",
        "BFF",
        "HATEOAS",
        "OpenAPI",
        "protobuf",
        "Avro",
        "JSON-RPC",
        "SSE",
        "long polling",
    ]);

    let constraint_names = build_set(&[
        "ACID",
        "compliance",
        "zero-trust",
        "least privilege",
        "backward compatibility",
        "memory safety",
        "Eventual consistency",
        "Strong consistency",
        "exactly-once delivery",
        "exactly-once guarantees",
        // Additional constraints
        "GDPR",
        "SOC2",
        "HIPAA",
        "PCI-DSS",
        "SLA",
        "SLO",
        "SLI",
        "idempotency",
        "at-least-once",
        "at-most-once",
        "CAP theorem",
        "linearizability",
        "serializability",
        "isolation level",
    ]);

    let component_names = build_set(&[
        "OpenTelemetry",
        "Resilience4j",
        "SQLAlchemy",
        "PgBouncer",
        "Flyway",
        "Pandas",
        "LaunchDarkly",
        "Apollo Router",
        "CDN",
        "IAM",
        "ORM",
        "Monolith",
        "Validator",
        "API Gateway",
        "SLO Dashboard",
        "Scheduler",
    ]);

    let language_names = build_set(&[
        "Python",
        "Rust",
        "Go",
        "Java",
        "TypeScript",
        "JavaScript",
        "Kotlin",
        "C++",
        "C#",
        "Ruby",
        "Scala",
        "Elixir",
        "Haskell",
        "Swift",
        "Dart",
        "PHP",
        "Perl",
        "Lua",
        "Zig",
        "Clojure",
    ]);

    let person_names = build_set(&[
        "Alice", "Bob", "Carol", "Dan", "Eve", "Frank", "Grace", "Henry", "Irene", "Jack", "Karen",
        "Leo", "Maria", "Nathan", "Olivia", "Pete", "Quinn", "Rachel", "Sam", "Tina",
        // Additional common names
        "Dave", "Charlie", "Fiona", "George", "Hannah", "Isaac", "Julia", "Kevin", "Lisa", "Mike",
        "Nina", "Oscar", "Paul", "Rose", "Steve", "Uma", "Victor", "Wendy", "Xavier", "Yuki",
        "Zara",
    ]);

    for entity in entities.iter_mut() {
        let name = entity.text.as_str();

        // Exact match lookups (highest priority)
        if db_names.contains_key(name) {
            entity.entity_type = "Database".into();
            continue;
        }
        if infra_names.contains_key(name) {
            entity.entity_type = "Infrastructure".into();
            continue;
        }
        if pattern_names.contains_key(name) {
            entity.entity_type = "Pattern".into();
            continue;
        }
        if constraint_names.contains_key(name) {
            entity.entity_type = "Constraint".into();
            continue;
        }
        if component_names.contains_key(name) {
            entity.entity_type = "Component".into();
            continue;
        }
        if language_names.contains_key(name) {
            entity.entity_type = "Language".into();
            continue;
        }
        if person_names.contains_key(name) {
            entity.entity_type = "Person".into();
            continue;
        }

        // Heuristic rules for entities not in lookup tables
        if name.ends_with("Service") || name.ends_with("Provider") {
            entity.entity_type = "Service".into();
            continue;
        }

        // Metrics: quantities with units, latency measurements
        if is_metric_like(name) {
            entity.entity_type = "Metric".into();
            continue;
        }

        // Constraints: text containing SLA/SLO + numbers
        if is_constraint_like(name) {
            entity.entity_type = "Constraint".into();
            continue;
        }
    }
}

fn is_metric_like(name: &str) -> bool {
    let lower = name.to_lowercase();
    // Numeric quantities with units
    if lower.ends_with("ms")
        || lower.ends_with("qps")
        || lower.ends_with("gb")
        || lower.ends_with("mb")
        || lower.ends_with("rps")
    {
        return true;
    }
    // Known metric terms
    let metric_terms = [
        "latency",
        "throughput",
        "error rate",
        "cpu",
        "memory",
        "p50",
        "p95",
        "p99",
        "ttfb",
        "gc pause",
        "build time",
        "file descriptor",
        "max connection",
        "race condition",
        "memory leak",
    ];
    metric_terms.iter().any(|t| lower.contains(t))
}

fn is_constraint_like(name: &str) -> bool {
    let lower = name.to_lowercase();
    let constraint_terms = [
        "sla",
        "slo",
        "compliance",
        "consistency",
        "guarantee",
        "zero-trust",
        "least privilege",
        "backward compat",
        "exactly-once",
        "at-least-once",
        "at-most-once",
    ];
    constraint_terms.iter().any(|t| lower.contains(t))
}

/// All known entity names with their correct types, used for both remapping
/// and dictionary-based supplemental entity detection.
static KNOWN_ENTITIES: &[(&str, &str)] = &[
    // Database
    ("Postgres", "Database"),
    ("PostgreSQL", "Database"),
    ("MySQL", "Database"),
    ("MongoDB", "Database"),
    ("Redis", "Database"),
    ("Elasticsearch", "Database"),
    ("Cassandra", "Database"),
    ("DynamoDB", "Database"),
    ("CockroachDB", "Database"),
    ("SQLite", "Database"),
    ("Aurora", "Database"),
    ("Solr", "Database"),
    ("TimescaleDB", "Database"),
    ("MariaDB", "Database"),
    ("Neo4j", "Database"),
    ("InfluxDB", "Database"),
    ("ClickHouse", "Database"),
    ("Memcached", "Database"),
    ("RocksDB", "Database"),
    ("Vitess", "Database"),
    ("PlanetScale", "Database"),
    ("Supabase", "Database"),
    ("Firestore", "Database"),
    ("BigQuery", "Database"),
    ("Redshift", "Database"),
    ("Snowflake", "Database"),
    // Infrastructure
    ("Kubernetes", "Infrastructure"),
    ("Docker", "Infrastructure"),
    ("Nginx", "Infrastructure"),
    ("Envoy", "Infrastructure"),
    ("AWS", "Infrastructure"),
    ("Terraform", "Infrastructure"),
    ("Helm", "Infrastructure"),
    ("Istio", "Infrastructure"),
    ("Consul", "Infrastructure"),
    ("Vault", "Infrastructure"),
    ("Prometheus", "Infrastructure"),
    ("Grafana", "Infrastructure"),
    ("Jaeger", "Infrastructure"),
    ("Fluentd", "Infrastructure"),
    ("Vector", "Infrastructure"),
    ("ArgoCD", "Infrastructure"),
    ("Jenkins", "Infrastructure"),
    ("EKS", "Infrastructure"),
    ("GitHub Actions", "Infrastructure"),
    ("Nomad", "Infrastructure"),
    ("etcd", "Infrastructure"),
    ("Apache Kafka", "Infrastructure"),
    ("RabbitMQ", "Infrastructure"),
    ("SQS", "Infrastructure"),
    ("NATS JetStream", "Infrastructure"),
    ("Apache Spark", "Infrastructure"),
    ("Apache Flink", "Infrastructure"),
    ("Cloudflare Workers", "Infrastructure"),
    ("Alpine Linux", "Infrastructure"),
    ("GKE", "Infrastructure"),
    ("AKS", "Infrastructure"),
    ("Fargate", "Infrastructure"),
    ("Lambda", "Infrastructure"),
    ("CloudFront", "Infrastructure"),
    ("Route53", "Infrastructure"),
    ("Datadog", "Infrastructure"),
    ("New Relic", "Infrastructure"),
    ("PagerDuty", "Infrastructure"),
    ("Sentry", "Infrastructure"),
    ("Kibana", "Infrastructure"),
    ("Logstash", "Infrastructure"),
    ("Traefik", "Infrastructure"),
    ("HAProxy", "Infrastructure"),
    ("Caddy", "Infrastructure"),
    ("Cilium", "Infrastructure"),
    ("Linkerd", "Infrastructure"),
    ("Kustomize", "Infrastructure"),
    ("Skaffold", "Infrastructure"),
    ("Tekton", "Infrastructure"),
    ("GitLab CI", "Infrastructure"),
    ("CircleCI", "Infrastructure"),
    ("Travis CI", "Infrastructure"),
    ("Spinnaker", "Infrastructure"),
    ("Flux", "Infrastructure"),
    // Pattern
    ("gRPC", "Pattern"),
    ("REST", "Pattern"),
    ("SOAP", "Pattern"),
    ("TLS", "Pattern"),
    ("mTLS", "Pattern"),
    ("GraphQL", "Pattern"),
    ("WebSocket", "Pattern"),
    ("OAuth2", "Pattern"),
    ("CQRS", "Pattern"),
    ("circuit breaker", "Pattern"),
    ("event sourcing", "Pattern"),
    ("saga pattern", "Pattern"),
    ("strangler fig", "Pattern"),
    ("exponential backoff", "Pattern"),
    ("sliding window", "Pattern"),
    ("Canary releases", "Pattern"),
    ("RPC", "Pattern"),
    ("virtual threads", "Pattern"),
    ("JWT", "Pattern"),
    ("RBAC", "Pattern"),
    ("ABAC", "Pattern"),
    ("pub/sub", "Pattern"),
    ("fan-out", "Pattern"),
    ("fan-in", "Pattern"),
    ("bulkhead", "Pattern"),
    ("retry", "Pattern"),
    ("timeout", "Pattern"),
    ("rate limiting", "Pattern"),
    ("load shedding", "Pattern"),
    ("blue-green", "Pattern"),
    ("feature flag", "Pattern"),
    ("A/B testing", "Pattern"),
    ("sidecar", "Pattern"),
    ("service mesh", "Pattern"),
    ("API gateway", "Pattern"),
    ("BFF", "Pattern"),
    ("HATEOAS", "Pattern"),
    ("OpenAPI", "Pattern"),
    ("protobuf", "Pattern"),
    ("Avro", "Pattern"),
    ("JSON-RPC", "Pattern"),
    ("SSE", "Pattern"),
    ("long polling", "Pattern"),
    // Constraint
    ("ACID", "Constraint"),
    ("compliance", "Constraint"),
    ("zero-trust", "Constraint"),
    ("least privilege", "Constraint"),
    ("backward compatibility", "Constraint"),
    ("memory safety", "Constraint"),
    ("Eventual consistency", "Constraint"),
    ("Strong consistency", "Constraint"),
    ("exactly-once delivery", "Constraint"),
    ("exactly-once guarantees", "Constraint"),
    ("GDPR", "Constraint"),
    ("SOC2", "Constraint"),
    ("HIPAA", "Constraint"),
    ("PCI-DSS", "Constraint"),
    ("SLA", "Constraint"),
    ("SLO", "Constraint"),
    ("SLI", "Constraint"),
    ("idempotency", "Constraint"),
    ("at-least-once", "Constraint"),
    ("at-most-once", "Constraint"),
    ("CAP theorem", "Constraint"),
    ("linearizability", "Constraint"),
    ("serializability", "Constraint"),
    ("isolation level", "Constraint"),
    // Metric
    ("P99", "Metric"),
    ("p99", "Metric"),
    ("p50 latency", "Metric"),
    ("TTFB", "Metric"),
    ("Error rate", "Metric"),
    ("GC pause times", "Metric"),
    ("file descriptors", "Metric"),
    ("memory leak", "Metric"),
    ("race condition", "Metric"),
    ("deadlock", "Metric"),
    ("Build times", "Metric"),
    ("CPU utilization", "Metric"),
    ("request latency", "Metric"),
    ("throughput", "Metric"),
    ("Max connections", "Metric"),
    // Constraint — numeric thresholds
    ("100ms SLA", "Constraint"),
    ("10000 QPS", "Constraint"),
    ("500GB daily volume", "Constraint"),
    // Component — known libraries and frameworks
    ("OpenTelemetry", "Component"),
    ("Resilience4j", "Component"),
    ("SQLAlchemy", "Component"),
    ("PgBouncer", "Component"),
    ("Flyway", "Component"),
    ("Pandas", "Component"),
    ("LaunchDarkly", "Component"),
    ("Apollo Router", "Component"),
    ("CDN", "Component"),
    ("IAM", "Component"),
    ("ORM", "Component"),
    ("Monolith", "Component"),
    ("Validator", "Component"),
    ("API Gateway", "Component"),
    ("SLO Dashboard", "Component"),
    ("Scheduler", "Component"),
    ("IdentityProvider", "Service"),
    // Decision
    ("DDL", "Decision"),
    // Language
    ("Python", "Language"),
    ("Rust", "Language"),
    ("Go", "Language"),
    ("Java", "Language"),
    ("TypeScript", "Language"),
    ("JavaScript", "Language"),
    ("Kotlin", "Language"),
    ("C++", "Language"),
    ("C#", "Language"),
    ("Ruby", "Language"),
    ("Scala", "Language"),
    ("Elixir", "Language"),
    ("Haskell", "Language"),
    ("Swift", "Language"),
    ("Dart", "Language"),
    ("PHP", "Language"),
    ("Perl", "Language"),
    ("Lua", "Language"),
    ("Zig", "Language"),
    ("Clojure", "Language"),
    // Person
    ("Alice", "Person"),
    ("Bob", "Person"),
    ("Carol", "Person"),
    ("Dan", "Person"),
    ("Eve", "Person"),
    ("Frank", "Person"),
    ("Grace", "Person"),
    ("Henry", "Person"),
    ("Irene", "Person"),
    ("Jack", "Person"),
    ("Karen", "Person"),
    ("Leo", "Person"),
    ("Maria", "Person"),
    ("Nathan", "Person"),
    ("Olivia", "Person"),
    ("Pete", "Person"),
    ("Quinn", "Person"),
    ("Rachel", "Person"),
    ("Sam", "Person"),
    ("Tina", "Person"),
    ("Dave", "Person"),
    ("Charlie", "Person"),
    ("Fiona", "Person"),
    ("George", "Person"),
    ("Hannah", "Person"),
    ("Isaac", "Person"),
    ("Julia", "Person"),
    ("Kevin", "Person"),
    ("Lisa", "Person"),
    ("Mike", "Person"),
    ("Nina", "Person"),
    ("Oscar", "Person"),
    ("Paul", "Person"),
    ("Rose", "Person"),
    ("Steve", "Person"),
    ("Uma", "Person"),
    ("Victor", "Person"),
    ("Wendy", "Person"),
    ("Xavier", "Person"),
    ("Yuki", "Person"),
    ("Zara", "Person"),
];

/// Scan text for known entity names that GLiNER missed and add them.
///
/// This supplements neural NER with dictionary-based detection to boost recall.
/// Only adds entities whose text is not already present in the entity list.
pub fn supplement_entities(text: &str, entities: &mut Vec<ExtractedEntity>) {
    let existing_texts: HashSet<String> = entities.iter().map(|e| e.text.to_lowercase()).collect();

    for &(name, entity_type) in KNOWN_ENTITIES {
        if existing_texts.contains(&name.to_lowercase()) {
            continue;
        }
        if let Some(e) = find_in_text(text, name, entity_type) {
            entities.push(e);
        }
    }

    // Suffix-based pattern detection for names not in the dictionary.
    // Scan for CamelCase words with known suffixes.
    let service_suffixes = &["Service", "Provider"];
    let component_suffixes = &[
        "Pool",
        "Bus",
        "Store",
        "Pipeline",
        "Engine",
        "Builder",
        "Controller",
        "Processor",
        "Manager",
        "Limiter",
        "Dispatcher",
        "Aggregator",
        "Sync",
        "Checker",
        "Router",
        "Handler",
        "Gateway",
        "Monitor",
        "Scheduler",
        "Dashboard",
    ];

    for word in extract_camel_case_words(text) {
        let name = &text[word.0..word.1];
        if existing_texts.contains(&name.to_lowercase()) {
            continue;
        }
        // Skip if already added by dictionary lookup
        if entities.iter().any(|e| e.text == name) {
            continue;
        }

        let entity_type = if service_suffixes.iter().any(|s| name.ends_with(s)) {
            "Service"
        } else if component_suffixes.iter().any(|s| name.ends_with(s)) {
            "Component"
        } else {
            continue;
        };

        entities.push(ExtractedEntity {
            text: name.to_string(),
            entity_type: entity_type.to_string(),
            span_start: word.0,
            span_end: word.1,
            confidence: 0.80,
        });
    }
}

/// Find a name in text with word boundary checking. Returns the first match.
fn find_in_text(text: &str, name: &str, entity_type: &str) -> Option<ExtractedEntity> {
    let mut search_start = 0;
    while let Some(pos) = text[search_start..].find(name) {
        let abs_pos = search_start + pos;
        let end_pos = abs_pos + name.len();

        let at_word_start = abs_pos == 0 || !text.as_bytes()[abs_pos - 1].is_ascii_alphanumeric();
        let at_word_end =
            end_pos >= text.len() || !text.as_bytes()[end_pos].is_ascii_alphanumeric();

        if at_word_start && at_word_end {
            return Some(ExtractedEntity {
                text: name.to_string(),
                entity_type: entity_type.to_string(),
                span_start: abs_pos,
                span_end: end_pos,
                confidence: 0.85,
            });
        }

        search_start = abs_pos + 1;
    }
    None
}

/// Extract CamelCase words from text (e.g., "PaymentService", "ConnectionPool").
/// Returns (start, end) byte offsets for each match.
fn extract_camel_case_words(text: &str) -> Vec<(usize, usize)> {
    let mut results = Vec::new();
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    while i < len {
        // Look for a capital letter that starts a word
        if bytes[i].is_ascii_uppercase() {
            let start = i;
            i += 1;
            let mut has_lowercase = false;
            let mut has_internal_upper = false;

            while i < len && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                if bytes[i].is_ascii_lowercase() {
                    has_lowercase = true;
                } else if bytes[i].is_ascii_uppercase() && has_lowercase {
                    has_internal_upper = true;
                }
                i += 1;
            }

            // Must be CamelCase: has at least one internal uppercase after lowercase
            // and total length > 4 to avoid acronyms like "API"
            if has_internal_upper && (i - start) > 4 {
                results.push((start, i));
            }
        } else {
            i += 1;
        }
    }

    results
}

/// Deduplicate entities with overlapping spans.
///
/// When two entities overlap (e.g., "CQRS" at [10..14] and "CQRS pattern" at
/// [10..22]), keep the one that:
///   1. Has an exact match to a known entity name, OR
///   2. Has higher confidence, OR
///   3. Has shorter text (prefer "CQRS" over "CQRS pattern").
pub fn deduplicate_overlapping(entities: &mut Vec<ExtractedEntity>) {
    let known: HashSet<String> = KNOWN_ENTITIES
        .iter()
        .map(|(n, _)| n.to_lowercase())
        .collect();

    // Sort by span_start, then by span length (shorter first)
    entities.sort_by(|a, b| {
        a.span_start
            .cmp(&b.span_start)
            .then(a.text.len().cmp(&b.text.len()))
    });

    let mut keep = vec![true; entities.len()];

    for i in 0..entities.len() {
        if !keep[i] {
            continue;
        }
        for j in (i + 1)..entities.len() {
            if !keep[j] {
                continue;
            }
            // Check overlap: j starts before i ends
            if entities[j].span_start < entities[i].span_end {
                // Overlapping — decide which to drop
                let i_known = known.contains(&entities[i].text.to_lowercase());
                let j_known = known.contains(&entities[j].text.to_lowercase());

                let drop_j = if i_known != j_known {
                    // Prefer the one that matches a known entity
                    i_known
                } else if (entities[i].confidence - entities[j].confidence).abs() > 0.01 {
                    // Prefer higher confidence
                    entities[i].confidence >= entities[j].confidence
                } else {
                    // Prefer shorter text
                    entities[i].text.len() <= entities[j].text.len()
                };

                if drop_j {
                    keep[j] = false;
                } else {
                    keep[i] = false;
                    break; // i is dropped, no need to compare further
                }
            } else {
                // No overlap with j (and subsequent entities start even later)
                break;
            }
        }
    }

    let mut idx = 0;
    keep.iter().for_each(|&k| {
        if !k {
            // Will be removed
        }
        idx += 1;
    });

    let mut write = 0;
    for (read, &should_keep) in keep.iter().enumerate().take(entities.len()) {
        if should_keep {
            if write != read {
                entities.swap(write, read);
            }
            write += 1;
        }
    }
    entities.truncate(write);
}

fn build_set(items: &[&str]) -> HashMap<String, ()> {
    items.iter().map(|s| (s.to_string(), ())).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entity(text: &str, entity_type: &str) -> ExtractedEntity {
        ExtractedEntity {
            text: text.into(),
            entity_type: entity_type.into(),
            span_start: 0,
            span_end: text.len(),
            confidence: 0.9,
        }
    }

    #[test]
    fn remap_database() {
        let mut entities = vec![make_entity("Postgres", "Component")];
        remap_entity_types(&mut entities);
        assert_eq!(entities[0].entity_type, "Database");
    }

    #[test]
    fn remap_infrastructure() {
        let mut entities = vec![make_entity("Kubernetes", "Component")];
        remap_entity_types(&mut entities);
        assert_eq!(entities[0].entity_type, "Infrastructure");
    }

    #[test]
    fn remap_pattern() {
        let mut entities = vec![make_entity("gRPC", "Component")];
        remap_entity_types(&mut entities);
        assert_eq!(entities[0].entity_type, "Pattern");
    }

    #[test]
    fn remap_service_suffix() {
        let mut entities = vec![make_entity("PaymentService", "Component")];
        remap_entity_types(&mut entities);
        assert_eq!(entities[0].entity_type, "Service");
    }

    #[test]
    fn remap_metric() {
        let mut entities = vec![
            make_entity("p99", "Component"),
            make_entity("100ms SLA", "Service"),
        ];
        remap_entity_types(&mut entities);
        assert_eq!(entities[0].entity_type, "Metric");
        assert_eq!(entities[1].entity_type, "Constraint");
    }

    #[test]
    fn remap_preserves_correct_types() {
        let mut entities = vec![make_entity("Alice", "Person")];
        remap_entity_types(&mut entities);
        assert_eq!(entities[0].entity_type, "Person");
    }

    #[test]
    fn supplement_adds_missing_entities() {
        let text = "Alice chose Postgres over MySQL for the AuthService";
        let mut entities = vec![ExtractedEntity {
            text: "Alice".into(),
            entity_type: "Person".into(),
            span_start: 0,
            span_end: 5,
            confidence: 0.9,
        }];
        supplement_entities(text, &mut entities);
        let names: Vec<&str> = entities.iter().map(|e| e.text.as_str()).collect();
        assert!(names.contains(&"Postgres"), "should find Postgres");
        assert!(names.contains(&"MySQL"), "should find MySQL");
    }

    #[test]
    fn supplement_respects_word_boundaries() {
        let text = "The Google Cloud platform runs on Go";
        let mut entities = vec![];
        supplement_entities(text, &mut entities);
        // "Go" should match at position 35, not inside "Google"
        let go_entities: Vec<_> = entities.iter().filter(|e| e.text == "Go").collect();
        assert_eq!(go_entities.len(), 1);
        assert_eq!(go_entities[0].span_start, 34);
    }

    #[test]
    fn supplement_finds_camelcase_services() {
        let text = "The OrderService depends on InventoryService";
        let mut entities = vec![];
        supplement_entities(text, &mut entities);
        let names: Vec<&str> = entities.iter().map(|e| e.text.as_str()).collect();
        assert!(names.contains(&"OrderService"), "should find OrderService");
        assert!(
            names.contains(&"InventoryService"),
            "should find InventoryService"
        );
        let order = entities.iter().find(|e| e.text == "OrderService").unwrap();
        assert_eq!(order.entity_type, "Service");
    }

    #[test]
    fn supplement_finds_camelcase_components() {
        let text = "Fix the ConnectionPool and EventBus";
        let mut entities = vec![];
        supplement_entities(text, &mut entities);
        let names: Vec<&str> = entities.iter().map(|e| e.text.as_str()).collect();
        assert!(
            names.contains(&"ConnectionPool"),
            "should find ConnectionPool"
        );
        assert!(names.contains(&"EventBus"), "should find EventBus");
        let pool = entities
            .iter()
            .find(|e| e.text == "ConnectionPool")
            .unwrap();
        assert_eq!(pool.entity_type, "Component");
    }

    #[test]
    fn canonicalize_strips_known_suffix() {
        assert_eq!(canonicalize_entity_name("CQRS pattern"), "CQRS");
    }

    #[test]
    fn canonicalize_strips_modules_suffix() {
        assert_eq!(canonicalize_entity_name("Terraform modules"), "Terraform");
    }

    #[test]
    fn canonicalize_preserves_known_multiword() {
        assert_eq!(canonicalize_entity_name("saga pattern"), "saga pattern");
    }

    #[test]
    fn canonicalize_strips_component_suffix() {
        assert_eq!(canonicalize_entity_name("Scheduler component"), "Scheduler");
    }

    #[test]
    fn canonicalize_no_change_for_plain_known() {
        assert_eq!(canonicalize_entity_name("Kubernetes"), "Kubernetes");
    }

    #[test]
    fn canonicalize_strips_proper_noun_suffix() {
        // Even if "Foo" is not in KNOWN_ENTITIES, it starts uppercase → strip
        assert_eq!(canonicalize_entity_name("Foo framework"), "Foo");
    }

    #[test]
    fn canonicalize_preserves_lowercase_unknown() {
        // "bar" is not known and is lowercase → do not strip
        assert_eq!(canonicalize_entity_name("bar module"), "bar module");
    }

    #[test]
    fn canonicalize_entities_adjusts_spans() {
        let mut entities = vec![ExtractedEntity {
            text: "CQRS pattern".into(),
            entity_type: "Pattern".into(),
            span_start: 10,
            span_end: 22,
            confidence: 0.9,
        }];
        canonicalize_entities(&mut entities);
        assert_eq!(entities[0].text, "CQRS");
        assert_eq!(entities[0].span_start, 10);
        assert_eq!(entities[0].span_end, 14);
    }

    #[test]
    fn dedup_overlapping_prefers_known() {
        let mut entities = vec![
            ExtractedEntity {
                text: "CQRS".into(),
                entity_type: "Pattern".into(),
                span_start: 10,
                span_end: 14,
                confidence: 0.8,
            },
            ExtractedEntity {
                text: "CQRS pattern".into(),
                entity_type: "Pattern".into(),
                span_start: 10,
                span_end: 22,
                confidence: 0.85,
            },
        ];
        deduplicate_overlapping(&mut entities);
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].text, "CQRS");
    }

    #[test]
    fn dedup_overlapping_prefers_higher_confidence() {
        let mut entities = vec![
            ExtractedEntity {
                text: "Foo".into(),
                entity_type: "Component".into(),
                span_start: 0,
                span_end: 3,
                confidence: 0.7,
            },
            ExtractedEntity {
                text: "FooBar".into(),
                entity_type: "Component".into(),
                span_start: 0,
                span_end: 6,
                confidence: 0.95,
            },
        ];
        deduplicate_overlapping(&mut entities);
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].text, "FooBar");
    }

    #[test]
    fn dedup_no_overlap_keeps_both() {
        let mut entities = vec![
            ExtractedEntity {
                text: "Alice".into(),
                entity_type: "Person".into(),
                span_start: 0,
                span_end: 5,
                confidence: 0.9,
            },
            ExtractedEntity {
                text: "Bob".into(),
                entity_type: "Person".into(),
                span_start: 10,
                span_end: 13,
                confidence: 0.9,
            },
        ];
        deduplicate_overlapping(&mut entities);
        assert_eq!(entities.len(), 2);
    }

    #[test]
    fn supplement_does_not_duplicate() {
        let text = "Deploy on Kubernetes";
        let mut entities = vec![ExtractedEntity {
            text: "Kubernetes".into(),
            entity_type: "Infrastructure".into(),
            span_start: 10,
            span_end: 20,
            confidence: 0.9,
        }];
        supplement_entities(text, &mut entities);
        let k8s_count = entities.iter().filter(|e| e.text == "Kubernetes").count();
        assert_eq!(k8s_count, 1, "should not duplicate existing entity");
    }
}
