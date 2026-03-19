// Rule-based coreference resolver: maps pronouns to preceding entity mentions.
// Simple but effective for well-structured technical writing.

use crate::ner::ExtractedEntity;

#[derive(Debug, Clone, Copy, PartialEq)]
enum PronounType {
    Person,
    Neuter,
    Plural,
}

pub struct CorefResolver;

impl CorefResolver {
    /// Given the original text and a list of extracted entities (from NER),
    /// find pronoun spans and resolve them to the most recently mentioned
    /// compatible entity. Returns additional entity mentions to add.
    pub fn resolve(text: &str, entities: &[ExtractedEntity]) -> Vec<ExtractedEntity> {
        if entities.is_empty() {
            return Vec::new();
        }

        // Sort entities by span_start ascending
        let mut sorted_entities: Vec<&ExtractedEntity> = entities.iter().collect();
        sorted_entities.sort_by_key(|e| e.span_start);

        let pronoun_spans = find_pronoun_spans(text);
        let mut result = Vec::new();

        for (pron_start, pron_end, pron_type) in &pronoun_spans {
            // Find most recently preceding entity compatible with pronoun type.
            // "preceding" means entity span_start < pronoun start (backward reference only).
            let candidate = sorted_entities
                .iter()
                .rev()
                .find(|e| {
                    e.span_start < *pron_start && is_compatible(pron_type, &e.entity_type)
                });

            if let Some(entity) = candidate {
                result.push(ExtractedEntity {
                    text: entity.text.clone(),
                    entity_type: entity.entity_type.clone(),
                    span_start: *pron_start,
                    span_end: *pron_end,
                    confidence: 0.45,
                });
            }
        }

        result
    }
}

fn is_compatible(pron_type: &PronounType, entity_type: &str) -> bool {
    match pron_type {
        PronounType::Person => entity_type.eq_ignore_ascii_case("Person"),
        PronounType::Neuter => !entity_type.eq_ignore_ascii_case("Person"),
        PronounType::Plural => true,
    }
}

/// Classify a lowercase word as a pronoun type, if it is one.
fn classify_pronoun(word: &str) -> Option<PronounType> {
    match word {
        "he" | "him" | "his" | "she" | "her" | "hers" => Some(PronounType::Person),
        "it" | "its" | "this" | "that" | "these" | "those" => Some(PronounType::Neuter),
        "they" | "them" | "their" | "theirs" | "we" | "our" | "us" => Some(PronounType::Plural),
        _ => None,
    }
}

/// Walk text and return (byte_start, byte_end, PronounType) for every pronoun found.
fn find_pronoun_spans(text: &str) -> Vec<(usize, usize, PronounType)> {
    let mut result = Vec::new();
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut i = 0usize;

    while i < len {
        // Skip non-alphabetic characters
        if !bytes[i].is_ascii_alphabetic() {
            i += 1;
            continue;
        }

        // Found start of a word — find its end
        let word_start = i;
        while i < len && bytes[i].is_ascii_alphabetic() {
            i += 1;
        }
        let word_end = i;

        // Lowercase for comparison
        let word_lower: String = bytes[word_start..word_end]
            .iter()
            .map(|b| b.to_ascii_lowercase() as char)
            .collect();

        if let Some(pron_type) = classify_pronoun(&word_lower) {
            result.push((word_start, word_end, pron_type));
        }
    }

    result
}
