use ctxgraph_extract::CorefResolver;
use ctxgraph_extract::ner::ExtractedEntity;

fn make_entity(text: &str, entity_type: &str, start: usize, end: usize) -> ExtractedEntity {
    ExtractedEntity {
        text: text.to_string(),
        entity_type: entity_type.to_string(),
        span_start: start,
        span_end: end,
        confidence: 0.9,
    }
}

/// "Alice chose Postgres. She said it was faster."
/// "She" → Alice (Person), "it" → Postgres (Component)
#[test]
fn test_person_and_neuter_pronoun_resolution() {
    let text = "Alice chose Postgres. She said it was faster.";
    //           0123456789012345678901234567890123456789012345
    //           Alice=0..5, Postgres=12..20, She=22..25, it=30..32

    let alice_start = text.find("Alice").unwrap();
    let alice_end = alice_start + "Alice".len();
    let postgres_start = text.find("Postgres").unwrap();
    let postgres_end = postgres_start + "Postgres".len();

    let entities = vec![
        make_entity("Alice", "Person", alice_start, alice_end),
        make_entity("Postgres", "Component", postgres_start, postgres_end),
    ];

    let coref = CorefResolver::resolve(text, &entities);

    // Find "She" resolution → should be Alice (Person)
    let she_pos = text.find("She").unwrap();
    let she_coref: Vec<_> = coref.iter().filter(|e| e.span_start == she_pos).collect();
    assert_eq!(she_coref.len(), 1, "Expected 1 coref for 'She'");
    assert_eq!(she_coref[0].text, "Alice");
    assert_eq!(she_coref[0].entity_type, "Person");
    assert!((she_coref[0].confidence - 0.45).abs() < 1e-9);

    // Find "it" resolution → should be Postgres (Component, non-Person)
    let it_pos = text.find(" it ").unwrap() + 1; // skip space before "it"
    let it_coref: Vec<_> = coref.iter().filter(|e| e.span_start == it_pos).collect();
    assert_eq!(it_coref.len(), 1, "Expected 1 coref for 'it'");
    assert_eq!(it_coref[0].text, "Postgres");
    assert_eq!(it_coref[0].entity_type, "Component");
}

/// "Bob and Carol reviewed the PR. He approved it."
/// "He" → most recently preceding Person (Carol appears last alphabetically but Bob is first in text).
/// The resolver picks the LAST Person before "He" which is Carol.
#[test]
fn test_plural_persons_he_resolves_to_most_recent() {
    let text = "Bob and Carol reviewed the PR. He approved it.";
    //           Bob=0..3, Carol=8..13, He=31..33

    let bob_start = text.find("Bob").unwrap();
    let bob_end = bob_start + "Bob".len();
    let carol_start = text.find("Carol").unwrap();
    let carol_end = carol_start + "Carol".len();

    let entities = vec![
        make_entity("Bob", "Person", bob_start, bob_end),
        make_entity("Carol", "Person", carol_start, carol_end),
    ];

    let coref = CorefResolver::resolve(text, &entities);

    // "He" should resolve to Carol (most recent Person before "He")
    let he_pos = text.find(" He ").unwrap() + 1;
    let he_coref: Vec<_> = coref.iter().filter(|e| e.span_start == he_pos).collect();
    assert_eq!(he_coref.len(), 1, "Expected 1 coref for 'He'");
    assert_eq!(
        he_coref[0].text, "Carol",
        "He should resolve to Carol (most recent Person)"
    );
}

/// No entities → no coref output (must not crash).
#[test]
fn test_no_entities_returns_empty() {
    let text = "She said it was faster.";
    let result = CorefResolver::resolve(text, &[]);
    assert!(
        result.is_empty(),
        "Expected no corefs when entities list is empty"
    );
}

/// Pronoun before entity → should NOT be resolved (no forward reference).
#[test]
fn test_pronoun_before_entity_not_resolved() {
    // "She" appears before "Alice" in text
    let text = "She approved the design. Alice came later.";
    //           She=0..3, Alice=25..30

    let alice_start = text.find("Alice").unwrap();
    let alice_end = alice_start + "Alice".len();

    let entities = vec![make_entity("Alice", "Person", alice_start, alice_end)];

    let coref = CorefResolver::resolve(text, &entities);

    // "She" at position 0 has no preceding Person entity → should not produce a coref
    let she_pos = text.find("She").unwrap();
    let she_corefs: Vec<_> = coref.iter().filter(|e| e.span_start == she_pos).collect();
    assert!(
        she_corefs.is_empty(),
        "Pronoun before entity should not be resolved, got: {:?}",
        she_corefs
    );
}
