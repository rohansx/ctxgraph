//! Test schema inference from sample text via LLM.
//!
//! Usage: cargo run --example infer_schema

use ctxgraph_extract::schema::ExtractionSchema;

fn main() {
    let samples = vec![
        "Novo Nordisk's Ozempic and Wegovy generated $28.3 billion in revenue. Eli Lilly's Mounjaro captured 34% US market share within 18 months.",
        "CRISPR Therapeutics and Vertex received FDA approval for Casgevy to treat sickle cell disease. The treatment costs $2.2 million per patient.",
        "The NHS launched the Federated Data Platform built by Palantir to unify patient records across 150 hospital trusts. GDPR concerns were raised by medConfidential.",
    ];

    let sample_refs: Vec<&str> = samples.iter().map(|s| *s).collect();

    // Try Ollama first (local), fall back to env vars
    let (url, key, model) = if reqwest::blocking::Client::new()
        .get("http://localhost:11434/api/tags")
        .timeout(std::time::Duration::from_secs(3))
        .send()
        .is_ok()
    {
        (
            "http://localhost:11434/v1/chat/completions",
            "ollama",
            std::env::var("CTXGRAPH_LLM_MODEL").unwrap_or_else(|_| "gemma3n:e4b".into()),
        )
    } else {
        (
            &*std::env::var("CTXGRAPH_LLM_URL")
                .unwrap_or_else(|_| "http://localhost:11434/v1/chat/completions".into()),
            &*std::env::var("CTXGRAPH_LLM_KEY").unwrap_or_else(|_| "ollama".into()),
            std::env::var("CTXGRAPH_LLM_MODEL").unwrap_or_else(|_| "gemma3n:e4b".into()),
        )
    };

    eprintln!("Inferring schema from {} samples using {model}...", samples.len());

    match ExtractionSchema::infer_from_text(&sample_refs, url, key, &model) {
        Ok(schema) => {
            eprintln!("\nInferred schema: {}", schema.name);
            eprintln!("\nEntity types:");
            for (k, v) in &schema.entity_types {
                eprintln!("  {k}: {v}");
            }
            eprintln!("\nRelation types:");
            for (k, v) in &schema.relation_types {
                eprintln!("  {k}: {}", v.description);
            }

            // Save to file
            let path = std::path::Path::new("/tmp/inferred_schema.toml");
            schema.save(path).expect("Failed to save schema");
            eprintln!("\nSaved to {}", path.display());

            // Print the TOML
            let content = std::fs::read_to_string(path).unwrap();
            println!("{content}");
        }
        Err(e) => {
            eprintln!("Schema inference failed: {e}");
            std::process::exit(1);
        }
    }
}
