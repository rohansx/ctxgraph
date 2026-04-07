//! Minimal extraction tool that reads text from stdin and outputs JSON.
//! Used by the real-world benchmark script.
//!
//! Usage: echo "some text" | cargo run --example extract_json

use chrono::Utc;
use ctxgraph_extract::pipeline::ExtractionPipeline;
use ctxgraph_extract::schema::ExtractionSchema;
use serde_json::json;
use std::io::Read;

fn main() {
    let models_dir = std::env::var("CTXGRAPH_MODELS_DIR").unwrap_or_else(|_| {
        let home = dirs::cache_dir().expect("no cache dir");
        home.join("ctxgraph").join("models").display().to_string()
    });

    let pipeline = ExtractionPipeline::new(
        ExtractionSchema::default(),
        std::path::Path::new(&models_dir),
        0.3,
    )
    .expect("Failed to create pipeline");

    let mut text = String::new();
    std::io::stdin().read_to_string(&mut text).expect("Failed to read stdin");
    let text = text.trim();

    if text.is_empty() {
        eprintln!("No input text");
        std::process::exit(1);
    }

    match pipeline.extract(text, Utc::now()) {
        Ok(result) => {
            let entities: Vec<_> = result
                .entities
                .iter()
                .map(|e| {
                    json!({
                        "name": e.text,
                        "entity_type": e.entity_type,
                        "confidence": e.confidence,
                    })
                })
                .collect();

            let relations: Vec<_> = result
                .relations
                .iter()
                .map(|r| {
                    json!({
                        "head": r.head,
                        "relation": r.relation,
                        "tail": r.tail,
                        "confidence": r.confidence,
                    })
                })
                .collect();

            let output = json!({
                "entities": entities,
                "relations": relations,
            });

            println!("{}", serde_json::to_string(&output).unwrap());
        }
        Err(e) => {
            let output = json!({"error": e.to_string(), "entities": [], "relations": []});
            println!("{}", serde_json::to_string(&output).unwrap());
        }
    }
}
