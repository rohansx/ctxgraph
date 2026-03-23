use std::env;

use ctxgraph::Graph;

pub fn run(name: Option<String>) -> ctxgraph::Result<()> {
    let dir = env::current_dir().map_err(ctxgraph::CtxGraphError::Io)?;
    let _graph = Graph::init(&dir)?;

    let project_name = name.unwrap_or_else(|| {
        dir.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unnamed")
            .to_string()
    });

    println!("Initialized ctxgraph for '{project_name}'");
    println!("  Database: .ctxgraph/graph.db");
    println!();
    println!("Get started:");
    println!("  ctxgraph models download    Download ONNX models for extraction");
    println!("  ctxgraph log \"Your first decision or event\"");
    println!("  ctxgraph query \"search for something\"");

    Ok(())
}
