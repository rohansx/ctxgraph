pub mod decisions;
pub mod entities;
pub mod init;
pub mod log;
pub mod models;
pub mod query;
pub mod stats;

use std::env;
use std::path::PathBuf;

use ctxgraph::Graph;

/// Find and open the nearest .ctxgraph/graph.db, searching up from cwd.
/// If extraction models are available, loads the extraction pipeline.
pub fn open_graph() -> ctxgraph::Result<Graph> {
    let db_path = find_db()?;
    let mut graph = Graph::open(&db_path)?;

    if let Some(models_dir) = find_models_dir(&db_path) {
        match graph.load_extraction_pipeline(&models_dir) {
            Ok(()) => {}
            Err(e) => {
                eprintln!(
                    "ctxgraph: extraction pipeline not loaded: {e}\n\
                     hint: place ONNX model files in {}",
                    models_dir.display()
                );
            }
        }
    }

    Ok(graph)
}

/// Locate models directory by checking (in order):
/// 1. `CTXGRAPH_MODELS_DIR` env var
/// 2. `~/.cache/ctxgraph/models`
/// 3. `.ctxgraph/models` next to the database
fn find_models_dir(db_path: &std::path::Path) -> Option<PathBuf> {
    // 1. Env var override
    if let Ok(val) = env::var("CTXGRAPH_MODELS_DIR") {
        let p = PathBuf::from(val);
        if p.is_dir() {
            return Some(p);
        }
    }

    // 2. ~/.cache/ctxgraph/models
    if let Ok(home) = env::var("HOME") {
        let p = PathBuf::from(home).join(".cache/ctxgraph/models");
        if p.is_dir() {
            return Some(p);
        }
    }

    // 3. .ctxgraph/models relative to the found .ctxgraph dir
    if let Some(ctxgraph_dir) = db_path.parent() {
        let p = ctxgraph_dir.join("models");
        if p.is_dir() {
            return Some(p);
        }
    }

    None
}

fn find_db() -> ctxgraph::Result<PathBuf> {
    let mut dir = env::current_dir().map_err(ctxgraph::CtxGraphError::Io)?;

    loop {
        let candidate = dir.join(".ctxgraph").join("graph.db");
        if candidate.exists() {
            return Ok(candidate);
        }
        if !dir.pop() {
            break;
        }
    }

    Err(ctxgraph::CtxGraphError::NotFound(
        "no .ctxgraph/ found in current or parent directories. Run `ctxgraph init` first."
            .to_string(),
    ))
}
