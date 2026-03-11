pub mod decisions;
pub mod entities;
pub mod init;
pub mod log;
pub mod query;
pub mod stats;

use std::env;
use std::path::PathBuf;

use ctxgraph_core::Graph;

/// Find and open the nearest .ctxgraph/graph.db, searching up from cwd.
pub fn open_graph() -> ctxgraph_core::Result<Graph> {
    let db_path = find_db()?;
    Graph::open(&db_path)
}

fn find_db() -> ctxgraph_core::Result<PathBuf> {
    let mut dir = env::current_dir().map_err(ctxgraph_core::CtxGraphError::Io)?;

    loop {
        let candidate = dir.join(".ctxgraph").join("graph.db");
        if candidate.exists() {
            return Ok(candidate);
        }
        if !dir.pop() {
            break;
        }
    }

    Err(ctxgraph_core::CtxGraphError::NotFound(
        "no .ctxgraph/ found in current or parent directories. Run `ctxgraph init` first."
            .to_string(),
    ))
}
