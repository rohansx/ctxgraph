use crate::display;

use super::open_graph;

pub fn list(
    _after: Option<String>,
    _source: Option<String>,
    limit: usize,
) -> ctxgraph_core::Result<()> {
    let graph = open_graph()?;
    let episodes = graph.list_episodes(limit, 0)?;

    if episodes.is_empty() {
        println!("No episodes found. Log some decisions first:");
        println!("  ctxgraph log \"Chose Postgres for billing service\"");
        return Ok(());
    }

    println!("{:<12} {:<12} CONTENT", "ID", "SOURCE");
    println!("{}", "-".repeat(70));

    for episode in &episodes {
        display::print_episode_row(episode);
    }

    println!("\n{} episodes total", episodes.len());

    Ok(())
}

pub fn show(id: String) -> ctxgraph_core::Result<()> {
    let graph = open_graph()?;

    let episode = graph.get_episode(&id)?;
    let Some(episode) = episode else {
        return Err(ctxgraph_core::CtxGraphError::NotFound(format!(
            "episode '{id}'"
        )));
    };

    display::print_episode(&episode, None);

    Ok(())
}
