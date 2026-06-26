use crate::display;

use super::open_graph;

pub fn run(
    text: String,
    limit: usize,
    _after: Option<String>,
    _source: Option<String>,
) -> ctxgraph::Result<()> {
    let graph = open_graph()?;

    let safe_query = super::query_universal::sanitize_query(&text);

    let results = graph.search(&safe_query, limit)?;

    if results.is_empty() {
        println!("No results found for '{text}'");
        return Ok(());
    }

    println!("Found {} result(s) for '{text}':\n", results.len());

    for (episode, score) in &results {
        display::print_episode(episode, Some(*score));
    }

    Ok(())
}
