use super::open_graph;

pub fn run() -> ctxgraph_core::Result<()> {
    let graph = open_graph()?;
    let stats = graph.stats()?;

    println!("ctxgraph stats");
    println!("{}", "-".repeat(30));
    println!("Episodes:  {}", stats.episode_count);
    println!("Entities:  {}", stats.entity_count);
    println!("Edges:     {}", stats.edge_count);

    if !stats.sources.is_empty() {
        let sources: Vec<String> = stats
            .sources
            .iter()
            .map(|(name, count)| format!("{name} ({count})"))
            .collect();
        println!("Sources:   {}", sources.join(", "));
    }

    println!("DB size:   {}", format_bytes(stats.db_size_bytes));

    Ok(())
}

fn format_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}
