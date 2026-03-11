use ctxgraph_core::Episode;

use super::open_graph;

pub fn run(
    text: String,
    source: Option<String>,
    tags: Option<String>,
) -> ctxgraph_core::Result<()> {
    let graph = open_graph()?;

    let mut builder = Episode::builder(&text);

    if let Some(src) = &source {
        builder = builder.source(src);
    }

    if let Some(tags_str) = &tags {
        for tag in tags_str.split(',') {
            builder = builder.tag(tag.trim());
        }
    }

    let episode = builder.build();
    let result = graph.add_episode(episode)?;

    println!("Episode stored: {}", &result.episode_id[..8]);

    if result.entities_extracted > 0 {
        println!("  Extracted {} entities", result.entities_extracted);
    }
    if result.edges_created > 0 {
        println!("  Created {} edges", result.edges_created);
    }

    Ok(())
}
