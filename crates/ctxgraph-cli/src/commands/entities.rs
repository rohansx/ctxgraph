use crate::display;

use super::open_graph;

pub fn list(entity_type: Option<String>, limit: usize) -> ctxgraph::Result<()> {
    let graph = open_graph()?;
    let entities = graph.list_entities(entity_type.as_deref(), limit)?;

    if entities.is_empty() {
        println!("No entities found.");
        println!("Entities will be auto-extracted in v0.2. For now, add them via the Rust API.");
        return Ok(());
    }

    println!("{:<12} {:<14} NAME", "ID", "TYPE");
    println!("{}", "-".repeat(50));

    for entity in &entities {
        display::print_entity_row(entity);
    }

    println!("\n{} entities total", entities.len());

    Ok(())
}

pub fn show(id: String) -> ctxgraph::Result<()> {
    let graph = open_graph()?;

    // Try by ID first, then by name
    let entity = graph.get_entity(&id)?.or(graph.get_entity_by_name(&id)?);

    let Some(entity) = entity else {
        return Err(ctxgraph::CtxGraphError::NotFound(format!("entity '{id}'")));
    };

    let context = graph.get_entity_context(&entity.id)?;
    display::print_entity_context(&context);

    Ok(())
}
