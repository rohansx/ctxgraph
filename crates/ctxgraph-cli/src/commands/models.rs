use ctxgraph_extract::model_manager::{
    gliner_large_v21_int8, gliner_large_v21_tokenizer, gliner_multitask_large,
    gliner_multitask_tokenizer, ModelManager,
};

pub fn download() -> ctxgraph::Result<()> {
    let manager = ModelManager::new().map_err(|e| {
        ctxgraph::CtxGraphError::Extraction(format!("failed to initialize model manager: {e}"))
    })?;

    println!(
        "Downloading models to {}",
        manager.model_path(&gliner_large_v21_int8())
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "~/.cache/ctxgraph/models".into())
    );

    let specs = [
        ("GLiNER v2.1 NER model (INT8)", gliner_large_v21_int8()),
        ("GLiNER v2.1 tokenizer", gliner_large_v21_tokenizer()),
        ("GLiNER Multitask relation model (INT8)", gliner_multitask_large()),
        ("GLiNER Multitask tokenizer", gliner_multitask_tokenizer()),
    ];

    for (label, spec) in &specs {
        if manager.is_cached(spec) {
            println!("  {label}: cached");
        } else {
            println!("  {label}: downloading...");
            manager.get_or_download(spec).map_err(|e| {
                ctxgraph::CtxGraphError::Extraction(format!("download failed for {label}: {e}"))
            })?;
            println!("  {label}: done");
        }
    }

    println!("\nAll models ready. Run `ctxgraph init` to get started.");
    Ok(())
}
