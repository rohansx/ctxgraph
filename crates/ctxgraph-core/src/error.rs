use thiserror::Error;

#[derive(Error, Debug)]
pub enum CtxGraphError {
    #[error("storage error: {0}")]
    Storage(#[from] rusqlite::Error),

    #[error("not found: {0}")]
    NotFound(String),

    #[error("already exists: {0}")]
    AlreadyExists(String),

    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error("schema error: {0}")]
    Schema(String),

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, CtxGraphError>;
