//! Error types for minisearch

use std::io;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    #[error("Field not found: {0}")]
    FieldNotFound(String),

    #[error("Invalid field type: expected {expected}, got {got}")]
    InvalidFieldType { expected: String, got: String },

    #[error("Index corruption: {0}")]
    Corruption(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Index is closed")]
    IndexClosed,

    #[error("Document not found: {0}")]
    DocumentNotFound(u32),

    #[error("Schema error: {0}")]
    Schema(String),

    #[error("Query error: {0}")]
    Query(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, Error>;
