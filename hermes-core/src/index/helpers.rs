//! Indexing helper functions
//!
//! This module provides high-level helper functions for creating indexes
//! and indexing documents, used by hermes-tool, hermes-server, and hermes-core-python.

use std::io::BufRead;
use std::path::Path;

use crate::directories::{Directory, DirectoryWriter, FsDirectory};
use crate::dsl::{Document, Schema, SchemaBuilder, parse_single_index};
use crate::error::{Error, Result};
use crate::index::{IndexConfig, IndexWriter};

/// Schema configuration from JSON format
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct SchemaFieldConfig {
    /// Field name
    pub name: String,
    /// Field type: text, u64, i64, f64, bytes, json, sparse_vector, dense_vector
    #[serde(rename = "type")]
    pub field_type: String,
    /// Whether field is indexed (default: true)
    #[serde(default = "default_true")]
    pub indexed: bool,
    /// Whether field is stored (default: true)
    #[serde(default = "default_true")]
    pub stored: bool,
    /// Dimension for dense_vector fields
    #[serde(default)]
    pub dimension: usize,
}

fn default_true() -> bool {
    true
}

/// JSON schema configuration
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct SchemaConfig {
    /// List of field definitions
    pub fields: Vec<SchemaFieldConfig>,
}

impl SchemaConfig {
    /// Build a Schema from this configuration
    pub fn build(&self) -> Result<Schema> {
        let mut builder = SchemaBuilder::default();

        for field in &self.fields {
            match field.field_type.as_str() {
                "text" => {
                    builder.add_text_field(&field.name, field.indexed, field.stored);
                }
                "u64" => {
                    builder.add_u64_field(&field.name, field.indexed, field.stored);
                }
                "i64" => {
                    builder.add_i64_field(&field.name, field.indexed, field.stored);
                }
                "f64" => {
                    builder.add_f64_field(&field.name, field.indexed, field.stored);
                }
                "bytes" => {
                    builder.add_bytes_field(&field.name, field.stored);
                }
                "json" => {
                    builder.add_json_field(&field.name, field.stored);
                }
                "sparse_vector" => {
                    builder.add_sparse_vector_field(&field.name, field.indexed, field.stored);
                }
                "dense_vector" => {
                    builder.add_dense_vector_field(
                        &field.name,
                        field.dimension,
                        field.indexed,
                        field.stored,
                    );
                }
                other => {
                    return Err(Error::Schema(format!("Unknown field type: {}", other)));
                }
            }
        }

        Ok(builder.build())
    }
}

/// Parse schema from a string (auto-detects JSON or SDL format)
pub fn parse_schema(content: &str) -> Result<Schema> {
    let trimmed = content.trim();

    // Detect SDL format (starts with "index " or "#" for comments)
    if trimmed.starts_with("index ") || trimmed.starts_with('#') {
        let index_def = parse_single_index(content)
            .map_err(|e| Error::Schema(format!("Failed to parse SDL: {}", e)))?;
        Ok(index_def.to_schema())
    } else {
        // Try JSON format
        let config: SchemaConfig = serde_json::from_str(content)
            .map_err(|e| Error::Schema(format!("Failed to parse JSON schema: {}", e)))?;
        config.build()
    }
}

/// Create a new index at the given path with the provided schema
pub async fn create_index_at_path(
    path: impl AsRef<Path>,
    schema: Schema,
    config: IndexConfig,
) -> Result<IndexWriter<FsDirectory>> {
    let path = path.as_ref();

    std::fs::create_dir_all(path).map_err(|e| {
        Error::Io(std::io::Error::new(
            e.kind(),
            format!("Failed to create index directory {:?}: {}", path, e),
        ))
    })?;

    let dir = FsDirectory::new(path);
    IndexWriter::create(dir, schema, config).await
}

/// Create a new index from an SDL schema string
pub async fn create_index_from_sdl(
    path: impl AsRef<Path>,
    sdl: &str,
    config: IndexConfig,
) -> Result<IndexWriter<FsDirectory>> {
    let schema = parse_schema(sdl)?;
    create_index_at_path(path, schema, config).await
}

/// Indexing statistics
#[derive(Debug, Clone, Default)]
pub struct IndexingStats {
    /// Number of documents indexed
    pub indexed: usize,
    /// Number of documents that failed to parse
    pub errors: usize,
    /// Total time in seconds
    pub elapsed_secs: f64,
}

impl IndexingStats {
    /// Documents per second rate
    pub fn docs_per_sec(&self) -> f64 {
        if self.elapsed_secs > 0.0 {
            self.indexed as f64 / self.elapsed_secs
        } else {
            0.0
        }
    }
}

/// Index documents from a JSONL reader
///
/// Each line should be a valid JSON object. Documents are parsed according
/// to the schema and indexed. Returns statistics about the indexing operation.
pub async fn index_documents_from_reader<D, R>(
    writer: &mut IndexWriter<D>,
    reader: R,
    progress_callback: Option<&dyn Fn(usize)>,
) -> Result<IndexingStats>
where
    D: Directory + DirectoryWriter,
    R: BufRead,
{
    let schema = writer.schema();
    let mut stats = IndexingStats::default();
    let start_time = std::time::Instant::now();

    for line in reader.lines() {
        let line = line.map_err(Error::Io)?;
        if line.trim().is_empty() {
            continue;
        }

        let json: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => {
                stats.errors += 1;
                continue;
            }
        };

        let doc = match Document::from_json(&json, schema) {
            Some(d) => d,
            None => {
                stats.errors += 1;
                continue;
            }
        };

        writer.add_document(doc)?;
        stats.indexed += 1;

        if let Some(callback) = progress_callback {
            callback(stats.indexed);
        }
    }

    writer.commit().await?;
    stats.elapsed_secs = start_time.elapsed().as_secs_f64();

    Ok(stats)
}

/// Index a single document from JSON
pub async fn index_json_document<D>(writer: &IndexWriter<D>, json: &serde_json::Value) -> Result<()>
where
    D: Directory + DirectoryWriter,
{
    let schema = writer.schema();
    let doc = Document::from_json(json, schema)
        .ok_or_else(|| Error::Document("Failed to parse JSON document".to_string()))?;
    writer.add_document(doc)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::directories::RamDirectory;

    #[test]
    fn test_schema_config_json() {
        let json = r#"{
            "fields": [
                {"name": "title", "type": "text", "indexed": true, "stored": true},
                {"name": "body", "type": "text"},
                {"name": "score", "type": "f64", "indexed": false}
            ]
        }"#;

        let config: SchemaConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.fields.len(), 3);

        let schema = config.build().unwrap();
        assert!(schema.get_field("title").is_some());
        assert!(schema.get_field("body").is_some());
        assert!(schema.get_field("score").is_some());
    }

    #[test]
    fn test_parse_schema_json() {
        let json = r#"{"fields": [{"name": "text", "type": "text"}]}"#;
        let schema = parse_schema(json).unwrap();
        assert!(schema.get_field("text").is_some());
    }

    #[test]
    fn test_parse_schema_sdl() {
        let sdl = r#"
            index test {
                field text: text [indexed, stored]
            }
        "#;
        let schema = parse_schema(sdl).unwrap();
        assert!(schema.get_field("text").is_some());
    }

    #[tokio::test]
    async fn test_index_documents_from_reader() {
        let mut builder = SchemaBuilder::default();
        let _title = builder.add_text_field("title", true, true);
        let schema = builder.build();

        let dir = RamDirectory::new();
        let config = IndexConfig::default();
        let mut writer = IndexWriter::create(dir, schema, config).await.unwrap();

        let jsonl = r#"{"title": "Doc 1"}
{"title": "Doc 2"}
{"title": "Doc 3"}"#;

        let reader = std::io::Cursor::new(jsonl);
        let stats = index_documents_from_reader(&mut writer, reader, None)
            .await
            .unwrap();

        assert_eq!(stats.indexed, 3);
        assert_eq!(stats.errors, 0);
    }
}
