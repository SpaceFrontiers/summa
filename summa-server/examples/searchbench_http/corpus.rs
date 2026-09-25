//! Benchmark schema and feed translation; persistence remains owned by core.

use std::io::BufRead;
use std::path::Path;

use anyhow::{Context, Result, bail};
use summa_core::directories::MmapDirectory;
use summa_core::dsl::{Document, PositionMode, Schema, SchemaBuilder};
use summa_core::{IndexConfig, IndexWriter};

pub const BODY_TOKENIZER: &str = "lex(segmenter: unicode, stem: none, stop_words: false, variants: false, fold: false, max_token_length: 255)";

pub fn schema(reorder: bool, tokenizer: &str) -> Schema {
    let mut schema = SchemaBuilder::default();
    let body = schema.add_text_field_with_tokenizer("body", true, true, tokenizer);
    schema.set_positions(body, PositionMode::TokenPosition);
    schema.set_reorder(body, reorder);
    schema.set_default_fields(vec!["body".into()]);
    for name in [
        "id",
        "cat_s",
        "cat100_s",
        "cat1k_s",
        "cat10k_s",
        "cat100k_s",
        "cat1m_s",
        "cat5m_s",
        "catu2m_s",
        "catid_s",
        "sel99_s",
        "sel90_s",
        "sel50_s",
        "sel10_s",
        "sel1_s",
        "sel01_s",
        "sel001_s",
        "date_dt",
    ] {
        // Date is retained as an exact string column: not queried by this
        // full-text campaign. This differs from the references' numeric date.
        let field = schema.add_text_field_with_tokenizer(name, true, false, "raw");
        schema.set_fast(field, true);
    }
    for name in ["sort_i", "price_i"] {
        let field = schema.add_u64_field(name, true, false);
        schema.set_fast(field, true);
    }
    schema.build()
}

pub async fn build(
    path: &Path,
    input: &Path,
    config: IndexConfig,
    reorder: bool,
    tokenizer: &str,
) -> Result<()> {
    if path.exists() {
        bail!("output index already exists: {}", path.display());
    }
    std::fs::create_dir_all(path)?;
    let schema = schema(reorder, tokenizer);
    let mut writer = IndexWriter::create(MmapDirectory::new(path), schema.clone(), config).await?;
    let source = std::io::BufReader::new(std::fs::File::open(input)?);
    let mut count = 0u64;
    for line in source.lines() {
        let value: serde_json::Value = serde_json::from_str(&line?)?;
        let object = value.as_object().context("corpus row must be an object")?;
        for name in object.keys() {
            if schema.get_field(name).is_none() {
                bail!("unknown corpus field {name}");
            }
        }
        for name in ["id", "body"] {
            if object.get(name).and_then(|value| value.as_str()).is_none() {
                bail!("corpus row requires string {name}");
            }
        }
        let doc = Document::from_json(&value, &schema).context("invalid corpus document")?;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(300);
        loop {
            match writer.add_document(doc.clone()) {
                Ok(()) => break,
                Err(summa_core::Error::QueueFull) if std::time::Instant::now() < deadline => {
                    tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                }
                Err(error) => return Err(error.into()),
            }
        }
        count += 1;
        if count.is_multiple_of(100_000) {
            eprintln!("indexed {count}");
        }
    }
    writer.commit().await?;
    writer.force_merge().await?;
    if reorder {
        eprintln!("starting RGB reorder");
        writer.reorder().await?;
        eprintln!("finished RGB reorder");
    }
    writer.shutdown().await?;
    eprintln!("committed and merged {count} documents");
    Ok(())
}
