//! Regression test: on the wasm branch the in-memory document store buffer
//! (`SegmentBuilder::store_buffer`) must be counted in the builder's
//! estimated memory. Otherwise the wasm `IndexWriter`'s memory-budget
//! auto-flush never fires for stored-heavy workloads and the buffer grows
//! unboundedly until the browser tab OOMs, losing all uncommitted documents.
//!
//! Run with: `cargo test -p hermes-core --no-default-features \
//!   --features wasm,http --test wasm_builder_store_memory_regression`
#![cfg(all(feature = "wasm", not(feature = "native")))]

use hermes_core::directories::RamDirectory;
use hermes_core::{Document, IndexConfig, Schema, WasmIndexWriter};

#[tokio::test]
async fn test_wasm_store_buffer_counts_against_memory_budget() {
    let mut builder = Schema::builder();
    // Stored-but-unindexed field: contributes nothing to postings memory,
    // everything to the in-memory store buffer.
    let body = builder.add_text_field("body", false, true);
    let schema = builder.build();

    let config = IndexConfig {
        max_indexing_memory_bytes: 1024 * 1024, // 1 MiB budget
        ..Default::default()
    };
    let mut writer = WasmIndexWriter::create(RamDirectory::new(), schema, config)
        .await
        .expect("create");

    // 120 docs x 64 KiB stored payload ~= 7.5 MiB of store buffer — far past
    // the 1 MiB budget (the auto-flush also requires >= 100 buffered docs).
    let payload = "x".repeat(64 * 1024);
    for i in 0..120 {
        let mut doc = Document::new();
        doc.add_text(body, format!("{i} {payload}"));
        writer.add_document(doc).await.expect("add_document");
    }
    writer.commit().await.expect("commit");

    assert!(
        writer.metadata().segment_metas.len() >= 2,
        "stored-heavy workload must trip the memory-budget auto-flush; the \
         store buffer is invisible to estimated_memory — got {} segment(s)",
        writer.metadata().segment_metas.len()
    );
}
