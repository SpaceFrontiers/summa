//! Test document retrieval from the test index

use hermes_core::{Index, IndexConfig, MmapDirectory};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = MmapDirectory::new("data/test_index");
    let config = IndexConfig::default();
    let index = Index::open(dir, config).await?;

    // Get reader and searcher for querying
    let reader = index.reader().await?;
    let searcher = reader.searcher().await?;

    println!("Num docs: {}", searcher.num_docs());

    let seg_id = searcher.segment_readers()[0].meta().id;
    if let Some(doc) = searcher.doc(seg_id, 0).await? {
        println!("Doc 0 field_values count: {}", doc.field_values().len());
        for (field, value) in doc.field_values() {
            println!("  Field {:?}: {:?}", field, value);
        }

        let json = doc.to_json(index.schema());
        println!("Doc as JSON: {}", serde_json::to_string_pretty(&json)?);
    } else {
        println!("Doc 0 not found");
    }

    // Note: query() method removed - use search_segment directly if needed
    println!("\nSearch functionality available via reader.segment_readers()");

    Ok(())
}
