//! Test document retrieval from the test index

use hermes_core::{FsDirectory, Index, IndexConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = FsDirectory::new("data/test_index");
    let config = IndexConfig::default();
    let index = Index::open(dir, config).await?;

    println!("Num docs: {}", index.num_docs());

    if let Some(doc) = index.doc(0).await? {
        println!("Doc 0 field_values count: {}", doc.field_values().len());
        for (field, value) in doc.field_values() {
            println!("  Field {:?}: {:?}", field, value);
        }

        let json = doc.to_json(index.schema());
        println!("Doc as JSON: {}", serde_json::to_string_pretty(&json)?);
    } else {
        println!("Doc 0 not found");
    }

    // Test search
    let results = index.query("rust", 5).await?;
    println!("\nSearch results: {} hits", results.hits.len());
    for hit in &results.hits {
        println!("  Doc {}: score={:.4}", hit.address.doc_id, hit.score,);
    }

    Ok(())
}
