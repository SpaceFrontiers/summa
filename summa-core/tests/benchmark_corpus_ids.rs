#[path = "../benches/summa_benchmark/corpus.rs"]
mod corpus;

use summa_core::directories::RamDirectory;
use summa_core::{Document, Index, IndexConfig, SchemaBuilder};

#[tokio::test]
async fn benchmark_identity_survives_commits_and_force_merge() {
    let mut schema = SchemaBuilder::default();
    let text = schema.add_text_field("content", true, false);
    let original = schema.add_u64_field("corpus_id", false, true);
    let index = Index::create(RamDirectory::new(), schema.build(), IndexConfig::default())
        .await
        .unwrap();
    let mut writer = index.writer();
    // Deliberately use row IDs different from every possible segment-local ID.
    for row in [91, 37, 82] {
        let mut document = Document::new();
        document.add_text(text, "search");
        document.add_u64(original, row);
        writer.add_document(document).unwrap();
        writer.commit().await.unwrap();
    }
    for merge in [false, true] {
        if merge {
            writer.force_merge().await.unwrap();
        }
        let results = index.query("search", 10).await.unwrap();
        let mut ids = corpus::corpus_ids(&index, &results.hits).await;
        ids.sort_unstable();
        assert_eq!(ids, vec![37, 82, 91]);
    }
}
