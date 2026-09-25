//! Resolve scored addresses to stable corpus rows outside the timed search.

use summa_core::Index;
use summa_core::directories::RamDirectory;
use summa_core::query::SearchHit;

pub async fn corpus_ids(index: &Index<RamDirectory>, hits: &[SearchHit]) -> Vec<usize> {
    let field = index
        .schema()
        .get_field("corpus_id")
        .expect("corpus ID field");
    let mut ids = Vec::with_capacity(hits.len());
    for hit in hits {
        let document = index
            .get_document(&hit.address)
            .await
            .expect("read benchmark hit")
            .expect("benchmark hit exists");
        ids.push(
            document
                .get_first(field)
                .and_then(|value| value.as_u64())
                .expect("stored corpus ID") as usize,
        );
    }
    ids
}
