//! Shared document hydration and fixture policy for the I/O probes.
use crate::{E, payload_io::PayloadDirectory};
use summa_core::{
    Document,
    index::{IndexConfig, Searcher},
    query::DocAddress,
};
pub(crate) fn config(cache: usize) -> IndexConfig {
    IndexConfig {
        num_threads: 1,
        num_indexing_threads: 1,
        store_cache_budget_bytes: cache,
        merge_policy: Box::new(summa_core::merge::NoMergePolicy),
        ..Default::default()
    }
}
pub(crate) async fn hydrate(
    searcher: &Searcher<PayloadDirectory>,
    addresses: &[DocAddress],
    batch: bool,
) -> Result<Vec<Document>, E> {
    let mut docs = Vec::with_capacity(addresses.len());
    if batch && searcher.has_lazy_document_reads() {
        let prepared = searcher.prepare_document_reads(addresses).await?;
        for p in 0..addresses.len() {
            docs.push(prepared.get(p, None).await?.unwrap());
        }
    } else {
        for address in addresses {
            docs.push(searcher.get_document(address).await?.unwrap());
        }
    }
    Ok(docs)
}
