//! Native full-query comparisons; no alternate nomination or scoring executor.
use crate::directories::MmapDirectory;
use crate::dsl::{BinaryDenseVectorConfig, Document, Field, IvfRoutingMode, SchemaBuilder};
use crate::index::{Index, IndexConfig, Searcher};
use crate::query::{BinaryDenseVectorQuery, BinaryRetrievalOptions, MultiValueCombiner};
use std::sync::Arc;

fn config(budget: usize) -> IndexConfig {
    IndexConfig {
        num_threads: 2,
        num_indexing_threads: 1,
        num_compression_threads: 1,
        background_reload: true,
        binary_prefix_cache_budget_bytes: budget,
        merge_policy: Box::new(crate::NoMergePolicy),
        ..Default::default()
    }
}
fn code(row: usize) -> Vec<u8> {
    let mut seed = (row as u64 + 1).wrapping_mul(0x9e3779b97f4a7c15);
    (0..320)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed as u8
        })
        .collect()
}
async fn fixture(dir: &MmapDirectory) {
    let mut schema = SchemaBuilder::default();
    let id = schema.add_text_field("id", true, true);
    schema.set_primary_key(id);
    let vector = schema.add_binary_dense_vector_field_with_config(
        "vector",
        true,
        false,
        BinaryDenseVectorConfig::new(2560)
            .with_ivf(Some(8), 4)
            .with_ivf_routing(IvfRoutingMode::Hnsw)
            .with_prefix(1024, Some(12))
            .with_soar(crate::structures::SoarConfig::full()),
    );
    schema.set_multi(vector, true);
    let index = Index::create(dir.clone(), schema.build(), config(0))
        .await
        .unwrap();
    let mut writer = index.writer();
    writer.init_primary_key_dedup().await.unwrap();
    for batch in 0..4 {
        for within in 0..12 {
            let row = batch * 12 + within;
            let mut doc = Document::new();
            doc.add_text(id, row.to_string());
            for ordinal in 0..if row == 47 { 73 } else { 3 } {
                doc.add_binary_dense_vector(vector, code(row * 97 + ordinal));
            }
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();
    }
    writer.build_vector_index().await.unwrap();
    writer.shutdown().await.unwrap();
}
type Controls = Vec<Vec<(u128, u32, u32)>>;
async fn controls(searcher: &Searcher<MmapDirectory>) -> Controls {
    let mut controls = Vec::new();
    for qi in 0..8 {
        for combiner in [MultiValueCombiner::Max, MultiValueCombiner::default()] {
            for nprobe in [1, 8] {
                for prefix_rerank in [0, 1, 1000] {
                    for k in [1, 8] {
                        let query = BinaryDenseVectorQuery::new(Field(1), code(200 + qi))
                            .with_combiner(combiner)
                            .with_retrieval_options(BinaryRetrievalOptions {
                                nprobe,
                                prefix_rerank,
                                ..Default::default()
                            });
                        controls.push(
                            searcher
                                .search(&query, k)
                                .await
                                .unwrap()
                                .into_iter()
                                .map(|h| (h.segment_id, h.doc_id, h.score.to_bits()))
                                .collect(),
                        );
                    }
                }
            }
        }
    }
    controls
}
async fn assert_mapped_results(dir: &MmapDirectory, actual: &Searcher<MmapDirectory>) {
    let baseline = Index::open(dir.clone(), config(0)).await.unwrap();
    let reader = baseline.reader().await.unwrap();
    assert_eq!(reader.binary_prefix_cache_stats().allocated_bytes, 0);
    assert_eq!(
        controls(actual).await,
        controls(&reader.searcher().await.unwrap()).await
    );
}

#[tokio::test]
async fn resident_binary_prefix_cache_preserves_native_results_through_merge_deletion_and_compaction()
 {
    let temp = tempfile::tempdir().unwrap();
    let dir = MmapDirectory::new(temp.path());
    fixture(&dir).await;
    let (index, mut writer) = Index::open_with_writer(dir.clone(), config(1024 * 1024))
        .await
        .unwrap();
    writer.init_primary_key_dedup().await.unwrap();
    let budget = Arc::clone(&index.search_resources.binary_prefix_cache);
    let reader = index.reader().await.unwrap();
    let old = reader.searcher().await.unwrap();
    let old_controls = controls(&old).await;
    assert_eq!(reader.binary_prefix_cache_stats().completed_builds, 4);
    assert!(
        old.segment_readers()
            .iter()
            .all(|s| s.resident_binary_prefix_bytes() > 0)
    );
    assert_mapped_results(&dir, &old).await;
    writer.delete_primary_key("0").unwrap();
    writer.delete_primary_key("17").unwrap();
    writer.commit().await.unwrap();
    reader.reload().await.unwrap();
    let deleted = reader.searcher().await.unwrap();
    assert_eq!(
        reader.binary_prefix_cache_stats().completed_builds,
        4,
        "visibility reload reused cache owners"
    );
    assert_eq!(controls(&old).await, old_controls);
    assert_mapped_results(&dir, &deleted).await;
    writer.force_merge().await.unwrap();
    reader.reload().await.unwrap();
    let merged = reader.searcher().await.unwrap();
    assert_eq!(merged.segment_readers().len(), 1);
    assert!(merged.segment_readers()[0].resident_binary_prefix_bytes() > 0);
    assert_mapped_results(&dir, &merged).await;
    assert_eq!(writer.compact(1024 * 1024).await.unwrap(), 1);
    reader.reload().await.unwrap();
    let compacted = reader.searcher().await.unwrap();
    assert_eq!(compacted.segment_readers()[0].num_docs(), 46);
    assert_mapped_results(&dir, &compacted).await;
    assert_eq!(controls(&old).await, old_controls);
    assert!(reader.binary_prefix_cache_stats().allocated_bytes <= 1024 * 1024);
    drop((old, deleted, merged, compacted));
    writer.shutdown().await.unwrap();
    drop(writer);
    drop(index);
    assert_eq!(budget.stats().allocated_bytes, 0);
}

#[tokio::test]
async fn resident_binary_prefix_cache_falls_back_during_merge_and_retries_unchanged_generation() {
    let temp = tempfile::tempdir().unwrap();
    let dir = MmapDirectory::new(temp.path());
    fixture(&dir).await;
    let measured = Index::open(dir.clone(), config(1024 * 1024)).await.unwrap();
    let cap = measured
        .reader()
        .await
        .unwrap()
        .binary_prefix_cache_stats()
        .allocated_bytes;
    assert!(cap > 0);
    drop(measured);
    let (index, mut writer) = Index::open_with_writer(dir.clone(), config(cap))
        .await
        .unwrap();
    writer.init_primary_key_dedup().await.unwrap();
    let reader = index.reader().await.unwrap();
    let old = reader.searcher().await.unwrap();
    let expected = controls(&old).await;
    assert_eq!(reader.binary_prefix_cache_stats().allocated_bytes, cap);
    writer.force_merge().await.unwrap();
    reader.reload().await.unwrap();
    let current = reader.searcher().await.unwrap();
    assert_eq!(
        current.segment_readers()[0].resident_binary_prefix_bytes(),
        0
    );
    assert!(reader.binary_prefix_cache_stats().refused_builds > 0);
    assert_mapped_results(&dir, &current).await;
    assert_eq!(controls(&old).await, expected);
    drop(old);
    reader.refresh().await.unwrap();
    reader.reload().await.unwrap();
    assert!(
        current.segment_readers()[0].resident_binary_prefix_bytes() > 0,
        "unchanged reader uses newly admitted immutable pool"
    );
    let stats = reader.binary_prefix_cache_stats();
    assert!(stats.allocated_bytes <= cap);
    assert!(stats.peak_allocated_bytes <= cap);
    assert_mapped_results(&dir, &current).await;
    writer.shutdown().await.unwrap();
}

#[tokio::test]
async fn resident_binary_prefix_cache_build_cancellation_rolls_back_and_retries() {
    use crate::segment::{PrefixCacheBudget, VectorIndex};
    let temp = tempfile::tempdir().unwrap();
    let dir = MmapDirectory::new(temp.path());
    fixture(&dir).await;
    let index = Index::open(dir, config(0)).await.unwrap();
    let searcher = index.reader().await.unwrap().searcher().await.unwrap();
    let segment = &searcher.segment_readers()[0];
    let Some(VectorIndex::BinaryIvf(ann)) = segment.get_vector_index(Field(1)) else {
        panic!("binary IVF")
    };
    let flat = &segment.flat_vectors()[&1];
    let budget = PrefixCacheBudget::new(1024 * 1024);
    let expected = controls(&searcher).await;
    let mut pending = Box::pin(ann.get().prepare_prefix_cache(flat, &budget));
    assert!(futures::poll!(&mut pending).is_pending());
    assert!(budget.stats().allocated_bytes > 0);
    drop(pending);
    assert_eq!(budget.stats().allocated_bytes, 0);
    assert_eq!(budget.stats().aborted_builds, 1);
    ann.get().prepare_prefix_cache(flat, &budget).await.unwrap();
    assert!(budget.stats().allocated_bytes > 0);
    assert_eq!(controls(&searcher).await, expected);
}
