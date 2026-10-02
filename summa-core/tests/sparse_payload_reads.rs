#![cfg(feature = "native")]

use std::sync::Arc;
use summa_core::directories::{MmapDirectory, PayloadReadBackend, PayloadReadService};
use summa_core::index::{Index, IndexConfig};
use summa_core::query::{FusionMethod, MultiValueCombiner, Query, SearchResult, SparseVectorQuery};
use summa_core::structures::{SparseFormat, SparseVectorConfig};
use summa_core::{Document, SchemaBuilder};

fn identities(hits: &[SearchResult]) -> Vec<(u128, u32, u32)> {
    hits.iter()
        .map(|hit| (hit.segment_id, hit.doc_id, hit.score.to_bits()))
        .collect()
}

#[test]
fn sparse_payload_reads_preserve_results_and_use_service_on_both_runtimes() {
    for multithread in [false, true] {
        let mut builder = if multithread {
            let mut builder = tokio::runtime::Builder::new_multi_thread();
            builder.worker_threads(2);
            builder
        } else {
            tokio::runtime::Builder::new_current_thread()
        };
        builder.enable_all().build().unwrap().block_on(async {
            let mut backends = vec![PayloadReadBackend::Pool];
            if cfg!(all(target_os = "linux", feature = "io-uring")) {
                backends.push(PayloadReadBackend::IoUring);
            }
            for backend in backends {
                check_backend(backend).await;
            }
        });
    }
}

async fn check_backend(backend: PayloadReadBackend) {
    let temp = tempfile::tempdir().unwrap();
    let mut schema = SchemaBuilder::default();
    let field = schema.add_sparse_vector_field_with_config(
        "sparse",
        true,
        false,
        SparseVectorConfig {
            format: SparseFormat::MaxScore,
            ..Default::default()
        },
    );
    let config = IndexConfig {
        num_threads: 2,
        num_indexing_threads: 1,
        merge_policy: Box::new(summa_core::merge::NoMergePolicy),
        ..Default::default()
    };
    let index = Index::create(
        MmapDirectory::new(temp.path()),
        schema.build(),
        config.clone(),
    )
    .await
    .unwrap();
    let mut writer = index.writer();
    for n in 0..640 {
        let mut document = Document::new();
        if n % 11 != 0 {
            document.add_sparse_vector(
                field,
                vec![(1, (n % 37 + 1) as f32), (3, (n % 7 + 1) as f32)],
            );
            if n % 5 == 0 {
                document.add_sparse_vector(field, vec![(1, 1.5), (7, (n % 29 + 1) as f32)]);
            }
        }
        writer.add_document(document).unwrap();
        if n == 319 || n == 639 {
            writer.commit().await.unwrap();
        }
    }
    writer.shutdown().await.unwrap();
    let mapped_reader = index.reader().await.unwrap();
    let mapped = mapped_reader.searcher().await.unwrap();
    let query: Arc<dyn Query> = Arc::new(SparseVectorQuery::new(
        field,
        vec![(1, 1.0), (3, 0.5), (7, 2.0)],
    ));
    let expected = mapped.search(query.as_ref(), 25).await.unwrap();
    let service = PayloadReadService::new(backend).unwrap();
    let stored_only = Index::open(
        MmapDirectory::new(temp.path()).with_payload_reads(service.clone()),
        config.clone(),
    )
    .await
    .unwrap();
    let stored_reader = stored_only.reader().await.unwrap();
    let stored_searcher = stored_reader.searcher().await.unwrap();
    let before = service.stats().submitted;
    assert_eq!(
        identities(&stored_searcher.search(query.as_ref(), 25).await.unwrap()),
        identities(&expected)
    );
    assert_eq!(
        service.stats().submitted,
        before,
        "stored-document backend must not opt sparse blocks in"
    );
    let directory = MmapDirectory::new(temp.path()).with_sparse_payload_reads(service.clone());
    let explicit_index = Index::open(directory.clone(), config).await.unwrap();
    let explicit_reader = explicit_index.reader().await.unwrap();
    let explicit = explicit_reader.searcher().await.unwrap();
    let before = service.stats().submitted;
    assert_eq!(
        identities(&explicit.search(query.as_ref(), 25).await.unwrap()),
        identities(&expected)
    );
    assert!(
        service.stats().submitted > before,
        "sparse search bypassed explicit payload service"
    );
    // Exhaust all postings: each of three short lists in two segments fits
    // one bounded read window, independent of the number of encoded blocks.
    let before = service.stats().submitted;
    assert_eq!(
        identities(&explicit.search(query.as_ref(), 640).await.unwrap()),
        identities(&mapped.search(query.as_ref(), 640).await.unwrap())
    );
    assert!(
        service.stats().submitted - before <= 6,
        "adjacent sparse blocks should share a bounded cursor read"
    );
    let before = service.stats().submitted;
    let actual = explicit
        .search_candidate_lists(std::slice::from_ref(&query), 25, None)
        .await
        .unwrap();
    assert_eq!(identities(&actual[0].0), identities(&expected));
    assert!(
        service.stats().submitted > before,
        "candidate lists bypassed explicit payload service"
    );
    let positioned = explicit
        .search_with_positions(query.as_ref(), 25)
        .await
        .unwrap();
    let expected_positioned = mapped
        .search_with_positions(query.as_ref(), 25)
        .await
        .unwrap();
    assert_eq!(
        serde_json::to_vec(&positioned).unwrap(),
        serde_json::to_vec(&expected_positioned).unwrap()
    );
    let boolean = summa_core::query::BooleanQuery::new()
        .must(SparseVectorQuery::new(field, vec![(1, 1.0)]))
        .should(SparseVectorQuery::new(field, vec![(7, 2.0)]));
    let actual_boolean = explicit.search_with_positions(&boolean, 25).await.unwrap();
    let expected_boolean = mapped.search_with_positions(&boolean, 25).await.unwrap();
    assert_eq!(
        serde_json::to_vec(&actual_boolean).unwrap(),
        serde_json::to_vec(&expected_boolean).unwrap()
    );
    let branches = [(query.as_ref(), 1.0)];
    let method = FusionMethod::Rrf { k: 60.0 };
    let expected_fused = mapped
        .search_fused(&branches, 25, 25, method, MultiValueCombiner::Max)
        .await
        .unwrap();
    let before = service.stats().submitted;
    let actual = explicit
        .search_fused(&branches, 25, 25, method, MultiValueCombiner::Max)
        .await
        .unwrap();
    assert_eq!(identities(&actual), identities(&expected_fused));
    assert!(
        service.stats().submitted > before,
        "fusion bypassed explicit payload service"
    );
    #[cfg(feature = "sync")]
    {
        let before = service.stats().submitted;
        for (left, right) in mapped
            .segment_readers()
            .iter()
            .zip(explicit.segment_readers())
        {
            let collect = |segment: &summa_core::segment::SegmentReader| {
                let mut scorer = query.scorer_sync(segment, 25).unwrap();
                let mut hits = Vec::new();
                while scorer.doc() != summa_core::structures::TERMINATED {
                    hits.push((scorer.doc(), scorer.score().to_bits()));
                    scorer.advance();
                }
                hits
            };
            assert_eq!(collect(left), collect(right));
        }
        assert_eq!(
            service.stats().submitted,
            before,
            "explicit sync APIs must retain their mapped path"
        );
    }
    // Retiring the owner rejects subsequent cold reads even while an old searcher lives.
    directory.retire_payload_reads().await.unwrap();
    assert!(explicit.search(query.as_ref(), 25).await.is_err());
    assert_eq!(
        identities(&mapped.search(query.as_ref(), 25).await.unwrap()),
        identities(&expected)
    );
    service.shutdown().await.unwrap();
    assert_eq!(service.stats().active, 0);
    assert_eq!(service.stats().quarantined_bytes, 0);
}

#[tokio::test]
async fn sparse_payload_roles_share_existing_cache_and_service_owners() {
    use summa_core::directories::{CachingDirectory, Directory, SliceCachingDirectory};
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("blocks"), b"encoded sparse blocks").unwrap();
    let service = PayloadReadService::new(PayloadReadBackend::Pool).unwrap();
    let mapped = MmapDirectory::new(root.path()).with_sparse_payload_reads(service.clone());
    let cached = SliceCachingDirectory::new(CachingDirectory::new(mapped, 4096), 4096);
    let path = std::path::Path::new("blocks");
    let sparse = cached.open_sparse_payload(path).await.unwrap().unwrap();
    assert_eq!(
        sparse.read_bytes_range(0..7).await.unwrap().as_slice(),
        b"encoded"
    );
    let completed = service.stats().completed;
    assert!(completed > 0);
    // Both roles use the same slice cache, not independently retained buffers.
    let ordinary = cached.open_payload(path).await.unwrap();
    assert_eq!(
        ordinary.read_bytes_range(0..7).await.unwrap().as_slice(),
        b"encoded"
    );
    assert_eq!(service.stats().completed, completed);
    service.shutdown().await.unwrap();
}
