mod basic;
#[cfg(feature = "native")]
mod binary_prefix_cache;
mod bmp;
mod boolean;
mod chunked;
mod config_policy;
mod maintenance;
mod merge;
mod merge_bounds;
mod physical_ties;
mod pin;
mod posting_codecs;
mod primary_key;
mod range;
mod search;
#[cfg(feature = "sync")]
mod search_cpu;
#[cfg(feature = "sync")]
mod search_pool_liveness;
mod seismic_lifecycle;
mod seismic_operations;

mod vector;
mod word_pairs;

#[cfg(feature = "sync")]
#[test]
fn search_cpu_pool_is_bounded_and_reused_by_width() {
    let first = super::shared_search_pool(1).unwrap();
    let second = super::shared_search_pool(1).unwrap();

    assert!(std::sync::Arc::ptr_eq(&first, &second));
    assert_eq!(first.install(rayon::current_num_threads), 1);
}

#[cfg(feature = "native")]
fn per_segment(blocks: usize, budget_bytes: Option<usize>) -> crate::segment::TermCachePolicy {
    crate::segment::TermCachePolicy::PerSegment {
        blocks,
        budget_bytes,
    }
}

#[cfg(feature = "native")]
#[test]
fn zero_search_threads_is_rejected() {
    assert!(super::searcher::SearcherResources::new(per_segment(1, None), 1, 0, 1).is_err());
}

#[cfg(feature = "native")]
#[test]
fn zero_sparse_io_concurrency_is_rejected() {
    assert!(super::searcher::SearcherResources::new(per_segment(1, None), 1, 1, 0).is_err());
}

#[cfg(feature = "native")]
#[test]
fn oversized_term_cache_block_cap_is_rejected_at_load() {
    for process_bytes in [0, 1 << 20] {
        let config = super::IndexConfig {
            term_cache_blocks: super::MAX_TERM_CACHE_BLOCKS + 1,
            term_cache_process_bytes: process_bytes,
            ..Default::default()
        };
        let error = super::searcher::SearcherResources::from_config(&config)
            .err()
            .expect("cap above MAX_TERM_CACHE_BLOCKS must fail");
        assert!(error.to_string().contains("term_cache_blocks"), "{error}");
    }
    let config = super::IndexConfig {
        term_cache_blocks: super::MAX_TERM_CACHE_BLOCKS,
        term_cache_budget_bytes: Some(0),
        term_cache_process_bytes: 0,
        ..Default::default()
    };
    assert!(
        super::searcher::SearcherResources::from_config(&config).is_ok_and(|resources| matches!(
            resources.term_cache,
            crate::segment::TermCachePolicy::PerSegment {
                budget_bytes: Some(0),
                ..
            }
        ))
    );
}

#[cfg(feature = "native")]
#[test]
fn indexes_with_equal_term_cache_budgets_share_one_process_cache() {
    let config = super::IndexConfig {
        term_cache_process_bytes: 3 << 20,
        ..Default::default()
    };
    let shared = |config: &super::IndexConfig| match super::term_cache_policy(config).unwrap() {
        crate::segment::TermCachePolicy::Shared(cache) => cache,
        policy => panic!("expected the process-wide cache, got {policy:?}"),
    };
    let first = shared(&config);
    let second = shared(&config);
    assert!(std::sync::Arc::ptr_eq(&first, &second));
    assert_eq!(first.max_bytes(), 3 << 20);
    let other = shared(&super::IndexConfig {
        term_cache_process_bytes: 5 << 20,
        ..Default::default()
    });
    assert!(!std::sync::Arc::ptr_eq(&first, &other));
}

#[cfg(feature = "sync")]
#[tokio::test]
async fn sparse_io_gate_shares_capacity_between_sync_and_async_paths() {
    use std::time::Duration;

    let gate = super::SparseIoGate::new(1);
    let blocking = gate.acquire();
    let mut async_waiter = Box::pin(gate.acquire_async());
    assert!(
        tokio::time::timeout(Duration::from_millis(10), &mut async_waiter)
            .await
            .is_err()
    );

    drop(blocking);
    let async_permit = tokio::time::timeout(Duration::from_millis(100), &mut async_waiter)
        .await
        .expect("async sparse scorer should wake when the shared slot is released");
    drop(async_permit);

    let _blocking_again = gate.acquire();
}

#[cfg(feature = "sync")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_sparse_io_gate_state_reports_blocked_and_parked_waiters() {
    use std::time::Duration;

    let gate = std::sync::Arc::new(super::SparseIoGate::new(1));
    let held = gate.acquire();
    let blocked = {
        let gate = std::sync::Arc::clone(&gate);
        std::thread::spawn(move || drop(gate.acquire()))
    };
    let mut parked = Box::pin(gate.acquire_async());
    assert!(
        tokio::time::timeout(Duration::from_millis(10), &mut parked)
            .await
            .is_err()
    );
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while gate.state().waiting < 2 {
        assert!(
            std::time::Instant::now() < deadline,
            "waiters were not counted"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(
        gate.state(),
        super::SparseIoGateState {
            limit: 1,
            active: 1,
            waiting: 2
        }
    );

    // A cancelled async acquisition stops counting as a waiter.
    drop(parked);
    assert_eq!(gate.state().waiting, 1);
    drop(held);
    blocked.join().unwrap();
    assert_eq!(
        gate.state(),
        super::SparseIoGateState {
            limit: 1,
            active: 0,
            waiting: 0
        }
    );
}

#[cfg(feature = "sync")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_search_runtime_state_counts_pool_jobs_until_they_finish() {
    use crate::directories::RamDirectory;
    use crate::dsl::{Document, SchemaBuilder};

    let mut sb = SchemaBuilder::default();
    let title = sb.add_text_field("title", true, true);
    let schema = sb.build();
    // A width no other test uses, so this test owns the pool.
    let config = super::IndexConfig {
        num_threads: 3,
        ..Default::default()
    };
    let dir = RamDirectory::new();
    let mut writer = super::IndexWriter::create(dir.clone(), schema, config.clone())
        .await
        .unwrap();
    let mut doc = Document::new();
    doc.add_text(title, "hello");
    writer.add_document(doc).unwrap();
    writer.commit().await.unwrap();
    let index = super::Index::open(dir, config).await.unwrap();
    let searcher = index.reader().await.unwrap().searcher().await.unwrap();

    let baseline = super::search_runtime_state();
    let (release, wait) = std::sync::mpsc::channel::<()>();
    let job = {
        let searcher = std::sync::Arc::clone(&searcher);
        tokio::spawn(async move { searcher.run_on_search_pool(move || wait.recv()).await })
    };
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while super::search_runtime_state().running_pool_jobs <= baseline.running_pool_jobs {
        assert!(
            std::time::Instant::now() < deadline,
            "running job was not counted"
        );
        tokio::task::yield_now().await;
    }
    release.send(()).unwrap();
    job.await.unwrap().unwrap().unwrap();
    // Counters are process-wide; other tests may run jobs concurrently, so
    // only require that this job no longer counts.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while super::search_runtime_state().running_pool_jobs > baseline.running_pool_jobs {
        assert!(
            std::time::Instant::now() < deadline,
            "finished job still counted"
        );
        tokio::task::yield_now().await;
    }
}

#[cfg(feature = "native")]
#[test]
fn reorder_gate_clamps_oversubscription() {
    let gate = super::ReorderConcurrencyGate::new(usize::MAX);
    assert_eq!(gate.limit(), super::MAX_CONCURRENT_REORDER_PASSES);
}

#[cfg(feature = "native")]
#[tokio::test]
async fn automatic_merge_reorder_leaves_capacity_for_optimizer() {
    use super::ReorderPriority;
    use std::sync::Arc;
    use std::time::Duration;

    let gate = Arc::new(super::ReorderConcurrencyGate::new(2));
    let first_merge = gate.acquire(ReorderPriority::AutomaticMerge).await.unwrap();

    // A second automatic merge must wait even though one total slot remains.
    let second_merge = tokio::time::timeout(
        Duration::from_millis(25),
        gate.acquire(ReorderPriority::AutomaticMerge),
    )
    .await;
    assert!(second_merge.is_err());

    // The remaining slot is deliberately available to the optimizer so short
    // fresh-segment passes cannot queue behind minutes-long merge BP work.
    let optimizer = tokio::time::timeout(
        Duration::from_millis(100),
        gate.acquire(ReorderPriority::Optimizer),
    )
    .await
    .expect("optimizer should retain one whole-pass slot")
    .unwrap();

    drop(optimizer);
    drop(first_merge);

    tokio::time::timeout(
        Duration::from_millis(100),
        gate.acquire(ReorderPriority::AutomaticMerge),
    )
    .await
    .expect("automatic merges should resume when their slot is released")
    .unwrap();
}

#[cfg(feature = "native")]
#[tokio::test]
async fn standalone_optimizer_reorders_are_serialized() {
    use super::ReorderPriority;
    use std::sync::Arc;
    use std::time::Duration;

    let gate = Arc::new(super::ReorderConcurrencyGate::new(2));
    let first_optimizer = gate.acquire(ReorderPriority::Optimizer).await.unwrap();

    // A second optimizer pass would duplicate the full per-pass memory and
    // sparse-blob IO working set, even though both passes share one Rayon
    // pool. It must wait rather than occupying the second whole-pass slot.
    assert!(
        tokio::time::timeout(
            Duration::from_millis(25),
            gate.acquire(ReorderPriority::Optimizer),
        )
        .await
        .is_err()
    );

    // The remaining common slot is still useful: merge-time BP has its own
    // class permit and may make compaction progress alongside the optimizer.
    let merge = tokio::time::timeout(
        Duration::from_millis(100),
        gate.acquire(ReorderPriority::AutomaticMerge),
    )
    .await
    .expect("merge-time BP should retain its independent slot")
    .unwrap();

    drop(first_optimizer);
    let second_optimizer = tokio::time::timeout(
        Duration::from_millis(100),
        gate.acquire(ReorderPriority::Optimizer),
    )
    .await
    .expect("next optimizer should start when the active pass finishes")
    .unwrap();

    drop(second_optimizer);
    drop(merge);
}

#[cfg(feature = "native")]
#[tokio::test]
async fn foreground_reorder_pauses_new_optimizer_passes() {
    use super::ReorderPriority;
    use std::sync::Arc;
    use std::time::Duration;

    let gate = Arc::new(super::ReorderConcurrencyGate::new(2));
    let existing = gate.acquire(ReorderPriority::Optimizer).await.unwrap();
    let foreground = gate.begin_foreground().await.unwrap();

    // Free the one non-reserved slot. An optimizer waiter must still remain
    // paused while a force merge is active, whereas foreground BP can use it.
    drop(existing);
    assert!(
        tokio::time::timeout(
            Duration::from_millis(25),
            gate.acquire(ReorderPriority::Optimizer),
        )
        .await
        .is_err()
    );
    let foreground_pass = tokio::time::timeout(
        Duration::from_millis(100),
        gate.acquire(ReorderPriority::Foreground),
    )
    .await
    .expect("foreground BP should get the unreserved slot")
    .unwrap();
    drop(foreground_pass);
    drop(foreground);

    let _resumed = tokio::time::timeout(
        Duration::from_millis(100),
        gate.acquire(ReorderPriority::Optimizer),
    )
    .await
    .expect("optimizer BP should resume when force merge finishes")
    .unwrap();
}

#[cfg(feature = "native")]
#[tokio::test]
async fn cancelled_foreground_wait_resumes_optimizer_passes() {
    use super::ReorderPriority;
    use std::sync::Arc;
    use std::time::Duration;

    let gate = Arc::new(super::ReorderConcurrencyGate::new(2));
    let first = gate.acquire(ReorderPriority::Optimizer).await.unwrap();
    let second = gate.acquire(ReorderPriority::AutomaticMerge).await.unwrap();

    // begin_foreground marks the gate active before waiting for its reserved
    // permit. Cancelling that wait must not leave background admission paused.
    assert!(
        tokio::time::timeout(Duration::from_millis(25), gate.begin_foreground())
            .await
            .is_err()
    );
    drop(first);
    let resumed = tokio::time::timeout(
        Duration::from_millis(100),
        gate.acquire(ReorderPriority::Optimizer),
    )
    .await
    .expect("cancelled foreground reservation must release optimizer waiters")
    .unwrap();
    drop(resumed);
    drop(second);
}
mod deletion;

mod pattern_counts;
