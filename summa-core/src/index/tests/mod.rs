mod basic;
mod bmp;
mod boolean;
mod chunked;
mod config_policy;
mod format_migration;
mod maintenance;
mod merge;
mod merge_bounds;
mod pin;
mod posting_codecs;
mod primary_key;
mod range;
mod search;
#[cfg(feature = "sync")]
mod search_cpu;
mod seismic_lifecycle;
mod seismic_operations;
mod tq_bench;
mod vector;

#[cfg(feature = "sync")]
#[test]
fn search_cpu_pool_is_bounded_and_reused_by_width() {
    let first = super::shared_search_pool(1).unwrap();
    let second = super::shared_search_pool(1).unwrap();

    assert!(std::sync::Arc::ptr_eq(&first, &second));
    assert_eq!(first.install(rayon::current_num_threads), 1);
}

#[cfg(feature = "native")]
#[test]
fn zero_search_threads_is_rejected() {
    assert!(super::searcher::SearcherResources::new(1, None, 1, 0, 1).is_err());
}

#[cfg(feature = "native")]
#[test]
fn zero_sparse_io_concurrency_is_rejected() {
    assert!(super::searcher::SearcherResources::new(1, None, 1, 1, 0).is_err());
}

#[cfg(feature = "native")]
#[test]
fn oversized_term_cache_block_cap_is_rejected_at_load() {
    let error =
        super::searcher::SearcherResources::new(super::MAX_TERM_CACHE_BLOCKS + 1, None, 1, 1, 1)
            .err()
            .expect("cap above MAX_TERM_CACHE_BLOCKS must fail");
    assert!(error.to_string().contains("term_cache_blocks"), "{error}");
    assert!(
        super::searcher::SearcherResources::new(super::MAX_TERM_CACHE_BLOCKS, Some(0), 1, 1, 1)
            .is_ok_and(|resources| resources.term_cache_budget_bytes == Some(0))
    );
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
