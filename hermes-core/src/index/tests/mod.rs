mod basic;
mod bmp;
mod boolean;
mod merge;
mod pin;
mod primary_key;
mod range;
mod search;

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
    assert!(super::searcher::SearcherResources::new(1, 1, 0, 1).is_err());
}

#[cfg(feature = "native")]
#[test]
fn zero_bmp_io_concurrency_is_rejected() {
    assert!(super::searcher::SearcherResources::new(1, 1, 1, 0).is_err());
}

#[cfg(feature = "native")]
#[tokio::test]
async fn bmp_io_gate_shares_capacity_between_sync_and_async_paths() {
    use std::time::Duration;

    let gate = super::BmpIoGate::new(1);
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
        .expect("async BMP scorer should wake when the shared slot is released");
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
async fn foreground_reorder_pauses_new_background_passes() {
    use super::ReorderPriority;
    use std::sync::Arc;
    use std::time::Duration;

    let gate = Arc::new(super::ReorderConcurrencyGate::new(2));
    let existing = gate.acquire(ReorderPriority::Background).await.unwrap();
    let foreground = gate.begin_foreground().await.unwrap();

    // Free the one non-reserved slot. A background waiter must still remain
    // paused while a force merge is active, whereas foreground BP can use it.
    drop(existing);
    assert!(
        tokio::time::timeout(
            Duration::from_millis(25),
            gate.acquire(ReorderPriority::Background),
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
        gate.acquire(ReorderPriority::Background),
    )
    .await
    .expect("background BP should resume when force merge finishes")
    .unwrap();
}

#[cfg(feature = "native")]
#[tokio::test]
async fn cancelled_foreground_wait_resumes_background_passes() {
    use super::ReorderPriority;
    use std::sync::Arc;
    use std::time::Duration;

    let gate = Arc::new(super::ReorderConcurrencyGate::new(2));
    let first = gate.acquire(ReorderPriority::Background).await.unwrap();
    let second = gate.acquire(ReorderPriority::Background).await.unwrap();

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
        gate.acquire(ReorderPriority::Background),
    )
    .await
    .expect("cancelled foreground reservation must release background waiters")
    .unwrap();
    drop(resumed);
    drop(second);
}
