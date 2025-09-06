use crate::{Index, IndexConfig, RamDirectory, Schema};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

async fn searcher() -> Arc<crate::Searcher<RamDirectory>> {
    let index = Index::create(
        RamDirectory::new(),
        Schema::builder().build(),
        IndexConfig::default(),
    )
    .await
    .unwrap();
    index.reader().await.unwrap().searcher().await.unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn scoring_polls_resume_on_the_search_pool_after_pending_io() {
    let searcher = searcher().await;
    let expected = tokio::runtime::Handle::current().id();
    let borrowed = String::from("borrowed request");
    let value = searcher
        .run_search_cpu(async {
            assert!(rayon::current_thread_index().is_some());
            assert_eq!(tokio::runtime::Handle::current().id(), expected);
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
            assert!(rayon::current_thread_index().is_some());
            assert_eq!(tokio::runtime::Handle::current().id(), expected);
            borrowed.as_str()
        })
        .await;
    assert_eq!(value, borrowed);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancelling_pending_scoring_drops_borrowed_work_and_propagates_failure() {
    let searcher = searcher().await;
    struct Dropped(Arc<AtomicBool>);
    impl Drop for Dropped {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }
    let dropped = Arc::new(AtomicBool::new(false));
    let guard = Dropped(dropped.clone());
    let mut pending = Box::pin(searcher.run_search_cpu(async move {
        let _guard = guard;
        std::future::pending::<()>().await;
    }));
    assert!(futures::poll!(&mut pending).is_pending());
    assert!(!dropped.load(Ordering::SeqCst));
    drop(pending);
    assert!(dropped.load(Ordering::SeqCst));

    use futures::FutureExt;
    let panic = std::panic::AssertUnwindSafe(searcher.run_search_cpu(async {
        panic!("scoring failure");
    }))
    .catch_unwind()
    .await;
    assert!(panic.is_err());
    assert_eq!(
        searcher
            .run_search_cpu(async { Err::<(), _>("read failed") })
            .await,
        Err("read failed")
    );
}

#[tokio::test]
async fn current_thread_scoring_can_await_io_without_block_in_place() {
    let searcher = searcher().await;
    let thread = std::thread::current().id();
    searcher
        .run_search_cpu(async {
            assert_eq!(std::thread::current().id(), thread);
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
            assert_eq!(std::thread::current().id(), thread);
        })
        .await;
}
