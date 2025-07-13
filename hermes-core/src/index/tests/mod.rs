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
    assert!(super::searcher::SearcherResources::new(1, 1, 0).is_err());
}
