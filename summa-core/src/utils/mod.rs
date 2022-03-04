pub mod random;
pub mod sync;

pub fn current_time() -> u64 {
    (instant::now() / 1000.0) as u64
}

pub fn transpose<T>(v: Vec<Vec<T>>) -> Vec<Vec<T>> {
    assert!(!v.is_empty());
    let len = v[0].len();
    let mut iters: Vec<_> = v.into_iter().map(|n| n.into_iter()).collect();
    (0..len)
        .map(|_| iters.iter_mut().map(|n| n.next().expect("wrong length")).collect::<Vec<T>>())
        .collect()
}
