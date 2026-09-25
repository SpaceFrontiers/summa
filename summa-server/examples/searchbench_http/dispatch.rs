//! Benchmark dispatch policy. The closure owns admission and reader lifetimes.
use std::panic::{AssertUnwindSafe, catch_unwind};

#[derive(Clone, Copy, Debug, Default)]
pub enum Dispatch {
    #[default]
    Blocking,
    InPlace,
}

impl std::str::FromStr for Dispatch {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "blocking" => Ok(Self::Blocking),
            "in-place" => Ok(Self::InPlace),
            _ => anyhow::bail!("DISPATCH must be blocking or in-place"),
        }
    }
}

impl Dispatch {
    pub async fn run<T: Send + 'static>(
        self,
        work: impl FnOnce() -> T + Send + 'static,
    ) -> Result<T, String> {
        match self {
            Self::Blocking => tokio::task::spawn_blocking(work)
                .await
                .map_err(|e| e.to_string()),
            Self::InPlace => tokio::task::block_in_place(|| catch_unwind(AssertUnwindSafe(work)))
                .map_err(|_| "search worker panicked".to_owned()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, mpsc};
    use std::time::Duration;
    use tokio::sync::Semaphore;

    fn runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .max_blocking_threads(2)
            .enable_all()
            .build()
            .unwrap()
    }

    #[test]
    fn full_admission_makes_progress_and_panics_release_permits() {
        for mode in [Dispatch::Blocking, Dispatch::InPlace] {
            runtime().block_on(async {
                let admission = Arc::new(Semaphore::new(64));
                let mut tasks = Vec::new();
                for _ in 0..64 {
                    let permit = admission.clone().try_acquire_owned().unwrap();
                    tasks.push(tokio::spawn(async move {
                        mode.run(move || {
                            let _permit = permit;
                            std::thread::sleep(Duration::from_millis(1));
                            42
                        })
                        .await
                        .unwrap()
                    }));
                }
                for task in tasks {
                    assert_eq!(
                        tokio::time::timeout(Duration::from_secs(10), task)
                            .await
                            .unwrap()
                            .unwrap(),
                        42
                    );
                }
                assert_eq!(admission.available_permits(), 64);
                let permit = admission.clone().try_acquire_owned().unwrap();
                assert!(
                    mode.run(move || {
                        let _permit = permit;
                        panic!("injected worker panic")
                    })
                    .await
                    .is_err()
                );
                assert_eq!(admission.available_permits(), 64);
            });
        }
    }

    #[test]
    fn cancellation_retains_owner_and_shutdown_drains_started_work() {
        for mode in [Dispatch::Blocking, Dispatch::InPlace] {
            let runtime = runtime();
            let admission = Arc::new(Semaphore::new(1));
            let owner = Arc::new(());
            let held_owner = owner.clone();
            let permit = admission.clone().try_acquire_owned().unwrap();
            let (started_tx, started_rx) = mpsc::channel();
            let (release_tx, release_rx) = mpsc::channel();
            let task = runtime.spawn(async move {
                mode.run(move || {
                    let _permit = permit;
                    let _owner = held_owner;
                    started_tx.send(()).unwrap();
                    release_rx.recv_timeout(Duration::from_secs(10)).unwrap();
                })
                .await
            });
            started_rx.recv_timeout(Duration::from_secs(10)).unwrap();
            task.abort();
            assert_eq!(admission.available_permits(), 0);
            assert_eq!(Arc::strong_count(&owner), 2);
            let (drained_tx, drained_rx) = mpsc::channel();
            let shutdown = std::thread::spawn(move || {
                drop(runtime);
                drained_tx.send(()).unwrap();
            });
            assert!(drained_rx.recv_timeout(Duration::from_millis(30)).is_err());
            release_tx.send(()).unwrap();
            drained_rx.recv_timeout(Duration::from_secs(10)).unwrap();
            shutdown.join().unwrap();
            assert_eq!(admission.available_permits(), 1);
            assert_eq!(Arc::strong_count(&owner), 1);
        }
    }
}
