//! Bounded fixed-arrival scheduling; query execution remains in the probe owner.
use crate::E;
use std::{
    future::Future,
    time::{Duration, Instant},
};
use tokio::task::JoinSet;

pub(super) struct Config {
    rate: usize,
    requests: usize,
    concurrency: usize,
}
impl Config {
    pub(super) fn new(rate: usize, requests: usize, concurrency: usize) -> Result<Self, E> {
        if !(1..=10_000).contains(&rate)
            || !(1..=65_536).contains(&requests)
            || !(1..=32).contains(&concurrency)
            || requests > rate * 120
        {
            return Err("load requires rate 1..=10000, requests 1..=65536, concurrency 1..=32 and at most 120 seconds".into());
        }
        Ok(Self {
            rate,
            requests,
            concurrency,
        })
    }
    fn offset(&self, sequence: usize) -> Duration {
        Duration::from_nanos(sequence as u64 * 1_000_000_000 / self.rate as u64)
    }
}

pub(super) struct Accepted<T> {
    pub sequence: usize,
    pub scheduled_ns: u64,
    pub admitted_ns: u64,
    pub started_ns: u64,
    pub completed_ns: u64,
    pub output: T,
}
pub(super) struct Rejected {
    pub sequence: usize,
    pub scheduled_ns: u64,
    pub observed_ns: u64,
}
pub(super) struct Run<T> {
    pub origin: Instant,
    pub elapsed: Duration,
    pub arrival_window: Duration,
    pub max_in_flight: usize,
    pub accepted: Vec<Accepted<T>>,
    pub rejected: Vec<Rejected>,
}

pub(super) async fn run<T, F, Fut>(config: Config, mut work: F) -> Result<Run<T>, E>
where
    T: Send + 'static,
    F: FnMut(usize) -> Fut,
    Fut: Future<Output = Result<T, E>> + Send + 'static,
{
    let origin = Instant::now();
    let mut jobs = JoinSet::new();
    let result = async {
        let mut accepted = Vec::with_capacity(config.requests);
        let mut rejected = Vec::new();
        let mut max_in_flight = 0;
        for sequence in 0..config.requests {
            let offset = config.offset(sequence);
            let deadline = tokio::time::Instant::from_std(origin + offset);
            loop {
                tokio::select! {
                    biased;
                    result = jobs.join_next(), if !jobs.is_empty() => {
                        accepted.push(result.unwrap()??);
                    }
                    _ = tokio::time::sleep_until(deadline) => break,
                }
            }
            // Completed tasks must not be counted as occupied admission slots.
            while let Some(result) = jobs.try_join_next() {
                accepted.push(result??);
            }
            let scheduled_ns = offset.as_nanos() as u64;
            let admitted_ns = origin.elapsed().as_nanos() as u64;
            if jobs.len() == config.concurrency {
                rejected.push(Rejected {
                    sequence,
                    scheduled_ns,
                    observed_ns: admitted_ns,
                });
                continue;
            }
            let future = work(sequence);
            jobs.spawn(async move {
                let started_ns = origin.elapsed().as_nanos() as u64;
                let output = future.await?;
                Ok::<_, E>(Accepted {
                    sequence,
                    scheduled_ns,
                    admitted_ns,
                    started_ns,
                    completed_ns: origin.elapsed().as_nanos() as u64,
                    output,
                })
            });
            max_in_flight = max_in_flight.max(jobs.len());
        }
        while let Some(result) = jobs.join_next().await {
            accepted.push(result??);
        }
        tokio::time::sleep_until((origin + config.offset(config.requests)).into()).await;
        let elapsed = origin.elapsed();
        accepted.sort_unstable_by_key(|row| row.sequence);
        Ok(Run {
            origin,
            elapsed,
            arrival_window: config.offset(config.requests),
            max_in_flight,
            accepted,
            rejected,
        })
    }
    .await;
    if result.is_err() {
        jobs.abort_all();
        while jobs.join_next().await.is_some() {}
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_limits_bound_observations_tasks_and_arrival_window() {
        for (rate, requests, concurrency) in [
            (0, 1, 1),
            (1, 0, 1),
            (1, 1, 0),
            (10_001, 1, 1),
            (10_000, 65_537, 1),
            (1, 1, 33),
            (1, 121, 1),
        ] {
            assert!(Config::new(rate, requests, concurrency).is_err());
        }
        let config = Config::new(3, 360, 32).unwrap();
        assert_eq!(config.offset(3), Duration::from_secs(1));
        assert_eq!(config.offset(360), Duration::from_secs(120));
    }

    #[tokio::test]
    async fn overloaded_arrivals_are_accounted_and_accepted_work_is_drained() {
        let run = run(Config::new(10_000, 16, 2).unwrap(), |sequence| async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            Ok(sequence)
        })
        .await
        .unwrap();
        assert_eq!(run.accepted.len() + run.rejected.len(), 16);
        assert!(!run.rejected.is_empty());
        assert!(run.max_in_flight <= 2);
        let mut sequences = Vec::new();
        for sample in run.accepted {
            assert_eq!(sample.output, sample.sequence);
            assert!(sample.scheduled_ns <= sample.admitted_ns);
            assert!(sample.admitted_ns <= sample.started_ns);
            assert!(sample.started_ns <= sample.completed_ns);
            sequences.push(sample.sequence);
        }
        sequences.extend(run.rejected.into_iter().map(|sample| sample.sequence));
        sequences.sort_unstable();
        assert_eq!(sequences, (0..16).collect::<Vec<_>>());
    }

    #[tokio::test]
    async fn completed_tasks_release_admission_before_later_arrivals() {
        let run = run(Config::new(20, 4, 1).unwrap(), |sequence| async move {
            Ok(sequence)
        })
        .await
        .unwrap();
        assert_eq!(run.accepted.len(), 4);
        assert!(run.rejected.is_empty());
        for sample in run.accepted {
            assert_eq!(sample.scheduled_ns, sample.sequence as u64 * 50_000_000);
        }
    }

    #[tokio::test]
    async fn failed_work_cancels_and_drains_other_tasks_before_returning_error() {
        use std::sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        };
        struct DropNotice(Arc<AtomicBool>);
        impl Drop for DropNotice {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Relaxed);
            }
        }
        let dropped = Arc::new(AtomicBool::new(false));
        let notice = dropped.clone();
        let error = run(Config::new(100, 4, 2).unwrap(), move |sequence| {
            let notice = notice.clone();
            async move {
                if sequence == 0 {
                    let _notice = DropNotice(notice);
                    std::future::pending::<()>().await;
                }
                Err::<(), E>("injected query failure".into())
            }
        })
        .await
        .err()
        .unwrap();
        assert_eq!(error.to_string(), "injected query failure");
        assert!(dropped.load(Ordering::Relaxed));
    }
}
