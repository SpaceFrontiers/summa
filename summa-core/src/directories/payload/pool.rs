//! Persistent positional workers; shared admission/replies live in the service.
use super::{Admission, Counters, DEPTH, Job, finish, prepare};
use std::{io, sync::Arc, thread::JoinHandle};

pub(super) fn start(
    receiver: async_channel::Receiver<Job>,
    counters: Arc<Counters>,
    admission: Arc<Admission>,
) -> io::Result<Vec<JoinHandle<io::Result<()>>>> {
    let mut workers = Vec::with_capacity(DEPTH);
    for _ in 0..DEPTH {
        let reads = receiver.clone();
        let stats = counters.clone();
        match super::spawn_worker(
            "summa-payload-pool",
            receiver.clone(),
            counters.clone(),
            admission.clone(),
            move || {
                while let Ok(job) = reads.recv_blocking() {
                    if let Some(mut read) = prepare(job, &stats) {
                        let len = read.job.len();
                        let result = super::super::local::read_exact_at(
                            &read.job.file,
                            &mut read.buffer[..len],
                            read.job.range.start,
                        );
                        finish(read, result, &stats);
                    }
                }
                Ok(())
            },
        ) {
            Ok(worker) => workers.push(worker),
            Err(error) => {
                receiver.close();
                admission.close();
                for worker in workers {
                    let _ = worker.join();
                }
                return Err(error);
            }
        }
    }
    Ok(workers)
}
