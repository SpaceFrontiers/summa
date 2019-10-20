use crate::errors::Error;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

pub struct ThreadHandler {
    join_handle: Box<JoinHandle<Result<(), Error>>>,
    running: Arc<AtomicBool>,
}

impl ThreadHandler {
    pub fn new(
        join_handle: Box<JoinHandle<Result<(), Error>>>,
        running: Arc<AtomicBool>,
    ) -> ThreadHandler {
        ThreadHandler {
            join_handle,
            running,
        }
    }
    pub fn stop(self) -> std::thread::Result<Result<(), Error>> {
        self.running.store(false, Ordering::Release);
        self.join_handle.thread().unpark();
        self.join_handle.join()
    }
}
