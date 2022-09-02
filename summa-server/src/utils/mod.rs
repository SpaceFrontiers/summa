pub(crate) mod thread_handler;

use async_broadcast::{broadcast, Receiver};
use tokio::signal::ctrl_c;
use tokio::task;
use tracing::error;

use crate::errors::SummaServerResult;
pub use crate::utils::thread_handler::{ControlMessage, ThreadHandler};

/// Spawns a thread for processing `SignalKind` and returns `oneshot::Receiver` for a signal event
pub fn signal_channel() -> SummaServerResult<Receiver<ControlMessage>> {
    let (sender, receiver) = broadcast::<ControlMessage>(1);
    #[cfg(unix)]
    let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
    let ctrl_c_sig = ctrl_c();
    task::spawn(async move {
        #[cfg(unix)]
        tokio::select! {
            _ = sigterm.recv() => {}
            _ = ctrl_c_sig => {}
        }
        #[cfg(windows)]
        tokio::select! {
            _ = ctrl_c_sig => {}
        }
        if let Err(error) = sender.broadcast(ControlMessage::Shutdown).await {
            error!(action = "signal_channel_termination", error = ?error)
        }
    });
    Ok(receiver)
}

#[cfg(unix)]
pub fn increase_fd_limit() -> std::io::Result<u64> {
    const DEFAULT_NOFILE_LIMIT: u64 = 65536;
    const MIN_NOFILE_LIMIT: u64 = 2048;

    let (_, hard) = rlimit::Resource::NOFILE.get()?;
    let target = std::cmp::min(hard, DEFAULT_NOFILE_LIMIT);
    rlimit::Resource::NOFILE.set(target, hard)?;
    let (soft, _) = rlimit::Resource::NOFILE.get()?;
    if soft < MIN_NOFILE_LIMIT {
        return Err(std::io::Error::new(std::io::ErrorKind::Other, format!("NOFILE limit too low: {soft}")));
    }
    Ok(soft)
}

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    pub fn acquire_free_port() -> usize {
        static PORT: AtomicUsize = AtomicUsize::new(50000);
        PORT.fetch_add(1, Ordering::SeqCst)
    }
}
