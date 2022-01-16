use std::net::SocketAddr;
use std::time::Duration;

use async_broadcast::{broadcast, Receiver};
use summa_core::utils::thread_handler::ControlMessage;
use tokio::net::TcpStream;
use tokio::signal::unix::{signal, SignalKind};
use tokio::task;
use tracing::error;

use crate::errors::{Error, SummaServerResult};

/// Spawns a thread for processing `SignalKind` and returns `oneshot::Receiver` for a signal event
pub fn signal_channel() -> SummaServerResult<Receiver<ControlMessage>> {
    let (sender, receiver) = broadcast::<ControlMessage>(1);
    let mut sigint = signal(SignalKind::interrupt())?;
    let mut sigterm = signal(SignalKind::terminate())?;
    task::spawn(async move {
        tokio::select! {
            _ = sigint.recv() => {}
            _ = sigterm.recv() => {}
        }
        if let Err(error) = sender.broadcast(ControlMessage::Shutdown).await {
            error!(action = "signal_channel_termination", error = ?error)
        }
    });
    Ok(receiver)
}

pub async fn wait_for_addr(socket_addr: SocketAddr, timeout: Duration) -> SummaServerResult<()> {
    if tokio::time::timeout(timeout, TcpStream::connect(socket_addr)).await.is_err() {
        return Err(Error::Timeout(format!("cannot connect to {socket_addr:?} for {timeout:?}")));
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    pub fn acquire_free_port() -> usize {
        static PORT: AtomicUsize = AtomicUsize::new(50000);
        PORT.fetch_add(1, Ordering::SeqCst)
    }
}
