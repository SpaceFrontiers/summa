use async_broadcast::{broadcast, Receiver};
use rand::{distributions::Alphanumeric, Rng};
use summa_core::errors::SummaResult;
use summa_core::utils::thread_handler::ControlMessage;
use tokio::signal::unix::{signal, SignalKind};
use tokio::task;
use tracing::error;

/// Spawns a thread for processing `SignalKind` and returns `oneshot::Receiver` for a signal event
pub fn signal_channel() -> SummaResult<Receiver<ControlMessage>> {
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

pub fn random_string(length: usize) -> String {
    rand::thread_rng().sample_iter(&Alphanumeric).take(length).map(char::from).collect()
}
