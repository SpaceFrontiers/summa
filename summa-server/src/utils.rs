use async_broadcast::{broadcast, Receiver};
use rand::{distributions::Alphanumeric, Rng};
use summa_core::utils::thread_handler::ControlMessage;
use tokio::signal::unix::{signal, SignalKind};
use tokio::task;

/// Spawns a thread for processing `SignalKind` and returns `oneshot::Receiver` for a signal event
pub fn signal_channel() -> Receiver<ControlMessage> {
    let (sender, receiver) = broadcast::<ControlMessage>(1);
    task::spawn(async move {
        let mut sigint = signal(SignalKind::interrupt()).unwrap();
        let mut sigterm = signal(SignalKind::terminate()).unwrap();
        tokio::select! {
            _ = sigint.recv() => {}
            _ = sigterm.recv() => {}
        }
        sender.broadcast(ControlMessage::Shutdown).await.unwrap()
    });
    receiver
}

pub fn random_string(length: usize) -> String {
    rand::thread_rng().sample_iter(&Alphanumeric).take(length).map(char::from).collect()
}
