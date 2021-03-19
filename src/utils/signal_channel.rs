use tokio::signal::unix::{signal, SignalKind};
use tokio::sync::oneshot;
use tokio::task;

pub fn signal_channel() -> oneshot::Receiver<()> {
    let (tx, rx) = oneshot::channel::<()>();
    task::spawn(async move {
        let mut sigint = signal(SignalKind::interrupt()).unwrap();
        let mut sigterm = signal(SignalKind::terminate()).unwrap();
        tokio::select! {
            _ = sigint.recv() => {}
            _ = sigterm.recv() => {}
        }
        tx.send(()).unwrap();
    });
    rx
}
