use crate::errors::SummaResult;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

#[derive(Debug)]
pub enum ControlMessage {
    Shutdown,
}

#[derive(Debug)]
pub struct ThreadHandler {
    join_handle: JoinHandle<SummaResult<()>>,
    shutdown_trigger: oneshot::Sender<ControlMessage>,
}

impl ThreadHandler {
    pub fn new(join_handle: JoinHandle<SummaResult<()>>, shutdown_trigger: oneshot::Sender<ControlMessage>) -> ThreadHandler {
        ThreadHandler { join_handle, shutdown_trigger }
    }

    pub async fn stop(self) -> SummaResult<()> {
        self.shutdown_trigger.send(ControlMessage::Shutdown).unwrap();
        self.join_handle.await?
    }
}
