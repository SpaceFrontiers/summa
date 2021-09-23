use crate::errors::SummaServerResult;
use async_broadcast::Sender;
use tokio::task::JoinHandle;

#[derive(Clone, Debug)]
pub enum ControlMessage {
    Shutdown,
}

/// Holds `JoinHandle` together with its `shutdown_trigger`
#[derive(Debug)]
pub struct ThreadHandler {
    join_handle: JoinHandle<SummaServerResult<()>>,
    shutdown_trigger: Sender<ControlMessage>,
}

impl ThreadHandler {
    pub fn new(join_handle: JoinHandle<SummaServerResult<()>>, shutdown_trigger: Sender<ControlMessage>) -> ThreadHandler {
        ThreadHandler { join_handle, shutdown_trigger }
    }

    pub async fn stop(self) -> SummaServerResult<()> {
        self.shutdown_trigger.broadcast(ControlMessage::Shutdown).await.unwrap();
        self.join_handle.await?
    }
}
