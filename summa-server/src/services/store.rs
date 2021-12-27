use std::future::Future;
use std::sync::Arc;

use async_broadcast::Receiver;
use iroh_rpc_types::store::StoreAddr;
use summa_core::configs::ConfigProxy;
use summa_core::utils::thread_handler::ControlMessage;
use tracing::{info, info_span, instrument, Instrument};

use crate::errors::SummaServerResult;

pub struct Store {
    config: crate::configs::store::Config,
    store: iroh_store::Store,
    rpc_addr: StoreAddr,
}

impl Store {
    pub async fn new(config: &Arc<dyn ConfigProxy<crate::configs::server::Config>>) -> SummaServerResult<Store> {
        let config = config.read().await.get().store.clone();
        let rpc_addr: StoreAddr = config.endpoint.parse()?;
        let iroh_store_config = iroh_store::Config::new_with_rpc(config.path.clone(), rpc_addr.clone());
        let store = if config.path.exists() {
            info!(action = "open_store", path = ?config.path);
            iroh_store::Store::open(iroh_store_config).await?
        } else {
            info!(action = "create_store", path = ?config.path);
            iroh_store::Store::create(iroh_store_config).await?
        };

        Ok(Store { config, store, rpc_addr })
    }

    pub fn rpc_addr(&self) -> &StoreAddr {
        &self.rpc_addr
    }

    #[instrument("lifecycle", skip_all)]
    pub async fn start(self, mut terminator: Receiver<ControlMessage>) -> SummaServerResult<impl Future<Output = SummaServerResult<()>>> {
        let rpc_addr = self.config.endpoint.parse()?;
        let store_task = tokio::spawn(iroh_store::rpc::new(rpc_addr, self.store));

        info!(action = "binded", endpoint = ?self.config.endpoint);
        Ok(async move {
            let signal_result = terminator.recv().await;
            info!(action = "sigterm_received", received = ?signal_result);
            store_task.abort();
            if let Err(e) = store_task.await {
                if !e.is_cancelled() {
                    info!(action = "terminated", error = ?e);
                    return Ok(());
                }
            }
            info!(action = "terminated");
            Ok(())
        }
        .instrument(info_span!(parent: None, "lifecycle")))
    }
}
