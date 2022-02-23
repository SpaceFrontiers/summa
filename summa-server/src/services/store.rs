use std::future::Future;
use std::time::Duration;

use async_broadcast::Receiver;
use iroh_rpc_types::store::StoreAddr;
use summa_core::utils::parse_endpoint;
use tracing::{info, info_span, instrument, Instrument};

use crate::errors::SummaServerResult;
use crate::utils::thread_handler::ControlMessage;
use crate::utils::wait_for_addr;

/// Store splits data into chunks and make them available through IPFS network.
#[derive(Clone)]
pub struct Store {
    config: crate::configs::store::Config,
    store: iroh_store::Store,
    rpc_addr: StoreAddr,
    content_loader: iroh_unixfs::content_loader::FullLoader,
}

impl Store {
    pub async fn new(
        config: crate::configs::store::Config,
        iroh_rpc_config: &iroh_rpc_client::Config,
        content_loader: iroh_unixfs::content_loader::FullLoader,
    ) -> SummaServerResult<Store> {
        let rpc_addr: StoreAddr = parse_endpoint(&config.endpoint)?;
        let iroh_store_config = iroh_store::Config {
            path: config.path.clone(),
            rpc_client: iroh_rpc_config.clone(),
        };
        let store = if config.path.exists() {
            info!(action = "open_store", path = ?config.path, endpoint = ?rpc_addr);
            iroh_store::Store::open(iroh_store_config).await?
        } else {
            info!(action = "create_store", path = ?config.path, endpoint = ?rpc_addr);
            tokio::fs::create_dir_all(&config.path).await?;
            iroh_store::Store::create(iroh_store_config).await?
        };

        Ok(Store {
            config,
            store,
            rpc_addr,
            content_loader,
        })
    }

    pub fn rpc_addr(&self) -> &StoreAddr {
        &self.rpc_addr
    }

    pub fn content_loader(&self) -> &iroh_unixfs::content_loader::FullLoader {
        &self.content_loader
    }

    pub fn store(&self) -> &iroh_store::Store {
        &self.store
    }

    pub fn default_chunk_size(&self) -> u64 {
        self.config.default_chunk_size
    }

    #[instrument("lifecycle", skip_all)]
    pub async fn prepare_serving_future(&self, mut terminator: Receiver<ControlMessage>) -> SummaServerResult<impl Future<Output = SummaServerResult<()>>> {
        let rpc_addr: StoreAddr = parse_endpoint(&self.config.endpoint)?;
        let store_task = tokio::spawn(iroh_store::rpc::new(rpc_addr.clone(), self.store.clone()));
        wait_for_addr(rpc_addr.try_as_socket_addr().expect("not socket addr"), Duration::from_secs(10)).await?;
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
