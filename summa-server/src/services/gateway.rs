use std::future::Future;
use std::sync::Arc;

use async_broadcast::Receiver;
use iroh_rpc_types::gateway::GatewayAddr;
use summa_core::components::IrohClient;
use summa_core::utils::thread_handler::ControlMessage;
use tracing::{info, info_span, instrument, Instrument};

use crate::errors::SummaServerResult;

pub struct Gateway {
    config: crate::configs::gateway::Config,
    core: iroh_gateway::core::Core<iroh_unixfs::content_loader::FullLoader>,
    rpc_addr: GatewayAddr,
}

impl Gateway {
    pub async fn new(config: crate::configs::gateway::Config, iroh_client: &IrohClient) -> SummaServerResult<Gateway> {
        let rpc_addr: GatewayAddr = config.p2p_endpoint.parse()?;
        let mut gateway_config = config.derive_iroh_gateway_config();
        gateway_config.rpc_client.store_addr = iroh_client.iroh_config().store_addr.clone();
        gateway_config.rpc_client.p2p_addr = iroh_client.iroh_config().p2p_addr.clone();
        gateway_config.redirect_to_subdomain = false;
        println!("{:#?}", gateway_config);
        let core = iroh_gateway::core::Core::new(
            Arc::new(gateway_config),
            rpc_addr.clone(),
            Arc::new(None),
            iroh_client.content_loader().clone(),
            config.dns_resolver.clone(),
        )
        .await?;
        Ok(Gateway { config, core, rpc_addr })
    }

    pub fn rpc_addr(&self) -> &GatewayAddr {
        &self.rpc_addr
    }

    #[instrument("lifecycle", skip_all)]
    pub async fn start(self, mut terminator: Receiver<ControlMessage>) -> SummaServerResult<impl Future<Output = SummaServerResult<()>>> {
        let server = self.core.server();
        let endpoint = server.local_addr();
        let gateway_task = tokio::task::spawn(async move { server.await.unwrap() });
        info!(action = "binded", endpoint = ?endpoint.to_string());
        Ok(async move {
            let signal_result = terminator.recv().await;
            info!(action = "sigterm_received", received = ?signal_result);
            gateway_task.abort();
            if let Err(e) = gateway_task.await {
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
