use std::future::Future;
use std::sync::Arc;

use async_broadcast::Receiver;
use iroh_p2p::{DiskStorage, Keychain, Libp2pConfig, Node, DEFAULT_BOOTSTRAP};
use summa_core::configs::ConfigProxy;
use summa_core::utils::thread_handler::ControlMessage;
use tracing::{info, info_span, instrument, Instrument};

use crate::errors::SummaServerResult;

pub struct P2p {
    config: crate::configs::p2p::Config,
    node: Node<DiskStorage>,
}

impl P2p {
    pub async fn new(config: &Arc<dyn ConfigProxy<crate::configs::server::Config>>) -> SummaServerResult<P2p> {
        let config = config.read().await.get().p2p.clone();
        let bootstrap = config.bootstrap.clone();
        let key_chain = Keychain::<DiskStorage>::new(config.key_store_path.clone()).await?;
        let rpc_addr = config.endpoint.parse()?;

        let bootstrap_peers = DEFAULT_BOOTSTRAP
            .iter()
            .map(|x| x.to_string())
            .chain(bootstrap.into_iter())
            .map(|node| node.parse().unwrap())
            .collect();
        let mut libp2p = Libp2pConfig::default();
        libp2p.bootstrap_peers = bootstrap_peers;
        Ok(P2p {
            config: config.clone(),
            node: Node::new(
                iroh_p2p::Config {
                    libp2p,
                    rpc_client: iroh_rpc_client::Config::default_network(),
                    metrics: iroh_metrics::config::Config::default(),
                    key_store_path: config.key_store_path,
                },
                rpc_addr,
                key_chain,
            )
            .await?,
        })
    }

    #[instrument("lifecycle", skip_all)]
    pub async fn start(mut self, mut terminator: Receiver<ControlMessage>) -> SummaServerResult<impl Future<Output = SummaServerResult<()>>> {
        let p2p_task = tokio::task::spawn(async move { self.node.run().await });
        info!(action = "binded", endpoint = ?self.config.endpoint);
        Ok(async move {
            let signal_result = terminator.recv().await;
            info!(action = "sigterm_received", received = ?signal_result);
            p2p_task.abort();
            let task_result = p2p_task.await;
            info!(action = "terminated", result = ?task_result);
            Ok(())
        }
        .instrument(info_span!(parent: None, "lifecycle")))
    }
}
