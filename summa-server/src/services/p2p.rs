use std::future::Future;

use async_broadcast::Receiver;
use iroh_p2p::{DiskStorage, Keychain, Libp2pConfig, Node, DEFAULT_BOOTSTRAP};
use iroh_rpc_types::p2p::P2pAddr;
use summa_core::utils::thread_handler::ControlMessage;
use tracing::{info, info_span, instrument, Instrument};

use crate::errors::SummaServerResult;

pub struct P2p {
    config: crate::configs::p2p::Config,
    node: Node<DiskStorage>,
    rpc_addr: P2pAddr,
}

impl P2p {
    pub async fn new(config: crate::configs::p2p::Config) -> SummaServerResult<P2p> {
        tokio::fs::create_dir_all(&config.key_store_path).await?;
        let key_chain = Keychain::<DiskStorage>::new(config.key_store_path.clone()).await?;
        let rpc_addr: P2pAddr = config.endpoint.parse()?;

        let bootstrap_peers = config
            .bootstrap
            .clone()
            .unwrap_or_else(|| DEFAULT_BOOTSTRAP.iter().map(|x| x.to_string()).collect())
            .iter()
            .map(|node| node.parse().expect("incorrect bootstrap node"))
            .collect();
        let mut libp2p = Libp2pConfig::default();
        libp2p.bootstrap_peers = bootstrap_peers;
        libp2p.gossipsub = false;
        libp2p.max_conns_out = 256;
        Ok(P2p {
            config: config.clone(),
            node: Node::new(
                iroh_p2p::Config {
                    libp2p,
                    rpc_client: iroh_rpc_client::Config::default_network(),
                    metrics: iroh_metrics::config::Config::default(),
                    key_store_path: config.key_store_path,
                },
                rpc_addr.clone(),
                key_chain,
            )
            .await?,
            rpc_addr,
        })
    }

    pub fn rpc_addr(&self) -> &P2pAddr {
        &self.rpc_addr
    }

    #[instrument("lifecycle", skip_all)]
    pub async fn start(mut self, mut terminator: Receiver<ControlMessage>) -> SummaServerResult<impl Future<Output = SummaServerResult<()>>> {
        info!(action = "p2p", local_peer_id = ?self.node.local_peer_id(), listen_addrs = ?self.node.listen_addrs());
        let p2p_task = tokio::task::spawn(async move { self.node.run().await });
        info!(action = "binded", endpoint = ?self.config.endpoint);
        Ok(async move {
            let signal_result = terminator.recv().await;
            info!(action = "sigterm_received", received = ?signal_result);
            p2p_task.abort();
            if let Err(e) = p2p_task.await {
                if !e.is_cancelled() {
                    info!(action = "terminated", result = ?e);
                    return Ok(());
                }
            }
            info!(action = "terminated");
            Ok(())
        }
        .instrument(info_span!(parent: None, "lifecycle")))
    }
}
