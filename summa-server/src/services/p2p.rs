use std::future::Future;
use std::time::Duration;

use async_broadcast::Receiver;
use iroh_p2p::{DiskStorage, Keychain, Libp2pConfig, Node, DEFAULT_BOOTSTRAP};
use iroh_rpc_types::p2p::P2pAddr;
use summa_core::utils::thread_handler::ControlMessage;
use tracing::{info, info_span, instrument, Instrument};

use crate::errors::SummaServerResult;
use crate::utils::wait_for_addr;

/// Allows to exchange data blocks with IPFS network
pub struct P2p {
    config: crate::configs::p2p::Config,
    rpc_addr: P2pAddr,
    libp2p_config: Libp2pConfig,
}

impl P2p {
    pub async fn new(config: crate::configs::p2p::Config) -> SummaServerResult<P2p> {
        tokio::fs::create_dir_all(&config.key_store_path).await?;
        let rpc_addr: P2pAddr = format!("irpc://{}", config.endpoint).parse()?;

        let bootstrap_peers = config
            .bootstrap
            .clone()
            .unwrap_or_else(|| DEFAULT_BOOTSTRAP.iter().map(|x| x.to_string()).collect())
            .iter()
            .map(|node| node.parse().expect("incorrect bootstrap node"))
            .collect();
        let listening_multiaddrs = config
            .listening_multiaddrs
            .clone()
            .iter()
            .map(|node| node.parse().expect("incorrect multiaddr"))
            .collect();
        let mut libp2p_config = Libp2pConfig::default();
        libp2p_config.bootstrap_peers = bootstrap_peers;
        libp2p_config.gossipsub = false;
        libp2p_config.max_conns_out = config.max_conns_out;
        libp2p_config.listening_multiaddrs = listening_multiaddrs;
        Ok(P2p {
            config: config.clone(),
            rpc_addr,
            libp2p_config,
        })
    }

    pub fn rpc_addr(&self) -> &P2pAddr {
        &self.rpc_addr
    }

    #[instrument("lifecycle", skip_all)]
    pub async fn prepare_serving_future(&self, mut terminator: Receiver<ControlMessage>) -> SummaServerResult<impl Future<Output = SummaServerResult<()>>> {
        let key_chain = Keychain::<DiskStorage>::new(self.config.key_store_path.clone()).await?;
        let mut node = Node::new(
            iroh_p2p::Config {
                libp2p: self.libp2p_config.clone(),
                rpc_client: iroh_rpc_client::Config::default_network(),
                key_store_path: self.config.key_store_path.clone(),
            },
            self.rpc_addr.clone(),
            key_chain,
        )
        .await?;
        info!(action = "p2p", local_peer_id = ?node.local_peer_id(), listen_addrs = ?node.listen_addrs());
        let p2p_task = tokio::task::spawn(async move { node.run().await });
        wait_for_addr(self.rpc_addr.try_as_socket_addr().expect("not socket addr"), Duration::from_secs(10)).await?;
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
