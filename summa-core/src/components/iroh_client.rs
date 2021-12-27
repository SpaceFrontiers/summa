use iroh_unixfs::content_loader::GatewayUrl;

use crate::errors::SummaResult;

#[derive(Clone)]
pub struct IrohClient {
    content_loader: iroh_unixfs::content_loader::FullLoader,
    iroh_config: iroh_rpc_client::Config,
}

impl IrohClient {
    pub async fn new(http_gateways: Vec<GatewayUrl>, iroh_config: iroh_rpc_client::Config) -> SummaResult<IrohClient> {
        let iroh_rpc_client = iroh_rpc_client::Client::new(iroh_config.clone()).await?;
        let content_loader =
            iroh_unixfs::content_loader::FullLoader::new(iroh_rpc_client, iroh_unixfs::content_loader::FullLoaderConfig { http_gateways, indexer: None })?;
        Ok(IrohClient { content_loader, iroh_config })
    }

    pub fn content_loader(&self) -> &iroh_unixfs::content_loader::FullLoader {
        &self.content_loader
    }

    pub fn iroh_config(&self) -> &iroh_rpc_client::Config {
        &self.iroh_config
    }
}
