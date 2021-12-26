use iroh_unixfs::content_loader::GatewayUrl;

use crate::errors::SummaResult;

#[derive(Clone)]
pub struct IrohClient {
    content_loader: iroh_unixfs::content_loader::FullLoader,
}

impl IrohClient {
    pub async fn new(http_gateways: Vec<GatewayUrl>) -> SummaResult<IrohClient> {
        let iroh_rpc_client = iroh_rpc_client::Client::new(iroh_rpc_client::Config::default_network()).await?;
        let content_loader =
            iroh_unixfs::content_loader::FullLoader::new(iroh_rpc_client, iroh_unixfs::content_loader::FullLoaderConfig { http_gateways, indexer: None })?;
        Ok(IrohClient { content_loader })
    }

    pub fn content_loader(&self) -> &iroh_unixfs::content_loader::FullLoader {
        &self.content_loader
    }
}
