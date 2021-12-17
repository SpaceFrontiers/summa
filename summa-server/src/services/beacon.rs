use std::sync::Arc;

use iroh_api::{FileBuilder, UnixfsEntry};
use iroh_unixfs::chunker::Chunker;
use summa_core::components::ComponentFile;
use summa_core::components::IndexHolder;
use summa_core::configs::ConfigProxy;
use summa_core::utils::sync::Handler;
use summa_proto::proto;
use tracing::{info, instrument};

use crate::errors::SummaServerResult;

#[derive(Clone)]
pub struct Beacon {
    iroh: iroh_api::Api,
}

impl Beacon {
    pub async fn new(server_config: &Arc<dyn ConfigProxy<crate::configs::server::Config>>) -> SummaServerResult<Beacon> {
        let iroh = iroh_api::Api::new(iroh_api::Config::default()).await?;
        Ok(Beacon { iroh })
    }

    #[instrument(skip_all)]
    pub async fn publish_index(&self, index_holder: Handler<IndexHolder>) -> SummaServerResult<String> {
        let index_path = {
            match &index_holder.index_engine_config() {
                proto::IndexEngineConfig {
                    config: Some(proto::index_engine_config::Config::File(config)),
                } => &config.path,
                _ => unimplemented!(),
            }
        };
        let hash = index_holder
            .index_writer_holder()
            .write()
            .await
            .lock_files(index_path.clone(), |files: Vec<ComponentFile>| async move {
                info!(files = ?files);
                let mut entries = vec![];
                for file in files.into_iter() {
                    entries.push(iroh_unixfs::builder::Entry::File(
                        FileBuilder::new()
                            .name(file.path().file_name().expect("should have name").to_string_lossy().to_string())
                            .chunker(Chunker::Fixed(iroh_unixfs::chunker::Fixed { chunk_size: 64 * 1024 }))
                            .content_reader(tokio::fs::File::open(file.path()).await?)
                            .build()
                            .await
                            .map_err(|e| summa_core::errors::Error::External(e.to_string()))?,
                    ))
                }
                info!(entries = ?entries);
                Ok(self
                    .iroh
                    .add(UnixfsEntry::Directory(iroh_unixfs::builder::Directory::basic("".to_string(), entries)))
                    .await
                    .map_err(|e| summa_core::errors::Error::External(e.to_string()))?
                    .to_string())
            })
            .await?;
        Ok(hash)
    }
}
