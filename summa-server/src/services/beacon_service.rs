use std::fmt::Debug;
use std::sync::Arc;

use summa_core::components::{IndexFilePath, IndexHolder};
use summa_core::configs::{IndexEngine, IpfsConfig};
use tracing::instrument;

use crate::errors::SummaServerResult;
use crate::ipfs_client::IpfsClient;

#[derive(Clone, Debug, Default)]
pub struct BeaconService {
    ipfs_client: IpfsClient,
}

impl BeaconService {
    pub fn new(ipfs_config: IpfsConfig) -> BeaconService {
        let ipfs_client = IpfsClient::new(ipfs_config);
        BeaconService { ipfs_client }
    }

    #[instrument(skip_all, fields(index_name = ?index_holder.index_name()))]
    pub async fn publish_index(&self, index_holder: Arc<IndexHolder>, payload: Option<String>, copy: bool) -> SummaServerResult<String> {
        let no_copy = !copy;
        let index_path = {
            match &index_holder.index_config_proxy().read().await.get().index_engine {
                IndexEngine::File(index_path) => index_path.to_path_buf(),
                _ => unreachable!(),
            }
        };
        let index_name = index_holder.index_name().to_string();
        let index_updater = index_holder.index_updater();
        let key = {
            let mut index_updater = index_updater.write().await;
            index_updater
                .prepare_index_publishing(index_path.clone(), payload, |files: Vec<IndexFilePath>| async move {
                    let mutable_files = files.iter().filter_map(|file| (!file.is_immutable()).then(|| file.clone())).collect::<Vec<_>>();
                    self.ipfs_client.add(&index_path, &mutable_files, false).await.unwrap();
                    let added_files = self.ipfs_client.add(&index_path, &files, no_copy).await.unwrap();
                    let new_root = added_files.into_iter().find(|added_file| added_file.name == index_name).unwrap();
                    let old_key = self.ipfs_client.key_list().await.unwrap().keys.into_iter().find(|key| key.name == index_name);
                    let key = match old_key {
                        None => self.ipfs_client.key_gen(&index_name).await.unwrap(),
                        Some(old_key) => {
                            let resolved = self.ipfs_client.name_resolve(&old_key.id).await.unwrap();
                            self.ipfs_client.pin_rm(&resolved.path).await.unwrap();
                            self.ipfs_client.repo_gc().await.unwrap();
                            old_key
                        }
                    };
                    self.ipfs_client.name_publish(&new_root.hash, &index_name).await.unwrap();
                    // ToDo: Ok(key)
                    Ok(key.id)
                })
                .await?
        };
        Ok(key)
    }
}
