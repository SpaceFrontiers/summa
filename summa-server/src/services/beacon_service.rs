use crate::configs::{IndexEngine, IpfsConfig};
use crate::errors::SummaServerResult;
use crate::ipfs_client::IpfsClient;
use crate::search_engine::{IndexFilePath, IndexHolder};
use crate::utils::sync::Handler;
use std::fmt::Debug;
use tracing::instrument;

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
    pub async fn publish_index(&self, index_holder: Handler<IndexHolder>, copy: bool) -> SummaServerResult<crate::ipfs_client::Key> {
        let no_copy = !copy;
        let index_path = {
            match &index_holder.index_config_proxy().read().await.index_engine {
                IndexEngine::File(index_path) => index_path.to_path_buf(),
                _ => unreachable!(),
            }
        };
        let index_name = index_holder.index_name().to_string();
        let index_updater = index_holder.index_updater();
        let key = {
            let mut index_updater = index_updater.write().await;
            index_updater
                .prepare_index_publishing(index_path.clone(), |files: Vec<IndexFilePath>| async move {
                    let mutable_files = files.iter().filter_map(|file| (!file.is_immutable()).then(|| file.clone())).collect::<Vec<_>>();
                    self.ipfs_client.add(&index_path, &mutable_files, false).await?;
                    let added_files = self.ipfs_client.add(&index_path, &files, no_copy).await?;
                    let new_root = added_files.into_iter().find(|added_file| added_file.name == index_name).unwrap();
                    let old_key = self.ipfs_client.key_list().await?.keys.into_iter().find(|key| key.name == index_name);
                    let key = match old_key {
                        None => self.ipfs_client.key_gen(&index_name).await?,
                        Some(old_key) => {
                            let resolved = self.ipfs_client.name_resolve(&old_key.id).await?;
                            self.ipfs_client.pin_rm(&resolved.path).await?;
                            self.ipfs_client.repo_gc().await?;
                            old_key
                        }
                    };
                    self.ipfs_client.name_publish(&new_root.hash, &index_name).await?;
                    Ok(key)
                })
                .await?
        };
        Ok(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::AsyncWriteExt;

    #[tokio::test]
    async fn test_base() {
        let beacon_service = BeaconService::new(IpfsConfig::default());

        let test_directory = tempdir::TempDir::new("test_base").unwrap();

        let test_file_name = "test_file.txt";
        let full_test_file_path = test_directory.path().join(test_file_name);
        File::create(full_test_file_path).await.unwrap().write_all(b"Hello, world!").await.unwrap();

        let meta_file_name = "meta.json";
        let full_meta_file_path = test_directory.path().join(meta_file_name);
        File::create(full_meta_file_path).await.unwrap().write_all(b"{}").await.unwrap();

        beacon_service
            .add(
                test_directory.into_path(),
                &[
                    IndexFilePath::new(PathBuf::from(test_file_name), true),
                    IndexFilePath::new(PathBuf::from(meta_file_name), false),
                ],
                false,
            )
            .await
            .unwrap();
        panic!();
    }
}
