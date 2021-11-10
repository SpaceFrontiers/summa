use std::fs::File;
use std::sync::Arc;

use ipfs_api::request::{FilesMkdir, FilesWrite};
use ipfs_api::{IpfsApi, IpfsClient, TryFromUri};
use summa_core::components::{ComponentFile, IndexHolder};
use summa_core::configs::{IndexEngine, IpfsConfig};
use tracing::{info, instrument};

use crate::errors::{Error, SummaServerResult};
use crate::services::differential_updater::RequiredOperation;
use crate::services::DifferentialUpdater;
use crate::utils::random_string;

#[derive(Clone)]
pub struct BeaconService {
    ipfs_client: IpfsClient,
}

impl BeaconService {
    pub fn new(ipfs_config: IpfsConfig) -> SummaServerResult<BeaconService> {
        let ipfs_client = IpfsClient::from_str(&format!("http://{}", ipfs_config.api_endpoint)).unwrap();
        Ok(BeaconService { ipfs_client })
    }

    #[instrument(skip_all, fields(index_name = ?index_holder.index_name()))]
    pub async fn publish_index(&self, mfs_path: &str, index_holder: Arc<IndexHolder>, payload: Option<String>) -> SummaServerResult<()> {
        let index_path = {
            match &index_holder.index_config_proxy().read().await.get().index_engine {
                IndexEngine::File(index_path) => index_path.to_path_buf(),
                _ => unreachable!(),
            }
        };
        let index_updater = index_holder.index_updater();
        let hash = Some("blake2b-256");
        let mut index_updater = index_updater.write().await;
        index_updater
            .lock_files(index_path.clone(), payload, |files: Vec<ComponentFile>| async move {
                self.ipfs_client
                    .files_mkdir_with_options(FilesMkdir {
                        path: mfs_path,
                        parents: Some(true),
                        hash,
                        ..Default::default()
                    })
                    .await
                    .map_err(Error::from)?;
                let stored_files = self.ipfs_client.files_ls(Some(mfs_path)).await.map_err(Error::from)?.entries;

                let differential_updater = DifferentialUpdater::from_source(stored_files.into_iter());
                let operations = differential_updater.target_state(files.into_iter());

                let temporary_path = format!("/tmp/{}", random_string(12));
                info!(action = "create_temporary_directory", mfs_path = mfs_path, temporary_path = temporary_path);
                self.ipfs_client.files_cp(mfs_path, &temporary_path).await.map_err(Error::from)?;
                for operation in operations {
                    match operation {
                        RequiredOperation::Remove(files_entry) => {
                            let mfs_file_path = format!("{}/{}", temporary_path, files_entry.name);
                            info!(action = "remove_file", mfs_file_path = mfs_file_path);
                            self.ipfs_client.files_rm(&mfs_file_path, false).await.map_err(Error::from)?
                        }
                        RequiredOperation::Add(component_file) => {
                            let component_file_path = component_file.path().to_string_lossy();
                            let local_file_path = format!("{}/{}", index_path.to_string_lossy(), component_file_path);
                            let mfs_file_path = format!("{}/{}", temporary_path, component_file_path);
                            info!(action = "write_file", local_file_path = local_file_path, mfs_file_path = mfs_file_path);
                            self.ipfs_client
                                .files_write_with_options(
                                    FilesWrite {
                                        path: &mfs_file_path,
                                        create: Some(true),
                                        truncate: Some(true),
                                        hash,
                                        ..Default::default()
                                    },
                                    File::open(local_file_path)?,
                                )
                                .await
                                .map_err(Error::from)?;
                        }
                    }
                }
                info!(action = "committing_files", mfs_path = mfs_path, temporary_path = temporary_path);
                self.ipfs_client.files_mv(&temporary_path, mfs_path).await.map_err(Error::from)?;
                Ok(())
            })
            .await?;
        Ok(())
    }
}
