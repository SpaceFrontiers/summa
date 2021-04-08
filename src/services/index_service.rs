use crate::errors::{Error, SummaResult, ValidationError};
use crate::requests::{CreateConsumerRequest, CreateIndexRequest, DeleteConsumerRequest, DeleteIndexRequest};
use crate::search_engine::index_layout::IndexLayout;
use crate::search_engine::IndexHolder;
use crate::services::AliasService;
use crate::utils::sync::{Handler, OwningHandler};
use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::collections::HashMap;
use std::fs::DirEntry;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// The main struct responsible for indices lifecycle. Here lives indices creation and deletion as well as committing and indexing new documents.
#[derive(Clone, Debug)]
pub struct IndexService {
    root_path: PathBuf,
    alias_service: AliasService,
    /// RwLock for index modification operations like creating new ones or deleting old ones
    index_holders: Arc<RwLock<HashMap<String, OwningHandler<IndexHolder>>>>,
}

#[derive(Default)]
pub struct DeleteIndexResult {
    pub deleted_aliases: Vec<String>,
    pub deleted_consumers: Vec<String>,
}

///
impl IndexService {
    pub async fn new(root_path: &Path, alias_service: &AliasService) -> SummaResult<IndexService> {
        let mut index_holders: HashMap<String, OwningHandler<IndexHolder>> = HashMap::new();
        for index_path in IndexService::indices_on_disk(root_path)? {
            let index_layout = IndexLayout::open(&index_path.path()).await?;
            let index_name = String::from(index_path.file_name().to_str().unwrap());
            index_holders.insert(index_name.clone(), OwningHandler::new(IndexHolder::open(&index_name, &index_layout)?));
        }

        let index_service = IndexService {
            root_path: root_path.to_path_buf(),
            alias_service: alias_service.clone(),
            index_holders: Arc::new(RwLock::new(index_holders)),
        };
        Ok(index_service)
    }

    pub fn indices_on_disk(root_path: &Path) -> SummaResult<Vec<DirEntry>> {
        Ok(std::fs::read_dir(root_path)?
            .filter_map(|possible_index_path| possible_index_path.ok())
            .filter(|possible_index_path| possible_index_path.path().join("index.yaml").exists())
            .collect())
    }

    pub fn index_holders(&self) -> RwLockReadGuard<'_, HashMap<String, OwningHandler<IndexHolder>>> {
        self.index_holders.read()
    }
    pub fn index_holders_mut(&self) -> RwLockWriteGuard<'_, HashMap<String, OwningHandler<IndexHolder>>> {
        self.index_holders.write()
    }

    pub fn get_index_holder(&self, index_alias_or_name: &str) -> SummaResult<Handler<IndexHolder>> {
        let index_name = self.alias_service.resolve_index_alias(index_alias_or_name);
        let index_name = index_name.as_deref().unwrap_or(index_alias_or_name);
        let index_holder = self
            .index_holders
            .read()
            .get(index_name)
            .ok_or_else(|| Error::ValidationError(ValidationError::MissingIndexError(index_alias_or_name.to_string())))?
            .handler();
        Ok(index_holder)
    }

    /// Create consumer and insert it into the consumer registry. Add it to the `IndexHolder` afterwards.
    pub async fn create_index(&self, create_index_request: CreateIndexRequest) -> SummaResult<()> {
        let mut index_holders = self.index_holders.write();
        let index_path = self.root_path.join(&create_index_request.index_name);
        let index_layout = IndexLayout::create(&index_path).await?;
        let index_holder = IndexHolder::new(
            &create_index_request.index_name,
            create_index_request.index_config,
            &index_layout,
            &create_index_request.schema,
        )?;
        index_holders.insert(create_index_request.index_name.to_string(), OwningHandler::new(index_holder));
        Ok(())
    }

    fn check_delete_conditions(&self, aliases: &Vec<String>, index_holder: &IndexHolder) -> SummaResult<()> {
        if aliases.len() > 0 {
            Err(ValidationError::AliasedError(aliases.join(", ")))?;
        }
        let consumer_names = index_holder.get_consumers_names();
        if index_holder.get_consumers_names().len() > 0 {
            Err(ValidationError::ExistingConsumersError(consumer_names.join(", ")))?;
        }
        Ok(())
    }

    async fn prepare_delete(&self, delete_index_request: &DeleteIndexRequest, index_holder: &IndexHolder) -> SummaResult<DeleteIndexResult> {
        let mut delete_index_result = DeleteIndexResult::default();
        let aliases = self.alias_service.get_index_aliases_for_index(&delete_index_request.index_name);

        if delete_index_request.cascade {
            self.alias_service.delete_index_aliases(&aliases);
            delete_index_result.deleted_aliases.extend(aliases);
            index_holder.delete_all_consumers().await?;
        } else {
            self.check_delete_conditions(&aliases, index_holder)?;
        }

        Ok(delete_index_result)
    }

    pub async fn delete_index(&self, delete_index_request: DeleteIndexRequest) -> SummaResult<DeleteIndexResult> {
        let (delete_index_result, index_holder) = {
            let mut index_holders = self.index_holders.write();
            let index_holder = match index_holders.get(&delete_index_request.index_name) {
                Some(index_holder) => index_holder,
                None => Err(ValidationError::MissingIndexError(delete_index_request.index_name.to_string()))?,
            };
            (
                self.prepare_delete(&delete_index_request, index_holder).await?,
                index_holders.remove(&delete_index_request.index_name).unwrap(),
            )
        };
        let inner = tokio::task::spawn_blocking(move || index_holder.into_inner()).await?;
        inner.delete().await?;
        Ok(delete_index_result)
    }

    /// Create consumer and insert it into the consumer registry. Add it to the `IndexHolder` afterwards.
    pub fn get_consumers(&self) -> SummaResult<Vec<(String, String)>> {
        let index_holders = self.index_holders.read();
        let mut result = Vec::new();
        for index_holder in index_holders.values() {
            result.extend(
                index_holder
                    .get_consumers_names()
                    .iter()
                    .map(|consumer_name| (index_holder.index_name().to_string(), consumer_name.to_string())),
            )
        }
        Ok(result)
    }

    /// Create consumer and insert it into the consumer registry. Add it to the `IndexHolder` afterwards.
    pub async fn create_consumer(&self, create_consumer_request: CreateConsumerRequest) -> SummaResult<()> {
        let index_holder = self.get_index_holder(&create_consumer_request.index_name)?;
        index_holder
            .create_consumer(&create_consumer_request.consumer_name, create_consumer_request.consumer_config)
            .await?;
        Ok(())
    }

    /// Delete consumer from the consumer registry and from `IndexHolder` afterwards.
    pub async fn delete_consumer(&self, delete_consumer_request: DeleteConsumerRequest) -> SummaResult<()> {
        let index_holder = self.get_index_holder(&delete_consumer_request.index_name)?;
        index_holder.delete_consumer(&delete_consumer_request.consumer_name).await?;
        Ok(())
    }

    /// Stopping index holders
    pub async fn stop(self) -> SummaResult<()> {
        for (_, index_holder) in self.index_holders_mut().drain() {
            index_holder.into_inner().stop().await?;
        }
        Ok(())
    }
}
