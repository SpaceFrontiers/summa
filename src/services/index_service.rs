use crate::configs::{ApplicationConfigHolder, ConsumerConfig, IndexConfigBuilder, IndexConfigProxy, IndexEngine};
use crate::errors::{Error, SummaResult, ValidationError};
use crate::proto;
use crate::requests::{AlterIndexRequest, CreateConsumerRequest, CreateIndexRequest, DeleteConsumerRequest, DeleteIndexRequest};
use crate::search_engine::IndexHolder;
use crate::utils::sync::{Handler, OwningHandler};
use futures_util::future::join_all;
use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tantivy::IndexSettings;
use tracing::instrument;

/// The main struct responsible for indices lifecycle. Here lives indices creation and deletion as well as committing and indexing new documents.
#[derive(Clone, Default, Debug)]
pub struct IndexService {
    application_config: ApplicationConfigHolder,
    /// RwLock for index modification operations like creating new ones or deleting old ones
    /// `OwningHandler` that wraps `IndexHolder` allows safe cloning of `IndexHolder` objects.
    /// Every destructable operation like `stop` or `delete`
    /// will wait corresponding `IndexHolder` object until it will be released by other users
    index_holders: Arc<RwLock<HashMap<String, OwningHandler<IndexHolder>>>>,
}

#[derive(Default)]
pub struct DeleteIndexResult {
    pub deleted_index_aliases: Vec<String>,
    pub deleted_index_consumers: Vec<String>,
}

impl Into<proto::DeleteIndexResponse> for DeleteIndexResult {
    fn into(self) -> proto::DeleteIndexResponse {
        proto::DeleteIndexResponse {
            deleted_index_aliases: self.deleted_index_aliases,
            deleted_index_consumers: self.deleted_index_consumers,
        }
    }
}

/// The main entry point for managing Summa indices
impl IndexService {
    /// Creates new `IndexService` with `ApplicationConfigHolder`
    pub fn new(application_config_holder: &ApplicationConfigHolder) -> IndexService {
        IndexService {
            application_config: application_config_holder.clone(),
            index_holders: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create `IndexHolder`s from config
    pub(crate) async fn setup_index_holders(&self) -> SummaResult<()> {
        let mut index_holders = HashMap::new();
        for (index_name, _index_config) in &self.application_config.read().indices {
            index_holders.insert(
                index_name.clone(),
                OwningHandler::new(IndexHolder::open(index_name, IndexConfigProxy::new(&self.application_config, index_name)).await?),
            );
        }
        *self.index_holders.write() = index_holders;
        Ok(())
    }

    /// Read-locked `HashMap` of all indices
    pub fn index_holders(&self) -> RwLockReadGuard<'_, HashMap<String, OwningHandler<IndexHolder>>> {
        self.index_holders.read()
    }

    /// Write-locked `HashMap` of all indices
    ///
    /// Taking this lock means locking metadata modification
    pub fn index_holders_mut(&self) -> RwLockWriteGuard<'_, HashMap<String, OwningHandler<IndexHolder>>> {
        self.index_holders.write()
    }

    fn get_index_holder_by_name(&self, index_name: &str) -> SummaResult<Handler<IndexHolder>> {
        Ok(self
            .index_holders
            .read()
            .get(index_name)
            .ok_or_else(|| Error::ValidationError(ValidationError::MissingIndexError(index_name.to_owned())))?
            .handler())
    }
    /// Returns `Handler` to `IndexHolder`.
    ///
    /// It is safe to keep `Handler<IndexHolder>` cause `Index` won't be deleted until `Handler` is alive.
    /// Though, `IndexHolder` can be removed from the registry of `IndexHolder`s to prevent new queries
    pub fn get_index_holder(&self, index_alias_or_name: &str) -> SummaResult<Handler<IndexHolder>> {
        self.get_index_holder_by_name(
            self.application_config
                .read()
                .resolve_index_alias(index_alias_or_name)
                .as_deref()
                .unwrap_or(index_alias_or_name),
        )
    }

    fn insert_config(&self, create_index_request: &CreateIndexRequest) -> SummaResult<()> {
        let mut application_config = self.application_config.write();
        let mut index_config_builder = IndexConfigBuilder::default();
        index_config_builder
            .index_engine(match create_index_request.index_engine {
                proto::IndexEngine::Memory => IndexEngine::Memory(create_index_request.schema.clone()),
                proto::IndexEngine::File => IndexEngine::File(application_config.get_path_for_index_data(&create_index_request.index_name)),
            })
            .primary_key(create_index_request.primary_key.to_owned())
            .default_fields(create_index_request.default_fields.clone())
            .multi_fields(HashSet::from_iter(create_index_request.multi_fields.clone().into_iter()))
            .stop_words(create_index_request.stop_words.clone())
            .autocommit_interval_ms(create_index_request.autocommit_interval_ms);
        if let Some(writer_threads) = create_index_request.writer_threads {
            index_config_builder.writer_threads(writer_threads);
        }
        if let Some(writer_heap_size_bytes) = create_index_request.writer_heap_size_bytes {
            index_config_builder.writer_heap_size_bytes(writer_heap_size_bytes);
        }
        let index_config = index_config_builder.build().unwrap();
        match application_config.autosave().indices.entry(create_index_request.index_name.to_owned()) {
            Entry::Occupied(o) => Err(ValidationError::ExistingIndexError(o.key().to_owned())),
            Entry::Vacant(v) => {
                v.insert(index_config);
                Ok(())
            }
        }?;
        Ok(())
    }

    /// Create consumer and insert it into the consumer registry. Add it to the `IndexHolder` afterwards.
    #[instrument(skip_all)]
    pub async fn create_index(&self, create_index_request: CreateIndexRequest) -> SummaResult<Handler<IndexHolder>> {
        self.insert_config(&create_index_request)?;
        let index_settings = IndexSettings {
            docstore_compression: create_index_request.compression.clone(),
            sort_by_field: create_index_request.sort_by_field.clone(),
            ..Default::default()
        };
        let owning_handler = OwningHandler::new(
            IndexHolder::create(
                &create_index_request.index_name,
                &create_index_request.schema,
                IndexConfigProxy::new(&self.application_config, &create_index_request.index_name),
                index_settings,
            )
            .await?,
        );
        let handler = owning_handler.handler();
        self.index_holders.write().insert(create_index_request.index_name.to_owned(), owning_handler);
        Ok(handler)
    }

    /// Alters index
    #[instrument(skip_all)]
    pub async fn alter_index(&self, alter_index_request: AlterIndexRequest) -> SummaResult<Handler<IndexHolder>> {
        let index_holder = self.get_index_holder_by_name(&alter_index_request.index_name)?;
        let index_updater = index_holder.index_updater();
        let mut index_updater = index_updater.write();
        if let Some(compression) = alter_index_request.compression {
            index_updater.index_mut().settings_mut().docstore_compression = compression
        }
        if let Some(sort_by_field) = alter_index_request.sort_by_field {
            index_updater.index_mut().settings_mut().sort_by_field = match sort_by_field.field.as_str() {
                "" => None,
                _ => Some(sort_by_field),
            }
        }
        index_updater.commit().await?;
        Ok(index_holder)
    }

    fn check_delete_conditions(&self, aliases: &Vec<String>, index_holder: &IndexHolder) -> SummaResult<()> {
        if aliases.len() > 0 {
            Err(ValidationError::AliasedError(aliases.join(", ")))?;
        }
        if index_holder.index_updater().read().has_consumers() {
            Err(ValidationError::ExistingConsumersError(index_holder.index_name().to_owned()))?;
        }
        Ok(())
    }

    async fn prepare_delete(&self, delete_index_request: &DeleteIndexRequest, index_holder: &IndexHolder) -> SummaResult<DeleteIndexResult> {
        let mut application_config = self.application_config.write();
        let mut delete_index_result = DeleteIndexResult::default();
        let aliases = application_config.get_index_aliases_for_index(&delete_index_request.index_name);

        if delete_index_request.cascade {
            let consumers = index_holder.index_updater().write().delete_all_consumers().await?;
            delete_index_result.deleted_index_consumers.extend(consumers);

            application_config.autosave().delete_index_aliases(&aliases);
            delete_index_result.deleted_index_aliases.extend(aliases);
        } else {
            self.check_delete_conditions(&aliases, index_holder)?;
        }

        Ok(delete_index_result)
    }

    /// Delete index, optionally with all its aliases and consumers
    #[instrument(skip_all)]
    pub async fn delete_index(&self, delete_index_request: DeleteIndexRequest) -> SummaResult<DeleteIndexResult> {
        let mut index_holders = self.index_holders.write();
        let index_holder_entry = match index_holders.entry(delete_index_request.index_name.to_owned()) {
            Entry::Occupied(entry) => entry,
            Entry::Vacant(_) => Err(ValidationError::MissingIndexError(delete_index_request.index_name.to_owned()))?,
        };
        let delete_index_result = self.prepare_delete(&delete_index_request, index_holder_entry.get()).await?;
        let index_holder = index_holder_entry.remove();
        let inner = tokio::task::spawn_blocking(move || index_holder.into_inner()).await?;
        inner.delete().await?;
        Ok(delete_index_result)
    }

    /// Returns all existent consumers for all indices
    pub fn get_consumers(&self) -> SummaResult<Vec<(String, String)>> {
        let application_config = self.application_config.read();
        let mut result = Vec::new();
        for (index_name, index_config) in &application_config.indices {
            result.extend(
                index_config
                    .consumer_configs
                    .keys()
                    .map(|consumer_name| (index_name.to_owned(), consumer_name.to_owned())),
            )
        }
        Ok(result)
    }

    /// Create consumer and insert it into the consumer registry. Add it to the `IndexHolder` afterwards.
    #[instrument(skip_all, fields(consumer_name = ?create_consumer_request.consumer_name))]
    pub async fn create_consumer(&self, create_consumer_request: &CreateConsumerRequest) -> SummaResult<String> {
        let index_holder = self.get_index_holder(&create_consumer_request.index_alias)?;
        index_holder
            .index_updater()
            .write()
            .create_consumer(&create_consumer_request.consumer_name, &create_consumer_request.consumer_config)
            .await?;
        Ok(index_holder.index_name().to_owned())
    }

    /// Delete consumer from the consumer registry and from `IndexHolder` afterwards.
    #[instrument(skip_all, fields(consumer_name = ?delete_consumer_request.consumer_name))]
    pub async fn delete_consumer(&self, delete_consumer_request: DeleteConsumerRequest) -> SummaResult<()> {
        self.get_index_holder(&delete_consumer_request.index_alias)?
            .index_updater()
            .write()
            .delete_consumer(&delete_consumer_request.consumer_name)
            .await
    }

    /// Stopping index holders
    pub async fn stop(self) -> SummaResult<()> {
        join_all(self.index_holders_mut().drain().map(|(_, index_holder)| index_holder.into_inner().stop()))
            .await
            .into_iter()
            .collect::<SummaResult<Vec<_>>>()?;
        Ok(())
    }

    /// Returns `ConsumerConfig`
    pub(crate) fn get_consumer_config(&self, index_alias: &str, consumer_name: &str) -> SummaResult<ConsumerConfig> {
        Ok(self.get_index_holder(index_alias)?.get_consumer_config(consumer_name)?)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::configs::application_config::tests::create_test_application_config_holder;
    use crate::logging;
    use crate::requests::{CreateIndexRequestBuilder, DeleteIndexRequestBuilder};
    use crate::search_engine::index_holder::tests::create_test_schema;
    use std::path::Path;

    pub(crate) async fn create_test_index_service(data_path: &Path) -> IndexService {
        IndexService::new(&create_test_application_config_holder(&data_path))
    }

    #[tokio::test]
    async fn test_same_name_index() {
        logging::tests::initialize_default_once();
        let index_service = IndexService::default();
        assert!(index_service
            .create_index(
                CreateIndexRequestBuilder::default()
                    .index_name("test_index".to_owned())
                    .index_engine(proto::IndexEngine::Memory)
                    .schema(create_test_schema())
                    .build()
                    .unwrap()
            )
            .await
            .is_ok());

        assert!(index_service
            .create_index(
                CreateIndexRequestBuilder::default()
                    .index_name("test_index".to_owned())
                    .index_engine(proto::IndexEngine::Memory)
                    .schema(create_test_schema())
                    .build()
                    .unwrap()
            )
            .await
            .is_err());
    }

    #[tokio::test]
    async fn test_index_create_delete() {
        logging::tests::initialize_default_once();
        let root_path = tempdir::TempDir::new("summa_test").unwrap();
        let data_path = root_path.path().join("data");
        let application_config_holder = create_test_application_config_holder(&data_path);

        let index_service = IndexService::new(&application_config_holder);
        assert!(index_service
            .create_index(
                CreateIndexRequestBuilder::default()
                    .index_name("test_index".to_owned())
                    .index_engine(proto::IndexEngine::Memory)
                    .schema(create_test_schema())
                    .build()
                    .unwrap()
            )
            .await
            .is_ok());
        assert!(index_service
            .delete_index(DeleteIndexRequestBuilder::default().index_name("test_index".to_owned()).build().unwrap())
            .await
            .is_ok());
        assert!(index_service
            .create_index(
                CreateIndexRequestBuilder::default()
                    .index_name("test_index".to_owned())
                    .index_engine(proto::IndexEngine::Memory)
                    .schema(create_test_schema())
                    .build()
                    .unwrap()
            )
            .await
            .is_ok());
    }
}
