use crate::configs::{ApplicationConfigHolder, ConsumerConfig, IndexConfigBuilder, IndexConfigProxy, IndexEngine};
use crate::errors::{Error, SummaResult, ValidationError};
use crate::proto;
use crate::requests::{CreateConsumerRequest, CreateIndexRequest, DeleteConsumerRequest, DeleteIndexRequest};
use crate::search_engine::IndexHolder;
use crate::utils::sync::{Handler, OwningHandler};
use futures_util::future::join_all;
use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::sync::Arc;
use tantivy::directory::GarbageCollectionResult;
use tracing::instrument;

/// The main struct responsible for indices lifecycle. Here lives indices creation and deletion as well as committing and indexing new documents.
#[derive(Clone, Debug)]
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
    pub async fn new(application_config_holder: &ApplicationConfigHolder) -> SummaResult<IndexService> {
        let mut index_holders: HashMap<String, OwningHandler<IndexHolder>> = HashMap::new();
        {
            let application_config = application_config_holder.read();
            for (index_name, _index_config) in &application_config.indices {
                let index_config_proxy = IndexConfigProxy::new(application_config_holder, index_name);
                let index_holder = IndexHolder::open(index_name, index_config_proxy).await?;
                index_holders.insert(index_name.clone(), OwningHandler::new(index_holder));
            }
        }
        let index_service = IndexService {
            application_config: application_config_holder.clone(),
            index_holders: Arc::new(RwLock::new(index_holders)),
        };
        Ok(index_service)
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

    /// Returns `Handler` to `IndexHolder`.
    ///
    /// It is safe to keep `IndexHolder` cause `Index` won't be deleted until `Handler` is alive.
    /// Though, `IndexHolder` can be removed from `index_holders` to prevent new queries
    pub fn get_index_holder(&self, index_alias_or_name: &str) -> SummaResult<Handler<IndexHolder>> {
        let index_name = self.application_config.read().resolve_index_alias(index_alias_or_name);
        let index_name = index_name.as_deref().unwrap_or(index_alias_or_name);
        let index_holder = self
            .index_holders
            .read()
            .get(index_name)
            .ok_or_else(|| Error::ValidationError(ValidationError::MissingIndexError(index_alias_or_name.to_owned())))?
            .handler();
        Ok(index_holder)
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
            .sort_by_field(create_index_request.sort_by_field.clone())
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
    #[instrument(skip_all, fields(index_name = ?create_index_request.index_name))]
    pub async fn create_index(&self, create_index_request: CreateIndexRequest) -> SummaResult<Handler<IndexHolder>> {
        self.insert_config(&create_index_request)?;
        let owning_handler = OwningHandler::new(
            IndexHolder::create(
                &create_index_request.index_name,
                &create_index_request.schema,
                IndexConfigProxy::new(&self.application_config, &create_index_request.index_name),
            )
            .await?,
        );
        let handler = owning_handler.handler();
        self.index_holders.write().insert(create_index_request.index_name.to_owned(), owning_handler);
        Ok(handler)
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

    #[instrument(skip_all, fields(index_name = ?delete_index_request.index_name))]
    pub async fn delete_index(&self, delete_index_request: DeleteIndexRequest) -> SummaResult<DeleteIndexResult> {
        let mut index_holders = self.index_holders.write();
        let index_holder = match index_holders.get(&delete_index_request.index_name) {
            Some(index_holder) => index_holder,
            None => Err(ValidationError::MissingIndexError(delete_index_request.index_name.to_owned()))?,
        };
        let delete_index_result = self.prepare_delete(&delete_index_request, index_holder).await?;
        match index_holders.remove(&delete_index_request.index_name) {
            Some(index_holder) => {
                let inner = tokio::task::spawn_blocking(move || index_holder.into_inner()).await?;
                inner.delete().await?;
            }
            None => Err(ValidationError::MissingIndexError(delete_index_request.index_name.to_owned()))?,
        }
        Ok(delete_index_result)
    }

    pub async fn vacuum_index(&self, index_name: &str) -> SummaResult<GarbageCollectionResult> {
        let index_holders = self.index_holders.read();
        let index_holder = match index_holders.get(index_name) {
            Some(index_holder) => index_holder,
            None => Err(ValidationError::MissingIndexError(index_name.to_owned()))?,
        };
        index_holder.index_updater().read().vacuum().await
    }

    /// Create consumer and insert it into the consumer registry. Add it to the `IndexHolder` afterwards.
    pub fn get_consumers(&self) -> SummaResult<Vec<(String, String)>> {
        let application_config = self.application_config.read();
        let mut result = Vec::new();
        for (index_name, index_config) in &application_config.indices {
            result.extend(index_config.consumer_configs.keys().map(|consumer_name| (index_name.to_owned(), consumer_name.to_owned())))
        }
        Ok(result)
    }

    /// Create consumer and insert it into the consumer registry. Add it to the `IndexHolder` afterwards.
    #[instrument(skip_all, fields(consumer_name = ?create_consumer_request.consumer_name))]
    pub async fn create_consumer(&self, create_consumer_request: &CreateConsumerRequest) -> SummaResult<()> {
        let index_holder = self.get_index_holder(&create_consumer_request.index_name)?;
        index_holder
            .index_updater()
            .write()
            .create_consumer(&create_consumer_request.consumer_name, &create_consumer_request.consumer_config)
            .await?;

        Ok(())
    }

    /// Delete consumer from the consumer registry and from `IndexHolder` afterwards.
    #[instrument(skip_all, fields(consumer_name = ?delete_consumer_request.consumer_name))]
    pub async fn delete_consumer(&self, delete_consumer_request: DeleteConsumerRequest) -> SummaResult<()> {
        self.get_index_holder(&delete_consumer_request.index_name)?
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

    pub(crate) fn get_consumer_config(&self, index_name: &str, consumer_name: &str) -> SummaResult<ConsumerConfig> {
        Ok(self
            .application_config
            .read()
            .indices
            .get(index_name)
            .ok_or(ValidationError::MissingIndexError(index_name.to_owned()))?
            .consumer_configs
            .get(consumer_name)
            .ok_or(ValidationError::MissingConsumerError(consumer_name.to_owned()))?
            .clone())
    }
}
