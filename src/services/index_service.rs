use crate::configs::RuntimeConfigHolder;
use crate::consumers::kafka::KafkaConsumer;
use crate::consumers::ConsumerRegistry;
use crate::errors::{BadRequestError, SummaResult, ValidationError};
use crate::requests::{CreateConsumerRequest, CreateIndexRequest, DeleteConsumerRequest, DeleteIndexRequest};
use crate::search_engine::index_layout::IndexLayout;
use crate::search_engine::IndexHolder;
use crate::services::AliasService;
use crate::utils::sync::{Handler, OwningHandler};
use futures_util::future::join_all;
use parking_lot::{RwLock, RwLockReadGuard, RwLockUpgradableReadGuard, RwLockWriteGuard};
use std::collections::HashMap;
use std::fs::DirEntry;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// The main struct responsible for indices lifecycle. Here lives indices creation and deletion as well as committing and indexing new documents.
#[derive(Clone, Debug)]
pub struct IndexService {
    root_path: PathBuf,
    alias_service: AliasService,
    consumer_registry: ConsumerRegistry,
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
    pub async fn new(root_path: &Path, runtime_config: &Arc<RwLock<RuntimeConfigHolder>>, alias_service: &AliasService) -> SummaResult<IndexService> {
        let consumer_registry = ConsumerRegistry::new(runtime_config)?;
        let mut index_holders: HashMap<String, OwningHandler<IndexHolder>> = HashMap::new();
        for index_path in IndexService::indices_on_disk(root_path)? {
            let index_layout = IndexLayout::open(&index_path.path()).await?;
            let index_name = String::from(index_path.file_name().to_str().unwrap());
            let consumer_configs = consumer_registry.get_consumer_configs_for_index(&index_name);
            let consumers = consumer_configs
                .iter()
                .map(|(consumer_name, consumer_config)| KafkaConsumer::new(consumer_name, consumer_config));
            let consumers: SummaResult<Vec<_>> = join_all(consumers).await.into_iter().collect();
            index_holders.insert(index_name.clone(), OwningHandler::new(IndexHolder::open(&index_name, &index_layout, consumers?)?));
        }

        let index_service = IndexService {
            root_path: root_path.to_path_buf(),
            alias_service: alias_service.clone(),
            consumer_registry,
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

    pub fn consumer_registry(&self) -> &ConsumerRegistry {
        &self.consumer_registry
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
            .ok_or_else(|| BadRequestError::NotFoundError(index_alias_or_name.to_string()))?
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
            Vec::new(),
        )?;
        index_holders.insert(create_index_request.index_name.to_string(), OwningHandler::new(index_holder));
        Ok(())
    }

    pub async fn delete_index(&self, delete_index_request: DeleteIndexRequest) -> SummaResult<DeleteIndexResult> {
        let index_holders = self.index_holders.upgradable_read();
        if !index_holders.contains_key(&delete_index_request.index_name) {
            Err(BadRequestError::NotFoundError(delete_index_request.index_name.to_string()))?;
        }

        let mut delete_index_result = DeleteIndexResult::default();

        let aliases = self.alias_service.get_index_aliases_for_index(&delete_index_request.index_name);
        let consumer_configs = self.consumer_registry.get_consumer_configs_for_index(&delete_index_request.index_name);
        let consumer_config_names: Vec<_> = consumer_configs.keys().map(|x| x.to_string()).collect();

        if delete_index_request.cascade {
            self.alias_service.delete_index_aliases(&aliases);
            delete_index_result.deleted_aliases.extend(aliases);
            self.consumer_registry.delete_consumer_configs(&consumer_config_names);
            delete_index_result.deleted_consumers.extend(consumer_config_names);
        } else {
            if aliases.len() > 0 {
                Err(BadRequestError::AliasedError(aliases.join(", ")))?;
            }
            if consumer_configs.len() > 0 {
                Err(BadRequestError::ExistingConsumersError(consumer_config_names.join(", ")))?;
            }
        }

        let index = RwLockUpgradableReadGuard::<'_, HashMap<String, OwningHandler<IndexHolder>>>::upgrade(index_holders)
            .remove(&delete_index_request.index_name)
            .unwrap();

        let inner = tokio::task::spawn_blocking(move || index.into_inner()).await?;
        inner.delete().await?;
        Ok(delete_index_result)
    }

    /// Create consumer and insert it into the consumer registry. Add it to the `IndexHolder` afterwards.
    pub async fn create_consumer(&self, create_consumer_request: CreateConsumerRequest) -> SummaResult<()> {
        let index_holder = self.get_index_holder(&create_consumer_request.index_name)?;
        if self.consumer_registry.get_consumer_config(&create_consumer_request.consumer_name).is_ok() {
            Err(ValidationError::ExistingConsumerError(create_consumer_request.consumer_name.to_string()))?
        };
        let kafka_consumer = KafkaConsumer::new(&create_consumer_request.consumer_name, &create_consumer_request.consumer_config).await?;
        index_holder.index_updater().write().add_consumer(kafka_consumer)?;
        self.consumer_registry
            .insert_consumer_config(&create_consumer_request.consumer_name, &create_consumer_request.consumer_config)?;
        Ok(())
    }

    /// Delete consumer from the consumer registry and from `IndexHolder` afterwards.
    pub async fn delete_consumer(&self, delete_consumer_request: DeleteConsumerRequest) -> SummaResult<()> {
        let index_name = self.consumer_registry.get_consumer_config(&delete_consumer_request.consumer_name)?.index_name;
        let index_holder = self.get_index_holder(&index_name)?;
        self.consumer_registry.delete_consumer_config(&delete_consumer_request.consumer_name)?;
        index_holder.index_updater().write().delete_consumer(&delete_consumer_request.consumer_name).await?;
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
