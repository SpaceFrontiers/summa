use crate::configurator::configs::RuntimeConfigHolder;
use crate::consumers::kafka::KafkaConsumer;
use crate::consumers::ConsumerRegistry;
use crate::errors::{BadRequestError, Error, SummaResult, ValidationError};
use crate::requests::{CreateConsumerRequest, CreateIndexRequest};
use crate::search_engine::IndexHolder;
use crate::services::AliasService;
use futures::future::join_all;
use opentelemetry::metrics::{BatchObserverResult, Unit};
use opentelemetry::{global, KeyValue};
use parking_lot::{RwLock, RwLockReadGuard, RwLockUpgradableReadGuard, RwLockWriteGuard};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Indices creation and deletion as well as committing and indexing new documents
#[derive(Clone, Debug)]
pub struct IndexService {
    root_path: PathBuf,
    alias_service: AliasService,
    consumer_registry: ConsumerRegistry,
    index_holders: Arc<RwLock<HashMap<String, IndexHolder>>>,
}

#[derive(Default)]
pub struct DeleteIndexResult {
    pub deleted_aliases: Vec<String>,
    pub deleted_consumers: Vec<String>,
}

impl IndexService {
    pub fn new(root_path: &Path, runtime_config: &Arc<RwLock<RuntimeConfigHolder>>, alias_service: &AliasService) -> SummaResult<IndexService> {
        let consumer_registry = ConsumerRegistry::new(runtime_config)?;
        let index_holders = std::fs::read_dir(root_path)?
            .filter_map(|possible_index_path| possible_index_path.ok())
            .filter(|possible_index_path| possible_index_path.path().join("index.yaml").exists())
            .map(|index_path| {
                let index_config_filepath = index_path.path().join("index.yaml");
                let index_name = String::from(index_path.file_name().to_str().unwrap());
                Ok((index_name.clone(), IndexHolder::open(&index_name, &index_config_filepath, &index_path.path().join("data"))?))
            })
            .collect::<SummaResult<HashMap<String, IndexHolder>>>()?;

        let index_service = IndexService {
            root_path: root_path.to_path_buf(),
            alias_service: alias_service.clone(),
            consumer_registry,
            index_holders: Arc::new(RwLock::new(index_holders)),
        };
        index_service.register_meter();
        Ok(index_service)
    }

    pub fn consumer_registry(&self) -> &ConsumerRegistry {
        &self.consumer_registry
    }

    pub fn index_holders(&self) -> RwLockReadGuard<'_, HashMap<String, IndexHolder>> {
        self.index_holders.read()
    }
    pub fn index_holders_mut(&self) -> RwLockWriteGuard<'_, HashMap<String, IndexHolder>> {
        self.index_holders.write()
    }

    pub fn get_index_holder(&self, index_alias_or_name: &str) -> SummaResult<IndexHolder> {
        let index_name = self.alias_service.resolve_index_alias(index_alias_or_name);
        let index_name = index_name.as_deref().unwrap_or(index_alias_or_name);
        Ok(self
            .index_holders
            .read()
            .get(index_name)
            .ok_or_else(|| BadRequestError::NotFoundError(index_alias_or_name.to_string()))?
            .clone())
    }

    pub async fn create_index(&self, create_index_request: CreateIndexRequest) -> SummaResult<()> {
        // ToDo: Move validation to CreateIndexRequest class
        let index_path = self.root_path.join(&create_index_request.index_name);
        if index_path.exists() {
            Err(ValidationError::ExistingIndexError(create_index_request.index_name.clone()))?;
        }
        let data_path = index_path.join("data");
        std::fs::create_dir_all(&data_path).map_err(|e| Error::IOError((e, Some(data_path.clone()))))?;
        let index_holder = IndexHolder::new(
            &create_index_request.index_name,
            create_index_request.index_config,
            &index_path.join("index.yaml"),
            &data_path,
            &create_index_request.schema,
        )?;
        self.index_holders.write().insert(create_index_request.index_name.to_string(), index_holder);
        Ok(())
    }

    pub async fn delete_index(&self, index_name: &str, cascade: bool) -> SummaResult<DeleteIndexResult> {
        let index_holders = self.index_holders.upgradable_read();
        index_holders.get(index_name).ok_or_else(|| BadRequestError::NotFoundError(index_name.to_string()))?;

        let mut delete_index_result = DeleteIndexResult::default();

        let aliases = self.alias_service.get_index_aliases_for_index(index_name);
        if aliases.len() > 0 {
            if cascade {
                self.alias_service.delete_index_aliases(&aliases);
                delete_index_result.deleted_aliases.extend(aliases);
            } else {
                Err(BadRequestError::AliasedError(aliases.join(", ")))?;
            }
        }

        let consumer_configs = self.consumer_registry.get_consumer_configs_for_index(index_name);
        let consumer_config_names: Vec<_> = consumer_configs.keys().map(|x| x.to_string()).collect();
        if consumer_configs.len() > 0 {
            if cascade {
                self.consumer_registry.delete_consumer_configs(&consumer_config_names);
                delete_index_result.deleted_consumers.extend(consumer_config_names);
            } else {
                Err(BadRequestError::ExistingConsumersError(consumer_config_names.join(", ")))?;
            }
        }

        RwLockUpgradableReadGuard::<'_, HashMap<std::string::String, IndexHolder>>::upgrade(index_holders)
            .remove(index_name)
            .unwrap()
            .delete()
            .await?;
        Ok(delete_index_result)
    }

    pub async fn create_consumer(&self, create_consumer_request: CreateConsumerRequest) -> SummaResult<()> {
        self.consumer_registry
            .insert_consumer_config(&create_consumer_request.consumer_name, &create_consumer_request.consumer_config)?;
        let index_holder = self.get_index_holder(&create_consumer_request.index_name)?;
        index_holder
            .index_updater()
            .add_consumer(KafkaConsumer::new(&create_consumer_request.consumer_name, &create_consumer_request.consumer_config).await?)?;
        Ok(())
    }

    pub async fn delete_consumer(&self, consumer_name: &str) -> SummaResult<()> {
        let consumer_config = self.consumer_registry.delete_consumer_config(consumer_name)?;
        let index_holder = self.get_index_holder(&consumer_config.index_name)?;
        index_holder.index_updater().delete_consumer(consumer_name).await?;
        Ok(())
    }

    fn register_meter(&self) {
        let meter = global::meter("summa");
        let index_holders = self.index_holders.clone();
        meter.batch_observer(move |batch| {
            let index_holders = index_holders.clone();
            let total_memory_usage = batch
                .u64_value_observer("total_memory_usage")
                .with_description("Total memory usage in bytes, including all large subcomponents. Does not account for smaller things like meta.json")
                .with_unit(Unit::new("bytes"))
                .init();
            let document_count = batch.u64_value_observer("document_count").with_description("Total document count").init();
            let segment_count = batch.u64_value_observer("segment_count").with_description("Total segment count").init();

            move |result: BatchObserverResult| {
                for index_holder in index_holders.read().values() {
                    let space_usage = index_holder.space_usage();
                    let segments = space_usage.segments();

                    result.observe(
                        &[KeyValue::new("index_name", index_holder.index_name().to_string())],
                        &[
                            document_count.observation(index_holder.count().try_into().unwrap()),
                            total_memory_usage.observation(space_usage.total().try_into().unwrap()),
                            segment_count.observation(segments.len().try_into().unwrap()),
                        ],
                    );
                }
            }
        });
    }

    /// Reading consumers configs and injecting consumers into appropriate indices
    pub async fn start(&self) -> SummaResult<()> {
        for index_holder in self.index_holders().values() {
            let consumer_configs = self.consumer_registry.get_consumer_configs_for_index(&index_holder.index_name());
            let consumers: Vec<_> = consumer_configs
                .iter()
                .map(|(consumer_name, consumer_config)| KafkaConsumer::new(consumer_name, consumer_config))
                .collect();
            let consumers = join_all(consumers).await.into_iter().collect::<SummaResult<Vec<_>>>()?;
            index_holder.start(consumers)?;
        }
        Ok(())
    }

    /// Stopping index holders
    pub async fn stop(self) -> SummaResult<()> {
        for (_, index_holder) in self.index_holders_mut().drain() {
            index_holder.stop().await?;
        }
        Ok(())
    }
}
