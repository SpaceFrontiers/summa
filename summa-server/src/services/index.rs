use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use iroh_unixfs::content_loader::GatewayUrl;
use summa_core::components::{IndexHolder, IndexRegistry, IrohClient};
use summa_core::configs::ConfigProxy;
use summa_core::configs::PartialProxy;
use summa_core::directories::DefaultExternalRequestGenerator;
use summa_core::utils::sync::{Handler, OwningHandler};
use summa_core::utils::thread_handler::ThreadHandler;
use summa_proto::proto;
use tantivy::IndexBuilder;
use tokio::sync::RwLock;
use tracing::{info, info_span, instrument, warn, Instrument};

use crate::components::{ConsumerManager, PreparedConsumption};
use crate::errors::SummaServerResult;
use crate::errors::ValidationError;
use crate::hyper_external_request::HyperExternalRequest;
use crate::requests::{AttachIndexRequest, CreateConsumerRequest, CreateIndexRequest, DeleteConsumerRequest, DeleteIndexRequest};

/// The main struct responsible for indices lifecycle. Here lives indices creation and deletion as well as committing and indexing new documents.
#[derive(Clone)]
pub struct Index {
    server_config: Arc<dyn ConfigProxy<crate::configs::server::Config>>,
    index_registry: IndexRegistry,
    consumer_manager: Arc<RwLock<ConsumerManager>>,
    should_terminate: Arc<AtomicBool>,
    autocommit_thread: Arc<RwLock<Option<ThreadHandler<SummaServerResult<()>>>>>,
    iroh_client: IrohClient,
}

#[derive(Default)]
pub struct DeleteIndexResult {
    index_name: String,
}

impl DeleteIndexResult {
    pub fn new(index_name: &str) -> DeleteIndexResult {
        DeleteIndexResult {
            index_name: index_name.to_string(),
        }
    }
}

impl From<DeleteIndexResult> for proto::DeleteIndexResponse {
    fn from(delete_index_request: DeleteIndexResult) -> Self {
        proto::DeleteIndexResponse {
            index_name: delete_index_request.index_name,
        }
    }
}

/// The main entry point for managing Summa indices
impl Index {
    /// `HashMap` of all indices
    pub fn index_holders(&self) -> &Arc<RwLock<HashMap<String, OwningHandler<IndexHolder>>>> {
        self.index_registry.index_holders()
    }

    pub async fn index_holders_cloned(&self) -> HashMap<String, Handler<IndexHolder>> {
        self.index_registry
            .index_holders()
            .read()
            .await
            .iter()
            .map(|(index_name, handler)| (index_name.to_string(), handler.handler()))
            .collect()
    }

    /// Returns `ConsumerManager`
    pub fn consumer_manager(&self) -> &Arc<RwLock<ConsumerManager>> {
        &self.consumer_manager
    }

    /// Returns server config
    pub fn server_config(&self) -> &Arc<dyn ConfigProxy<crate::configs::server::Config>> {
        &self.server_config
    }

    /// Flag meaning that all dependent operations should be ended as soon as possible
    pub fn should_terminate(&self) -> bool {
        self.should_terminate.load(Ordering::Relaxed)
    }

    /// Returns `Handler` to `IndexHolder`.
    ///
    /// It is safe to keep `Handler<IndexHolder>` cause `Index` won't be deleted until `Handler` is alive.
    /// Though, `IndexHolder` can be removed from the registry of `IndexHolder`s to prevent new queries
    pub async fn get_index_holder(&self, index_alias: &str) -> SummaServerResult<Handler<IndexHolder>> {
        Ok(self
            .index_registry
            .get_index_holder_by_name(
                self.server_config
                    .read()
                    .await
                    .get()
                    .resolve_index_alias(index_alias)
                    .as_deref()
                    .unwrap_or(index_alias),
            )
            .await?)
    }

    /// Creates new `IndexService` with `ConfigHolder`
    pub async fn new(server_config_holder: &Arc<dyn ConfigProxy<crate::configs::server::Config>>) -> SummaServerResult<Index> {
        let config = server_config_holder.read().await.get().clone();
        Ok(Index {
            server_config: server_config_holder.clone(),
            index_registry: IndexRegistry::default(),
            consumer_manager: Arc::default(),
            should_terminate: Arc::default(),
            autocommit_thread: Arc::default(),
            iroh_client: IrohClient::new(
                config
                    .p2p
                    .http_gateways
                    .into_iter()
                    .map(|x| GatewayUrl::from_str(&x))
                    .collect::<Result<_, _>>()?,
            )
            .await?,
        })
    }

    async fn derive_configs(
        &self,
        index_name: &str,
    ) -> (
        Arc<dyn ConfigProxy<summa_core::configs::core::Config>>,
        Arc<dyn ConfigProxy<proto::IndexEngineConfig>>,
    ) {
        let core_config_holder = Arc::new(PartialProxy::new(
            &self.server_config,
            |server_config| &server_config.core,
            |server_config| &mut server_config.core,
        ));
        let index_engine_config_holder = Arc::new(PartialProxy::new(
            &self.server_config,
            {
                let index_name = index_name.to_string();
                move |server_config| server_config.core.indices.get(&index_name).unwrap()
            },
            {
                let index_name = index_name.to_string();
                move |server_config| server_config.core.indices.get_mut(&index_name).unwrap()
            },
        ));
        (core_config_holder, index_engine_config_holder)
    }

    /// Create `IndexHolder`s from config
    pub(crate) async fn setup_index_holders(&self) -> SummaServerResult<()> {
        let mut index_holders = HashMap::new();
        for (index_name, index_engine_config) in self.server_config.read().await.get().core.indices.clone().into_iter() {
            let index = self.create_index_from(&index_engine_config).await?;
            let core_config = self.server_config.read().await.get().core.clone();
            let (core_config_holder, index_engine_config_holder) = self.derive_configs(&index_name).await;
            let index_holder = tokio::task::spawn_blocking(move || {
                IndexHolder::create_holder(core_config_holder, &core_config, index, Some(&index_name), index_engine_config_holder)
            })
            .await
            .unwrap()?;
            index_holders.insert(index_holder.index_name().to_string(), OwningHandler::new(index_holder));
        }
        info!(action = "setting_index_holders", index_holders = ?index_holders);
        *self.index_registry.index_holders().write().await = index_holders;

        for (consumer_name, consumer_config) in self.server_config.read().await.get().consumers.iter() {
            let index_holder = self.get_index_holder(&consumer_config.index_name).await?;
            let prepared_consumption = PreparedConsumption::from_config(consumer_name, consumer_config)?;
            self.consumer_manager.write().await.start_consuming(&index_holder, prepared_consumption).await?;
        }

        Ok(())
    }

    pub(crate) async fn insert_config(&self, index_name: &str, index_engine_config: &proto::IndexEngineConfig) -> SummaServerResult<()> {
        let mut server_config = self.server_config.write().await;
        match server_config.get_mut().core.indices.entry(index_name.to_owned()) {
            Entry::Occupied(o) => Err(ValidationError::ExistingIndex(format!("{:?}", o.key()))),
            Entry::Vacant(v) => {
                v.insert(index_engine_config.clone());
                Ok(())
            }
        }?;
        server_config.commit().await?;
        Ok(())
    }

    /// Create consumer and insert it into the consumer registry. Add it to the `IndexHolder` afterwards.
    #[instrument(skip_all, fields(index_name = ?attach_index_request.index_name))]
    pub async fn attach_index(&self, attach_index_request: AttachIndexRequest) -> SummaServerResult<Handler<IndexHolder>> {
        let index_path = self.server_config.read().await.get().get_path_for_index_data(&attach_index_request.index_name);
        let (index, index_engine_config) = match attach_index_request.attach_index_request {
            proto::attach_index_request::Request::AttachFileEngineRequest(proto::AttachFileEngineRequest {}) => {
                if !index_path.exists() {
                    return Err(ValidationError::MissingIndex(attach_index_request.index_name.to_string()).into());
                }
                let file_engine_config = proto::FileEngineConfig {
                    path: index_path.to_string_lossy().to_string(),
                };
                let index = IndexHolder::attach_file_index(&file_engine_config).await?;
                let index_engine_config = proto::IndexEngineConfig {
                    config: Some(proto::index_engine_config::Config::File(file_engine_config)),
                };
                (index, index_engine_config)
            }
            proto::attach_index_request::Request::AttachIpfsEngineRequest(proto::AttachIpfsEngineRequest { cid }) => {
                let ipfs_index_engine = proto::IpfsEngineConfig {
                    cid: cid.to_string(),
                    chunked_cache_config: Some(proto::ChunkedCacheConfig {
                        chunk_size: 65536,
                        cache_size: Some(536870912),
                    }),
                    path: index_path.to_string_lossy().to_string(),
                };
                let index = IndexHolder::attach_ipfs_index(&ipfs_index_engine, &self.iroh_client).await?;
                let index_engine_config = proto::IndexEngineConfig {
                    config: Some(proto::index_engine_config::Config::Ipfs(ipfs_index_engine)),
                };
                (index, index_engine_config)
            }
            _ => unimplemented!(),
        };
        self.insert_config(&attach_index_request.index_name, &index_engine_config).await?;
        let core_config = self.server_config.read().await.get().core.clone();
        let (core_config_holder, index_engine_config_holder) = self.derive_configs(&attach_index_request.index_name).await;
        Ok(self
            .index_registry
            .add(IndexHolder::create_holder(
                core_config_holder,
                &core_config,
                index,
                Some(&attach_index_request.index_name),
                index_engine_config_holder,
            )?)
            .await)
    }

    /// Create consumer and insert it into the consumer registry. Add it to the `IndexHolder` afterwards.
    #[instrument(skip_all, fields(index_name = ?create_index_request.index_name))]
    pub async fn create_index(&self, create_index_request: CreateIndexRequest) -> SummaServerResult<Handler<IndexHolder>> {
        let index_attributes = create_index_request.index_attributes;
        let index_settings = tantivy::IndexSettings {
            docstore_compression: create_index_request.compression,
            docstore_blocksize: create_index_request.blocksize.unwrap_or(16384),
            sort_by_field: create_index_request.sort_by_field.clone(),
            ..Default::default()
        };
        let index_builder = tantivy::Index::builder()
            .schema(create_index_request.schema.clone())
            .settings(index_settings)
            .index_attributes(index_attributes);
        let (index, index_engine_config) = match create_index_request.index_engine {
            proto::CreateIndexEngineRequest::File => {
                let index_path = self.server_config.read().await.get().get_path_for_index_data(&create_index_request.index_name);
                let index = IndexHolder::create_file_index(&index_path, index_builder).await?;
                let index_engine_config = proto::IndexEngineConfig {
                    config: Some(proto::index_engine_config::Config::File(proto::FileEngineConfig {
                        path: index_path.to_string_lossy().to_string(),
                    })),
                };
                (index, index_engine_config)
            }
            proto::CreateIndexEngineRequest::Memory => {
                let index = index_builder.create_in_ram()?;
                let index_engine_config = proto::IndexEngineConfig {
                    config: Some(proto::index_engine_config::Config::Memory(proto::MemoryEngineConfig {
                        schema: serde_yaml::to_string(&index.schema()).expect("cannot serialize"),
                    })),
                };
                (index, index_engine_config)
            }
        };
        self.insert_config(&create_index_request.index_name, &index_engine_config).await?;
        let core_config = self.server_config.read().await.get().core.clone();
        let (core_config_holder, index_engine_config_holder) = self.derive_configs(&create_index_request.index_name).await;
        Ok(self
            .index_registry
            .add(IndexHolder::create_holder(
                core_config_holder,
                &core_config,
                index,
                Some(&create_index_request.index_name),
                index_engine_config_holder,
            )?)
            .await)
    }

    /// Delete index, optionally with all its aliases and consumers
    #[instrument(skip_all, fields(index_name = ?delete_index_request.index_name))]
    pub async fn delete_index(&self, delete_index_request: DeleteIndexRequest) -> SummaServerResult<DeleteIndexResult> {
        let index_holder = {
            let mut server_config = self.server_config.write().await;
            let aliases = server_config.get().get_index_aliases_for_index(&delete_index_request.index_name);
            match (
                self.index_registry
                    .index_holders()
                    .write()
                    .await
                    .entry(delete_index_request.index_name.to_owned()),
                server_config.get_mut().core.indices.entry(delete_index_request.index_name.to_owned()),
            ) {
                (Entry::Occupied(index_holder_entry), Entry::Occupied(config_entry)) => {
                    let index_holder = index_holder_entry.get();
                    if !aliases.is_empty() {
                        return Err(ValidationError::Aliased(aliases.join(", ")).into());
                    }
                    if let Some(consumer_name) = self
                        .consumer_manager
                        .read()
                        .await
                        .get_consumer_for(&index_holder.handler())
                        .await
                        .map(|consumer_thread| consumer_thread.consumer_name().to_string())
                    {
                        return Err(ValidationError::ExistingConsumer(consumer_name).into());
                    }
                    config_entry.remove();
                    let index_holder = index_holder_entry.remove();
                    server_config.commit().await?;
                    index_holder
                }
                (Entry::Vacant(_), Entry::Vacant(_)) => return Err(ValidationError::MissingIndex(delete_index_request.index_name.to_owned()).into()),
                (Entry::Occupied(index_holder_entry), Entry::Vacant(_)) => {
                    warn!(error = "missing_config");
                    index_holder_entry.remove()
                }
                (Entry::Vacant(_), Entry::Occupied(config_entry)) => {
                    warn!(error = "missing_index_holder");
                    config_entry.remove();
                    server_config.commit().await?;
                    return Err(ValidationError::MissingIndex(delete_index_request.index_name.to_owned()).into());
                }
            }
        };
        let index_name = index_holder.index_name().to_string();
        index_holder.into_inner().await.delete().await?;
        Ok(DeleteIndexResult::new(&index_name))
    }

    /// Returns all existent consumers for all indices
    pub async fn get_consumers(&self) -> HashMap<String, crate::configs::consumer::Config> {
        self.server_config.read().await.get().consumers.clone()
    }

    /// Create consumer and insert it into the consumer registry. Add it to the `IndexHolder` afterwards.
    #[instrument(skip_all, fields(consumer_name = ?create_consumer_request.consumer_name))]
    pub async fn create_consumer(&self, create_consumer_request: &CreateConsumerRequest) -> SummaServerResult<String> {
        let index_holder = self.get_index_holder(&create_consumer_request.consumer_config.index_name).await?;
        let prepared_consumption = PreparedConsumption::from_config(&create_consumer_request.consumer_name, &create_consumer_request.consumer_config)?;
        prepared_consumption.on_create().await?;
        self.consumer_manager.write().await.start_consuming(&index_holder, prepared_consumption).await?;
        let mut server_config = self.server_config.write().await;
        server_config.get_mut().consumers.insert(
            create_consumer_request.consumer_name.to_string(),
            create_consumer_request.consumer_config.clone(),
        );
        server_config.commit().await?;
        Ok(index_holder.index_name().to_owned())
    }

    /// Delete consumer from the consumer registry and from `IndexHolder` afterwards.
    #[instrument(skip_all, fields(consumer_name = ?delete_consumer_request.consumer_name))]
    pub async fn delete_consumer(&self, delete_consumer_request: DeleteConsumerRequest) -> SummaServerResult<proto::DeleteConsumerResponse> {
        let mut server_config = self.server_config.write().await;
        if server_config.get_mut().consumers.remove(&delete_consumer_request.consumer_name).is_none() {
            return Err(ValidationError::MissingConsumer(delete_consumer_request.consumer_name.to_string()).into());
        }
        server_config.commit().await?;
        let index_holder = self.consumer_manager.read().await.find_index_holder_for(&delete_consumer_request.consumer_name);
        if let Some(index_holder) = index_holder {
            let prepared_consumption = self.commit(&index_holder).await?;
            if let Some(prepared_consumption) = prepared_consumption {
                prepared_consumption.on_delete().await?;
            }
        }
        Ok(proto::DeleteConsumerResponse {
            consumer_name: delete_consumer_request.consumer_name.to_string(),
        })
    }

    /// Stopping index holders
    pub async fn stop(self, force: bool) -> SummaServerResult<()> {
        self.should_terminate.store(true, Ordering::Relaxed);
        if !force {
            self.consumer_manager.write().await.stop().await?;
        }
        Ok(())
    }

    /// Returns `ConsumerConfig`
    pub(crate) async fn get_consumer_config(&self, consumer_name: &str) -> Option<crate::configs::consumer::Config> {
        self.consumer_manager.read().await.find_consumer_config_for(consumer_name).cloned()
    }

    /// Commits all and restarts consuming threads
    pub async fn commit_and_restart_consumption(&self, index_holder: &Handler<IndexHolder>) -> SummaServerResult<()> {
        let prepared_consumption = self.commit(index_holder).await?;
        if let Some(prepared_consumption) = prepared_consumption {
            self.consumer_manager.write().await.start_consuming(index_holder, prepared_consumption).await?;
        }
        Ok(())
    }

    /// Commits all and restarts consuming threads
    pub async fn try_commit_and_restart_consumption(&self, index_holder: &Handler<IndexHolder>) -> SummaServerResult<()> {
        let prepared_consumption = self.try_commit(index_holder).await?;
        if let Some(prepared_consumption) = prepared_consumption {
            self.consumer_manager.write().await.start_consuming(index_holder, prepared_consumption).await?;
        }
        Ok(())
    }

    /// Commits all without restarting consuming threads
    #[instrument(skip(self, index_holder), fields(index_name = ?index_holder.index_name()))]
    pub async fn commit(&self, index_holder: &Handler<IndexHolder>) -> SummaServerResult<Option<PreparedConsumption>> {
        let stopped_consumption = self.consumer_manager.write().await.stop_consuming_for(index_holder).await?;
        index_holder.index_writer_holder().write().await.commit(None).await?;
        Ok(match stopped_consumption {
            Some(stopped_consumption) => Some(stopped_consumption.commit_offsets().await?),
            None => None,
        })
    }

    /// Commits immediately or returns all without restarting consuming threads
    #[instrument(skip(self, index_holder), fields(index_name = ?index_holder.index_name()))]
    pub async fn try_commit(&self, index_holder: &Handler<IndexHolder>) -> SummaServerResult<Option<PreparedConsumption>> {
        let mut index_writer = index_holder.index_writer_holder().try_write()?;
        let stopped_consumption = self.consumer_manager.write().await.stop_consuming_for(index_holder).await?;
        index_writer.commit(None).await?;
        Ok(match stopped_consumption {
            Some(stopped_consumption) => Some(stopped_consumption.commit_offsets().await?),
            None => None,
        })
    }

    async fn setup_autocommit_thread(&mut self) {
        let interval_ms = match self.server_config.read().await.get().core.autocommit_interval_ms {
            Some(interval_ms) => interval_ms,
            None => return,
        };

        let index_service = self.clone();
        let (shutdown_trigger, mut shutdown_tripwire) = async_broadcast::broadcast(1);
        let mut tick_task = tokio::time::interval(Duration::from_millis(interval_ms));

        *self.autocommit_thread.write().await = Some(ThreadHandler::new(
            tokio::spawn(
                async move {
                    info!(action = "spawning_autocommit_thread", interval_ms = interval_ms);
                    // The first tick ticks immediately so we skip it
                    tick_task.tick().await;
                    loop {
                        tokio::select! {
                            _ = tick_task.tick() => {
                                info!(action = "autocommit_thread_tick");
                                let index_holders = index_service.index_holders_cloned().await;
                                for index_holder in index_holders.into_values() {
                                    let result = index_service.try_commit_and_restart_consumption(&index_holder).await;
                                    if let Err(error) = result {
                                        warn!(error = ?error);
                                    }
                                }
                            }
                            _ = &mut shutdown_tripwire.recv() => {
                                info!(action = "shutdown_autocommit_thread");
                                break;
                            }
                        }
                    }
                    Ok(())
                }
                .instrument(info_span!(parent: None, "autocommit_thread")),
            ),
            shutdown_trigger,
        ));
    }

    /// Starts autocommitting thread
    #[instrument(skip(self))]
    pub async fn start_autocommit_thread(&mut self) {
        self.setup_autocommit_thread().await;
    }

    /// Stops autocommitting thread
    #[instrument(skip(self))]
    pub async fn stop_autocommit_thread(&mut self) -> SummaServerResult<()> {
        if let Some(autocommit_thread) = self.autocommit_thread.write().await.take() {
            autocommit_thread.stop().await??;
        }
        Ok(())
    }

    /// Opens index and sets it up via `setup`
    #[instrument(skip_all)]
    pub async fn create_index_from(&self, index_engine_config: &proto::IndexEngineConfig) -> SummaServerResult<tantivy::Index> {
        let index = match &index_engine_config.config {
            Some(proto::index_engine_config::Config::File(config)) => tantivy::Index::open_in_dir(&config.path)?,
            Some(proto::index_engine_config::Config::Memory(config)) => IndexBuilder::new().schema(serde_yaml::from_str(&config.schema)?).create_in_ram()?,
            Some(proto::index_engine_config::Config::Remote(config)) => {
                IndexHolder::attach_remote_index::<HyperExternalRequest, DefaultExternalRequestGenerator<HyperExternalRequest>>(config).await?
            }
            Some(proto::index_engine_config::Config::Ipfs(config)) => IndexHolder::attach_ipfs_index(config, &self.iroh_client).await?,
            _ => unimplemented!(),
        };
        info!(action = "from_config", index = ?index);
        Ok(index)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::default::Default;
    use std::path::Path;

    use rdkafka::admin::ConfigSource::Default;
    use summa_core::components::SummaDocument;
    use summa_proto::proto_traits::collector::shortcuts::{scored_doc, top_docs_collector, top_docs_collector_output, top_docs_collector_with_eval_expr};
    use summa_proto::proto_traits::query::shortcuts::match_query;
    use tantivy::doc;
    use tantivy::schema::{IndexRecordOption, Schema, TextFieldIndexing, TextOptions, FAST, INDEXED, STORED};

    use super::*;
    use crate::configs::server::tests::create_test_server_config_holder;
    use crate::logging;
    use crate::requests::{CreateIndexRequestBuilder, DeleteIndexRequestBuilder};

    pub async fn create_test_index_service(data_path: &Path) -> Index {
        Index::new(&create_test_server_config_holder(&data_path)).await.unwrap()
    }
    pub async fn create_test_index_holder(index_service: &Index, schema: &Schema) -> SummaServerResult<Handler<IndexHolder>> {
        index_service
            .create_index(
                CreateIndexRequestBuilder::default()
                    .index_name("test_index".to_owned())
                    .index_attributes(proto::IndexAttributes {
                        default_fields: vec!["title".to_owned(), "body".to_owned()],
                        ..Default::default()
                    })
                    .index_engine(proto::CreateIndexEngineRequest::Memory)
                    .schema(schema.clone())
                    .build()
                    .unwrap(),
            )
            .await
    }

    pub fn create_test_schema() -> Schema {
        let mut schema_builder = Schema::builder();

        schema_builder.add_i64_field("id", FAST | INDEXED | STORED);
        schema_builder.add_i64_field("issued_at", FAST | INDEXED | STORED);
        schema_builder.add_text_field(
            "title",
            TextOptions::default().set_stored().set_indexing_options(
                TextFieldIndexing::default()
                    .set_tokenizer("summa")
                    .set_index_option(IndexRecordOption::WithFreqsAndPositions),
            ),
        );
        schema_builder.add_text_field(
            "body",
            TextOptions::default().set_stored().set_indexing_options(
                TextFieldIndexing::default()
                    .set_tokenizer("summa")
                    .set_index_option(IndexRecordOption::WithFreqsAndPositions),
            ),
        );

        schema_builder.build()
    }

    #[tokio::test]
    async fn test_same_name_index() {
        logging::tests::initialize_default_once();
        let index_service = Index::default();
        assert!(index_service
            .create_index(
                CreateIndexRequestBuilder::default()
                    .index_name("test_index".to_owned())
                    .index_engine(proto::CreateIndexEngineRequest::Memory)
                    .schema(create_test_schema())
                    .build()
                    .unwrap(),
            )
            .await
            .is_ok());

        assert!(index_service
            .create_index(
                CreateIndexRequestBuilder::default()
                    .index_name("test_index".to_owned())
                    .index_engine(proto::CreateIndexEngineRequest::Memory)
                    .schema(create_test_schema())
                    .build()
                    .unwrap(),
            )
            .await
            .is_err());
    }

    #[tokio::test]
    async fn test_index_create_delete() -> SummaServerResult<()> {
        logging::tests::initialize_default_once();
        let root_path = tempdir::TempDir::new("summa_test").unwrap();
        let data_path = root_path.path().join("data");
        let server_config_holder = create_test_server_config_holder(&data_path);

        let index_service = Index::new(&server_config_holder).await;
        index_service
            .create_index(
                CreateIndexRequestBuilder::default()
                    .index_name("test_index".to_owned())
                    .index_engine(proto::CreateIndexEngineRequest::File)
                    .schema(create_test_schema())
                    .build()
                    .unwrap(),
            )
            .await?;
        assert!(index_service
            .delete_index(DeleteIndexRequestBuilder::default().index_name("test_index".to_owned()).build().unwrap())
            .await
            .is_ok());
        assert!(index_service
            .create_index(
                CreateIndexRequestBuilder::default()
                    .index_name("test_index".to_owned())
                    .index_engine(proto::CreateIndexEngineRequest::Memory)
                    .schema(create_test_schema())
                    .build()
                    .unwrap()
            )
            .await
            .is_ok());
        Ok(())
    }

    #[tokio::test]
    async fn test_search() -> SummaServerResult<()> {
        logging::tests::initialize_default_once();
        let schema = create_test_schema();

        let root_path = tempdir::TempDir::new("summa_test").unwrap();
        let data_path = root_path.path().join("data");

        let index_service = create_test_index_service(&data_path).await;
        let index_holder = create_test_index_holder(&index_service, &schema).await?;

        index_holder.index_document(SummaDocument::TantivyDocument(doc!(
            schema.get_field("id").unwrap() => 1i64,
            schema.get_field("title").unwrap() => "Headcrab",
            schema.get_field("body").unwrap() => "Physically, headcrabs are frail: a few bullets or a single strike from the player's melee weapon being sufficient to dispatch them. \
            They are also relatively slow-moving and their attacks inflict very little damage. However, they can leap long distances and heights. \
            Headcrabs seek out larger human hosts, which are converted into zombie-like mutants that attack any living lifeform nearby. \
            The converted humans are more resilient than an ordinary human would be and inherit the headcrab's resilience toward toxic and radioactive materials. \
            Headcrabs and headcrab zombies die slowly when they catch fire. \
            The games also establish that while headcrabs are parasites that prey on humans, they are also the prey of the creatures of their homeworld. \
            Bullsquids, Vortigaunts, barnacles and antlions will all eat headcrabs and Vortigaunts can be seen cooking them in several locations in-game.",
            schema.get_field("issued_at").unwrap() => 1652986134i64
        ))).await?;
        index_holder.index_writer_holder().write().await.commit(None).await?;
        index_holder.index_reader().reload().unwrap();
        assert_eq!(
            index_holder.search("index", &match_query("headcrabs"), &vec![top_docs_collector(10)]).await?,
            vec![top_docs_collector_output(
                vec![scored_doc(
                    "{\
                        \"body\":\"Physically, headcrabs are frail: a few bullets or a single strike from the player's melee weapon being sufficient \
                        to dispatch them. They are also relatively slow-moving and their attacks inflict very little damage. However, they can leap long distances \
                        and heights. Headcrabs seek out larger human hosts, which are converted into zombie-like mutants that attack any living lifeform nearby. \
                        The converted humans are more resilient than an ordinary human would be and inherit the headcrab's resilience toward toxic and radioactive materials. \
                        Headcrabs and headcrab zombies die slowly when they catch fire. \
                        The games also establish that while headcrabs are parasites that prey on humans, they are also the prey of the creatures of their homeworld. \
                        Bullsquids, Vortigaunts, barnacles and antlions will all eat headcrabs and Vortigaunts can be seen cooking them in several locations in-game.\",\
                        \"id\":1,\
                        \"issued_at\":1652986134,\
                        \"title\":\"Headcrab\"}",
                    0.5126588344573975,
                    0
                )],
                false
            )]
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_custom_ranking() -> SummaServerResult<()> {
        logging::tests::initialize_default_once();
        let schema = create_test_schema();

        let root_path = tempdir::TempDir::new("summa_test").unwrap();
        let data_path = root_path.path().join("data");

        let index_service = create_test_index_service(&data_path).await;
        let index_holder = create_test_index_holder(&index_service, &schema).await?;

        index_holder
            .index_document(SummaDocument::TantivyDocument(doc!(
                schema.get_field("id").unwrap() => 1i64,
                schema.get_field("title").unwrap() => "term1 term2",
                schema.get_field("body").unwrap() => "term3 term4 term5 term6",
                schema.get_field("issued_at").unwrap() => 100i64
            )))
            .await?;
        index_holder
            .index_document(SummaDocument::TantivyDocument(doc!(
                schema.get_field("id").unwrap() => 2i64,
                schema.get_field("title").unwrap() => "term2 term3",
                schema.get_field("body").unwrap() => "term1 term7 term8 term9 term10",
                schema.get_field("issued_at").unwrap() => 110i64
            )))
            .await?;
        index_holder.index_writer_holder().write().await.commit(None).await?;
        index_holder.index_reader().reload().unwrap();
        assert_eq!(
            index_holder
                .search("index", &match_query("term1"), &vec![top_docs_collector_with_eval_expr(10, "issued_at")])
                .await?,
            vec![top_docs_collector_output(
                vec![
                    scored_doc(
                        "{\"body\":\"term1 term7 term8 term9 term10\",\"id\":2,\"issued_at\":110,\"title\":\"term2 term3\"}",
                        110.0,
                        0
                    ),
                    scored_doc(
                        "{\"body\":\"term3 term4 term5 term6\",\"id\":1,\"issued_at\":100,\"title\":\"term1 term2\"}",
                        100.0,
                        1
                    )
                ],
                false
            )]
        );
        assert_eq!(
            index_holder
                .search("index", &match_query("term1"), &vec![top_docs_collector_with_eval_expr(10, "-issued_at")])
                .await?,
            vec![top_docs_collector_output(
                vec![
                    scored_doc(
                        "{\"body\":\"term3 term4 term5 term6\",\"id\":1,\"issued_at\":100,\"title\":\"term1 term2\"}",
                        -100.0,
                        0
                    ),
                    scored_doc(
                        "{\"body\":\"term1 term7 term8 term9 term10\",\"id\":2,\"issued_at\":110,\"title\":\"term2 term3\"}",
                        -110.0,
                        1
                    ),
                ],
                false
            )]
        );
        Ok(())
    }
}
