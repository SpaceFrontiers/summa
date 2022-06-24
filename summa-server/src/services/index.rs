use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_broadcast::Receiver;
use futures_util::future::join_all;
use summa_core::components::{cleanup_index, IndexHolder, IndexRegistry};
use summa_core::configs::ConfigProxy;
use summa_core::configs::PartialProxy;
use summa_core::directories::DefaultExternalRequestGenerator;
use summa_core::errors::SummaResult;
use summa_core::hyper_external_request::HyperExternalRequest;
use summa_core::proto_traits::Wrapper;
use summa_core::utils::sync::{Handler, OwningHandler};
use summa_core::validators;
use summa_proto::proto;
use tantivy::store::ZstdCompressor;
use tantivy::{IndexBuilder, SegmentId};
use tokio::sync::RwLock;
use tracing::{error, info, info_span, instrument, warn, Instrument};

use crate::components::{ConsumerManager, PreparedConsumption};
use crate::errors::SummaServerResult;
use crate::errors::ValidationError;
use crate::utils::thread_handler::{ControlMessage, ThreadHandler};

/// `services::Index` is responsible for indices lifecycle. Here lives indices creation and deletion as well as committing and indexing new documents.
#[derive(Clone)]
pub struct Index {
    server_config: Arc<dyn ConfigProxy<crate::configs::server::Config>>,
    index_registry: IndexRegistry,
    consumer_manager: Arc<RwLock<ConsumerManager>>,
    should_terminate: Arc<AtomicBool>,
    autocommit_thread: Arc<RwLock<Option<ThreadHandler<SummaServerResult<()>>>>>,
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
            deleted_index_name: delete_index_request.index_name,
        }
    }
}

/// The main entry point for managing Summa indices
impl Index {
    pub fn index_registry(&self) -> &IndexRegistry {
        &self.index_registry
    }

    pub async fn get_index_holder(&self, index_alias: &str) -> SummaServerResult<Handler<IndexHolder>> {
        Ok(self.index_registry.get_index_holder(index_alias).await?)
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
    pub(crate) fn should_terminate(&self) -> bool {
        self.should_terminate.load(Ordering::Relaxed)
    }

    /// Creates new `IndexService` with `ConfigHolder`
    pub fn new(server_config_holder: &Arc<dyn ConfigProxy<crate::configs::server::Config>>) -> SummaServerResult<Index> {
        let core_config_holder = Arc::new(PartialProxy::new(
            server_config_holder,
            |server_config| &server_config.core,
            |server_config| &mut server_config.core,
        )) as Arc<dyn ConfigProxy<_>>;
        Ok(Index {
            server_config: server_config_holder.clone(),
            index_registry: IndexRegistry::new(&core_config_holder),
            consumer_manager: Arc::default(),
            should_terminate: Arc::default(),
            autocommit_thread: Arc::default(),
        })
    }

    async fn derive_configs(&self, index_name: &str) -> Arc<dyn ConfigProxy<proto::IndexEngineConfig>> {
        Arc::new(PartialProxy::new(
            &self.server_config,
            {
                let index_name = index_name.to_string();
                move |server_config| server_config.core.indices.get(&index_name).expect("index disappeared")
            },
            {
                let index_name = index_name.to_string();
                move |server_config| server_config.core.indices.get_mut(&index_name).expect("index disappeared")
            },
        ))
    }

    /// Create `IndexHolder`s from config
    pub(crate) async fn setup_indices(&self) -> SummaServerResult<()> {
        let mut index_holders = HashMap::new();
        for (index_name, index_engine_config) in self.server_config.read().await.get().core.indices.clone().into_iter() {
            info!(action = "from_config", index = ?index_name);
            let index = self
                .open_index_from_config(index_engine_config)
                .instrument(info_span!("open_index_from_config", index_name = ?index_name))
                .await?;
            let core_config = self.server_config.read().await.get().core.clone();
            let index_engine_config_holder = self.derive_configs(&index_name).await;
            let merge_policy = core_config.indices[&index_name].merge_policy.clone();
            let query_parser_config = core_config.indices[&index_name].query_parser_config.as_ref().cloned().unwrap_or_default();
            let default_fields = query_parser_config.default_fields.clone();
            let index_name_clone = index_name.clone();
            let index_holder = tokio::task::spawn_blocking(move || {
                IndexHolder::create_holder(
                    &core_config,
                    index,
                    &index_name_clone,
                    index_engine_config_holder,
                    merge_policy,
                    query_parser_config,
                )
            })
            .await??;
            index_holder.partial_warmup(false, &default_fields).await?;
            index_holders.insert(index_name, OwningHandler::new(index_holder));
        }
        info!(action = "setting_index_holders", indices = ?index_holders.keys().collect::<Vec<_>>());
        *self.index_registry.index_holders().write().await = index_holders;

        for (consumer_name, consumer_config) in self.server_config.read().await.get().consumers.iter() {
            let index_holder = self.index_registry.get_index_holder(&consumer_config.index_name).await?;
            let prepared_consumption = PreparedConsumption::from_config(consumer_name, consumer_config)?;
            self.consumer_manager.write().await.start_consuming(&index_holder, prepared_consumption).await?;
        }
        info!(action = "indices_ready");
        Ok(())
    }

    #[instrument("lifecycle", skip_all)]
    pub(crate) async fn prepare_serving_future(
        &self,
        mut terminator: Receiver<ControlMessage>,
    ) -> SummaServerResult<impl Future<Output = SummaServerResult<()>>> {
        self.setup_indices().await?;
        let this = self.clone();
        Ok(async move {
            let signal_result = terminator.recv().await;
            info!(action = "sigterm_received", received = ?signal_result);
            let force = matches!(signal_result, Ok(ControlMessage::ForceShutdown));
            match this.stop(force).await {
                Ok(_) => info!(action = "terminated"),
                Err(error) => info!(action = "terminated", error = ?error),
            };
            Ok(())
        }
        .instrument(info_span!(parent: None, "lifecycle")))
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

    async fn insert_index(
        &self,
        index_name: &str,
        index: tantivy::Index,
        index_engine_config: &proto::IndexEngineConfig,
    ) -> SummaServerResult<Handler<IndexHolder>> {
        self.insert_config(index_name, index_engine_config).await?;
        let core_config = self.server_config.read().await.get().core.clone();
        let index_engine_config_holder = self.derive_configs(index_name).await;
        let merge_policy = index_engine_config.merge_policy.clone();
        let query_parser_config = index_engine_config.query_parser_config.as_ref().cloned().unwrap_or_default();
        let index_name = index_name.to_string();
        Ok(self
            .index_registry
            .add(
                tokio::task::spawn_blocking(move || {
                    IndexHolder::create_holder(&core_config, index, &index_name, index_engine_config_holder, merge_policy, query_parser_config)
                })
                .await??,
            )
            .await?)
    }

    /// Create consumer and insert it into the consumer registry. Add it to the `IndexHolder` afterwards.
    #[instrument(skip_all, fields(index_name = ?attach_index_request.index_name))]
    pub async fn attach_index(&self, attach_index_request: proto::AttachIndexRequest) -> SummaServerResult<Handler<IndexHolder>> {
        let index_path = self.server_config.read().await.get().get_path_for_index_data(&attach_index_request.index_name);
        let query_parser_config = attach_index_request.query_parser_config;
        let (index, index_engine_config) = match attach_index_request.index_engine {
            None | Some(proto::attach_index_request::IndexEngine::File(proto::AttachFileEngineRequest {})) => {
                if !index_path.exists() {
                    return Err(ValidationError::MissingIndex(attach_index_request.index_name.to_string()).into());
                }
                let file_engine_config = proto::FileEngineConfig {
                    path: index_path.to_string_lossy().to_string(),
                };
                let index = IndexHolder::open_file_index(&file_engine_config).await?;
                let index_engine_config = proto::IndexEngineConfig {
                    config: Some(proto::index_engine_config::Config::File(file_engine_config)),
                    merge_policy: attach_index_request.merge_policy,
                    query_parser_config: query_parser_config.clone(),
                };
                (index, index_engine_config)
            }
            Some(proto::attach_index_request::IndexEngine::Remote(proto::AttachRemoteEngineRequest {
                config: Some(remote_engine_config),
            })) => {
                let index = IndexHolder::open_remote_index::<HyperExternalRequest, DefaultExternalRequestGenerator<HyperExternalRequest>>(
                    remote_engine_config.clone(),
                    true,
                )
                .await?;
                let index_engine_config = proto::IndexEngineConfig {
                    config: Some(proto::index_engine_config::Config::Remote(remote_engine_config)),
                    merge_policy: attach_index_request.merge_policy,
                    query_parser_config: query_parser_config.clone(),
                };
                (index, index_engine_config)
            }
            _ => unimplemented!(),
        };
        let index_holder = self.insert_index(&attach_index_request.index_name, index, &index_engine_config).await?;
        index_holder
            .partial_warmup(false, &query_parser_config.map(|q| q.default_fields).unwrap_or_default())
            .await?;
        Ok(index_holder)
    }

    #[instrument(skip_all, fields(source_index_name = ?copy_documents_request.source_index_name, target_index_name = ?copy_documents_request.target_index_name))]
    pub async fn copy_documents(&self, copy_documents_request: proto::CopyDocumentsRequest) -> SummaServerResult<u32> {
        let target_index_holder = self.get_index_holder(&copy_documents_request.target_index_name).await?;
        let mut target_index_writer = target_index_holder.index_writer_holder()?.clone().read_owned().await;
        let conflict_strategy = copy_documents_request
            .conflict_strategy
            .and_then(proto::ConflictStrategy::from_i32)
            .unwrap_or(target_index_holder.conflict_strategy());
        let source_index_holder = self.get_index_holder(&copy_documents_request.source_index_name).await?;
        let searcher = source_index_holder.index_reader().searcher();
        let mut source_documents_receiver = source_index_holder.documents(&searcher, Some)?;
        let mut documents = 0u32;
        while let Some(document) = source_documents_receiver.recv().await {
            target_index_writer
                .index_document(document, conflict_strategy)
                .map_err(crate::errors::Error::from)?;
            documents += 1;
            target_index_writer = if documents % 100_000 == 0 {
                info!(action = "copied", documents = documents);
                drop(target_index_writer);
                target_index_holder.index_writer_holder()?.clone().read_owned().await
            } else {
                target_index_writer
            }
        }
        Ok(documents)
    }

    /// Create consumer and insert it into the consumer registry. Add it to the `IndexHolder` afterwards.
    #[instrument(skip_all, fields(index_name = ?create_index_request.index_name))]
    pub async fn create_index(&self, create_index_request: proto::CreateIndexRequest) -> SummaServerResult<Handler<IndexHolder>> {
        let schema = validators::parse_schema(&create_index_request.schema)?;

        let mut index_attributes = create_index_request.index_attributes.unwrap_or_default();
        let query_parser_config = create_index_request.query_parser_config;
        let default_fields = query_parser_config.as_ref().map(|q| q.default_fields.clone()).unwrap_or_default();
        validators::parse_fields(&schema, &default_fields)?;
        validators::parse_fields(&schema, &index_attributes.multi_fields)?;
        validators::parse_fields(&schema, &index_attributes.unique_fields)?;

        index_attributes.created_at = SystemTime::now().duration_since(UNIX_EPOCH).expect("cannot retrieve time").as_secs();

        let index_settings = tantivy::IndexSettings {
            docstore_compression: proto::Compression::from_i32(create_index_request.compression)
                .map(|c| Wrapper::from(c).into())
                .unwrap_or(tantivy::store::Compressor::Zstd(ZstdCompressor::default())),
            docstore_blocksize: create_index_request.blocksize.unwrap_or(32768) as usize,
            sort_by_field: create_index_request.sort_by_field.map(|s| Wrapper::from(s).into()),
            ..Default::default()
        };
        let index_builder = tantivy::Index::builder()
            .schema(schema)
            .settings(index_settings)
            .index_attributes(index_attributes);
        let (index, index_engine_config) = match create_index_request.index_engine {
            None | Some(proto::create_index_request::IndexEngine::File(proto::CreateFileEngineRequest {})) => {
                let index_path = self.server_config.read().await.get().get_path_for_index_data(&create_index_request.index_name);
                let index = IndexHolder::create_file_index(&index_path, index_builder).await?;
                let index_engine_config = proto::IndexEngineConfig {
                    config: Some(proto::index_engine_config::Config::File(proto::FileEngineConfig {
                        path: index_path.to_string_lossy().to_string(),
                    })),
                    merge_policy: create_index_request.merge_policy,
                    query_parser_config,
                };
                (index, index_engine_config)
            }
            Some(proto::create_index_request::IndexEngine::Memory(proto::CreateMemoryEngineRequest {})) => {
                let index = IndexHolder::create_memory_index(index_builder)?;
                let index_engine_config = proto::IndexEngineConfig {
                    config: Some(proto::index_engine_config::Config::Memory(proto::MemoryEngineConfig {
                        schema: serde_yaml::to_string(&index.schema()).expect("cannot serialize"),
                    })),
                    merge_policy: create_index_request.merge_policy,
                    query_parser_config,
                };
                (index, index_engine_config)
            }
        };
        let index_holder = self.insert_index(&create_index_request.index_name, index, &index_engine_config).await?;
        index_holder.partial_warmup(false, &default_fields).await?;
        Ok(index_holder)
    }

    /// Delete index, optionally with all its aliases and consumers
    #[instrument(skip_all, fields(index_name = ?delete_index_request.index_name))]
    pub async fn delete_index(&self, delete_index_request: proto::DeleteIndexRequest) -> SummaServerResult<DeleteIndexResult> {
        let mut server_config = self.server_config.write().await;
        let aliases = server_config.get().core.get_index_aliases_for_index(&delete_index_request.index_name);
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
                let removed_config_entry = config_entry.remove();
                server_config.commit().await?;
                index_holder_entry.remove();
                cleanup_index(removed_config_entry).await?;
                Ok(DeleteIndexResult::new(&delete_index_request.index_name))
            }
            _ => Err(ValidationError::MissingIndex(delete_index_request.index_name.to_owned()).into()),
        }
    }

    /// Returns all existent consumers for all indices
    pub async fn get_consumers(&self) -> HashMap<String, crate::configs::consumer::Config> {
        self.server_config.read().await.get().consumers.clone()
    }

    /// Create consumer and insert it into the consumer registry. Add it to the `IndexHolder` afterwards.
    #[instrument(skip_all, fields(consumer_name = ?create_consumer_request.consumer_name))]
    pub async fn create_consumer(&self, create_consumer_request: proto::CreateConsumerRequest) -> SummaServerResult<String> {
        let index_holder = self.index_registry.get_index_holder(&create_consumer_request.index_name).await?;
        let consumer_config = crate::configs::consumer::Config::new(
            &create_consumer_request.index_name,
            &create_consumer_request.bootstrap_servers,
            &create_consumer_request.group_id,
            &create_consumer_request.topics,
        )?;
        let prepared_consumption = PreparedConsumption::from_config(&create_consumer_request.consumer_name, &consumer_config)?;
        prepared_consumption.on_create().await?;
        self.consumer_manager.write().await.start_consuming(&index_holder, prepared_consumption).await?;
        let mut server_config = self.server_config.write().await;
        server_config
            .get_mut()
            .consumers
            .insert(create_consumer_request.consumer_name.to_string(), consumer_config);
        server_config.commit().await?;
        Ok(index_holder.index_name().to_owned())
    }

    /// Delete consumer from the consumer registry and from `IndexHolder` afterwards.
    #[instrument(skip_all, fields(consumer_name = ?delete_consumer_request.consumer_name))]
    pub async fn delete_consumer(&self, delete_consumer_request: proto::DeleteConsumerRequest) -> SummaServerResult<proto::DeleteConsumerResponse> {
        let mut server_config = self.server_config.write().await;
        if server_config.get_mut().consumers.remove(&delete_consumer_request.consumer_name).is_none() {
            return Err(ValidationError::MissingConsumer(delete_consumer_request.consumer_name.to_string()).into());
        }
        server_config.commit().await?;
        drop(server_config);
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
    pub async fn stop(&self, force: bool) -> SummaServerResult<()> {
        self.should_terminate.store(true, Ordering::Relaxed);
        if !force {
            self.consumer_manager.write().await.stop().await?;
            for index_holder in self.index_registry.index_holders_cloned().await.values() {
                self.commit(index_holder).await?;
            }
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

    /// Commits everything and restarts consuming threads
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
        let mut index_writer = index_holder.index_writer_holder()?.clone().write_owned().await;
        let index_holder = index_holder.clone();

        let span = tracing::Span::current();
        tokio::task::spawn_blocking(move || {
            span.in_scope(|| {
                index_writer.commit_and_prepare(true)?;
                index_holder.index_reader().reload()?;
                Ok::<_, crate::errors::Error>(())
            })
        })
        .await??;

        let prepared_consumption = match stopped_consumption {
            Some(stopped_consumption) => Some(stopped_consumption.commit_offsets().await?),
            None => None,
        };
        Ok(prepared_consumption)
    }

    /// Rollbacks everything without restarting consuming threads
    #[instrument(skip(self, index_holder), fields(index_name = ?index_holder.index_name()))]
    pub async fn rollback(&self, index_holder: &Handler<IndexHolder>) -> SummaServerResult<Option<PreparedConsumption>> {
        let mut index_writer = index_holder.index_writer_holder()?.clone().try_write_owned()?;
        let stopped_consumption = self.consumer_manager.write().await.stop_consuming_for(index_holder).await?;
        let span = tracing::Span::current();
        tokio::task::spawn_blocking(move || span.in_scope(|| index_writer.rollback())).await??;
        Ok(stopped_consumption.map(|c| c.ignore()))
    }

    /// Commits immediately or returns all without restarting consuming threads
    #[instrument(skip(self, index_holder), fields(index_name = ?index_holder.index_name()))]
    pub async fn try_commit(&self, index_holder: &Handler<IndexHolder>) -> SummaServerResult<Option<PreparedConsumption>> {
        let mut index_writer = index_holder.index_writer_holder()?.clone().try_write_owned()?;
        let stopped_consumption = self.consumer_manager.write().await.stop_consuming_for(index_holder).await?;
        let span = tracing::Span::current();
        tokio::task::spawn_blocking(move || span.in_scope(|| index_writer.commit_and_prepare(true))).await??;
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
                                let index_holders = index_service.index_registry.index_holders_cloned().await;
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
    pub async fn open_index_from_config(&self, index_engine_config: proto::IndexEngineConfig) -> SummaServerResult<tantivy::Index> {
        let index = match index_engine_config.config {
            Some(proto::index_engine_config::Config::File(config)) => tantivy::Index::open_in_dir(config.path)?,
            Some(proto::index_engine_config::Config::Memory(config)) => IndexBuilder::new().schema(serde_yaml::from_str(&config.schema)?).create_in_ram()?,
            Some(proto::index_engine_config::Config::Remote(config)) => {
                IndexHolder::open_remote_index::<HyperExternalRequest, DefaultExternalRequestGenerator<HyperExternalRequest>>(config, true).await?
            }
            _ => unimplemented!(),
        };
        Ok(index)
    }

    /// Copies index with changing engine
    #[instrument(skip(self, copy_index_request), fields(source_index_name = copy_index_request.source_index_name, target_index_name = copy_index_request.target_index_name))]
    pub async fn copy_index(&self, copy_index_request: proto::CopyIndexRequest) -> SummaServerResult<Handler<IndexHolder>> {
        unimplemented!()
    }

    /// Search documents
    pub async fn search(&self, search_request: proto::SearchRequest) -> SummaServerResult<Vec<proto::CollectorOutput>> {
        let futures = self.index_registry.search_futures(search_request.index_queries)?;
        let collector_outputs = join_all(futures.into_iter().map(tokio::spawn))
            .await
            .into_iter()
            .collect::<Result<SummaResult<Vec<_>>, _>>()??;
        Ok(self.index_registry.finalize_extraction(collector_outputs).await?)
    }

    /// Merge several segments into a single one
    #[instrument(skip(self, merge_segments_request), fields(index_name = merge_segments_request.index_name))]
    pub async fn merge_segments(&self, merge_segments_request: proto::MergeSegmentsRequest) -> SummaServerResult<Option<SegmentId>> {
        let index_holder = self.get_index_holder(&merge_segments_request.index_name).await?;
        let index_writer_holder = index_holder
            .index_writer_holder()
            .map_err(crate::errors::Error::from)?
            .clone()
            .read_owned()
            .await;
        let span = tracing::Span::current();
        let segment_ids: Vec<_> = merge_segments_request
            .segment_ids
            .iter()
            .map(|segment_id| SegmentId::from_uuid_string(segment_id))
            .collect::<Result<Vec<_>, _>>()
            .expect("wrong uuid");
        let segment_meta = tokio::task::spawn_blocking(move || span.in_scope(|| index_writer_holder.merge(&segment_ids, None))).await??;
        Ok(segment_meta.map(|segment_meta| segment_meta.id()))
    }

    /// Merges segments and removes tombstones
    #[instrument(skip(self, vacuum_index_request), fields(index_name = vacuum_index_request.index_name))]
    pub async fn vacuum_index(&self, vacuum_index_request: proto::VacuumIndexRequest) -> SummaServerResult<u64> {
        let index_holder = self.get_index_holder(&vacuum_index_request.index_name).await?;
        let index_writer_holder = index_holder
            .index_writer_holder()
            .map_err(crate::errors::Error::from)?
            .clone()
            .read_owned()
            .await;
        let before_size: u64 = index_holder.space_usage()?.segments().iter().map(|s| s.total().get_bytes()).sum();
        let span = tracing::Span::current();
        tokio::task::spawn_blocking(move || {
            span.in_scope(|| {
                let result = index_writer_holder.vacuum(None, vacuum_index_request.excluded_segments);
                if let Err(error) = result {
                    error!(error = ?error)
                }
            });
        })
        .await?;
        let after_size: u64 = index_holder.space_usage()?.segments().iter().map(|s| s.total().get_bytes()).sum();
        Ok(before_size - after_size)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::default::Default;
    use std::path::Path;
    use std::sync::atomic::AtomicI64;

    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};
    use summa_core::components::test_utils::create_test_schema;
    use summa_core::components::test_utils::{generate_documents, generate_documents_with_doc_id_gen_and_rng, generate_unique_document};
    use summa_core::configs::DirectProxy;
    use summa_proto::proto_traits::collector::shortcuts::{top_docs_collector, top_docs_collector_with_eval_expr};
    use summa_proto::proto_traits::query::shortcuts::match_query;
    use tantivy::schema::Schema;

    use super::*;
    use crate::configs::server::tests::create_test_server_config_holder;
    use crate::logging;

    pub async fn create_test_index_service(data_path: &Path) -> Index {
        Index::new(&create_test_server_config_holder(&data_path)).unwrap()
    }

    pub async fn create_test_index_holder(
        index_service: &Index,
        schema: &Schema,
        index_engine: proto::create_index_request::IndexEngine,
    ) -> SummaServerResult<Handler<IndexHolder>> {
        index_service
            .create_index(proto::CreateIndexRequest {
                index_name: "test_index".to_owned(),
                schema: serde_yaml::to_string(schema).unwrap(),
                compression: 0,
                blocksize: None,
                sort_by_field: None,
                index_attributes: Some(proto::IndexAttributes { ..Default::default() }),
                index_engine: Some(index_engine),
                merge_policy: None,
                query_parser_config: None,
            })
            .await
    }

    #[tokio::test]
    async fn test_same_name_index() -> SummaServerResult<()> {
        logging::tests::initialize_default_once();
        let index_service = Index::new(&(Arc::new(DirectProxy::default()) as Arc<dyn ConfigProxy<_>>))?;
        assert!(index_service
            .create_index(proto::CreateIndexRequest {
                index_name: "test_index".to_owned(),
                schema: serde_yaml::to_string(&create_test_schema()).unwrap(),
                compression: 0,
                blocksize: None,
                sort_by_field: None,
                index_attributes: None,
                index_engine: Some(proto::create_index_request::IndexEngine::Memory(proto::CreateMemoryEngineRequest {})),
                merge_policy: None,
                query_parser_config: None,
            },)
            .await
            .is_ok());
        assert!(index_service
            .create_index(proto::CreateIndexRequest {
                index_name: "test_index".to_owned(),
                schema: serde_yaml::to_string(&create_test_schema()).unwrap(),
                compression: 0,
                blocksize: None,
                sort_by_field: None,
                index_attributes: None,
                index_engine: Some(proto::create_index_request::IndexEngine::Memory(proto::CreateMemoryEngineRequest {})),
                merge_policy: None,
                query_parser_config: None,
            },)
            .await
            .is_err());
        Ok(())
    }

    #[tokio::test]
    async fn test_index_create_delete() -> SummaServerResult<()> {
        logging::tests::initialize_default_once();
        let root_path = tempdir::TempDir::new("summa_test").unwrap();
        let data_path = root_path.path().join("data");

        let server_config_holder = create_test_server_config_holder(&data_path);

        let index_service = Index::new(&server_config_holder)?;
        index_service
            .create_index(proto::CreateIndexRequest {
                index_name: "test_index".to_owned(),
                schema: serde_yaml::to_string(&create_test_schema()).unwrap(),
                compression: 0,
                blocksize: None,
                sort_by_field: None,
                index_attributes: None,
                index_engine: Some(proto::create_index_request::IndexEngine::File(proto::CreateFileEngineRequest {})),
                merge_policy: None,
                query_parser_config: None,
            })
            .await?;
        assert!(index_service
            .delete_index(proto::DeleteIndexRequest {
                index_name: "test_index".to_string()
            })
            .await
            .is_ok());
        assert!(index_service
            .create_index(proto::CreateIndexRequest {
                index_name: "test_index".to_owned(),
                schema: serde_yaml::to_string(&create_test_schema()).unwrap(),
                compression: 0,
                blocksize: None,
                sort_by_field: None,
                index_attributes: None,
                index_engine: Some(proto::create_index_request::IndexEngine::Memory(proto::CreateMemoryEngineRequest {})),
                merge_policy: None,
                query_parser_config: None,
            },)
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
        let index_holder = create_test_index_holder(
            &index_service,
            &schema,
            proto::create_index_request::IndexEngine::Memory(proto::CreateMemoryEngineRequest {}),
        )
        .await?;

        for d in generate_documents(index_holder.schema(), 1000) {
            index_holder.index_document(d).await?;
        }
        index_holder
            .index_document(generate_unique_document(index_holder.schema(), "testtitle"))
            .await?;
        index_service.commit(&index_holder).await?;

        let search_response = index_holder
            .search(
                "index",
                match_query("testtitle", vec!["title".to_string(), "body".to_string()]),
                vec![top_docs_collector(10)],
                None,
            )
            .await?;
        assert_eq!(search_response.len(), 1);

        drop(index_holder);
        index_service
            .delete_index(proto::DeleteIndexRequest {
                index_name: "test_index".to_string(),
            })
            .await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_custom_ranking() -> SummaServerResult<()> {
        logging::tests::initialize_default_once();
        let schema = create_test_schema();

        let root_path = tempdir::TempDir::new("summa_test").unwrap();
        let data_path = root_path.path().join("data");

        let index_service = create_test_index_service(&data_path).await;
        let index_holder = create_test_index_holder(
            &index_service,
            &schema,
            proto::create_index_request::IndexEngine::File(proto::CreateFileEngineRequest {}),
        )
        .await?;

        let mut rng = SmallRng::seed_from_u64(42);
        for _ in 0..4 {
            for d in generate_documents_with_doc_id_gen_and_rng(AtomicI64::new(1), &mut rng, &schema, 300) {
                index_holder.index_document(d).await?;
            }
            index_service.commit(&index_holder).await?;
        }
        let index_holder_clone = index_holder.clone();
        let index_writer = index_holder.index_writer_holder().unwrap().clone().write_owned().await;
        tokio::task::spawn_blocking(move || {
            index_writer.vacuum(None, vec![])?;
            index_holder_clone.index_reader().reload()?;
            Ok::<_, crate::errors::Error>(())
        })
        .await??;
        assert!(index_holder
            .search(
                "test_index",
                match_query("title1", vec!["title".to_string(), "body".to_string()]),
                vec![top_docs_collector_with_eval_expr(1, "issued_at")],
                None
            )
            .await
            .is_ok());
        Ok(())
    }
}
