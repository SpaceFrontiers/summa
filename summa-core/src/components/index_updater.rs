use std::collections::hash_map::Entry;
use std::fmt::{Debug, Formatter};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use futures::future::join_all;
use rdkafka::error::{KafkaError, RDKafkaErrorCode};
use rdkafka::message::BorrowedMessage;
use rdkafka::Message;
use summa_proto::proto;
use tantivy::directory::MmapDirectory;
use tantivy::schema::Schema;
use tantivy::{Directory, Index, Opstamp, SegmentId, SegmentMeta};
use tokio::sync::{OwnedRwLockReadGuard, RwLock, RwLockReadGuard, RwLockWriteGuard, TryLockError};
use tokio::time;
use tracing::{info, info_span, instrument, warn, Instrument};

use super::index_writer_holder::IndexWriterHolder;
use super::SummaDocument;
use crate::components::frozen_log_merge_policy::FrozenLogMergePolicy;
use crate::components::segment_attributes::SummaSegmentAttributes;
use crate::configs::{ConfigProxy, ConsumerConfig, IndexConfig};
use crate::consumers::kafka::status::{KafkaConsumingError, KafkaConsumingStatus};
use crate::consumers::kafka::Consumer;
use crate::directories::write_hotcache;
use crate::errors::{Error, SummaResult, ValidationError};
use crate::utils::thread_handler::ThreadHandler;

pub fn process_message(
    index_writer_holder: &OwnedRwLockReadGuard<IndexWriterHolder>,
    schema: &Schema,
    message: Result<BorrowedMessage<'_>, KafkaError>,
) -> Result<KafkaConsumingStatus, KafkaConsumingError> {
    let message = message.map_err(KafkaConsumingError::Kafka)?;
    let payload = message.payload().ok_or(KafkaConsumingError::EmptyPayload)?;
    let proto_message: proto::IndexOperation = prost::Message::decode(payload).map_err(KafkaConsumingError::ProtoDecode)?;
    let index_operation = proto_message.operation.ok_or(KafkaConsumingError::EmptyOperation)?;
    match index_operation {
        proto::index_operation::Operation::IndexDocument(index_document_operation) => {
            let parsed_document = SummaDocument::BoundJsonBytes((schema, &index_document_operation.document))
                .try_into()
                .map_err(KafkaConsumingError::ParseDocument)?;
            index_writer_holder.index_document(parsed_document).map_err(KafkaConsumingError::Index)?;
            Ok(KafkaConsumingStatus::Consumed)
        }
    }
}

fn create_index_writer_holder(index: &Index, index_config: &IndexConfig) -> SummaResult<IndexWriterHolder> {
    let index_writer = index.writer_with_num_threads(index_config.writer_threads as usize, index_config.writer_heap_size_bytes as usize)?;
    index_writer.set_merge_policy(Box::<FrozenLogMergePolicy>::default());
    IndexWriterHolder::new(
        index_writer,
        match index_config.primary_key {
            Some(ref primary_key) => index.schema().get_field(primary_key),
            None => None,
        },
    )
}

fn wait_merging_threads(index_writer_holder: &mut IndexWriterHolder, index: &Index, index_config: &IndexConfig) {
    take_mut::take(index_writer_holder, |index_writer_holder| {
        index_writer_holder.wait_merging_threads().unwrap();
        create_index_writer_holder(index, index_config).unwrap()
    });
}

#[derive(Clone)]
pub struct SegmentComponent {
    pub path: PathBuf,
    pub segment_component: tantivy::SegmentComponent,
}

impl Debug for SegmentComponent {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.path.fmt(f)
    }
}

#[derive(Clone, Debug)]
pub enum ComponentFile {
    SegmentComponent(SegmentComponent),
    Other(PathBuf),
}

impl ComponentFile {
    pub fn path(&self) -> &Path {
        match self {
            ComponentFile::SegmentComponent(segment_component) => &segment_component.path,
            ComponentFile::Other(path) => path,
        }
    }
}

pub struct IndexUpdater {
    autocommit_thread: Option<ThreadHandler<SummaResult<()>>>,
    inner_index_updater: Arc<RwLock<InnerIndexUpdater>>,
}

impl IndexUpdater {
    /// Creates new `IndexUpdater`
    pub(super) async fn new(index: Index, index_name: &str, index_config_proxy: Arc<dyn ConfigProxy<IndexConfig>>) -> SummaResult<IndexUpdater> {
        let index_config = index_config_proxy.read().await.get().clone();
        let index_writer_holder = Arc::new(RwLock::new(create_index_writer_holder(&index, &index_config)?));
        let consumers = index_config
            .consumer_configs
            .iter()
            .map(|(consumer_name, consumer_config)| Consumer::new(consumer_name, consumer_config))
            .into_iter()
            .collect::<SummaResult<_>>()?;

        let inner_index_updater = InnerIndexUpdater {
            index_config_proxy,
            consumers,
            index,
            index_name: index_name.to_owned(),
            index_writer_holder,
        };
        let mut index_updater = IndexUpdater {
            autocommit_thread: None,
            inner_index_updater: Arc::new(RwLock::new(inner_index_updater)),
        };

        index_updater.start_updates().await;
        Ok(index_updater)
    }

    pub async fn read(&self) -> RwLockReadGuard<'_, InnerIndexUpdater> {
        self.inner_index_updater.read().await
    }

    pub async fn write(&self) -> RwLockWriteGuard<'_, InnerIndexUpdater> {
        self.inner_index_updater.write().await
    }

    pub fn try_write(&self) -> Result<RwLockWriteGuard<'_, InnerIndexUpdater>, TryLockError> {
        self.inner_index_updater.try_write()
    }

    pub fn try_read(&self) -> Result<RwLockReadGuard<'_, InnerIndexUpdater>, TryLockError> {
        self.inner_index_updater.try_read()
    }

    async fn setup_autocommit_thread(&mut self) {
        let inner_index_updater = self.inner_index_updater.write().await;
        let index_name = inner_index_updater.index_name.clone();
        let interval_ms = match inner_index_updater.index_config_proxy.read().await.get().autocommit_interval_ms {
            Some(interval_ms) => interval_ms,
            None => return,
        };

        let (shutdown_trigger, mut shutdown_tripwire) = async_broadcast::broadcast(1);
        let mut tick_task = time::interval(Duration::from_millis(interval_ms));
        let inner_index_updater = self.inner_index_updater.clone();

        self.autocommit_thread = Some(ThreadHandler::new(
            tokio::spawn(
                async move {
                    info!(action = "spawning_autocommit_thread", interval_ms = interval_ms);
                    // The first tick ticks immediately so we skip it
                    tick_task.tick().await;
                    loop {
                        tokio::select! {
                            _ = tick_task.tick() => {
                                info!(action = "autocommit_thread_tick");
                                if let Ok(inner_index_updater) = inner_index_updater.try_read() {
                                    if let Err(error) = inner_index_updater.commit(None).await {
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
                .instrument(info_span!(parent: None, "autocommit_thread", index_name = ?index_name)),
            ),
            shutdown_trigger,
        ));
    }

    /// Starts all consumers
    #[instrument(skip(self))]
    pub async fn start_updates(&mut self) {
        self.inner_index_updater.read().await.start_consumers().await;
        self.setup_autocommit_thread().await;
    }

    /// Stops all consumers
    #[instrument(skip(self))]
    pub async fn stop_updates(&mut self) -> SummaResult<()> {
        if let Some(autocommit_thread) = self.autocommit_thread.take() {
            autocommit_thread.stop().await??;
        }
        self.inner_index_updater.write().await.stop_consumers().await?;
        Ok(())
    }

    /// Stops consumers, commits Tantivy and Kafka offsets
    #[instrument(skip(self))]
    pub async fn stop_updates_and_commit(self) -> SummaResult<Opstamp> {
        let inner_index_updater = self.read().await;
        inner_index_updater.stop_consumers().await?;
        let opstamp = inner_index_updater.commit_index(None).await?;
        inner_index_updater.commit_offsets().await?;
        Ok(opstamp)
    }
}

/// Index updating through consumers and via direct invocation
pub struct InnerIndexUpdater {
    index_config_proxy: Arc<dyn ConfigProxy<IndexConfig>>,
    consumers: Vec<Consumer>,
    index: Index,
    index_name: String,
    index_writer_holder: Arc<RwLock<IndexWriterHolder>>,
}

impl InnerIndexUpdater {
    /// Tantivy `Index`
    pub fn index(&self) -> &Index {
        &self.index
    }

    /// Mutable Tantivy `Index`
    pub fn index_mut(&mut self) -> &mut Index {
        &mut self.index
    }

    pub async fn start_consumers(&self) {
        for consumer in &self.consumers {
            let schema = self.index.schema();
            let index_writer_holder = self.index_writer_holder.clone().read_owned().await;
            consumer.start(move |message| process_message(&index_writer_holder, &schema, message)).await;
        }
    }

    pub async fn stop_consumers(&self) -> SummaResult<()> {
        join_all(self.consumers.iter().map(|consumer| consumer.stop())).await.into_iter().collect()
    }

    /// Add consumer and starts it
    pub async fn attach_consumer(&mut self, consumer: Consumer) {
        let schema = self.index.schema();
        let index_writer_holder = self.index_writer_holder.clone().read_owned().await;
        consumer.start(move |message| process_message(&index_writer_holder, &schema, message)).await;
        self.consumers.push(consumer);
    }

    /// Create new consumer and attaches it to the `IndexUpdater`
    #[instrument(skip(self, consumer_config))]
    pub async fn create_consumer(&mut self, consumer_name: &str, consumer_config: &ConsumerConfig) -> SummaResult<()> {
        {
            let mut index_config_proxy = self.index_config_proxy.write().await;
            match index_config_proxy.get_mut().consumer_configs.entry(consumer_name.to_owned()) {
                Entry::Occupied(o) => Err(ValidationError::ExistingConsumer(o.key().to_owned())),
                Entry::Vacant(v) => {
                    v.insert(consumer_config.clone());
                    Ok(())
                }
            }?;
            index_config_proxy.commit()?;
        }
        let consumer = Consumer::new(consumer_name, consumer_config)?;
        consumer.on_create().await?;
        self.attach_consumer(consumer).await;
        Ok(())
    }

    async fn inner_delete_consumer(&mut self, consumer_name: &str) -> SummaResult<()> {
        let position = self
            .consumers
            .iter()
            .position(|consumer| consumer.consumer_name() == consumer_name)
            .ok_or_else(|| ValidationError::MissingConsumer(consumer_name.to_string()))?;
        self.consumers.swap_remove(position).on_delete().await?;
        Ok(())
    }

    /// Deletes consumer
    ///
    /// Stops and commit all consumers. Required due to parallel nature of consuming because it prevents the possibility to commit every
    /// consumer separately.
    #[instrument(skip(self))]
    pub async fn delete_consumer(&mut self, consumer_name: &str) -> SummaResult<()> {
        self.stop_consumers().await?;
        self.commit_index(None).await?;
        if let Err(error) = self.commit_offsets().await {
            info!(action = "failed_commit", consumer_name = consumer_name, error = ?error);
        }
        {
            let mut index_config_proxy = self.index_config_proxy.write().await;
            index_config_proxy
                .get_mut()
                .consumer_configs
                .remove(consumer_name)
                .ok_or_else(|| ValidationError::MissingConsumer(consumer_name.to_owned()))?;
            index_config_proxy.commit()?;
        }
        self.inner_delete_consumer(consumer_name).await?;
        self.start_consumers().await;
        Ok(())
    }

    /// Deletes all consumers in a faster way than separate deletion of every consumer by their names
    #[instrument(skip(self))]
    pub async fn delete_all_consumers(&mut self) -> SummaResult<Vec<String>> {
        self.stop_consumers().await?;
        self.commit_index(None).await?;
        self.commit_offsets().await?;
        let mut deleted_consumers_names: Vec<_> = Vec::new();
        for consumer in self.consumers.drain(..) {
            consumer.on_delete().await?;
            deleted_consumers_names.push(consumer.consumer_name().to_owned());
        }
        Ok(deleted_consumers_names)
    }

    /// Return consumer names
    pub fn consumer_names(&self) -> Vec<String> {
        self.consumers.iter().map(|x| x.consumer_name().to_owned()).collect()
    }

    /// Delete `SummaDocument` by `primary_key`
    pub async fn delete_document(&self, primary_key_value: i64) -> SummaResult<Opstamp> {
        self.index_writer_holder.read().await.delete_document_by_primary_key(primary_key_value)
    }

    /// Index generic `SummaDocument`
    ///
    /// `IndexUpdater` bounds unbounded `SummaDocument` inside
    pub async fn index_document(&self, document: SummaDocument<'_>) -> SummaResult<Opstamp> {
        let document = document.bound_with(&self.index.schema()).try_into()?;
        self.index_writer_holder.read().await.index_document(document)
    }

    /// Index multiple documents at a time
    pub async fn index_bulk(&self, documents: &Vec<Vec<u8>>) -> (u64, u64) {
        let (mut success_docs, mut failed_docs) = (0u64, 0u64);
        for document in documents {
            match self.index_document(SummaDocument::UnboundJsonBytes(document)).await {
                Ok(_) => success_docs += 1,
                Err(error) => {
                    warn!(action = "error", error = ?error);
                    failed_docs += 1
                }
            }
        }
        (success_docs, failed_docs)
    }

    /// Commits Kafka offsets
    #[instrument(skip(self))]
    pub(super) async fn commit_offsets(&self) -> SummaResult<()> {
        for consumer in &self.consumers {
            let commit_result = consumer.commit().await;
            if let Err(Error::Kafka(KafkaError::ConsumerCommit(RDKafkaErrorCode::AssignmentLost))) = commit_result {
                let schema = self.index.schema();
                let index_writer_holder = self.index_writer_holder.clone().read_owned().await;
                consumer.start(move |message| process_message(&index_writer_holder, &schema, message)).await;
            }
        }
        info!(action = "committed_offsets");
        Ok(())
    }

    /// Commits index
    #[instrument(skip(self))]
    pub(super) async fn commit_index(&self, payload: Option<String>) -> SummaResult<Opstamp> {
        self.index_writer_holder.write().await.commit(payload).await
    }

    /// Commits all
    #[instrument(skip(self))]
    pub async fn commit(&self, payload: Option<String>) -> SummaResult<Opstamp> {
        self.stop_consumers().await?;
        let opstamp = self.commit_index(payload).await?;
        self.commit_offsets().await?;
        self.start_consumers().await;
        Ok(opstamp)
    }

    /// Vacuum index
    #[instrument(skip(self))]
    pub async fn vacuum_index(&self, payload: Option<String>, segment_attributes: Option<SummaSegmentAttributes>) -> SummaResult<Opstamp> {
        self.stop_consumers().await?;
        self.commit_index(payload.clone()).await?;
        self.index_writer_holder.write().await.vacuum(segment_attributes).await?;
        let opstamp = self.commit_index(payload).await?;
        self.commit_offsets().await?;
        self.start_consumers().await;
        Ok(opstamp)
    }

    /// Merge index
    #[instrument(skip(self))]
    pub async fn merge_index(&self, segment_ids: &[SegmentId], segment_attributes: Option<SummaSegmentAttributes>) -> SummaResult<Option<SegmentMeta>> {
        self.stop_consumers().await?;
        let segment_meta = self.index_writer_holder.write().await.merge(segment_ids, segment_attributes).await?;
        self.start_consumers().await;
        Ok(segment_meta)
    }

    /// Locking index files for executing operation on them
    pub async fn lock_files<P, Fut>(&mut self, index_path: P, payload: Option<String>, f: impl FnOnce(Vec<ComponentFile>) -> Fut) -> SummaResult<()>
    where
        P: AsRef<Path>,
        Fut: Future<Output = SummaResult<()>>,
    {
        let index = self.index.clone();
        let index_config = self.index_config_proxy.read().await.get().clone();

        let mut index_writer_holder = self.index_writer_holder.write().await;
        self.stop_consumers().await?;

        let segment_attributes = SummaSegmentAttributes { is_frozen: true };

        index_writer_holder.commit(None).await?;
        index_writer_holder.vacuum(Some(segment_attributes)).await?;
        index_writer_holder.commit(payload).await?;

        wait_merging_threads(&mut index_writer_holder, &index, &index_config);
        self.commit_offsets().await?;

        let mut hotcache_bytes = vec![];

        let read_directory = MmapDirectory::open(&index_path)?;
        write_hotcache(read_directory, 16384, &mut hotcache_bytes)?;
        index.directory().atomic_write(&PathBuf::from("hotcache.bin".to_string()), &hotcache_bytes)?;

        let segment_files = [
            ComponentFile::Other(PathBuf::from(".managed.json")),
            ComponentFile::Other(PathBuf::from("meta.json")),
            ComponentFile::Other(PathBuf::from("hotcache.bin")),
        ]
        .into_iter()
        .chain(self.get_index_files(index_path.as_ref().to_path_buf())?)
        .collect();
        f(segment_files).await?;

        self.start_consumers().await;
        Ok(())
    }

    /// Get segments
    fn get_index_files(&self, index_path: PathBuf) -> SummaResult<impl Iterator<Item = ComponentFile>> {
        Ok(self.index.searchable_segments()?.into_iter().flat_map(move |segment| {
            tantivy::SegmentComponent::iterator()
                .filter_map(|segment_component| {
                    let relative_path = segment.meta().relative_path(*segment_component);
                    index_path.join(relative_path).exists().then(|| {
                        ComponentFile::SegmentComponent(SegmentComponent {
                            path: segment.meta().relative_path(*segment_component),
                            segment_component: segment_component.clone(),
                        })
                    })
                })
                .collect::<Vec<_>>()
        }))
    }
}
