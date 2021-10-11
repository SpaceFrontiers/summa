use super::index_writer_holder::IndexWriterHolder;
use crate::configs::{ConsumerConfig, IndexConfig, IndexConfigProxy};
use crate::consumers::kafka::status::{KafkaConsumingError, KafkaConsumingStatus};
use crate::consumers::kafka::Consumer;
use crate::errors::{Error, SummaServerResult, ValidationError};
use crate::ipfs_client::Key;
use crate::search_engine::frozen_log_merge_policy::FrozenLogMergePolicy;
use crate::search_engine::segment_attributes::SummaSegmentAttributes;
use rdkafka::error::{KafkaError, RDKafkaErrorCode};
use rdkafka::message::BorrowedMessage;
use rdkafka::Message;
use std::collections::hash_map::Entry;
use std::collections::HashSet;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use summa_core::directories::write_hotcache;
use summa_core::SummaDocument;
use summa_proto::proto;
use tantivy::directory::MmapDirectory;
use tantivy::schema::Schema;
use tantivy::{Directory, Index, Opstamp, SegmentComponent, SegmentId, SegmentMeta};
use tracing::{info, instrument, warn};

fn process_message(
    schema: &Schema,
    index_writer_holder: &IndexWriterHolder,
    message: Result<BorrowedMessage<'_>, rdkafka::error::KafkaError>,
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

fn create_index_writer_holder(index: &Index, index_config: &IndexConfig) -> SummaServerResult<IndexWriterHolder> {
    let index_writer = index.writer_with_num_threads(
        index_config.writer_threads.try_into().unwrap(),
        index_config.writer_heap_size_bytes.try_into().unwrap(),
    )?;
    index_writer.set_merge_policy(Box::new(FrozenLogMergePolicy::default()));
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

#[derive(Clone, Debug, Default, Eq, Hash, PartialEq)]
pub struct IndexFilePath {
    file_path: PathBuf,
    is_immutable: bool,
}

impl IndexFilePath {
    pub fn new(file_path: PathBuf, is_immutable: bool) -> IndexFilePath {
        IndexFilePath { file_path, is_immutable }
    }
    pub fn path(&self) -> &Path {
        &self.file_path
    }
    pub fn is_immutable(&self) -> bool {
        self.is_immutable
    }
}

/// Index updating through consumers and via direct invocation
pub(crate) struct IndexUpdater {
    index_config_proxy: IndexConfigProxy,
    consumers: Vec<Consumer>,
    index: Index,
    index_name: String,
    index_writer_holder: Arc<IndexWriterHolder>,
}

impl IndexUpdater {
    /// Creates new `IndexUpdater`
    pub(super) async fn new(index: Index, index_name: &str, index_config_proxy: IndexConfigProxy) -> SummaServerResult<IndexUpdater> {
        let index_config = index_config_proxy.read().await.get().clone();
        let index_writer_holder = Arc::new(create_index_writer_holder(&index, &index_config)?);
        let consumers = index_config
            .consumer_configs
            .iter()
            .map(|(consumer_name, consumer_config)| Consumer::new(consumer_name, consumer_config).unwrap())
            .into_iter()
            .collect();
        let mut inner_index_updater = IndexUpdater {
            index_config_proxy,
            consumers,
            index,
            index_name: index_name.to_owned(),
            index_writer_holder,
        };
        inner_index_updater.start_consumers().await?;
        Ok(inner_index_updater)
    }

    /// Tantivy `Index`
    pub(crate) fn index(&self) -> &Index {
        &self.index
    }

    /// Mutable Tantivy `Index`
    pub(crate) fn index_mut(&mut self) -> &mut Index {
        &mut self.index
    }

    /// Stops all consumers
    async fn stop_consumers(&mut self) -> SummaServerResult<&mut IndexWriterHolder> {
        for consumer in &self.consumers {
            consumer.stop().await?;
        }
        Arc::get_mut(&mut self.index_writer_holder).ok_or(Error::ArcIndexWriterHolderLeaked)
    }

    /// Starts all consumers
    async fn start_consumers(&mut self) -> SummaServerResult<()> {
        for consumer in &self.consumers {
            let index_writer_holder = self.index_writer_holder.clone();
            let schema = self.index.schema();
            consumer.start(move |message| process_message(&schema, &index_writer_holder, message)).await?;
        }
        Ok(())
    }

    /// Add consumer and starts it
    pub(super) async fn attach_consumer(&mut self, consumer: Consumer) -> SummaServerResult<()> {
        let index_writer_holder = self.index_writer_holder.clone();
        let schema = self.index.schema();
        consumer.start(move |message| process_message(&schema, &index_writer_holder, message)).await?;
        self.consumers.push(consumer);
        Ok(())
    }

    /// Create new consumer and attaches it to the `IndexUpdater`
    #[instrument(skip(self, consumer_config))]
    pub(crate) async fn create_consumer(&mut self, consumer_name: &str, consumer_config: &ConsumerConfig) -> SummaServerResult<()> {
        {
            match self
                .index_config_proxy
                .write()
                .await
                .autosave()
                .get_mut()
                .consumer_configs
                .entry(consumer_name.to_owned())
            {
                Entry::Occupied(o) => Err(ValidationError::ExistingConsumer(o.key().to_owned())),
                Entry::Vacant(v) => {
                    v.insert(consumer_config.clone());
                    Ok(())
                }
            }?;
        }
        let consumer = Consumer::new(consumer_name, consumer_config)?;
        consumer.on_create().await?;
        self.attach_consumer(consumer).await?;
        Ok(())
    }

    async fn inner_delete_consumer(&mut self, consumer_name: &str) -> SummaServerResult<()> {
        let position = self.consumers.iter().position(|consumer| consumer.consumer_name() == consumer_name).unwrap();
        self.consumers.swap_remove(position).on_delete().await?;
        Ok(())
    }

    /// Deletes consumer
    ///
    /// Stops and commit all consumers. Required due to parallel nature of consuming because it prevents the possibility to commit every
    /// consumer separately.
    #[instrument(skip(self))]
    pub(crate) async fn delete_consumer(&mut self, consumer_name: &str) -> SummaServerResult<()> {
        self.stop_consumers().await?.commit(None).await?;
        if let Err(error) = self.commit_offsets().await {
            info!(action = "failed_commit", consumer_name = consumer_name, error = ?error);
        }
        self.index_config_proxy
            .write()
            .await
            .autosave()
            .get_mut()
            .consumer_configs
            .remove(consumer_name)
            .ok_or_else(|| ValidationError::MissingConsumer(consumer_name.to_owned()))?;
        self.inner_delete_consumer(consumer_name).await?;
        self.start_consumers().await?;
        Ok(())
    }

    /// Deletes all consumers in a faster way than separate deletion of every consumer by their names
    #[instrument(skip(self))]
    pub(crate) async fn delete_all_consumers(&mut self) -> SummaServerResult<Vec<String>> {
        self.stop_consumers().await?.commit(None).await?;
        self.commit_offsets().await?;
        let mut deleted_consumers_names: Vec<_> = Vec::new();
        for consumer in self.consumers.drain(..) {
            consumer.on_delete().await?;
            deleted_consumers_names.push(consumer.consumer_name().to_owned());
        }
        Ok(deleted_consumers_names)
    }

    /// Return consumer names
    pub(crate) fn consumer_names(&self) -> Vec<String> {
        self.consumers.iter().map(|x| x.consumer_name().to_owned()).collect()
    }

    /// Delete `SummaDocument` by `primary_key`
    pub(crate) fn delete_document(&self, primary_key_value: i64) -> SummaServerResult<Opstamp> {
        self.index_writer_holder.delete_document_by_primary_key(primary_key_value)
    }

    /// Index generic `SummaDocument`
    ///
    /// `IndexUpdater` bounds unbounded `SummaDocument` inside
    pub(crate) fn index_document(&self, document: SummaDocument<'_>) -> SummaServerResult<Opstamp> {
        let document = document.bound_with(&self.index.schema()).try_into()?;
        self.index_writer_holder.index_document(document)
    }

    /// Index multiple documents at a time
    pub(crate) fn index_bulk(&self, documents: &Vec<Vec<u8>>) -> (u64, u64) {
        let (mut success_docs, mut failed_docs) = (0u64, 0u64);
        for document in documents {
            match self.index_document(SummaDocument::UnboundJsonBytes(document)) {
                Ok(_) => success_docs += 1,
                Err(error) => {
                    warn!(action = "error", error = ?error);
                    failed_docs += 1
                }
            }
        }
        (success_docs, failed_docs)
    }

    /// Merges multiple segments, see `IndexWriterHolder::merge` for details
    #[instrument(skip(self))]
    pub(crate) async fn merge(&mut self, segment_ids: &[SegmentId]) -> SummaServerResult<Option<SegmentMeta>> {
        let index_writer_holder = self.stop_consumers().await?;
        let segment_meta = index_writer_holder.merge(segment_ids, None).await?;
        self.start_consumers().await?;
        Ok(segment_meta)
    }

    /// Commits Kafka offsets
    #[instrument(skip(self))]
    async fn commit_offsets(&self) -> SummaServerResult<()> {
        for consumer in &self.consumers {
            let commit_result = consumer.commit_offsets().await;
            if let Err(Error::Kafka(KafkaError::ConsumerCommit(RDKafkaErrorCode::AssignmentLost))) = commit_result {
                consumer.stop().await?;
                let schema = self.index.schema();
                let index_writer_holder = self.index_writer_holder.clone();
                consumer.start(move |message| process_message(&schema, &index_writer_holder, message)).await?;
            }
        }
        info!(action = "committed_offsets");
        Ok(())
    }

    /// Commit Tantivy index and Kafka offsets
    #[instrument(skip(self), fields(index_name = ?self.index_name))]
    pub(crate) async fn commit(&mut self) -> SummaServerResult<Opstamp> {
        let opstamp = self.stop_consumers().await?.commit(None).await?;
        self.commit_offsets().await?;
        self.start_consumers().await?;
        Ok(opstamp)
    }

    /// Vacuums garbage files and compacts segments
    #[instrument(skip(self))]
    pub(crate) async fn vacuum(&mut self) -> SummaServerResult<()> {
        let index_writer_holder = self.stop_consumers().await?;
        index_writer_holder.commit(None).await?;
        index_writer_holder.vacuum(None).await?;
        index_writer_holder.commit(None).await?;

        self.commit_offsets().await?;
        self.start_consumers().await?;
        Ok(())
    }

    /// Stops consumers
    #[instrument(skip(self))]
    pub(super) async fn stop(mut self) -> SummaServerResult<()> {
        self.stop_consumers().await?;
        Ok(())
    }

    /// Stops consumers, commits Tantivy and Kafka offsets
    #[instrument(skip(self))]
    pub(super) async fn stop_consumers_and_commit(mut self) -> SummaServerResult<Opstamp> {
        self.stop_consumers().await?;
        let opstamp = self.stop_consumers().await?.commit(None).await?;
        self.commit_offsets().await?;
        Ok(opstamp)
    }

    /// Set attributes
    pub async fn prepare_index_publishing<P, Fut>(
        &mut self,
        index_path: P,
        payload: Option<String>,
        publisher: impl FnOnce(Vec<IndexFilePath>) -> Fut,
    ) -> SummaServerResult<Key>
    where
        P: AsRef<Path>,
        Fut: Future<Output = SummaServerResult<Key>>,
    {
        let index = self.index.clone();
        let index_config = self.index_config_proxy.read().await.get().clone();

        let index_writer_holder = self.stop_consumers().await?;

        let segment_attributes = SummaSegmentAttributes { is_frozen: true };

        index_writer_holder.commit(None).await?;
        index_writer_holder.vacuum(Some(segment_attributes)).await?;
        index_writer_holder.commit(payload).await?;

        wait_merging_threads(index_writer_holder, &index, &index_config);
        self.commit_offsets().await?;

        let mut hotcache_bytes = vec![];

        let read_directory = MmapDirectory::open(index_path).unwrap();
        write_hotcache(read_directory, 16384, &mut hotcache_bytes).unwrap();
        index
            .directory()
            .atomic_write(&PathBuf::from("hotcache.bin".to_string()), &hotcache_bytes)
            .unwrap();

        let mut segment_files = self.get_immutable_segment_files().await?;
        segment_files.push(IndexFilePath::new(PathBuf::from(".managed.json"), false));
        segment_files.push(IndexFilePath::new(PathBuf::from("meta.json"), false));
        segment_files.push(IndexFilePath::new(PathBuf::from("hotcache.bin"), false));

        let result = publisher(segment_files).await?;

        self.start_consumers().await?;
        Ok(result)
    }

    /// Get segments
    async fn get_immutable_segment_files(&self) -> SummaServerResult<Vec<IndexFilePath>> {
        Ok(self
            .index
            .searchable_segments()?
            .into_iter()
            .flat_map(|segment| {
                SegmentComponent::iterator()
                    .filter_map(|segment_component| {
                        if *segment_component == SegmentComponent::TempStore
                            || (*segment_component == SegmentComponent::Delete && !segment.meta().has_deletes())
                        {
                            None
                        } else {
                            Some(IndexFilePath::new(segment.meta().relative_path(*segment_component), true))
                        }
                    })
                    .collect::<HashSet<_>>()
            })
            .collect::<Vec<_>>())
    }
}

#[cfg(test)]
pub mod tests {

    #[test]
    fn test_en_tokenizer() {}
}
