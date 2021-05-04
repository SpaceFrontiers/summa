use super::index_writer_holder::IndexWriterHolder;
use crate::configs::{ConsumerConfig, IndexConfigProxy};
use crate::consumers::kafka::status::{KafkaConsumingError, KafkaConsumingStatus};
use crate::consumers::kafka::Consumer;
use crate::errors::{Error, SummaResult, ValidationError};
use crate::proto;
use crate::search_engine::SummaDocument;
use rdkafka::message::BorrowedMessage;
use rdkafka::Message;
use std::collections::hash_map::Entry;
use std::sync::Arc;
use tantivy::directory::GarbageCollectionResult;
use tantivy::schema::Schema;
use tantivy::{Opstamp, SegmentId, SegmentMeta};
use tracing::instrument;

fn process_message(
    schema: &Schema,
    index_writer_holder: &IndexWriterHolder,
    message: Result<BorrowedMessage<'_>, rdkafka::error::KafkaError>,
) -> Result<KafkaConsumingStatus, KafkaConsumingError> {
    let message = message.map_err(|e| KafkaConsumingError::KafkaError(e))?;
    let payload = message.payload().ok_or(KafkaConsumingError::EmptyPayloadError)?;
    let proto_message: proto::IndexOperation = prost::Message::decode(payload).map_err(|e| KafkaConsumingError::ProtoDecodeError(e))?;
    let index_operation = proto_message.operation.ok_or(KafkaConsumingError::EmptyOperationError)?;
    match index_operation {
        proto::index_operation::Operation::IndexDocument(index_document_operation) => {
            let parsed_document = SummaDocument::BoundJsonBytes((schema, &index_document_operation.document))
                .try_into()
                .map_err(|e| KafkaConsumingError::ParseDocumentError(e))?;
            index_writer_holder.index_document(parsed_document).map_err(|e| KafkaConsumingError::IndexError(e))?;
            Ok(KafkaConsumingStatus::Consumed)
        }
    }
}

/// Index updating through consumers and via direct invocation
pub(crate) struct IndexUpdater {
    index_config_proxy: IndexConfigProxy,
    consumers: Vec<Consumer>,
    index_writer_holder: Arc<IndexWriterHolder>,
    schema: Schema,
}

impl IndexUpdater {
    /// Creates new `IndexUpdater`
    pub(super) fn new(schema: &Schema, index_config_proxy: IndexConfigProxy, index_writer_holder: IndexWriterHolder) -> SummaResult<IndexUpdater> {
        let index_writer_holder = Arc::new(index_writer_holder);
        let consumers = index_config_proxy
            .read()
            .consumer_configs
            .iter()
            .map(|(consumer_name, consumer_config)| Consumer::new(consumer_name, consumer_config).unwrap())
            .into_iter()
            .collect();
        let mut inner_index_updater = IndexUpdater {
            index_config_proxy,
            consumers,
            index_writer_holder,
            schema: schema.clone(),
        };
        inner_index_updater.start_consumers()?;
        Ok(inner_index_updater)
    }

    /// Stops all consumers
    async fn stop_consumers(&mut self) -> SummaResult<&mut IndexWriterHolder> {
        for consumer in &self.consumers {
            consumer.stop().await?;
        }
        Arc::get_mut(&mut self.index_writer_holder).ok_or(Error::ArcIndexWriterHolderLeakedError)
    }

    /// Starts all consumers
    fn start_consumers(&mut self) -> SummaResult<()> {
        for consumer in &self.consumers {
            let index_writer_holder = self.index_writer_holder.clone();
            let schema = self.schema.clone();
            consumer.start(move |message| process_message(&schema, &index_writer_holder, message))?;
        }
        Ok(())
    }

    /// Add consumer and starts it
    pub(super) fn attach_consumer(&mut self, consumer: Consumer) -> SummaResult<()> {
        let index_writer_holder = self.index_writer_holder.clone();
        let schema = self.schema.clone();
        consumer.start(move |message| process_message(&schema, &index_writer_holder, message))?;
        self.consumers.push(consumer);
        Ok(())
    }

    /// Create new consumer and attaches it to the `IndexUpdater`
    #[instrument(skip(self, consumer_config))]
    pub(crate) async fn create_consumer(&mut self, consumer_name: &str, consumer_config: &ConsumerConfig) -> SummaResult<()> {
        match self.index_config_proxy.write().autosave().get_mut().consumer_configs.entry(consumer_name.to_owned()) {
            Entry::Occupied(o) => Err(ValidationError::ExistingConsumerError(o.key().to_owned())),
            Entry::Vacant(v) => {
                v.insert(consumer_config.clone());
                Ok(())
            }
        }?;
        let consumer = Consumer::new(consumer_name, consumer_config)?;
        consumer.on_create().await?;
        self.attach_consumer(consumer)?;
        Ok(())
    }

    async fn inner_delete_consumer(&mut self, consumer_name: &str) -> SummaResult<()> {
        let position = self.consumers.iter().position(|consumer| consumer.consumer_name() == consumer_name).unwrap();
        self.consumers.swap_remove(position).on_delete().await?;
        Ok(())
    }

    /// Deletes consumer
    ///
    /// Stops and commit all consumers. Required due to parallel nature of consuming because it prevents the possibility to commit every
    /// consumer separately.
    #[instrument(skip(self))]
    pub(crate) async fn delete_consumer(&mut self, consumer_name: &str) -> SummaResult<()> {
        self.stop_consumers().await?.commit().await?;
        self.commit_offsets()?;
        self.index_config_proxy
            .write()
            .autosave()
            .get_mut()
            .consumer_configs
            .remove(consumer_name)
            .ok_or(ValidationError::MissingConsumerError(consumer_name.to_owned()))?;
        self.inner_delete_consumer(consumer_name).await?;
        self.start_consumers()?;
        Ok(())
    }

    /// Deletes all consumers in a faster way than separate deletion of every consumer by their names
    #[instrument(skip(self))]
    pub(crate) async fn delete_all_consumers(&mut self) -> SummaResult<Vec<String>> {
        self.stop_consumers().await?.commit().await?;
        self.commit_offsets()?;
        let mut deleted_consumers_names: Vec<_> = Vec::new();
        for consumer in self.consumers.drain(..) {
            consumer.on_delete().await?;
            deleted_consumers_names.push(consumer.consumer_name().to_owned());
        }
        Ok(deleted_consumers_names)
    }

    /// Check if any consumers for the `IndexUpdater` exists
    pub(crate) fn has_consumers(&self) -> bool {
        self.consumers.len() > 0
    }

    /// Index generic `SummaDocument`
    ///
    /// `IndexUpdater` bounds unbounded `SummaDocument` inside
    pub(crate) fn index_document(&self, document: SummaDocument<'_>) -> SummaResult<Opstamp> {
        let document = document.bound_with(&self.schema).try_into()?;
        self.index_writer_holder.index_document(document)
    }

    /// Merges multiple segments, see `IndexWriterHolder::merge` for details
    #[instrument(skip(self))]
    pub(crate) async fn merge(&mut self, segment_ids: &[SegmentId]) -> SummaResult<SegmentMeta> {
        let index_writer_holder = self.stop_consumers().await?;
        let segment_meta = index_writer_holder.merge(segment_ids).await?;
        index_writer_holder.commit().await?;
        self.commit_offsets()?;
        self.start_consumers()?;
        Ok(segment_meta)
    }

    /// Commits Kafka offsets
    #[instrument(skip(self))]
    fn commit_offsets(&self) -> SummaResult<()> {
        for consumer in &self.consumers {
            consumer.commit_offsets()?;
        }
        Ok(())
    }

    /// Commit Tantivy index and Kafka offsets
    #[instrument(skip(self))]
    pub(crate) async fn commit(&mut self) -> SummaResult<Opstamp> {
        let opstamp = self.stop_consumers().await?.commit().await?;
        self.commit_offsets()?;
        self.start_consumers()?;
        Ok(opstamp)
    }

    /// Vacuums garbage files and compacts segments (to be done)
    #[instrument(skip(self))]
    pub(crate) async fn vacuum(&self) -> SummaResult<GarbageCollectionResult> {
        self.index_writer_holder.garbage_collect_files().await
    }

    /// Stops consumers
    #[instrument(skip(self))]
    pub(super) async fn stop(mut self) -> SummaResult<()> {
        self.stop_consumers().await?;
        Ok(())
    }

    /// Stops consumers, commits Tantivy and Kafka offsets
    #[instrument(skip(self))]
    pub(super) async fn stop_consumers_and_commit(mut self) -> SummaResult<Opstamp> {
        self.stop_consumers().await?;
        let opstamp = self.stop_consumers().await?.commit().await?;
        self.commit_offsets()?;
        Ok(opstamp)
    }
}
