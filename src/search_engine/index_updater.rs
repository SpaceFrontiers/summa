use super::index_writer_holder::IndexWriterHolder;
use crate::configs::{IndexConfig, KafkaConsumerConfig};
use crate::consumers::kafka::status::{KafkaConsumingError, KafkaConsumingStatus};
use crate::consumers::kafka::KafkaConsumer;
use crate::errors::{Error, SummaResult};
use crate::proto;
use crate::search_engine::SummaDocument;
use parking_lot::RwLock;
use rdkafka::message::BorrowedMessage;
use rdkafka::Message;

use std::sync::Arc;
use tantivy::directory::GarbageCollectionResult;
use tantivy::schema::Schema;
use tantivy::{Opstamp, SegmentId, SegmentMeta};
use tracing::{instrument, warn};

/// Index updating through Kafka consumers and via direct invocation
pub(crate) struct IndexUpdater {
    inner: RwLock<InnerIndexUpdater>,
}

pub(crate) struct InnerIndexUpdater {
    consumers: Vec<KafkaConsumer>,
    index_writer_holder: Arc<IndexWriterHolder>,
    schema: Schema,
}

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
            index_writer_holder
                .index_document(parsed_document, index_document_operation.reindex)
                .map_err(|e| KafkaConsumingError::IndexError(e))?;
            Ok(KafkaConsumingStatus::Consumed)
        }
    }
}

impl InnerIndexUpdater {
    pub fn new(schema: &Schema, index_config: &IndexConfig, index_writer_holder: IndexWriterHolder) -> SummaResult<InnerIndexUpdater> {
        let index_writer_holder = Arc::new(index_writer_holder);
        let consumers = index_config
            .consumer_configs
            .iter()
            .map(|(consumer_name, consumer_config)| KafkaConsumer::new(consumer_name, consumer_config).unwrap())
            .into_iter()
            .collect();
        let mut inner_index_updater = InnerIndexUpdater {
            consumers,
            index_writer_holder,
            schema: schema.clone(),
        };
        inner_index_updater.start_consumers()?;
        Ok(inner_index_updater)
    }
    async fn stop_consumers(&mut self) -> SummaResult<()> {
        for consumer in &self.consumers {
            consumer.stop().await?;
        }
        Ok(())
    }

    fn start_consumers(&mut self) -> SummaResult<()> {
        for consumer in &self.consumers {
            let index_writer_holder = self.index_writer_holder.clone();
            let schema = self.schema.clone();
            consumer.start(move |message| process_message(&schema, &index_writer_holder, message))?;
        }
        Ok(())
    }

    fn commit_offsets(&self) -> SummaResult<()> {
        for consumer in &self.consumers {
            consumer.commit_offsets()?;
        }
        Ok(())
    }

    pub(crate) fn attach_consumer(&mut self, consumer: KafkaConsumer) -> SummaResult<()> {
        let index_writer_holder = self.index_writer_holder.clone();
        let schema = self.schema.clone();
        consumer.start(move |message| process_message(&schema, &index_writer_holder, message))?;
        self.consumers.push(consumer);
        Ok(())
    }

    pub(crate) async fn create_consumer(&mut self, consumer_name: &str, consumer_config: KafkaConsumerConfig) -> SummaResult<()> {
        let consumer = KafkaConsumer::new(consumer_name, &consumer_config)?;
        consumer.on_create().await?;
        self.attach_consumer(consumer)?;
        Ok(())
    }

    async fn inner_delete_consumer(&mut self, consumer_name: &str) -> SummaResult<()> {
        let position = self.consumers.iter().position(|consumer| consumer.consumer_name() == consumer_name).unwrap();
        self.consumers.swap_remove(position).on_delete().await?;
        Ok(())
    }

    pub(crate) async fn delete_consumer(&mut self, consumer_name: &str) -> SummaResult<()> {
        self.commit(false).await?;
        self.inner_delete_consumer(consumer_name).await?;
        self.start_consumers()?;
        Ok(())
    }

    pub(crate) async fn delete_all_consumers(&mut self) -> SummaResult<()> {
        self.commit(false).await?;
        for consumer in self.consumers.drain(..) {
            consumer.on_delete().await?;
        }
        Ok(())
    }

    pub(crate) fn has_consumers(&self) -> bool {
        self.consumers.len() > 0
    }

    pub(crate) fn index_document(&self, document: SummaDocument<'_>, reindex: bool) -> SummaResult<Opstamp> {
        let document = document.bound_with(&self.schema).try_into()?;
        self.index_writer_holder.index_document(document, reindex)
    }

    #[instrument(skip_all)]
    pub(crate) async fn merge(&mut self, segment_ids: &[SegmentId]) -> SummaResult<SegmentMeta> {
        self.stop_consumers().await?;
        let index_writer_holder = Arc::<IndexWriterHolder>::get_mut(&mut self.index_writer_holder).ok_or(Error::ArcIndexWriterHolderLeakedError)?;
        let segment_meta = index_writer_holder.merge(segment_ids).await?;
        index_writer_holder.commit().await?;
        self.commit_offsets()?;
        self.start_consumers()?;
        Ok(segment_meta)
    }

    #[instrument(skip_all)]
    pub(crate) async fn commit(&mut self, restart_consumers: bool) -> SummaResult<Opstamp> {
        self.stop_consumers().await?;
        let index_writer_holder = Arc::<IndexWriterHolder>::get_mut(&mut self.index_writer_holder).ok_or(Error::ArcIndexWriterHolderLeakedError)?;
        let opstamp = index_writer_holder.commit().await?;
        self.commit_offsets()?;
        if restart_consumers {
            self.start_consumers()?;
        }
        Ok(opstamp)
    }

    #[instrument(skip_all)]
    pub(crate) async fn vacuum(&self) -> SummaResult<GarbageCollectionResult> {
        self.index_writer_holder.garbage_collect_files().await
    }
}

impl IndexUpdater {
    pub(crate) fn new(schema: &Schema, index_config: &IndexConfig, index_writer_holder: IndexWriterHolder) -> SummaResult<IndexUpdater> {
        Ok(IndexUpdater {
            inner: RwLock::new(InnerIndexUpdater::new(schema, index_config, index_writer_holder)?),
        })
    }

    pub(crate) async fn create_consumer(&self, consumer_name: &str, consumer_config: KafkaConsumerConfig) -> SummaResult<()> {
        self.inner.write().create_consumer(consumer_name, consumer_config).await
    }

    pub(crate) async fn delete_consumer(&self, consumer_name: &str) -> SummaResult<()> {
        self.inner.write().delete_consumer(consumer_name).await
    }

    pub async fn delete_all_consumers(&self) -> SummaResult<()> {
        self.inner.write().delete_all_consumers().await
    }

    pub(crate) fn has_consumers(&self) -> bool {
        self.inner.read().has_consumers()
    }

    pub(crate) fn index_document(&self, document: SummaDocument<'_>, reindex: bool) -> SummaResult<Opstamp> {
        self.inner.read().index_document(document, reindex)
    }

    pub async fn try_commit(&self, restart_consumers: bool) -> Option<SummaResult<Opstamp>> {
        match self.inner.try_write() {
            Some(mut inner_index_updater) => Some(inner_index_updater.commit(restart_consumers).await),
            None => None,
        }
    }

    pub async fn try_commit_and_log(&self, restart_consumers: bool) -> Option<Opstamp> {
        match self.try_commit(restart_consumers).await {
            Some(result) => match result {
                Ok(opstamp) => Some(opstamp),
                Err(error) => {
                    warn!(action = "failed_commit", error = ?error);
                    None
                }
            },
            None => {
                warn!(error = "index_updater_busy");
                None
            }
        }
    }

    pub async fn vacuum(&self) -> SummaResult<GarbageCollectionResult> {
        self.inner.read().vacuum().await
    }

    pub async fn merge(&self, segment_ids: &[SegmentId]) -> SummaResult<SegmentMeta> {
        self.inner.write().merge(segment_ids).await
    }

    pub async fn commit(&self) -> SummaResult<Opstamp> {
        self.inner.write().commit(true).await
    }

    pub async fn last_commit(self) -> SummaResult<Opstamp> {
        self.inner.into_inner().commit(false).await
    }
}
