use super::index_writer_holder::IndexWriterHolder;
use crate::consumers::kafka::status::{KafkaConsumingError, KafkaConsumingStatus};
use crate::consumers::kafka::KafkaConsumer;
use crate::errors::{Error, SummaResult};
use crate::proto;
use parking_lot::RwLock;
use rdkafka::message::BorrowedMessage;
use rdkafka::Message;
use std::sync::Arc;
use tantivy::Opstamp;
use tracing::warn;

/// Index updating through Kafka consumers and via direct invocation
#[derive(Debug)]
pub(crate) struct IndexUpdater {
    inner: RwLock<InnerIndexUpdater>,
}

#[derive(Debug)]
pub(crate) struct InnerIndexUpdater {
    consumers: Vec<KafkaConsumer>,
    index_writer_holder: Arc<IndexWriterHolder>,
}

fn process_message(index_writer_holder: &IndexWriterHolder, message: Result<BorrowedMessage<'_>, rdkafka::error::KafkaError>) -> Result<KafkaConsumingStatus, KafkaConsumingError> {
    let message = message.map_err(|e| KafkaConsumingError::KafkaError(e))?;
    let payload = message.payload().ok_or(KafkaConsumingError::EmptyPayloadError)?;
    let proto_message: proto::IndexOperation = prost::Message::decode(payload).map_err(|e| KafkaConsumingError::ProtoDecodeError(e))?;
    let index_operation = proto_message.operation.ok_or(KafkaConsumingError::EmptyOperationError)?;
    match index_operation {
        proto::index_operation::Operation::IndexDocument(index_document_operation) => {
            index_writer_holder
                .index_document(&index_document_operation.document, index_document_operation.reindex)
                .map_err(|e| KafkaConsumingError::IndexError(e))?;
            Ok(KafkaConsumingStatus::Consumed)
        }
    }
}

impl InnerIndexUpdater {
    async fn stop_consumers(&mut self) -> SummaResult<()> {
        for consumer in &self.consumers {
            consumer.stop().await?;
        }
        Ok(())
    }

    async fn start_consumers(&mut self) -> SummaResult<()> {
        for consumer in &self.consumers {
            let index_writer_holder = self.index_writer_holder.clone();
            consumer.start(move |message| process_message(&index_writer_holder, message))?;
        }
        Ok(())
    }

    fn commit_offsets(&self) -> SummaResult<()> {
        for consumer in &self.consumers {
            consumer.commit()?;
        }
        Ok(())
    }

    pub(crate) fn add_consumer(&mut self, consumer: KafkaConsumer) -> SummaResult<()> {
        let index_writer_holder = self.index_writer_holder.clone();
        consumer.start(move |message| process_message(&index_writer_holder, message))?;
        self.consumers.push(consumer);
        Ok(())
    }

    pub(crate) async fn delete_consumer(&mut self, consumer_name: &str) -> SummaResult<()> {
        self.commit(false).await?;

        let position = self.consumers.iter().position(|consumer| consumer.consumer_name() == consumer_name).unwrap();
        self.consumers.swap_remove(position).clear().await?;

        self.start_consumers().await?;
        Ok(())
    }

    pub(crate) async fn index_document(&self, document: &[u8], reindex: bool) -> SummaResult<Opstamp> {
        self.index_writer_holder.index_document(document, reindex)
    }

    pub(crate) async fn commit(&mut self, restart_consumers: bool) -> SummaResult<()> {
        self.stop_consumers().await?;
        let index_writer_holder = Arc::<IndexWriterHolder>::get_mut(&mut self.index_writer_holder).ok_or(Error::ArcIndexWriterHolderLeakedError)?;
        index_writer_holder.commit()?;
        self.commit_offsets()?;
        if restart_consumers {
            self.start_consumers().await?;
        }
        Ok(())
    }
}

impl IndexUpdater {
    pub(crate) fn new(index_writer_holder: IndexWriterHolder) -> IndexUpdater {
        let index_writer_holder = Arc::new(index_writer_holder);
        let index_updater = IndexUpdater {
            inner: RwLock::new(InnerIndexUpdater {
                consumers: Vec::new(),
                index_writer_holder,
            }),
        };
        index_updater
    }

    pub(crate) fn add_consumer(&self, consumer: KafkaConsumer) -> SummaResult<()> {
        self.inner.write().add_consumer(consumer)
    }

    pub(crate) fn add_consumers(&self, mut consumers: Vec<KafkaConsumer>) -> SummaResult<()> {
        let mut inner_index_updater = self.inner.write();
        for consumer in consumers.drain(..) {
            inner_index_updater.add_consumer(consumer)?;
        }
        Ok(())
    }

    pub(crate) async fn delete_consumer(&self, consumer_name: &str) -> SummaResult<()> {
        self.inner.write().delete_consumer(consumer_name).await
    }

    pub(crate) async fn index_document(&self, document: &[u8], reindex: bool) -> SummaResult<Opstamp> {
        self.inner.read().index_document(document, reindex).await
    }

    pub async fn try_commit(&self, restart_consumers: bool) -> Option<SummaResult<()>> {
        match self.inner.try_write() {
            Some(mut inner_index_updater) => Some(inner_index_updater.commit(restart_consumers).await),
            None => None,
        }
    }

    pub async fn try_commit_and_log(&self, restart_consumers: bool) {
        match self.try_commit(restart_consumers).await {
            Some(result) => match result {
                Ok(_) => (),
                Err(error) => warn!(action = "commit", error = ?error),
            },
            None => warn!(error = "index_updater_busy"),
        };
    }

    pub async fn last_commit(self) -> SummaResult<()> {
        self.inner.into_inner().commit(false).await
    }
}
