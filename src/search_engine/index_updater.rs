use super::index_writer_holder::IndexWriterHolder;
use crate::consumers::kafka::status::{KafkaConsumingError, KafkaConsumingStatus};
use crate::consumers::kafka::KafkaConsumer;
use crate::errors::{Error, SummaResult};
use crate::proto;
use rdkafka::message::BorrowedMessage;
use rdkafka::Message;
use std::sync::Arc;

/// Index updating through Kafka consumers and via direct invocation
#[derive(Debug)]
pub(crate) struct IndexUpdater {
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

impl IndexUpdater {
    pub(crate) fn new(index_writer_holder: IndexWriterHolder) -> IndexUpdater {
        let index_writer_holder = Arc::new(index_writer_holder);
        let index_updater = IndexUpdater {
            consumers: Vec::new(),
            index_writer_holder,
        };
        index_updater
    }

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

    pub(crate) fn add_consumers(&mut self, mut consumers: Vec<KafkaConsumer>) -> SummaResult<()> {
        for consumer in consumers.drain(..) {
            self.add_consumer(consumer)?;
        }
        Ok(())
    }

    pub(crate) async fn delete_consumer(&mut self, consumer_name: &str) -> SummaResult<()> {
        self.commit(false).await?;

        let position = self.consumers.iter().position(|consumer| consumer.consumer_name() == consumer_name).unwrap();
        self.consumers.swap_remove(position).clear().await?;

        self.start_consumers().await?;
        Ok(())
    }

    pub(crate) async fn index_document(&self, document: &[u8], reindex: bool) -> SummaResult<()> {
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
