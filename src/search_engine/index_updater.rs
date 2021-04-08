use super::index_writer_holder::IndexWriterHolder;
use crate::configs::{IndexConfigHolder, KafkaConsumerConfig};
use crate::consumers::kafka::status::{KafkaConsumingError, KafkaConsumingStatus};
use crate::consumers::kafka::KafkaConsumer;
use crate::errors::{Error, SummaResult, ValidationError};
use crate::proto;
use parking_lot::RwLock;
use rdkafka::message::BorrowedMessage;
use rdkafka::Message;
use std::sync::Arc;
use tantivy::Opstamp;
use tracing::{instrument, warn};

/// Index updating through Kafka consumers and via direct invocation
#[derive(Debug)]
pub(crate) struct IndexUpdater {
    inner: RwLock<InnerIndexUpdater>,
}

#[derive(Debug)]
pub(crate) struct InnerIndexUpdater {
    consumers: Vec<KafkaConsumer>,
    index_config: IndexConfigHolder,
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
    pub fn new(index_config: IndexConfigHolder, index_writer_holder: IndexWriterHolder) -> SummaResult<InnerIndexUpdater> {
        let index_writer_holder = Arc::new(index_writer_holder);
        let consumers = index_config
            .consumer_configs
            .iter()
            .map(|(consumer_name, consumer_config)| KafkaConsumer::new(consumer_name, consumer_config).unwrap())
            .into_iter()
            .collect();
        let mut inner_index_updater = InnerIndexUpdater {
            consumers,
            index_config,
            index_writer_holder,
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
            consumer.start(move |message| process_message(&index_writer_holder, message))?;
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
        consumer.start(move |message| process_message(&index_writer_holder, message))?;
        self.consumers.push(consumer);
        Ok(())
    }

    pub(crate) async fn create_consumer(&mut self, consumer_name: &str, consumer_config: KafkaConsumerConfig) -> SummaResult<()> {
        if self.index_config.consumer_configs.contains_key(consumer_name) {
            Err(ValidationError::ExistingConsumerError(consumer_name.to_string()))?
        }
        let consumer = KafkaConsumer::new(consumer_name, &consumer_config)?;
        consumer.on_create().await?;
        self.attach_consumer(consumer)?;
        self.index_config.autosave().consumer_configs.insert(consumer_name.to_string(), consumer_config);
        Ok(())
    }

    async fn inner_delete_consumer(&mut self, consumer_name: &str) -> SummaResult<()> {
        self.index_config.autosave().consumer_configs.remove(consumer_name);
        let position = self.consumers.iter().position(|consumer| consumer.consumer_name() == consumer_name).unwrap();
        self.consumers.swap_remove(position).on_delete().await?;
        Ok(())
    }

    pub(crate) async fn delete_consumer(&mut self, consumer_name: &str) -> SummaResult<()> {
        if !self.index_config.consumer_configs.contains_key(consumer_name) {
            Err(ValidationError::MissingConsumerError(consumer_name.to_string()))?
        }
        self.commit(false).await?;
        self.inner_delete_consumer(consumer_name).await?;
        self.start_consumers()?;
        Ok(())
    }

    pub(crate) async fn delete_all_consumers(&mut self) -> SummaResult<()> {
        self.commit(false).await?;
        self.index_config.autosave().consumer_configs.clear();
        for consumer in self.consumers.drain(..) {
            consumer.on_delete().await?;
        }
        Ok(())
    }

    pub(crate) async fn index_document(&self, document: &[u8], reindex: bool) -> SummaResult<Opstamp> {
        self.index_writer_holder.index_document(document, reindex)
    }

    #[instrument(skip(self))]
    pub(crate) async fn commit(&mut self, restart_consumers: bool) -> SummaResult<Opstamp> {
        self.stop_consumers().await?;
        let index_writer_holder = Arc::<IndexWriterHolder>::get_mut(&mut self.index_writer_holder).ok_or(Error::ArcIndexWriterHolderLeakedError)?;
        let opstamp = index_writer_holder.commit()?;
        self.commit_offsets()?;
        if restart_consumers {
            self.start_consumers()?;
        }
        Ok(opstamp)
    }
}

impl IndexUpdater {
    pub(crate) fn new(index_config: IndexConfigHolder, index_writer_holder: IndexWriterHolder) -> SummaResult<IndexUpdater> {
        Ok(IndexUpdater {
            inner: RwLock::new(InnerIndexUpdater::new(index_config, index_writer_holder)?),
        })
    }

    pub(crate) fn get_consumer_config(&self, consumer_name: &str) -> Option<KafkaConsumerConfig> {
        self.inner.read().index_config.consumer_configs.get(consumer_name).map(|x| x.clone())
    }

    pub(crate) fn get_consumers_names(&self) -> Vec<String> {
        self.inner.read().index_config.consumer_configs.keys().map(|x| x.clone()).collect()
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

    pub(crate) async fn index_document(&self, document: &[u8], reindex: bool) -> SummaResult<Opstamp> {
        self.inner.read().index_document(document, reindex).await
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

    pub async fn commit(&self) -> SummaResult<Opstamp> {
        self.inner.write().commit(true).await
    }

    pub async fn last_commit(self) -> SummaResult<Opstamp> {
        self.inner.into_inner().commit(false).await
    }
}
