use std::collections::HashMap;
use std::future::Future;

use futures_util::future::join_all;
use rdkafka::error::{KafkaError, RDKafkaErrorCode};
use rdkafka::message::BorrowedMessage;
use rdkafka::Message;
use summa_core::components::{IndexHolder, IndexWriterHolder, SummaDocument};
use summa_core::utils::sync::Handler;
use summa_proto::proto;
use tantivy::schema::Schema;
use tokio::sync::OwnedRwLockReadGuard;
use tracing::{info, instrument, warn};

use crate::components::consumers::kafka::status::{KafkaConsumingError, KafkaConsumingStatus};
use crate::components::consumers::kafka::ConsumerThread;
use crate::errors::{Error, SummaServerResult, ValidationError};

pub fn process_message(
    index_writer_holder: &OwnedRwLockReadGuard<IndexWriterHolder>,
    conflict_strategy: proto::ConflictStrategy,
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
            index_writer_holder
                .index_document(parsed_document, conflict_strategy)
                .map_err(|e| KafkaConsumingError::Index(e.into()))?;
            Ok(KafkaConsumingStatus::Consumed)
        }
    }
}

#[derive(Debug)]
pub struct StoppedConsumption {
    consumer_thread: ConsumerThread,
}

impl StoppedConsumption {
    pub async fn commit_offsets(self) -> SummaServerResult<PreparedConsumption> {
        match self.consumer_thread.commit().await {
            Ok(_) => {
                info!(action = "committed_offsets");
                Ok(PreparedConsumption {
                    committed_consumer_thread: self.consumer_thread,
                })
            }
            Err(Error::Kafka(KafkaError::ConsumerCommit(RDKafkaErrorCode::AssignmentLost))) => {
                warn!(error = "assignment_lost");
                Ok(PreparedConsumption {
                    committed_consumer_thread: self.consumer_thread,
                })
            }
            Err(Error::Kafka(KafkaError::ConsumerCommit(RDKafkaErrorCode::NoOffset))) => {
                warn!(error = "no_offset");
                Ok(PreparedConsumption {
                    committed_consumer_thread: self.consumer_thread,
                })
            }
            Err(e) => Err(e),
        }
    }

    pub fn ignore(self) -> PreparedConsumption {
        PreparedConsumption {
            committed_consumer_thread: self.consumer_thread,
        }
    }
}

#[derive(Debug)]
pub struct PreparedConsumption {
    committed_consumer_thread: ConsumerThread,
}

impl PreparedConsumption {
    pub fn from_config(consumer_name: &str, consumer_config: &crate::configs::consumer::Config) -> SummaServerResult<PreparedConsumption> {
        Ok(PreparedConsumption {
            committed_consumer_thread: ConsumerThread::new(consumer_name, consumer_config)?,
        })
    }

    pub fn on_create(&self) -> impl Future<Output = SummaServerResult<()>> + '_ {
        self.committed_consumer_thread.on_create()
    }

    pub fn on_delete(&self) -> impl Future<Output = SummaServerResult<()>> + '_ {
        self.committed_consumer_thread.on_delete()
    }

    pub fn consumer_name(&self) -> &str {
        self.committed_consumer_thread.consumer_name()
    }
}

#[derive(Debug, Default)]
pub struct ConsumerManager {
    consumptions: HashMap<Handler<IndexHolder>, ConsumerThread>,
}

impl ConsumerManager {
    pub fn new() -> ConsumerManager {
        ConsumerManager { consumptions: HashMap::new() }
    }

    /// Starts prepared consumption to the index
    #[instrument(skip(self, index_holder, prepared_consumption), fields(index_name = index_holder.index_name(), consumer_name = prepared_consumption.consumer_name()))]
    pub async fn start_consuming(&mut self, index_holder: &Handler<IndexHolder>, prepared_consumption: PreparedConsumption) -> SummaServerResult<()> {
        if self.consumptions.contains_key(index_holder) {
            return Err(ValidationError::ExistingConsumer(index_holder.index_name().to_string()).into());
        }
        let index_writer_holder = index_holder.index_writer_holder()?.clone().read_owned().await;
        let schema = index_holder.schema().clone();
        let conflict_strategy = index_holder.conflict_strategy();
        prepared_consumption
            .committed_consumer_thread
            .start(move |message| process_message(&index_writer_holder, conflict_strategy, &schema, message))
            .await;
        self.consumptions.insert(index_holder.clone(), prepared_consumption.committed_consumer_thread);
        Ok(())
    }

    /// Stops all consuming threads
    #[instrument(skip(self))]
    pub async fn stop(&mut self) -> SummaServerResult<()> {
        info!(action = "stopping");
        join_all(self.consumptions.drain().map(|(index_holder, consumer_thread)| async move {
            consumer_thread.stop().await?;
            let stopped_consumption = StoppedConsumption { consumer_thread };
            let mut index_writer_holder = index_holder.index_writer_holder()?.clone().write_owned().await;
            tokio::task::spawn_blocking(move || index_writer_holder.commit()).await??;
            stopped_consumption.commit_offsets().await?;
            Ok(())
        }))
        .await
        .into_iter()
        .collect::<SummaServerResult<_>>()
    }

    /// Stops particular `IndexHolder`
    #[instrument(skip(self))]
    pub async fn stop_consuming_for(&mut self, index_holder: &Handler<IndexHolder>) -> SummaServerResult<Option<StoppedConsumption>> {
        if let Some(consumer_thread) = self.consumptions.remove(index_holder) {
            consumer_thread.stop().await?;
            Ok(Some(StoppedConsumption { consumer_thread }))
        } else {
            Ok(None)
        }
    }

    pub async fn get_consumer_for(&self, index_holder: &Handler<IndexHolder>) -> Option<&ConsumerThread> {
        self.consumptions.get(index_holder)
    }

    pub fn find_index_holder_for(&self, consumer_name: &str) -> Option<Handler<IndexHolder>> {
        for (index_holder, consumer_thread) in &self.consumptions {
            if consumer_thread.consumer_name() == consumer_name {
                return Some(index_holder.clone());
            }
        }
        None
    }

    pub fn find_consumer_config_for(&self, consumer_name: &str) -> Option<&crate::configs::consumer::Config> {
        for consumer_thread in self.consumptions.values() {
            if consumer_thread.consumer_name() == consumer_name {
                return Some(consumer_thread.config());
            }
        }
        None
    }
}
