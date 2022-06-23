use std::collections::HashMap;
use std::future::Future;

use futures_util::future::join_all;
use summa_core::components::IndexHolder;
use summa_core::utils::sync::Handler;
use tracing::{info, instrument};

use crate::components::consumers::ConsumerThread;
use crate::errors::{SummaServerResult, ValidationError};

#[derive(Debug)]
pub struct StoppedConsumption {
    consumer_thread: Box<dyn ConsumerThread>,
}

impl StoppedConsumption {
    pub async fn commit_offsets(self) -> SummaServerResult<PreparedConsumption> {
        self.consumer_thread.commit().await?;
        Ok(PreparedConsumption {
            committed_consumer_thread: self.consumer_thread,
        })
    }

    pub fn ignore(self) -> PreparedConsumption {
        PreparedConsumption {
            committed_consumer_thread: self.consumer_thread,
        }
    }
}

#[derive(Debug)]
pub struct PreparedConsumption {
    committed_consumer_thread: Box<dyn ConsumerThread>,
}

impl PreparedConsumption {
    #[cfg(not(feature = "kafka"))]
    pub fn from_config(_consumer_name: &str, _consumer_config: &crate::configs::consumer::Config) -> SummaServerResult<PreparedConsumption> {
        unimplemented!();
    }
    #[cfg(feature = "kafka")]
    pub fn from_config(consumer_name: &str, consumer_config: &crate::configs::consumer::Config) -> SummaServerResult<PreparedConsumption> {
        Ok(PreparedConsumption {
            committed_consumer_thread: Box::new(crate::components::consumers::kafka::KafkaConsumerThread::new(consumer_name, consumer_config)?)
                as Box<dyn ConsumerThread>,
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
    consumptions: HashMap<Handler<IndexHolder>, Box<dyn ConsumerThread>>,
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
            .start(index_writer_holder, conflict_strategy, schema)
            .await?;
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
            tokio::task::spawn_blocking(move || index_writer_holder.commit_and_prepare(true)).await??;
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

    pub async fn get_consumer_for(&self, index_holder: &Handler<IndexHolder>) -> Option<&Box<dyn ConsumerThread>> {
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
