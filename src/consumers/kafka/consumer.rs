use super::consumer_thread::ConsumerThread;
use super::status::{KafkaConsumingError, KafkaConsumingStatus};
use crate::configs::ConsumerConfig;
use crate::errors::SummaResult;
use rdkafka::admin::{AdminClient, AdminOptions, AlterConfig, NewTopic, ResourceSpecifier, TopicReplication};
use rdkafka::config::{ClientConfig, FromClientConfig};
use rdkafka::consumer::{Consumer as KafkaConsumer, StreamConsumer as KafkaStreamConsumer};
use rdkafka::error::KafkaError;
use rdkafka::message::BorrowedMessage;
use rdkafka::util::Timeout;
use std::str;
use tracing::{info, instrument};

/// Manages multiple consuming threads
#[derive(Clone, Debug)]
pub(crate) struct Consumer {
    consumer_name: String,
    config: ConsumerConfig,
    kafka_producer_config: ClientConfig,
    consumer_threads: Vec<ConsumerThread>,
}

impl Consumer {
    #[instrument]
    pub fn new(consumer_name: &str, config: &ConsumerConfig) -> SummaResult<Consumer> {
        let mut kafka_consumer_config = ClientConfig::new();
        kafka_consumer_config
            .set("bootstrap.servers", config.bootstrap_servers.join(","))
            .set("group.id", &config.group_id)
            .set("enable.partition.eof", "false")
            .set("session.timeout.ms", config.session_timeout_ms.to_string())
            .set("max.poll.interval.ms", config.max_poll_interval_ms.to_string())
            .set("auto.offset.reset", "earliest")
            .set("allow.auto.create.topics", "true");

        let mut kafka_producer_config = ClientConfig::new();
        kafka_producer_config.set("bootstrap.servers", config.bootstrap_servers.join(","));

        let consumer_threads: Vec<ConsumerThread> = (0..config.threads)
            .map(|n| {
                let stream_consumer: KafkaStreamConsumer = kafka_consumer_config.create()?;
                stream_consumer.subscribe(&config.topics.iter().map(String::as_str).collect::<Vec<_>>()).unwrap();
                Ok(ConsumerThread::new(&format!("{}-{}", consumer_name, n), stream_consumer))
            })
            .collect::<SummaResult<_>>()?;

        Ok(Consumer {
            consumer_name: consumer_name.to_owned(),
            config: config.clone(),
            kafka_producer_config,
            consumer_threads,
        })
    }

    pub fn consumer_name(&self) -> &str {
        &self.consumer_name
    }

    pub fn start<TProcessor>(&self, processor: TProcessor) -> SummaResult<()>
    where
        TProcessor: 'static + Fn(Result<BorrowedMessage<'_>, KafkaError>) -> Result<KafkaConsumingStatus, KafkaConsumingError> + Send + Sync + Clone,
    {
        for thread_controller in self.consumer_threads.iter() {
            let processor = processor.clone();
            thread_controller.start(processor);
        }
        Ok(())
    }

    pub fn commit_offsets(&self) -> SummaResult<()> {
        for thread_controller in self.consumer_threads.iter() {
            thread_controller.commit_offsets()?;
        }
        Ok(())
    }

    pub async fn stop(&self) -> SummaResult<()> {
        for thread_controller in self.consumer_threads.iter() {
            thread_controller.stop().await?;
        }
        Ok(())
    }

    #[instrument(skip(self))]
    async fn create_topics(&self) -> SummaResult<()> {
        let admin_client = AdminClient::from_config(&self.kafka_producer_config)?;
        let admin_options = AdminOptions::new().operation_timeout(Some(Timeout::Never));
        let new_topics: Vec<_> = self
            .config
            .topics
            .iter()
            .map(|topic_name| NewTopic::new(topic_name.as_str(), self.config.threads.try_into().unwrap(), TopicReplication::Fixed(1)))
            .collect();
        let alter_topics: Vec<_> = self
            .config
            .topics
            .iter()
            .map(|topic_name| {
                AlterConfig::new(ResourceSpecifier::Topic(topic_name.as_str()))
                    .set("retention.ms", "3600000")
                    .set("retention.bytes", "1073741824")
            })
            .collect();
        let response = admin_client.create_topics(&new_topics, &admin_options).await?;
        info!(action = "create_topics", topics = ?new_topics, response = ?response);
        let response = admin_client.alter_configs(&alter_topics, &admin_options).await?;
        info!(action = "alter_configs", topics = ?new_topics, response = ?response);
        Ok(())
    }

    #[instrument(skip(self))]
    async fn delete_topics(&self) -> SummaResult<()> {
        let admin_client = AdminClient::from_config(&self.kafka_producer_config)?;
        let topics: Vec<_> = self.config.topics.iter().map(String::as_str).collect();
        let response = admin_client.delete_topics(&topics, &AdminOptions::new().operation_timeout(Some(Timeout::Never))).await?;
        info!(action = "delete_topics", topics = ?topics, response = ?response);
        Ok(())
    }

    #[instrument]
    pub async fn on_create(&self) -> SummaResult<()> {
        if self.config.create_topics {
            self.create_topics().await
        } else {
            Ok(())
        }
    }

    #[instrument(skip(self), fields(consumer_name = ?self.consumer_name))]
    pub async fn on_delete(&self) -> SummaResult<()> {
        if self.config.delete_topics {
            self.delete_topics().await
        } else {
            Ok(())
        }
    }
}
