use super::consumer_thread::KafkaConsumerThreadController;
use super::status::{KafkaConsumingError, KafkaConsumingStatus};
use crate::configs::KafkaConsumerConfig;
use crate::errors::SummaResult;
use rdkafka::admin::{AdminClient, AdminOptions, AlterConfig, NewTopic, ResourceSpecifier, TopicReplication};
use rdkafka::config::{ClientConfig, FromClientConfig};
use rdkafka::consumer::{Consumer, StreamConsumer};
use rdkafka::error::KafkaError;
use rdkafka::message::BorrowedMessage;
use rdkafka::util::Timeout;
use std::str;
use tracing::{info, instrument};

/// Manages multiple consuming Kafka threads
#[derive(Clone, Debug)]
pub(crate) struct KafkaConsumer {
    consumer_name: String,
    config: KafkaConsumerConfig,
    kafka_producer_config: ClientConfig,
    thread_controllers: Vec<KafkaConsumerThreadController>,
}

impl KafkaConsumer {
    #[instrument]
    pub fn new(consumer_name: &str, config: &KafkaConsumerConfig) -> SummaResult<KafkaConsumer> {
        let mut kafka_consumer_config = ClientConfig::new();
        kafka_consumer_config
            .set("group.id", &config.group_id)
            .set("bootstrap.servers", config.bootstrap_servers.join(","))
            .set("enable.partition.eof", "false")
            .set("session.timeout.ms", "6000")
            .set("auto.offset.reset", "earliest")
            .set("allow.auto.create.topics", "true");

        let mut kafka_producer_config = ClientConfig::new();
        kafka_producer_config.set("bootstrap.servers", config.bootstrap_servers.join(","));

        let thread_controllers: SummaResult<Vec<KafkaConsumerThreadController>> = (0..config.threads)
            .map(|n| {
                let stream_consumer: StreamConsumer = kafka_consumer_config.create()?;
                stream_consumer.subscribe(&config.topics.iter().map(String::as_str).collect::<Vec<_>>()).unwrap();
                Ok(KafkaConsumerThreadController::new(&format!("{}-{}", consumer_name, n), stream_consumer))
            })
            .collect();

        let thread_controllers = thread_controllers?;

        Ok(KafkaConsumer {
            consumer_name: consumer_name.to_owned(),
            config: config.clone(),
            kafka_producer_config,
            thread_controllers,
        })
    }

    pub fn consumer_name(&self) -> &str {
        &self.consumer_name
    }

    pub fn start<TProcessor>(&self, processor: TProcessor) -> SummaResult<()>
    where
        TProcessor: 'static + Fn(Result<BorrowedMessage<'_>, KafkaError>) -> Result<KafkaConsumingStatus, KafkaConsumingError> + Send + Sync + Clone,
    {
        for thread_controller in self.thread_controllers.iter() {
            let processor = processor.clone();
            thread_controller.start(processor);
        }
        Ok(())
    }

    pub fn commit_offsets(&self) -> SummaResult<()> {
        for thread_controller in self.thread_controllers.iter() {
            thread_controller.commit_offsets()?;
        }
        Ok(())
    }

    pub async fn stop(&self) -> SummaResult<()> {
        for thread_controller in self.thread_controllers.iter() {
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
        info!(action = "create_topics", topics = ?new_topics);
        admin_client.create_topics(&new_topics, &admin_options).await?;
        admin_client.alter_configs(&alter_topics, &admin_options).await?;
        Ok(())
    }

    #[instrument(skip(self))]
    async fn delete_topics(&self) -> SummaResult<()> {
        let admin_client = AdminClient::from_config(&self.kafka_producer_config)?;
        let topics: Vec<_> = self.config.topics.iter().map(String::as_str).collect();
        info!(action = "delete_topics", topics = ?topics);
        admin_client.delete_topics(&topics, &AdminOptions::new().operation_timeout(Some(Timeout::Never))).await?;
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

    #[instrument]
    pub async fn on_delete(&self) -> SummaResult<()> {
        if self.config.delete_topics {
            self.delete_topics().await
        } else {
            Ok(())
        }
    }
}
