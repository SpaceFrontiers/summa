use super::consumer_thread::KafkaConsumerThreadController;
use super::status::{KafkaConsumingError, KafkaConsumingStatus};
use crate::configs::KafkaConsumerConfig;
use crate::errors::SummaResult;
use rdkafka::admin::{AdminClient, AdminOptions, NewTopic, TopicReplication};
use rdkafka::config::{ClientConfig, FromClientConfig};
use rdkafka::consumer::{Consumer, StreamConsumer};
use rdkafka::error::KafkaError;
use rdkafka::message::BorrowedMessage;
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
    pub async fn new(consumer_name: &str, config: &KafkaConsumerConfig) -> SummaResult<KafkaConsumer> {
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

        if config.create_topics {
            let config = config.clone();
            let kafka_producer_config = kafka_producer_config.clone();
            KafkaConsumer::create_topics(&kafka_producer_config, &config).await?;
        }

        let thread_controllers: SummaResult<Vec<KafkaConsumerThreadController>> = (0..config.threads)
            .map(|n| {
                let stream_consumer: StreamConsumer = kafka_consumer_config.create()?;
                stream_consumer.subscribe(&config.topics.iter().map(String::as_str).collect::<Vec<_>>()).unwrap();
                Ok(KafkaConsumerThreadController::new(&format!("{}-{}", consumer_name, n), stream_consumer))
            })
            .collect();

        let thread_controllers = thread_controllers?;

        Ok(KafkaConsumer {
            consumer_name: consumer_name.to_string(),
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

    pub fn commit(&self) -> SummaResult<()> {
        for thread_controller in self.thread_controllers.iter() {
            thread_controller.commit()?;
        }
        Ok(())
    }

    pub async fn stop(&self) -> SummaResult<()> {
        for thread_controller in self.thread_controllers.iter() {
            thread_controller.stop().await?;
        }
        Ok(())
    }

    #[instrument(skip(kafka_producer_config))]
    pub async fn create_topics(kafka_producer_config: &ClientConfig, config: &KafkaConsumerConfig) -> SummaResult<()> {
        let admin_client = AdminClient::from_config(kafka_producer_config)?;
        let topics: Vec<_> = config
            .topics
            .iter()
            .map(|topic_name| NewTopic::new(topic_name.as_str(), config.threads.try_into().unwrap(), TopicReplication::Fixed(1)))
            .collect();
        info!(action = "create_topics", topics = ?topics);
        admin_client.create_topics(&topics, &AdminOptions::new()).await?;
        Ok(())
    }

    #[instrument(skip(self))]
    pub async fn delete_topics(&self) -> SummaResult<()> {
        let admin_client = AdminClient::from_config(&self.kafka_producer_config)?;
        let topics: Vec<_> = self.config.topics.iter().map(String::as_str).collect();
        info!(action = "delete_topics", topics = ?topics);
        admin_client.delete_topics(&topics, &AdminOptions::new()).await?;
        Ok(())
    }

    #[instrument]
    pub async fn clear(&self) -> SummaResult<()> {
        if self.config.delete_topics {
            self.delete_topics().await
        } else {
            Ok(())
        }
    }
}
