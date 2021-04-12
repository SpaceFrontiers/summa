use super::status::{KafkaConsumingError, KafkaConsumingStatus};
use crate::errors::SummaResult;
use crate::utils::thread_handler::ThreadHandler;
use futures::StreamExt;
use opentelemetry::{global, KeyValue};
use parking_lot::Mutex;
use rdkafka::consumer::stream_consumer::StreamConsumer;
use rdkafka::consumer::{CommitMode, Consumer};
use rdkafka::error::KafkaError;
use rdkafka::message::BorrowedMessage;
use std::sync::Arc;
use tokio::sync::oneshot;
use tracing::{info, info_span, instrument, warn, Instrument};

/// Long-living container for Kafka consuming thread
#[derive(Clone)]
pub struct KafkaConsumerThreadController {
    thread_name: String,
    thread_handler: Arc<Mutex<Option<ThreadHandler>>>,
    stream_consumer: Arc<Mutex<StreamConsumer>>,
}

impl std::fmt::Debug for KafkaConsumerThreadController {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self.thread_name)
    }
}

impl KafkaConsumerThreadController {
    pub fn new(thread_name: &str, stream_consumer: StreamConsumer) -> KafkaConsumerThreadController {
        KafkaConsumerThreadController {
            thread_name: thread_name.to_owned(),
            thread_handler: Arc::new(Mutex::new(None)),
            stream_consumer: Arc::new(Mutex::new(stream_consumer)),
        }
    }

    #[instrument(skip_all, fields(thread_name=?self.thread_name))]
    pub fn start<TProcessor>(&self, processor: TProcessor)
    where
        TProcessor: 'static + Fn(Result<BorrowedMessage<'_>, KafkaError>) -> Result<KafkaConsumingStatus, KafkaConsumingError> + Send,
    {
        info!(action = "start");
        let (shutdown_trigger, shutdown_tripwire) = oneshot::channel();

        let stream_consumer = self.stream_consumer.clone();
        let thread_name = self.thread_name.clone();
        let stream_processor = {
            async move {
                let stream_consumer = stream_consumer.lock();
                let stream = stream_consumer.stream();
                let meter = global::meter("summa");
                let counter = meter.u64_counter("consume").with_description("Number of consumed events").init();
                let mut message_stream = stream.take_until(shutdown_tripwire);
                info!(action = "started");
                loop {
                    match message_stream.next().await {
                        Some(message) => {
                            match processor(message) {
                                Ok(_) => counter.add(1, &[KeyValue::new("status", "ok"), KeyValue::new("thread_name", thread_name.clone())]),
                                Err(error) => {
                                    warn!(action = "error", error = ?error);
                                    counter.add(1, &[KeyValue::new("status", "error"), KeyValue::new("thread_name", thread_name.clone())]);
                                }
                            };
                        }
                        None => {
                            info!(action = "stopped");
                            break;
                        }
                    }
                }
                Ok(())
            }
        }
        .instrument(info_span!(parent: None, "consumer", thread_name = ?self.thread_name));
        *self.thread_handler.lock() = Some(ThreadHandler::new(tokio::spawn(stream_processor), shutdown_trigger));
    }

    #[instrument(skip(self))]
    pub fn commit_offsets(&self) -> SummaResult<()> {
        info!(action = "commit_consumer_state");
        let result = self.stream_consumer.lock().commit_consumer_state(CommitMode::Sync);
        match result {
            Err(rdkafka::error::KafkaError::ConsumerCommit(rdkafka::error::RDKafkaErrorCode::NoOffset)) => Ok(()),
            left => left,
        }?;
        Ok(())
    }

    #[instrument(skip(self))]
    pub async fn stop(&self) -> SummaResult<()> {
        info!(action = "stop", thread_name = ?self.thread_name);
        if let Some(thread_handler) = self.thread_handler.lock().take() {
            thread_handler.stop().await?;
        }
        Ok(())
    }
}
