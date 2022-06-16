mod consumer;
pub(crate) mod status;

pub(crate) use consumer::KafkaConsumerThread;

use crate::errors::Error;

impl From<rdkafka::error::KafkaError> for Error {
    fn from(error: rdkafka::error::KafkaError) -> Self {
        Error::Consumer(error.to_string())
    }
}
