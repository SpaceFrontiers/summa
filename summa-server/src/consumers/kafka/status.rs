pub enum KafkaConsumingStatus {
    Consumed,
}

#[derive(thiserror::Error, Debug)]
pub enum KafkaConsumingError {
    #[error("empty_payload_error")]
    EmptyPayload,
    #[error("empty_operation_error")]
    EmptyOperation,
    #[error("index_error: {0}")]
    Index(crate::errors::Error),
    #[error("kafka_error: {0}")]
    Kafka(rdkafka::error::KafkaError),
    #[error("parse_document_error: {0}")]
    ParseDocument(summa_core::errors::Error),
    #[error("proto_decode_error: {0}")]
    ProtoDecode(prost::DecodeError),
}
