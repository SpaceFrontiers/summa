pub enum KafkaConsumingStatus {
    Consumed,
}

#[derive(thiserror::Error, Debug)]
pub enum KafkaConsumingError {
    #[error("empty_payload_error")]
    EmptyPayloadError,
    #[error("empty_operation_error")]
    EmptyOperationError,
    #[error("index_error: {0}")]
    IndexError(crate::errors::Error),
    #[error("kafka_error: {0}")]
    KafkaError(rdkafka::error::KafkaError),
    #[error("proto_decode_error: {0}")]
    ProtoDecodeError(prost::DecodeError),
}
