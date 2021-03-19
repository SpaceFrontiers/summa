//! Consuming documents from Kafka

mod consumer_registry;
pub(crate) mod kafka;

pub(crate) use consumer_registry::ConsumerRegistry;
