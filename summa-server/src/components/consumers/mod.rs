//! Consuming documents from Kafka

mod consumer_thread;
#[cfg(feature = "kafka")]
pub mod kafka;

pub use consumer_thread::ConsumerThread;
