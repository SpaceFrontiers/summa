//! Consuming documents from Kafka

mod consumer_thread;
pub mod dummy;
#[cfg(feature = "kafka")]
pub mod kafka;

pub use consumer_thread::ConsumerThread;
