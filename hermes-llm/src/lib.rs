pub mod config;
pub mod data;
pub mod distributed;
pub mod io;
pub mod model;
pub mod tokenizer;
pub mod training;

pub use config::Config;
pub use distributed::{DistributedConfig, NcclCommunicator};
pub use model::GPT;
pub use training::Trainer;
