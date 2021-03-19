//! Storing and loading various Summa config files

mod config_holder;
pub mod configs;
mod configurator;

pub use config_holder::{ConfigHolder, Persistable};
pub use configurator::Configurator;
