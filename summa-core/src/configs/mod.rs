//! Storing and loading various Summa config files

mod config_proxy;
pub mod core_config;
#[cfg(feature = "fs")]
mod file_proxy;
mod partial_proxy;

pub use config_proxy::{ConfigProxy, ConfigReadProxy, ConfigWriteProxy, DirectProxy};
pub use core_config::{CoreConfig, CoreConfigBuilder};
#[cfg(feature = "fs")]
pub use file_proxy::{FileProxy, Loadable, Persistable};
pub use partial_proxy::{PartialProxy, PartialReadProxy, PartialWriteProxy};
