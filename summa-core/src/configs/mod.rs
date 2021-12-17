//! Storing and loading various Summa config files

mod config_proxy;
pub mod core;
#[cfg(feature = "fs")]
mod file_proxy;
mod partial_proxy;

pub use config_proxy::{ConfigProxy, ConfigReadProxy, ConfigWriteProxy, DirectProxy};
#[cfg(feature = "fs")]
pub use file_proxy::{FileProxy, Loadable, Persistable};
pub use partial_proxy::{PartialProxy, PartialReadProxy, PartialWriteProxy};
