pub use super::{ApplicationConfigHolder, RuntimeConfigHolder};
use crate::configs::config_holder::Persistable;
use crate::errors::{Error, SummaResult};
use parking_lot::RwLock;
use std::path::Path;
use std::sync::Arc;

#[derive(Clone)]
pub struct GlobalConfig {
    pub application_config: Arc<RwLock<ApplicationConfigHolder>>,
    pub runtime_config: Arc<RwLock<RuntimeConfigHolder>>,
}

impl GlobalConfig {
    pub fn new(application_config_filepath: &Path) -> SummaResult<GlobalConfig> {
        let application_config = ApplicationConfigHolder::from_file(application_config_filepath, None, true)?;

        std::fs::create_dir_all(&application_config.data_path).map_err(|e| Error::IOError((e, Some(application_config.data_path.clone()))))?;
        let runtime_config = RuntimeConfigHolder::from_file(&application_config.data_path.join("runtime_config.yaml"), None, false)?;
        runtime_config.save()?;

        Ok(GlobalConfig {
            application_config: Arc::new(RwLock::new(application_config)),
            runtime_config: Arc::new(RwLock::new(runtime_config)),
        })
    }
}
