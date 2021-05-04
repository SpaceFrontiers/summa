pub use super::ApplicationConfigHolder;
use crate::configs::config_holder::Loadable;
use crate::configs::{ApplicationConfig, ConfigHolder};
use crate::errors::{Error, SummaResult};
use std::path::Path;

#[derive(Clone)]
pub struct GlobalConfig {
    pub application_config: ApplicationConfigHolder,
}

impl GlobalConfig {
    pub fn new(application_config_filepath: &Path) -> SummaResult<GlobalConfig> {
        let application_config = ConfigHolder::<ApplicationConfig>::from_file(application_config_filepath, None)?;
        std::fs::create_dir_all(&application_config.data_path).map_err(|e| Error::IOError((e, Some(application_config.data_path.clone()))))?;

        Ok(GlobalConfig {
            application_config: ApplicationConfigHolder::new(application_config),
        })
    }
}
