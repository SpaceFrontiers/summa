use std::sync::Arc;

use tokio::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::errors::SummaResult;

#[async_trait]
pub trait ConfigProxy<TConfig>: Send + Sync {
    async fn read<'a>(&'a self) -> Box<dyn ConfigReadProxy<TConfig> + 'a>;
    async fn write<'a>(&'a self) -> Box<dyn ConfigWriteProxy<TConfig> + 'a>;
}

pub trait ConfigReadProxy<TConfig>: Send + Sync {
    fn get(&self) -> &TConfig;
}

#[async_trait]
pub trait ConfigWriteProxy<TConfig>: Send + Sync {
    fn get(&self) -> &TConfig;
    fn get_mut(&mut self) -> &mut TConfig;
    async fn commit(&self) -> SummaResult<()>;
}

pub struct DirectProxy<TConfig> {
    config: Arc<RwLock<TConfig>>,
}

impl<TConfig> DirectProxy<TConfig> {
    pub fn new(config: TConfig) -> DirectProxy<TConfig> {
        DirectProxy {
            config: Arc::new(RwLock::new(config)),
        }
    }
}

impl<TConfig: Default> Default for DirectProxy<TConfig> {
    fn default() -> Self {
        DirectProxy {
            config: Arc::new(RwLock::new(TConfig::default())),
        }
    }
}

#[async_trait]
impl<TConfig: Send + Sync> ConfigProxy<TConfig> for DirectProxy<TConfig> {
    async fn read<'a>(&'a self) -> Box<dyn ConfigReadProxy<TConfig> + 'a> {
        Box::new(DirectReadProxy {
            config: self.config.read().await,
        })
    }

    async fn write<'a>(&'a self) -> Box<dyn ConfigWriteProxy<TConfig> + 'a> {
        Box::new(DirectWriteProxy {
            config: self.config.write().await,
        })
    }
}

pub struct DirectReadProxy<'a, TConfig: Send + Sync> {
    config: RwLockReadGuard<'a, TConfig>,
}

impl<TConfig: Send + Sync> ConfigReadProxy<TConfig> for DirectReadProxy<'_, TConfig> {
    fn get(&self) -> &TConfig {
        &self.config
    }
}

pub struct DirectWriteProxy<'a, TConfig: Send + Sync> {
    config: RwLockWriteGuard<'a, TConfig>,
}

#[async_trait]
impl<TConfig: Send + Sync> ConfigWriteProxy<TConfig> for DirectWriteProxy<'_, TConfig> {
    fn get(&self) -> &TConfig {
        &self.config
    }

    fn get_mut(&mut self) -> &mut TConfig {
        &mut self.config
    }

    async fn commit(&self) -> SummaResult<()> {
        Ok(())
    }
}
