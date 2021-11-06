use std::sync::Arc;

use crate::errors::SummaResult;

#[async_trait]
pub trait ConfigProxy<TConfig>: Send + Sync {
    async fn read<'a>(&'a self) -> Box<dyn ConfigReadProxy<TConfig> + 'a>;
    async fn write<'a>(&'a self) -> Box<dyn ConfigWriteProxy<TConfig> + 'a>;
    async fn delete(self: Arc<Self>) -> SummaResult<TConfig>;
}

pub trait ConfigReadProxy<TConfig>: Send + Sync {
    fn get(&self) -> &TConfig;
}

pub trait ConfigWriteProxy<TConfig>: Send + Sync {
    fn get(&self) -> &TConfig;
    fn get_mut(&mut self) -> &mut TConfig;
    fn commit(&self) -> SummaResult<()>;
}

pub struct DirectProxy<TConfig> {
    config: TConfig,
}

impl<TConfig> DirectProxy<TConfig> {
    pub fn new(config: TConfig) -> DirectProxy<TConfig> {
        DirectProxy { config }
    }
}

#[async_trait]
impl<TConfig: Send + Sync> ConfigProxy<TConfig> for DirectProxy<TConfig> {
    async fn read<'a>(&'a self) -> Box<dyn ConfigReadProxy<TConfig> + 'a> {
        Box::new(self)
    }

    async fn write<'a>(&'a self) -> Box<dyn ConfigWriteProxy<TConfig> + 'a> {
        Box::new(self)
    }

    async fn delete(self: Arc<Self>) -> SummaResult<TConfig> {
        panic!()
    }
}

impl<TConfig: Send + Sync> ConfigReadProxy<TConfig> for &DirectProxy<TConfig> {
    fn get(&self) -> &TConfig {
        &self.config
    }
}

impl<TConfig: Send + Sync> ConfigWriteProxy<TConfig> for &DirectProxy<TConfig> {
    fn get(&self) -> &TConfig {
        &self.config
    }

    fn get_mut(&mut self) -> &mut TConfig {
        panic!()
    }

    fn commit(&self) -> SummaResult<()> {
        panic!()
    }
}
