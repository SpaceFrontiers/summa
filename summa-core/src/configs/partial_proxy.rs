use std::sync::Arc;

use crate::configs::{ConfigProxy, ConfigReadProxy, ConfigWriteProxy};
use crate::errors::SummaResult;

#[derive(Clone)]
pub struct PartialProxy<
    TRootConfig: Send + Sync,
    TConfig: Send,
    TPartialFn: Fn(&TRootConfig) -> &TConfig + Send + Sync,
    TPartialFnMut: Fn(&mut TRootConfig) -> &mut TConfig + Send + Sync,
> {
    root_config: Arc<dyn ConfigProxy<TRootConfig>>,
    partial_fn: TPartialFn,
    partial_fn_mut: TPartialFnMut,
}

impl<
        TRootConfig: Send + Sync,
        TConfig: Send,
        TPartialFn: Fn(&TRootConfig) -> &TConfig + Send + Sync,
        TPartialFnMut: Fn(&mut TRootConfig) -> &mut TConfig + Send + Sync,
    > PartialProxy<TRootConfig, TConfig, TPartialFn, TPartialFnMut>
{
    pub fn new(
        root_config: &Arc<dyn ConfigProxy<TRootConfig>>,
        partial_fn: TPartialFn,
        partial_fn_mut: TPartialFnMut,
    ) -> PartialProxy<TRootConfig, TConfig, TPartialFn, TPartialFnMut> {
        PartialProxy {
            root_config: root_config.clone(),
            partial_fn,
            partial_fn_mut,
        }
    }
}

#[async_trait]
impl<
        TRootConfig: Send + Sync,
        TConfig: Send,
        TPartialFn: Fn(&TRootConfig) -> &TConfig + Send + Sync,
        TPartialFnMut: Fn(&mut TRootConfig) -> &mut TConfig + Send + Sync,
    > ConfigProxy<TConfig> for PartialProxy<TRootConfig, TConfig, TPartialFn, TPartialFnMut>
{
    async fn read<'a>(&'a self) -> Box<dyn ConfigReadProxy<TConfig> + 'a> {
        Box::new(PartialReadProxy {
            root_config: self.root_config.read().await,
            partial_fn: &self.partial_fn,
        })
    }

    async fn write<'a>(&'a self) -> Box<dyn ConfigWriteProxy<TConfig> + 'a> {
        Box::new(PartialWriteProxy {
            root_config: self.root_config.write().await,
            partial_fn: &self.partial_fn,
            partial_fn_mut: &self.partial_fn_mut,
        })
    }
}

pub struct PartialReadProxy<'a, TRootConfig: Send + Sync, TConfig: Send, TPartialFn: Fn(&TRootConfig) -> &TConfig + Send + Sync> {
    root_config: Box<dyn ConfigReadProxy<TRootConfig> + 'a>,
    partial_fn: TPartialFn,
}

impl<TRootConfig: Send + Sync, TConfig: Send, TPartialFn: Fn(&TRootConfig) -> &TConfig + Send + Sync> ConfigReadProxy<TConfig>
    for PartialReadProxy<'_, TRootConfig, TConfig, TPartialFn>
{
    fn get(&self) -> &TConfig {
        (self.partial_fn)(self.root_config.get())
    }
}

pub struct PartialWriteProxy<
    'a,
    TRootConfig,
    TConfig: Send,
    TPartialFn: Fn(&TRootConfig) -> &TConfig + Send + Sync,
    TPartialFnMut: Fn(&mut TRootConfig) -> &mut TConfig + Send + Sync,
> {
    root_config: Box<dyn ConfigWriteProxy<TRootConfig> + 'a>,
    partial_fn: TPartialFn,
    partial_fn_mut: TPartialFnMut,
}
#[async_trait]
impl<TRootConfig, TConfig: Send, TPartialFn: Fn(&TRootConfig) -> &TConfig + Send + Sync, TPartialFnMut: Fn(&mut TRootConfig) -> &mut TConfig + Send + Sync>
    ConfigWriteProxy<TConfig> for PartialWriteProxy<'_, TRootConfig, TConfig, TPartialFn, TPartialFnMut>
{
    fn get(&self) -> &TConfig {
        (self.partial_fn)(self.root_config.get())
    }

    fn get_mut(&mut self) -> &mut TConfig {
        (self.partial_fn_mut)(self.root_config.get_mut())
    }

    async fn commit(&self) -> SummaResult<()> {
        self.root_config.commit().await
    }
}
