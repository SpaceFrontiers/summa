use std::fmt::Debug;
use std::future::Future;

use crate::errors::SummaResult;

#[derive(Clone)]
pub enum Driver {
    Native,
    #[cfg(feature = "tokio-rt")]
    Tokio(tokio::runtime::Handle),
}

impl Driver {
    #[cfg(feature = "tokio-rt")]
    pub fn current_tokio() -> Driver {
        Driver::Tokio(tokio::runtime::Handle::current())
    }

    #[inline]
    pub fn block_on<F: Debug + Send + 'static>(&self, f: impl Future<Output = F> + Send + 'static) -> F {
        match self {
            Driver::Native => unimplemented!("impossible to `block_on` without reactor"),
            #[cfg(feature = "tokio-rt")]
            Driver::Tokio(handle) => handle.block_on(f),
        }
    }

    #[inline]
    pub async fn execute_blocking<O: Send + 'static>(&self, f: impl FnOnce() -> O + Send + 'static) -> SummaResult<O> {
        match self {
            Driver::Native => Ok(f()),
            #[cfg(feature = "tokio-rt")]
            Driver::Tokio(handle) => Ok(handle.spawn_blocking(f).await?),
        }
    }
}
