use std::fmt::Debug;
use std::future::Future;

#[derive(Clone)]
pub enum Driver {
    Native,
    Tokio(tokio::runtime::Handle),
}

impl Driver {
    pub fn current_tokio() -> Driver {
        Driver::Tokio(tokio::runtime::Handle::current())
    }

    #[inline]
    pub fn block_on<F: Debug + Send + 'static>(&self, f: impl Future<Output = F> + Send + 'static) -> F {
        match self {
            Driver::Native => unimplemented!("impossible to `block_on` without reactor"),
            Driver::Tokio(handle) => handle.block_on(f),
        }
    }
}
