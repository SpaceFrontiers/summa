use std::fmt::Debug;
use std::future::Future;

#[derive(Clone, Debug)]
pub enum Executor {
    Tokio(tokio::runtime::Handle),
}

impl Executor {
    pub fn from_tokio_handle(handle: tokio::runtime::Handle) -> Executor {
        Executor::Tokio(handle)
    }

    pub fn spawn_blocking<F: Debug + Send + 'static>(&self, f: impl Future<Output = F> + Send + 'static) -> F {
        let (s, mut r) = tokio::sync::mpsc::unbounded_channel();
        match self {
            Executor::Tokio(handle) => {
                handle.spawn(async move { s.send(f.await).expect("cannot send to channel") });
                r.blocking_recv().expect("cannot block on channel")
            }
        }
    }
}
