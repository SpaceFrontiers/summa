use std::fmt::{Debug, Formatter};
use std::hash::{Hash, Hasher};
use std::mem::ManuallyDrop;
use std::ops::Deref;
use std::sync::Arc;

use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};

/// `Handler` wraps data with [Arc](std::sync::Arc) and notifies exteriors by channel when dropping
#[derive(Debug)]
pub struct Handler<T> {
    data: ManuallyDrop<Arc<T>>,
    sender: UnboundedSender<()>,
}

impl<T> Handler<T> {
    pub fn new(data: T, sender: UnboundedSender<()>) -> Handler<T> {
        Handler {
            sender,
            data: ManuallyDrop::new(Arc::new(data)),
        }
    }
}

impl<T> Clone for Handler<T> {
    fn clone(&self) -> Self {
        Handler {
            sender: self.sender.clone(),
            data: self.data.clone(),
        }
    }
}

impl<T> Drop for Handler<T> {
    fn drop(&mut self) {
        unsafe { ManuallyDrop::drop(&mut self.data) };
        self.sender.send(()).ok();
    }
}

impl<T> Deref for Handler<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

/// `OwningHandler` is like [Arc](std::sync::Arc) but with additional possibility to wait until
/// the last strong reference drops and then return wrapped data.
pub struct OwningHandler<T> {
    handler: Handler<T>,
    receiver: UnboundedReceiver<()>,
}

impl<T: Debug> Debug for OwningHandler<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        T::fmt(&**self.handler.data, f)
    }
}

impl<T: Default> Default for OwningHandler<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

impl<T> OwningHandler<T> {
    /// Wraps data with [Arc](std::sync::Arc) and creates a channel for listening drop events
    pub fn new(data: T) -> OwningHandler<T> {
        let (sender, receiver) = unbounded_channel();
        let handler = Handler::new(data, sender);
        OwningHandler { handler, receiver }
    }

    /// Takes new [Handler](crate::utils::sync::Handler)
    pub fn handler(&self) -> Handler<T> {
        self.handler.clone()
    }

    fn destruct(self) -> (UnboundedReceiver<()>, Handler<T>) {
        (self.receiver, self.handler)
    }

    /// Blocks until last strong references drops and return wrapped data afterwards
    pub async fn into_inner(self) -> T {
        let (mut receiver, handler) = self.destruct();
        let mut data: Arc<T> = Arc::clone(&handler.data);
        drop(handler);
        loop {
            match Arc::try_unwrap(data) {
                Ok(data) => return data,
                Err(arc_data) => {
                    data = arc_data;
                    receiver.recv().await.expect("channel unexpectedly closed");
                }
            }
        }
    }
}

impl<T> Deref for OwningHandler<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.handler.data
    }
}

impl<T: Hash> Hash for Handler<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.data.hash(state)
    }
}

impl<T: PartialEq> PartialEq<Self> for Handler<T> {
    fn eq(&self, other: &Self) -> bool {
        self.data.eq(&other.data)
    }
}

impl<T: PartialEq> Eq for Handler<T> {}
