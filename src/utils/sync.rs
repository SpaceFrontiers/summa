use crossbeam_channel::{bounded, Receiver, Sender};
use std::fmt::Debug;
use std::mem::ManuallyDrop;
use std::ops::Deref;
use std::sync::Arc;

pub struct Handler<T> {
    data: ManuallyDrop<Arc<T>>,
    sender: Sender<i32>,
}

impl<T> Handler<T> {
    pub fn new(data: T, sender: Sender<i32>) -> Handler<T> {
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
        self.sender.try_send(0).ok();
    }
}

impl<T> Deref for Handler<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T> Deref for OwningHandler<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.handler.data
    }
}

pub struct OwningHandler<T> {
    handler: Handler<T>,
    receiver: Receiver<i32>,
}

impl<T> OwningHandler<T> {
    pub fn new(data: T) -> OwningHandler<T> {
        let (sender, receiver) = bounded(1);
        let handler = Handler::new(data, sender);
        OwningHandler { handler, receiver }
    }

    pub fn handler(&self) -> Handler<T> {
        self.handler.clone()
    }

    pub fn destruct(self) -> (Receiver<i32>, Handler<T>) {
        (self.receiver, self.handler)
    }

    pub fn into_inner(self) -> T {
        let (receiver, mut handler) = self.destruct();
        let mut data = unsafe { ManuallyDrop::take(&mut handler.data) };
        loop {
            match Arc::try_unwrap(data) {
                Ok(data) => return data,
                Err(arc_data) => {
                    data = arc_data;
                    receiver.recv().unwrap();
                }
            }
        }
    }
}

impl<T: Debug> Debug for OwningHandler<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "OwningHandler({:?})", self.handler.data)
    }
}
