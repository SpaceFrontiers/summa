use std::sync::Arc;

use parking_lot::RwLock;
use serde::Serialize;
use summa_core::components::Tracker;

#[derive(Clone, Debug, Serialize)]
pub struct TrackerEvent {
    pub status: String,
}

#[derive(Default)]
pub struct SubscribeTrackerInner {
    status: String,
    subscribers: Vec<Box<dyn Fn(TrackerEvent)>>,
}

impl SubscribeTrackerInner {
    fn to_snapshot(&self) -> TrackerEvent {
        TrackerEvent { status: self.get_status() }
    }

    pub(super) fn get_status(&self) -> String {
        self.status.to_string()
    }

    pub(super) fn set_status(&mut self, new_status: &str) {
        self.status = new_status.to_string();
        self.notify();
    }

    pub(super) fn add_subscriber(&mut self, subscriber: Box<dyn Fn(TrackerEvent)>) {
        self.subscribers.push(subscriber)
    }

    fn notify(&self) {
        let snapshot = self.to_snapshot();
        for subscriber in &self.subscribers {
            subscriber(snapshot.clone())
        }
    }
}

#[derive(Clone, Default)]
pub struct SubscribeTracker {
    inner: Arc<RwLock<SubscribeTrackerInner>>,
}

impl Tracker for SubscribeTracker {
    fn get_status(&self) -> String {
        self.inner.read().get_status()
    }

    fn set_status(&self, new_status: &str) {
        self.inner.write().set_status(new_status)
    }
}

impl SubscribeTracker {
    pub fn add_subscriber(&mut self, subscriber: Box<dyn Fn(TrackerEvent)>) {
        self.inner.write().add_subscriber(subscriber)
    }
}
