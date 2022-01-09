use std::sync::Arc;

use parking_lot::RwLock;
use serde::Serialize;
use summa_core::components::{Tracker, TrackerEvent};

#[derive(Clone, Debug, Serialize)]
pub struct JsTrackerEvent {
    pub event: TrackerEvent,
}

#[derive(Default)]
pub struct SubscribeTrackerInner {
    subscribers: Vec<Box<dyn Fn(JsTrackerEvent)>>,
}

impl SubscribeTrackerInner {
    pub(super) fn add_subscriber(&mut self, subscriber: Box<dyn Fn(JsTrackerEvent)>) {
        self.subscribers.push(subscriber)
    }

    fn notify(&self, event: JsTrackerEvent) {
        for subscriber in &self.subscribers {
            subscriber(event.clone())
        }
    }
}

#[derive(Clone, Default)]
pub struct SubscribeTracker {
    inner: Arc<RwLock<SubscribeTrackerInner>>,
}

impl Tracker for SubscribeTracker {
    fn send_event(&self, event: TrackerEvent) {
        self.inner.write().notify(JsTrackerEvent { event })
    }
}

impl SubscribeTracker {
    pub fn add_subscriber(&mut self, subscriber: Box<dyn Fn(JsTrackerEvent)>) {
        self.inner.write().add_subscriber(subscriber)
    }
}
