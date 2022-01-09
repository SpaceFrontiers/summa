use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TrackerEvent {
    StartReadingFile(String),
    FinishReadingFile(String),
    ReadingHotcache,
    QueryingIndex,
    CollectingDocuments,
    WarmingUp,
}

impl TrackerEvent {
    pub fn start_reading_file(file_id: &str) -> TrackerEvent {
        TrackerEvent::StartReadingFile(file_id.to_string())
    }

    pub fn finish_reading_file(file_id: &str) -> TrackerEvent {
        TrackerEvent::FinishReadingFile(file_id.to_string())
    }
}

pub trait Tracker: Clone {
    fn send_event(&self, event: TrackerEvent);
}

#[derive(Clone, Default)]
pub struct NoTracker {}
impl Tracker for NoTracker {
    fn send_event(&self, _: TrackerEvent) {}
}
