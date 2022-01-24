use instant::Instant;
use tantivy::directory::OwnedBytes;

pub(super) struct StoredItem {
    last_access_time: Instant,
    payload: OwnedBytes,
}

impl StoredItem {
    pub fn new(payload: OwnedBytes, now: Instant) -> Self {
        StoredItem {
            last_access_time: now,
            payload,
        }
    }
}

impl StoredItem {
    pub fn payload(&mut self) -> OwnedBytes {
        self.last_access_time = Instant::now();
        self.payload.clone()
    }

    pub fn len(&self) -> usize {
        self.payload.len()
    }

    pub fn last_access_time(&self) -> Instant {
        self.last_access_time
    }
}
