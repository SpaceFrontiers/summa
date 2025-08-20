//! Per-backend gRPC channels and deadline propagation.
//!
//! One lazily-connected h2c channel per backend address, shared by the
//! search and index stubs, mirroring how hermes clients connect to
//! hermes-server. Channels are cached by address and dropped when discovery
//! removes the backend (or eviction wants a fresh connection on recovery).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use tokio::sync::Semaphore;
use tonic::Status;
use tonic::codec::CompressionEncoding;
use tonic::metadata::MetadataMap;
use tonic::transport::{Channel, Endpoint};

use crate::proto::hermes::index_service_client::IndexServiceClient;
use crate::proto::hermes::search_service_client::SearchServiceClient;

/// Shaved off a propagated client deadline to cover broker overhead, so the
/// backend gives up slightly before the client does and the client sees the
/// backend's status rather than a broker-side race.
const DEADLINE_EPSILON: Duration = Duration::from_millis(50);
const DEADLINE_FLOOR: Duration = Duration::from_millis(10);

#[derive(Clone)]
pub struct BackendChannels {
    pub search: SearchServiceClient<Channel>,
    pub index: IndexServiceClient<Channel>,
    /// Mirrors the backend's own --max-concurrent-searches admission so the
    /// broker never amplifies load past what the backend would admit.
    pub search_permits: Arc<Semaphore>,
}

pub struct ClientPool {
    channels: Mutex<HashMap<String, BackendChannels>>,
    backend_max_searches: usize,
    /// Decoded-message cap on broker→backend channels (--backend-max-decode-mb);
    /// must cover the backends' search/index encode caps.
    backend_max_decode: usize,
    /// Encoded-message cap on broker→backend channels (--backend-max-encode-mb);
    /// must cover whatever the broker accepted at its own edge
    /// (--index-max-decode-mb).
    backend_max_encode: usize,
}

impl ClientPool {
    pub fn new(
        backend_max_searches: usize,
        backend_max_decode: usize,
        backend_max_encode: usize,
    ) -> Self {
        Self {
            channels: Mutex::new(HashMap::new()),
            backend_max_searches,
            backend_max_decode,
            backend_max_encode,
        }
    }

    /// Get or lazily create the channel pair for a backend address.
    pub fn get(&self, addr: &str) -> Result<BackendChannels, Status> {
        if let Some(existing) = self.channels.lock().get(addr) {
            return Ok(existing.clone());
        }
        let endpoint = Endpoint::from_shared(format!("http://{addr}"))
            .map_err(|e| Status::internal(format!("invalid backend address '{addr}': {e}")))?
            .connect_timeout(Duration::from_secs(5))
            .tcp_keepalive(Some(Duration::from_secs(60)))
            .http2_keep_alive_interval(Duration::from_secs(30))
            .keep_alive_timeout(Duration::from_secs(10));
        let channel = endpoint.connect_lazy();
        let channels = BackendChannels {
            search: SearchServiceClient::new(channel.clone())
                .send_compressed(CompressionEncoding::Zstd)
                .accept_compressed(CompressionEncoding::Zstd)
                .accept_compressed(CompressionEncoding::Gzip)
                .max_decoding_message_size(self.backend_max_decode)
                .max_encoding_message_size(self.backend_max_encode),
            index: IndexServiceClient::new(channel)
                .send_compressed(CompressionEncoding::Zstd)
                .accept_compressed(CompressionEncoding::Zstd)
                .accept_compressed(CompressionEncoding::Gzip)
                .max_decoding_message_size(self.backend_max_decode)
                .max_encoding_message_size(self.backend_max_encode),
            search_permits: Arc::new(Semaphore::new(self.backend_max_searches)),
        };
        self.channels
            .lock()
            .entry(addr.to_string())
            .or_insert(channels.clone());
        Ok(channels)
    }

    /// Drop cached channels for addresses no longer discovered, so stale
    /// connections do not linger and a recovered backend reconnects fresh.
    pub fn retain(&self, live_addrs: &[&str]) {
        self.channels
            .lock()
            .retain(|addr, _| live_addrs.contains(&addr.as_str()));
    }
}

/// Parse the incoming `grpc-timeout` header (spec: 1-8 ASCII digits + one of
/// H/M/S/m/u/n). Returns the remaining budget the broker should propagate
/// downstream, epsilon-shaved with a floor. Absent or malformed → None: the
/// outbound RPC carries no deadline, which is what keeps untimed index-builder
/// channels and 24h admin Reorder/ForceMerge calls working through the broker.
pub fn forward_timeout(metadata: &MetadataMap) -> Option<Duration> {
    let raw = metadata.get("grpc-timeout")?.to_str().ok()?;
    if raw.len() < 2 || raw.len() > 9 {
        return None;
    }
    let (digits, unit) = raw.split_at(raw.len() - 1);
    let value: u64 = digits.parse().ok()?;
    let timeout = match unit {
        "H" => Duration::from_secs(value.checked_mul(3600)?),
        "M" => Duration::from_secs(value.checked_mul(60)?),
        "S" => Duration::from_secs(value),
        "m" => Duration::from_millis(value),
        "u" => Duration::from_micros(value),
        "n" => Duration::from_nanos(value),
        _ => return None,
    };
    Some(timeout.saturating_sub(DEADLINE_EPSILON).max(DEADLINE_FLOOR))
}

/// The admission-rejection message, byte-identical to hermes-server's so
/// client retry logic (which matches on the status) behaves the same whether
/// the broker or a backend rejected the request.
pub fn capacity_exhausted() -> Status {
    Status::resource_exhausted("Search capacity is full; retry with backoff")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn md(value: &str) -> MetadataMap {
        let mut m = MetadataMap::new();
        m.insert("grpc-timeout", value.parse().unwrap());
        m
    }

    #[test]
    fn parses_grpc_timeout_units() {
        assert_eq!(
            forward_timeout(&md("2S")),
            Some(Duration::from_millis(1950))
        );
        assert_eq!(
            forward_timeout(&md("500m")),
            Some(Duration::from_millis(450))
        );
        assert_eq!(
            forward_timeout(&md("1H")),
            Some(Duration::from_secs(3600) - DEADLINE_EPSILON)
        );
        assert_eq!(
            forward_timeout(&md("3M")),
            Some(Duration::from_secs(180) - DEADLINE_EPSILON)
        );
    }

    #[test]
    fn tiny_deadlines_clamp_to_floor_not_zero() {
        assert_eq!(forward_timeout(&md("10m")), Some(DEADLINE_FLOOR));
        assert_eq!(forward_timeout(&md("1u")), Some(DEADLINE_FLOOR));
        assert_eq!(forward_timeout(&md("1n")), Some(DEADLINE_FLOOR));
    }

    #[test]
    fn absent_or_malformed_header_means_no_deadline() {
        assert_eq!(forward_timeout(&MetadataMap::new()), None);
        assert_eq!(forward_timeout(&md("")), None);
        assert_eq!(forward_timeout(&md("S")), None);
        assert_eq!(forward_timeout(&md("12X")), None);
        assert_eq!(forward_timeout(&md("123456789S")), None);
    }
}
