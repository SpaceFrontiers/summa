//! Shared state handed to every gRPC service implementation.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use arc_swap::ArcSwap;
use tokio::sync::{Notify, Semaphore};
use tonic::Status;

use crate::client::ClientPool;
use crate::placement::PlacementRules;
use crate::topology::TopologySnapshot;

pub struct BrokerContext {
    pub snapshot: Arc<ArcSwap<TopologySnapshot>>,
    pub pool: Arc<ClientPool>,
    pub placement: Arc<PlacementRules>,
    /// Wakes the poller for an immediate re-poll (after broker-issued
    /// CreateIndex/DeleteIndex, and for the RefreshTopology admin RPC).
    pub refresh: Arc<Notify>,
    /// Optional broker-global search admission on top of the per-backend
    /// semaphores. None = per-backend caps only.
    pub global_search_permits: Option<Arc<Semaphore>>,
    /// Spreads reads across replicas of a shard.
    pub read_rotation: AtomicUsize,
    /// Set when the shutdown signal fires; new RPCs are refused while tonic
    /// drains in-flight ones.
    pub shutting_down: Arc<AtomicBool>,
}

impl BrokerContext {
    pub fn check_admission(&self) -> Result<(), Status> {
        if self.shutting_down.load(Ordering::Relaxed) {
            return Err(Status::unavailable("Hermes broker is shutting down"));
        }
        Ok(())
    }

    pub fn next_rotation(&self) -> usize {
        self.read_rotation.fetch_add(1, Ordering::Relaxed)
    }
}

/// Stable snake_case label for a gRPC status code (metric label values).
pub fn code_label(code: tonic::Code) -> &'static str {
    match code {
        tonic::Code::Ok => "ok",
        tonic::Code::Cancelled => "cancelled",
        tonic::Code::Unknown => "unknown",
        tonic::Code::InvalidArgument => "invalid_argument",
        tonic::Code::DeadlineExceeded => "deadline_exceeded",
        tonic::Code::NotFound => "not_found",
        tonic::Code::AlreadyExists => "already_exists",
        tonic::Code::PermissionDenied => "permission_denied",
        tonic::Code::ResourceExhausted => "resource_exhausted",
        tonic::Code::FailedPrecondition => "failed_precondition",
        tonic::Code::Aborted => "aborted",
        tonic::Code::OutOfRange => "out_of_range",
        tonic::Code::Unimplemented => "unimplemented",
        tonic::Code::Internal => "internal",
        tonic::Code::Unavailable => "unavailable",
        tonic::Code::DataLoss => "data_loss",
        tonic::Code::Unauthenticated => "unauthenticated",
    }
}
