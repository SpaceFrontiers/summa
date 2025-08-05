//! Per-backend index-map poller and health state machine.
//!
//! One loop owns all mutable topology state: it polls `ListIndexes` on every
//! discovered, ready backend (15s steady state; 5s while a backend is
//! Suspect/Evicted), applies the health transitions, and publishes immutable
//! `TopologySnapshot`s. Health lifecycle:
//!
//! Healthy --failure--> Suspect --grace elapsed--> Evicted
//! Suspect --success--> Healthy
//! Evicted --2 consecutive successes--> Healthy
//!
//! A Suspect backend keeps serving off its last-known index map (better a
//! possibly-stale answer than none while its siblings are also down); an
//! Evicted backend drops out of every route.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arc_swap::ArcSwap;
use log::{info, warn};
use tokio::sync::{Notify, watch};
use tonic::Status;

use crate::client::ClientPool;
use crate::metrics as m;
use crate::placement::PlacementRules;
use crate::proto::hermes::ListIndexesRequest;
use crate::topology::{BackendId, BackendKnowledge, DiscoveredEndpoint, Health, TopologySnapshot};

#[derive(Clone, Copy, Debug)]
pub struct PollerConfig {
    /// Steady-state interval between ListIndexes polls of a Healthy backend.
    pub poll_interval: Duration,
    /// Probe interval for Suspect/Evicted backends (half-open recovery).
    pub probe_interval: Duration,
    /// How long a backend may stay Suspect before eviction.
    pub grace: Duration,
    /// Per-poll ListIndexes deadline.
    pub list_timeout: Duration,
}

#[derive(Debug, Default)]
struct PollState {
    know: BackendKnowledge,
    suspect_since: Option<Instant>,
    consecutive_successes: u64,
    last_poll: Option<Instant>,
}

/// Successes required to bring an Evicted backend back into rotation.
const EVICTION_RECOVERY_SUCCESSES: u64 = 2;

fn apply_poll_outcome(
    state: &mut PollState,
    outcome: Result<Vec<String>, Status>,
    grace: Duration,
    id: &BackendId,
) {
    let now = Instant::now();
    match outcome {
        Ok(mut indexes) => {
            indexes.sort();
            state.know.indexes = indexes;
            state.know.last_index_refresh = Some(now);
            state.know.consecutive_failures = 0;
            match state.know.health {
                Health::Healthy => {}
                Health::Suspect => {
                    info!("backend {} recovered (suspect -> healthy)", id.0);
                    state.know.health = Health::Healthy;
                    state.suspect_since = None;
                }
                Health::Evicted => {
                    state.consecutive_successes += 1;
                    if state.consecutive_successes >= EVICTION_RECOVERY_SUCCESSES {
                        info!(
                            "backend {} recovered (evicted -> healthy after {} probes)",
                            id.0, state.consecutive_successes
                        );
                        state.know.health = Health::Healthy;
                        state.suspect_since = None;
                        state.consecutive_successes = 0;
                    }
                }
            }
        }
        Err(status) => {
            state.know.consecutive_failures += 1;
            state.consecutive_successes = 0;
            match state.know.health {
                Health::Healthy => {
                    warn!(
                        "backend {} poll failed ({}); healthy -> suspect",
                        id.0,
                        status.code()
                    );
                    state.know.health = Health::Suspect;
                    state.suspect_since = Some(now);
                }
                Health::Suspect => {
                    let since = *state.suspect_since.get_or_insert(now);
                    if now.duration_since(since) >= grace {
                        warn!(
                            "backend {} evicted after {:?} suspect ({} consecutive failures)",
                            id.0,
                            now.duration_since(since),
                            state.know.consecutive_failures
                        );
                        state.know.health = Health::Evicted;
                    }
                }
                Health::Evicted => {}
            }
        }
    }
}

async fn poll_backend(
    pool: &ClientPool,
    endpoint: &DiscoveredEndpoint,
    list_timeout: Duration,
) -> Result<Vec<String>, Status> {
    let channels = pool.get(&endpoint.addr)?;
    let mut request = tonic::Request::new(ListIndexesRequest {});
    request.set_timeout(list_timeout);
    let response = channels.index.clone().list_indexes(request).await?;
    Ok(response.into_inner().index_names)
}

fn publish_gauges(snapshot: &TopologySnapshot) {
    let mut healthy = 0.0;
    let mut suspect = 0.0;
    let mut evicted = 0.0;
    let mut unready = 0.0;
    for backend in snapshot.backends.values() {
        if !backend.endpoint.ready {
            unready += 1.0;
        }
        let value = match backend.know.health {
            Health::Healthy => {
                healthy += 1.0;
                1.0
            }
            Health::Suspect => {
                suspect += 1.0;
                0.5
            }
            Health::Evicted => {
                evicted += 1.0;
                0.0
            }
        };
        metrics::gauge!(
            m::BACKEND_HEALTHY,
            "backend" => backend.endpoint.id.0.clone(),
            "shard" => backend.endpoint.shard.0.clone(),
        )
        .set(value);
        if let Some(refreshed) = backend.know.last_index_refresh {
            metrics::gauge!(m::INDEX_MAP_AGE, "backend" => backend.endpoint.id.0.clone())
                .set(refreshed.elapsed().as_secs_f64());
        }
    }
    metrics::gauge!(m::BACKENDS, "state" => "healthy").set(healthy);
    metrics::gauge!(m::BACKENDS, "state" => "suspect").set(suspect);
    metrics::gauge!(m::BACKENDS, "state" => "evicted").set(evicted);
    metrics::gauge!(m::BACKENDS, "state" => "unready").set(unready);
}

#[allow(clippy::too_many_arguments)]
pub fn spawn_poller(
    mut endpoints_rx: watch::Receiver<Vec<DiscoveredEndpoint>>,
    pool: Arc<ClientPool>,
    placement: Arc<PlacementRules>,
    snapshot: Arc<ArcSwap<TopologySnapshot>>,
    health_reporter: tonic_health::server::HealthReporter,
    cfg: PollerConfig,
    mut shutdown: watch::Receiver<bool>,
) -> (tokio::task::JoinHandle<()>, Arc<Notify>) {
    let refresh = Arc::new(Notify::new());
    let refresh_task = Arc::clone(&refresh);
    let handle = tokio::spawn(async move {
        let mut states: HashMap<BackendId, PollState> = HashMap::new();
        let mut known_ids: Vec<BackendId> = Vec::new();
        let mut force = true;
        loop {
            let endpoints: Vec<DiscoveredEndpoint> = endpoints_rx.borrow_and_update().clone();

            // Discovery churn accounting + state/channel pruning.
            let current_ids: Vec<BackendId> = endpoints.iter().map(|e| e.id.clone()).collect();
            for id in &current_ids {
                if !known_ids.contains(id) {
                    info!("backend {} discovered", id.0);
                    metrics::counter!(m::DISCOVERY_EVENTS, "type" => "added").increment(1);
                }
            }
            for id in &known_ids {
                if !current_ids.contains(id) {
                    info!("backend {} removed from discovery", id.0);
                    metrics::counter!(m::DISCOVERY_EVENTS, "type" => "removed").increment(1);
                }
            }
            known_ids = current_ids;
            states.retain(|id, _| endpoints.iter().any(|e| &e.id == id));
            let live_addrs: Vec<&str> = endpoints.iter().map(|e| e.addr.as_str()).collect();
            pool.retain(&live_addrs);

            // Poll whichever ready backends are due.
            let now = Instant::now();
            let due: Vec<DiscoveredEndpoint> = endpoints
                .iter()
                .filter(|e| e.ready)
                .filter(|e| {
                    force
                        || match states.get(&e.id) {
                            None => true,
                            Some(state) => {
                                let interval = if state.know.health == Health::Healthy {
                                    cfg.poll_interval
                                } else {
                                    cfg.probe_interval
                                };
                                state
                                    .last_poll
                                    .is_none_or(|t| now.duration_since(t) >= interval)
                            }
                        }
                })
                .cloned()
                .collect();
            force = false;

            let results = futures::future::join_all(due.iter().map(|ep| {
                let pool = Arc::clone(&pool);
                async move {
                    let outcome = poll_backend(&pool, ep, cfg.list_timeout).await;
                    (ep.id.clone(), outcome)
                }
            }))
            .await;
            for (id, outcome) in results {
                let state = states.entry(id.clone()).or_default();
                state.last_poll = Some(Instant::now());
                apply_poll_outcome(state, outcome, cfg.grace, &id);
            }

            // Publish the new immutable view.
            let knowledge: HashMap<BackendId, BackendKnowledge> = states
                .iter()
                .map(|(id, state)| (id.clone(), state.know.clone()))
                .collect();
            let assembled = TopologySnapshot::assemble(&endpoints, &knowledge, &placement);
            publish_gauges(&assembled);
            let serving = assembled.any_healthy();
            snapshot.store(Arc::new(assembled));
            let status = if serving {
                tonic_health::ServingStatus::Serving
            } else {
                tonic_health::ServingStatus::NotServing
            };
            health_reporter.set_service_status("", status).await;

            tokio::select! {
                _ = tokio::time::sleep(Duration::from_secs(1)) => {}
                _ = refresh_task.notified() => {
                    force = true;
                }
                changed = endpoints_rx.changed() => {
                    if changed.is_err() {
                        warn!("discovery channel closed; poller exiting");
                        break;
                    }
                    force = true;
                }
                _ = shutdown.changed() => {
                    if *shutdown.borrow() {
                        break;
                    }
                }
            }
        }
    });
    (handle, refresh)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id() -> BackendId {
        BackendId("a".to_string())
    }

    fn ok(indexes: &[&str]) -> Result<Vec<String>, Status> {
        Ok(indexes.iter().map(|s| s.to_string()).collect())
    }

    fn fail() -> Result<Vec<String>, Status> {
        Err(Status::unavailable("down"))
    }

    #[test]
    fn first_success_promotes_new_backend_to_healthy() {
        let mut state = PollState::default();
        assert_eq!(state.know.health, Health::Suspect);
        apply_poll_outcome(&mut state, ok(&["b", "a"]), Duration::from_secs(60), &id());
        assert_eq!(state.know.health, Health::Healthy);
        assert_eq!(state.know.indexes, vec!["a", "b"]); // sorted
    }

    #[test]
    fn failure_demotes_healthy_to_suspect_then_grace_evicts() {
        let mut state = PollState::default();
        apply_poll_outcome(&mut state, ok(&["a"]), Duration::from_secs(60), &id());
        apply_poll_outcome(&mut state, fail(), Duration::from_secs(60), &id());
        assert_eq!(state.know.health, Health::Suspect);
        // Stale knowledge is retained through the grace window.
        assert_eq!(state.know.indexes, vec!["a"]);
        // Grace of zero: next failure evicts immediately.
        apply_poll_outcome(&mut state, fail(), Duration::ZERO, &id());
        assert_eq!(state.know.health, Health::Evicted);
    }

    #[test]
    fn eviction_recovery_needs_two_consecutive_successes() {
        let mut state = PollState::default();
        apply_poll_outcome(&mut state, ok(&["a"]), Duration::ZERO, &id());
        apply_poll_outcome(&mut state, fail(), Duration::ZERO, &id());
        apply_poll_outcome(&mut state, fail(), Duration::ZERO, &id());
        assert_eq!(state.know.health, Health::Evicted);

        apply_poll_outcome(&mut state, ok(&["a"]), Duration::ZERO, &id());
        assert_eq!(state.know.health, Health::Evicted); // one is not enough
        apply_poll_outcome(&mut state, fail(), Duration::ZERO, &id());
        apply_poll_outcome(&mut state, ok(&["a"]), Duration::ZERO, &id());
        assert_eq!(state.know.health, Health::Evicted); // failure reset the streak
        apply_poll_outcome(&mut state, ok(&["a"]), Duration::ZERO, &id());
        assert_eq!(state.know.health, Health::Healthy);
    }

    #[test]
    fn suspect_recovers_on_single_success() {
        let mut state = PollState::default();
        apply_poll_outcome(&mut state, ok(&["a"]), Duration::from_secs(60), &id());
        apply_poll_outcome(&mut state, fail(), Duration::from_secs(60), &id());
        assert_eq!(state.know.health, Health::Suspect);
        apply_poll_outcome(&mut state, ok(&["a", "b"]), Duration::from_secs(60), &id());
        assert_eq!(state.know.health, Health::Healthy);
        assert_eq!(state.know.indexes, vec!["a", "b"]);
    }
}
