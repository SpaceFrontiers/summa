//! Broker-only control surface (`hermes.broker.BrokerService`): topology and
//! backend inspection for operators and migration runbooks.
//!
//! `GetTopology` fans out live `GetIndexInfo` calls (bounded by a short
//! per-call deadline) so replica doc counts are real, not cached — the
//! migration runbook compares them before deleting a moved index's old copy.

use std::sync::Arc;
use std::time::{Duration, Instant};

use tonic::{Request, Response, Status};

use crate::context::BrokerContext;
use crate::proto::broker::broker_service_server::BrokerService;
use crate::proto::broker::{
    BackendState, GetBackendsRequest, GetBackendsResponse, GetTopologyRequest, GetTopologyResponse,
    IndexTopology, Partition, RefreshTopologyRequest, RefreshTopologyResponse, ReplicaState,
};
use crate::proto::hermes::GetIndexInfoRequest;
use crate::topology::{Backend, Role, TopologySnapshot};

/// Deadline for each live GetIndexInfo call made on behalf of GetTopology.
const INFO_TIMEOUT: Duration = Duration::from_secs(3);

pub struct BrokerAdminService {
    pub ctx: Arc<BrokerContext>,
}

/// Role as routing sees it: explicit label wins; the sole member of a shard
/// is the implicit master; anything else is unlabeled.
fn effective_role(snapshot: &TopologySnapshot, backend: &Backend) -> &'static str {
    match backend.endpoint.role {
        Some(Role::Master) => "master",
        Some(Role::Follower) => "follower",
        None => {
            let sole = snapshot
                .shards
                .get(&backend.endpoint.shard)
                .is_some_and(|group| group.members.len() == 1);
            if sole { "master" } else { "unlabeled" }
        }
    }
}

fn index_map_age_ms(backend: &Backend) -> u64 {
    backend
        .know
        .last_index_refresh
        .map(|t| t.elapsed().as_millis() as u64)
        .unwrap_or(u64::MAX)
}

#[tonic::async_trait]
impl BrokerService for BrokerAdminService {
    async fn get_topology(
        &self,
        request: Request<GetTopologyRequest>,
    ) -> Result<Response<GetTopologyResponse>, Status> {
        let filter = request.into_inner().index_name;
        let snapshot = self.ctx.snapshot.load_full();

        let mut indexes = Vec::new();
        for (name, route) in snapshot
            .indexes
            .iter()
            .filter(|(name, _)| filter.is_empty() || *name == &filter)
        {
            let mut partitions = Vec::new();
            for shard in &route.shards {
                let Some(group) = snapshot.shards.get(shard) else {
                    continue;
                };
                let replicas = futures::future::join_all(group.members.iter().filter_map(|id| {
                    let backend = snapshot.backends.get(id)?;
                    if !backend.know.indexes.iter().any(|n| n == name) {
                        return None;
                    }
                    Some(self.replica_state(&snapshot, backend, name))
                }))
                .await;
                partitions.push(Partition {
                    shard_id: shard.0.clone(),
                    replicas,
                });
            }
            indexes.push(IndexTopology {
                index_name: name.clone(),
                partitions,
                merge_policy: "passthrough".to_string(),
                primary_key_field: String::new(),
                ambiguous: route.ambiguous(),
            });
        }
        Ok(Response::new(GetTopologyResponse { indexes }))
    }

    async fn get_backends(
        &self,
        _request: Request<GetBackendsRequest>,
    ) -> Result<Response<GetBackendsResponse>, Status> {
        let snapshot = self.ctx.snapshot.load_full();
        let backends = snapshot
            .backends
            .values()
            .map(|backend| BackendState {
                backend_id: backend.endpoint.id.0.clone(),
                address: backend.endpoint.addr.clone(),
                shard_id: backend.endpoint.shard.0.clone(),
                role: effective_role(&snapshot, backend).to_string(),
                health: backend.know.health.as_str().to_string(),
                indexes: backend.know.indexes.clone(),
                index_map_age_ms: index_map_age_ms(backend),
                consecutive_failures: backend.know.consecutive_failures,
                ready: backend.endpoint.ready,
            })
            .collect();
        Ok(Response::new(GetBackendsResponse { backends }))
    }

    async fn refresh_topology(
        &self,
        _request: Request<RefreshTopologyRequest>,
    ) -> Result<Response<RefreshTopologyResponse>, Status> {
        let backends = self.ctx.snapshot.load().backends.len() as u32;
        self.ctx.refresh.notify_one();
        Ok(Response::new(RefreshTopologyResponse {
            backends_polled: backends,
        }))
    }
}

impl BrokerAdminService {
    async fn replica_state(
        &self,
        snapshot: &TopologySnapshot,
        backend: &Backend,
        index_name: &str,
    ) -> ReplicaState {
        let mut state = ReplicaState {
            backend_id: backend.endpoint.id.0.clone(),
            address: backend.endpoint.addr.clone(),
            role: effective_role(snapshot, backend).to_string(),
            health: backend.know.health.as_str().to_string(),
            num_docs: 0,
            num_segments: 0,
            index_map_age_ms: index_map_age_ms(backend),
        };
        if !backend.routable() {
            return state;
        }
        let Ok(channels) = self.ctx.pool.get(&backend.endpoint.addr) else {
            return state;
        };
        let mut request = Request::new(GetIndexInfoRequest {
            index_name: index_name.to_string(),
        });
        request.set_timeout(INFO_TIMEOUT);
        let started = Instant::now();
        match channels.search.clone().get_index_info(request).await {
            Ok(response) => {
                let info = response.into_inner();
                state.num_docs = info.num_docs;
                state.num_segments = info.num_segments;
            }
            Err(status) => {
                log::debug!(
                    "GetTopology: GetIndexInfo({index_name}) on {} failed after {:?}: {}",
                    backend.endpoint.id.0,
                    started.elapsed(),
                    status.code()
                );
            }
        }
        state
    }
}
