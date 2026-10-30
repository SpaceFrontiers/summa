//! Client of the broker's `summa.broker.BrokerService`: live topology and
//! backends for safety checks and the console.

use std::time::Duration;

use anyhow::Context as _;
use serde::Serialize;
use tonic::transport::Endpoint;

pub mod proto {
    tonic::include_proto!("summa.broker");
}

use proto::broker_service_client::BrokerServiceClient;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(3);
/// `GetTopology` asks every replica for `GetIndexInfo` with a 3 s deadline.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(15);

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Backend {
    pub backend_id: String,
    pub address: String,
    pub shard_id: String,
    pub role: String,
    pub health: String,
    pub indexes: Vec<String>,
    pub index_map_age_ms: u64,
    pub consecutive_failures: u64,
    pub ready: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct IndexTopology {
    pub index_name: String,
    pub partitions: Vec<Partition>,
    pub merge_policy: String,
    pub primary_key_field: String,
    pub ambiguous: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Partition {
    pub shard_id: String,
    pub weight: u32,
    pub replicas: Vec<Replica>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Replica {
    pub backend_id: String,
    pub role: String,
    pub health: String,
    pub num_docs: u32,
    pub num_segments: u32,
}

/// What the safety checks need to know about a live index.
#[derive(Debug, Clone)]
pub struct LiveIndex {
    pub name: String,
    pub partitioned: bool,
    /// Shards hosting a copy or a partition.
    pub hosts: Vec<String>,
}

impl From<&IndexTopology> for LiveIndex {
    fn from(index: &IndexTopology) -> Self {
        Self {
            name: index.index_name.clone(),
            partitioned: index.merge_policy != "passthrough",
            hosts: index
                .partitions
                .iter()
                .map(|p| p.shard_id.clone())
                .collect(),
        }
    }
}

#[derive(Clone)]
pub struct BrokerClient {
    address: String,
}

impl BrokerClient {
    pub fn new(address: impl Into<String>) -> Self {
        Self {
            address: address.into(),
        }
    }

    pub fn address(&self) -> &str {
        &self.address
    }

    async fn client(&self) -> anyhow::Result<BrokerServiceClient<tonic::transport::Channel>> {
        let channel = Endpoint::from_shared(format!("http://{}", self.address))
            .with_context(|| format!("invalid broker address '{}'", self.address))?
            .connect_timeout(CONNECT_TIMEOUT)
            .timeout(REQUEST_TIMEOUT)
            .connect()
            .await
            .with_context(|| format!("cannot connect to broker {}", self.address))?;
        Ok(BrokerServiceClient::new(channel))
    }

    pub async fn topology(&self) -> anyhow::Result<Vec<IndexTopology>> {
        let response = self
            .client()
            .await?
            .get_topology(proto::GetTopologyRequest::default())
            .await
            .with_context(|| format!("GetTopology on broker {}", self.address))?;
        Ok(response
            .into_inner()
            .indexes
            .into_iter()
            .map(|index| IndexTopology {
                index_name: index.index_name,
                merge_policy: index.merge_policy,
                primary_key_field: index.primary_key_field,
                ambiguous: index.ambiguous,
                partitions: index
                    .partitions
                    .into_iter()
                    .map(|partition| Partition {
                        shard_id: partition.shard_id,
                        weight: partition.weight,
                        replicas: partition
                            .replicas
                            .into_iter()
                            .map(|replica| Replica {
                                backend_id: replica.backend_id,
                                role: replica.role,
                                health: replica.health,
                                num_docs: replica.num_docs,
                                num_segments: replica.num_segments,
                            })
                            .collect(),
                    })
                    .collect(),
            })
            .collect())
    }

    pub async fn backends(&self) -> anyhow::Result<Vec<Backend>> {
        let response = self
            .client()
            .await?
            .get_backends(proto::GetBackendsRequest {})
            .await
            .with_context(|| format!("GetBackends on broker {}", self.address))?;
        Ok(response
            .into_inner()
            .backends
            .into_iter()
            .map(|backend| Backend {
                backend_id: backend.backend_id,
                address: backend.address,
                shard_id: backend.shard_id,
                role: backend.role,
                health: backend.health,
                indexes: backend.indexes,
                index_map_age_ms: backend.index_map_age_ms,
                consecutive_failures: backend.consecutive_failures,
                ready: backend.ready,
            })
            .collect())
    }

    pub async fn refresh(&self) -> anyhow::Result<u32> {
        let response = self
            .client()
            .await?
            .refresh_topology(proto::RefreshTopologyRequest {})
            .await
            .with_context(|| format!("RefreshTopology on broker {}", self.address))?;
        Ok(response.into_inner().backends_polled)
    }
}
