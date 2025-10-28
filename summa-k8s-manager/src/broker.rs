//! Client of a cluster's broker: live topology of its index and index
//! creation.

use std::time::Duration;

use anyhow::Context as _;
use serde::Serialize;
use tonic::transport::{Channel, Endpoint};

use summa_broker::proto;
use summa_broker::proto::broker::broker_service_client::BrokerServiceClient;
use summa_broker::proto::summa::index_service_client::IndexServiceClient;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(3);
/// `GetTopology` asks every shard for `GetIndexInfo` with a 3 s deadline.
const TOPOLOGY_TIMEOUT: Duration = Duration::from_secs(15);
/// Creating an index opens it on every shard.
const CREATE_TIMEOUT: Duration = Duration::from_secs(120);

/// One shard of the index as the broker sees it.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ShardStats {
    pub shard: String,
    pub health: String,
    pub num_docs: u32,
    pub num_segments: u32,
}

/// Endpoint of a cluster's broker, from a template with `{cluster}` and
/// `{namespace}`.
pub fn endpoint(template: &str, cluster: &str, namespace: &str) -> String {
    template
        .replace("{cluster}", cluster)
        .replace("{namespace}", namespace)
}

async fn channel(address: &str, timeout: Duration) -> anyhow::Result<Channel> {
    Endpoint::from_shared(format!("http://{address}"))
        .with_context(|| format!("invalid broker address '{address}'"))?
        .connect_timeout(CONNECT_TIMEOUT)
        .timeout(timeout)
        .connect()
        .await
        .with_context(|| format!("cannot connect to broker {address}"))
}

/// Shards hosting `index`, with their statistics. Empty when no shard has
/// the index.
pub async fn index_shards(address: &str, index: &str) -> anyhow::Result<Vec<ShardStats>> {
    let response = BrokerServiceClient::new(channel(address, TOPOLOGY_TIMEOUT).await?)
        .get_topology(proto::broker::GetTopologyRequest {
            index_name: index.to_string(),
        })
        .await
        .with_context(|| format!("GetTopology on broker {address}"))?;
    Ok(response
        .into_inner()
        .indexes
        .into_iter()
        .filter(|topology| topology.index_name == index)
        .flat_map(|topology| topology.partitions)
        .map(|partition| {
            // One replica per shard; report the healthiest view.
            let replica = partition.replicas.into_iter().max_by_key(|r| r.num_docs);
            ShardStats {
                shard: partition.shard_id,
                health: replica
                    .as_ref()
                    .map_or("missing".into(), |r| r.health.clone()),
                num_docs: replica.as_ref().map_or(0, |r| r.num_docs),
                num_segments: replica.as_ref().map_or(0, |r| r.num_segments),
            }
        })
        .collect())
}

/// Create `index` on every shard of the broker's placement rule.
pub async fn create_index(address: &str, index: &str, schema: &str) -> anyhow::Result<()> {
    IndexServiceClient::new(channel(address, CREATE_TIMEOUT).await?)
        .create_index(proto::summa::CreateIndexRequest {
            index_name: index.to_string(),
            schema: schema.to_string(),
        })
        .await
        .with_context(|| format!("CreateIndex {index} on broker {address}"))?;
    Ok(())
}
