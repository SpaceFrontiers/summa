//! `SearchCluster` custom resource: the desired shards, broker and placement
//! of one Summa cluster. See `docs/summa-k8s-manager.md`.

use k8s_openapi::api::core::v1::{EnvVar, ResourceRequirements};
use k8s_openapi::apimachinery::pkg::apis::meta::v1::Condition;
use kube::CustomResource;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(CustomResource, Clone, Debug, Default, Deserialize, Serialize, JsonSchema, PartialEq)]
#[kube(
    group = "summa.spacefrontiers.org",
    version = "v1alpha1",
    kind = "SearchCluster",
    namespaced,
    status = "SearchClusterStatus",
    printcolumn = r#"{"name":"Ready","type":"string","jsonPath":".status.conditions[?(@.type==\"Ready\")].status"}"#,
    printcolumn = r#"{"name":"Reason","type":"string","jsonPath":".status.conditions[?(@.type==\"Ready\")].reason"}"#
)]
#[serde(rename_all = "camelCase")]
pub struct SearchClusterSpec {
    /// summa-server image; the broker runs the `summa-broker` binary of the
    /// same image.
    pub image: String,
    /// `Always`, `IfNotPresent` or `Never`; omitted uses the Kubernetes
    /// default for the tag.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_pull_policy: Option<String>,
    /// Defaults for every shard replica.
    #[serde(default)]
    pub server: ServerSpec,
    #[serde(default)]
    pub shards: Vec<ShardSpec>,
    #[serde(default)]
    pub broker: BrokerSpec,
    #[serde(default)]
    pub placement: PlacementSpec,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ServerSpec {
    /// summa-server arguments after the operator's `--addr`, `--data-dir`
    /// and `--cache-dir`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub env: Vec<EnvVar>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resources: Option<ResourceRequirements>,
    /// Add CAP_IPC_LOCK, needed by `--pin-mode mlock`.
    #[serde(default)]
    pub ipc_lock: bool,
    /// Default 600: a shard commits admitted documents before it stops.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub termination_grace_period_seconds: Option<i64>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ShardSpec {
    /// Value of the `summa.spacefrontiers.org/shard-id` pod label.
    pub id: String,
    /// Appended to `server.args`; must not repeat a flag set there.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<String>,
    /// Replaces `server.resources` for this shard.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resources: Option<ResourceRequirements>,
    /// Exactly one until summa-server supports follower replication.
    #[serde(default)]
    pub replicas: Vec<ReplicaSpec>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ReplicaSpec {
    /// Deployment name; default `<cluster>-shard-<id>`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Node (`kubernetes.io/hostname`) the replica runs on.
    pub node: String,
    /// Existing host directory holding the shard's indexes.
    pub data_path: String,
    /// Existing host directory for the shard's cache.
    pub cache_path: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct BrokerSpec {
    /// Deployment and Service name; default `<cluster>-broker`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Broker image; default `spec.image`. Lets the broker upgrade without
    /// restarting the shards.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image: Option<String>,
    #[serde(default = "default_broker_replicas")]
    pub replicas: i32,
    /// Needs get/list/watch on pods in the namespace.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_account_name: Option<String>,
    /// In-flight searches admitted per backend across all broker replicas;
    /// each broker gets `backendMaxSearches / replicas`.
    #[serde(default = "default_backend_max_searches")]
    pub backend_max_searches: u32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub env: Vec<EnvVar>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resources: Option<ResourceRequirements>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub termination_grace_period_seconds: Option<i64>,
}

impl Default for BrokerSpec {
    fn default() -> Self {
        Self {
            name: None,
            image: None,
            replicas: default_broker_replicas(),
            service_account_name: None,
            backend_max_searches: default_backend_max_searches(),
            args: Vec::new(),
            env: Vec::new(),
            resources: None,
            termination_grace_period_seconds: None,
        }
    }
}

fn default_broker_replicas() -> i32 {
    1
}

fn default_backend_max_searches() -> u32 {
    16
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct PlacementSpec {
    #[serde(default)]
    pub default: PlacementDefault,
    /// First match wins; rule order is partition order.
    #[serde(default)]
    pub rules: Vec<PlacementRuleSpec>,
}

/// Where `CreateIndex` lands when no rule matches.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum PlacementDefault {
    /// The shard hosting the fewest indexes.
    #[default]
    Single,
    /// Refuse the CreateIndex.
    Reject,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct PlacementRuleSpec {
    /// Index-name glob (`*` matches any run of characters).
    pub pattern: String,
    /// One shard pins the index; several partition it in this order.
    pub partitions: Vec<PartitionRef>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct PartitionRef {
    pub shard: String,
    /// Share of rows (default 1); only for rules with several partitions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight: Option<u32>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SearchClusterStatus {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observed_generation: Option<i64>,
    #[serde(default)]
    pub shards: Vec<ShardStatus>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub broker: Option<BrokerStatus>,
    /// `--placement` arguments of the running broker Deployment.
    #[serde(default)]
    pub placement: Vec<String>,
    #[serde(default)]
    pub conditions: Vec<Condition>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ShardStatus {
    pub id: String,
    #[serde(default)]
    pub replicas: Vec<ReplicaStatus>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ReplicaStatus {
    pub name: String,
    pub node: String,
    pub ready: bool,
    /// Why the replica is not running its current spec, if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct BrokerStatus {
    pub name: String,
    pub replicas: i32,
    pub ready_replicas: i32,
}
