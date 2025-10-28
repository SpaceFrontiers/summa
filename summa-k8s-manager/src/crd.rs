//! `SearchCluster` custom resource: one Summa index with its shards and
//! broker. See `docs/summa-k8s-manager.md`.

use k8s_openapi::api::core::v1::{EnvVar, ResourceRequirements, Toleration};
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
    printcolumn = r#"{"name":"Routing","type":"string","jsonPath":".status.routing"}"#,
    printcolumn = r#"{"name":"Shard","type":"string","priority":1,"jsonPath":".spec.shard.image"}"#,
    printcolumn = r#"{"name":"Broker","type":"string","priority":1,"jsonPath":".spec.broker.image"}"#,
    printcolumn = r#"{"name":"Ready","type":"string","jsonPath":".status.conditions[?(@.type==\"Ready\")].status"}"#,
    printcolumn = r#"{"name":"Reason","type":"string","jsonPath":".status.conditions[?(@.type==\"Ready\")].reason"}"#
)]
#[serde(rename_all = "camelCase")]
pub struct SearchClusterSpec {
    /// Index schema (SDL). The operator creates the index from it once every
    /// shard serves; omit it for an index that already exists.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schema: Option<String>,
    /// Settings shared by every shard.
    pub shard: ShardTemplate,
    /// Shards in routing order. Count, order and weights are fixed once
    /// the index exists.
    pub shards: Vec<ShardSpec>,
    pub broker: BrokerSpec,
}

/// Image of a workload and how the kubelet pulls it.
#[derive(Clone, Debug, Default, Deserialize, Serialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ImageSpec {
    pub image: String,
    /// `Always`, `IfNotPresent` or `Never`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_pull_policy: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ShardTemplate {
    /// summa-server image of every shard.
    #[serde(flatten)]
    pub image: ImageSpec,
    /// summa-server arguments after the operator's `--addr`, `--data-dir`
    /// and `--cache-dir`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub env: Vec<EnvVar>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resources: Option<ResourceRequirements>,
    /// Add CAP_IPC_LOCK, needed by `--pin-mode mlock`.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub ipc_lock: bool,
    /// Default 600: a shard commits admitted documents before it stops.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub termination_grace_period_seconds: Option<i64>,
    /// Tolerations of shard pods, e.g. for dedicated data nodes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tolerations: Vec<Toleration>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ShardSpec {
    /// Node (`kubernetes.io/hostname`) the shard runs on.
    pub node: String,
    /// Host directory holding the shard's index; created if missing.
    pub data_path: String,
    /// Host cache directory; default `<dataPath>-cache`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_path: Option<String>,
    /// Share of rows (default 1); only for clusters with several shards.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight: Option<u32>,
    /// Appended to `shard.args`; must not repeat a flag set there.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<String>,
    /// Replaces `shard.resources` for this shard.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resources: Option<ResourceRequirements>,
    /// Scale the shard to zero for maintenance, keeping its routing.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub suspended: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct BrokerSpec {
    /// summa-broker image.
    #[serde(flatten)]
    pub image: ImageSpec,
    #[serde(default = "default_broker_replicas")]
    pub replicas: i32,
    /// Needs get/list/watch on pods in the namespace.
    #[serde(default = "default_service_account")]
    pub service_account_name: String,
    /// In-flight searches admitted per shard: every shard runs with
    /// `--max-concurrent-searches` set to it, and each broker replica admits
    /// `backendMaxSearches / replicas`.
    #[serde(default = "default_backend_max_searches")]
    pub backend_max_searches: u32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub env: Vec<EnvVar>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resources: Option<ResourceRequirements>,
}

impl Default for BrokerSpec {
    fn default() -> Self {
        Self {
            image: ImageSpec::default(),
            replicas: default_broker_replicas(),
            service_account_name: default_service_account(),
            backend_max_searches: default_backend_max_searches(),
            args: Vec::new(),
            env: Vec::new(),
            resources: None,
        }
    }
}

fn default_broker_replicas() -> i32 {
    1
}

fn default_service_account() -> String {
    "summa-broker".into()
}

fn default_backend_max_searches() -> u32 {
    16
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SearchClusterStatus {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observed_generation: Option<i64>,
    /// Placement rule of the index, fixed once the index exists.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routing: Option<String>,
    #[serde(default)]
    pub shards: Vec<ShardStatus>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub broker: Option<BrokerStatus>,
    #[serde(default)]
    pub conditions: Vec<Condition>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ShardStatus {
    pub ready: bool,
    /// Why the shard is not running its current spec, if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct BrokerStatus {
    pub ready_replicas: i32,
}

#[cfg(test)]
mod tests {
    use kube::CustomResourceExt;

    use super::SearchCluster;

    #[test]
    fn crd_requires_both_images_and_the_shard_placement() {
        let crd = serde_json::to_value(SearchCluster::crd()).unwrap();
        let spec = &crd["spec"]["versions"][0]["schema"]["openAPIV3Schema"]["properties"]["spec"];
        assert_eq!(
            spec["required"],
            serde_json::json!(["broker", "shard", "shards"])
        );
        assert_eq!(
            spec["properties"]["shard"]["required"],
            serde_json::json!(["image"])
        );
        assert_eq!(
            spec["properties"]["broker"]["required"],
            serde_json::json!(["image"])
        );
        assert_eq!(
            spec["properties"]["shards"]["items"]["required"],
            serde_json::json!(["dataPath", "node"])
        );
        // Broker defaults are applied by the API server at admission.
        let broker = &spec["properties"]["broker"]["properties"];
        assert_eq!(broker["replicas"]["default"], 1);
        assert_eq!(broker["backendMaxSearches"]["default"], 16);
        assert_eq!(broker["serviceAccountName"]["default"], "summa-broker");
    }
}
