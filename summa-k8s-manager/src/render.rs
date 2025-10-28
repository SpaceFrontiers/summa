//! Names, labels and workloads of a `SearchCluster`: one Deployment per
//! shard, the broker Deployment and the broker Service.

use k8s_openapi::api::apps::v1::Deployment;
use k8s_openapi::api::core::v1::Service;
use k8s_openapi::apimachinery::pkg::apis::meta::v1::OwnerReference;
use serde_json::{Value, json};
use summa_broker::placement_rules::{PartitionSpec as RulePartition, RuleSpec};

use crate::crd::{ImageSpec, SearchClusterSpec, ShardSpec};

pub const NAME_LABEL: &str = "app.kubernetes.io/name";
pub const COMPONENT_LABEL: &str = "app.kubernetes.io/component";
pub const MANAGED_BY_LABEL: &str = "app.kubernetes.io/managed-by";
pub const MANAGER: &str = "summa-k8s-manager";
pub const CLUSTER_LABEL: &str = "summa.spacefrontiers.org/cluster";
/// Broker discovery label (`summa-broker --shard-label`).
pub const SHARD_LABEL: &str = "summa.spacefrontiers.org/shard-id";
pub const SERVER_APP: &str = "summa-server";
pub const BROKER_APP: &str = "summa-broker";
pub const GRPC_PORT: i32 = 50051;
pub const METRICS_PORT: i32 = 9184;
const DEFAULT_GRACE_SECONDS: i64 = 600;

/// Flags rendered for every shard.
pub const SHARD_RESERVED_FLAGS: &[&str] = &[
    "--addr",
    "--data-dir",
    "--cache-dir",
    "--max-concurrent-searches",
];
/// Flags rendered for the broker.
pub const BROKER_RESERVED_FLAGS: &[&str] = &[
    "--addr",
    "--discovery",
    "--namespace",
    "--pod-selector",
    "--shard-label",
    "--backend-port",
    "--placement",
    "--placement-default",
    "--backend-max-searches",
];

/// Index served by a cluster: the cluster name with `-` as `_`.
pub fn index_name(cluster: &str) -> String {
    cluster.replace('-', "_")
}

pub fn shard_name(cluster: &str, shard: usize) -> String {
    format!("{cluster}-{shard}")
}

pub fn broker_name(cluster: &str) -> String {
    format!("{cluster}-broker")
}

pub fn cache_path(shard: &ShardSpec) -> String {
    shard
        .cache_path
        .clone()
        .unwrap_or_else(|| format!("{}-cache", shard.data_path.trim_end_matches('/')))
}

/// Host directories a shard opens: data, then cache.
pub fn shard_paths(shard: &ShardSpec) -> [String; 2] {
    [shard.data_path.clone(), cache_path(shard)]
}

/// Placement rule of the cluster's index: shards by ordinal, in order.
pub fn routing(cluster: &str, spec: &SearchClusterSpec) -> RuleSpec {
    RuleSpec {
        pattern: index_name(cluster),
        partitions: spec
            .shards
            .iter()
            .enumerate()
            .map(|(i, shard)| RulePartition {
                shard: i.to_string(),
                weight: shard.weight,
            })
            .collect(),
    }
}

/// Per-broker share of the per-shard search budget.
pub fn broker_backend_max_searches(spec: &SearchClusterSpec) -> u32 {
    let replicas = u32::try_from(spec.broker.replicas.max(1)).unwrap_or(1);
    (spec.broker.backend_max_searches / replicas).max(1)
}

/// Flag names in an argument list (`--flag value` and `--flag=value`).
pub fn flag_names(args: &[String]) -> impl Iterator<Item = &str> {
    args.iter()
        .filter(|arg| arg.starts_with("--"))
        .map(|arg| arg.split_once('=').map_or(arg.as_str(), |(flag, _)| flag))
}

/// The server admits exactly the searches all brokers together may send it.
pub fn shard_args(spec: &SearchClusterSpec, shard: &ShardSpec) -> Vec<String> {
    let mut args = strings(&[
        "--addr",
        &format!("0.0.0.0:{GRPC_PORT}"),
        "--data-dir",
        "/data",
        "--cache-dir",
        "/cache",
        "--max-concurrent-searches",
        &spec.broker.backend_max_searches.to_string(),
    ]);
    args.extend(spec.shard.args.iter().cloned());
    args.extend(shard.args.iter().cloned());
    args
}

pub fn broker_args(
    cluster: &str,
    namespace: &str,
    spec: &SearchClusterSpec,
    routing: &str,
) -> Vec<String> {
    let mut args = strings(&[
        "--addr",
        &format!("0.0.0.0:{GRPC_PORT}"),
        "--discovery",
        "kubernetes",
        "--namespace",
        namespace,
        "--pod-selector",
        &format!("{CLUSTER_LABEL}={cluster}"),
        "--placement",
        routing,
        "--placement-default",
        "reject",
        "--backend-max-searches",
        &broker_backend_max_searches(spec).to_string(),
    ]);
    args.extend(spec.broker.args.iter().cloned());
    args
}

fn strings(values: &[&str]) -> Vec<String> {
    values.iter().map(ToString::to_string).collect()
}

/// Identity of the cluster that owns the rendered objects.
pub struct Owner<'a> {
    pub cluster: &'a str,
    pub namespace: &'a str,
    pub reference: OwnerReference,
}

impl Owner<'_> {
    fn metadata(&self, name: &str, component: &str, app: &str) -> Value {
        json!({
            "name": name,
            "namespace": self.namespace,
            "labels": {
                NAME_LABEL: app,
                COMPONENT_LABEL: component,
                CLUSTER_LABEL: self.cluster,
                MANAGED_BY_LABEL: MANAGER,
            },
            "ownerReferences": [self.reference],
        })
    }
}

/// Container fields both workloads share; optional ones only when set.
fn container(
    name: &str,
    image: &ImageSpec,
    args: Vec<String>,
    env: &Value,
    resources: &Value,
) -> Value {
    let mut container = json!({
        "name": name,
        "image": image.image,
        "command": [name],
        "args": args,
        "ports": [
            { "name": "grpc", "containerPort": GRPC_PORT, "protocol": "TCP" },
            { "name": "metrics", "containerPort": METRICS_PORT, "protocol": "TCP" },
        ],
        "securityContext": { "allowPrivilegeEscalation": false },
        // gRPC health is NOT_SERVING until the process can route or serve.
        "readinessProbe": {
            "grpc": { "port": GRPC_PORT },
            "initialDelaySeconds": 5,
            "periodSeconds": 10,
        },
    });
    for (key, value) in [
        ("imagePullPolicy", &json!(image.image_pull_policy)),
        ("env", env),
        ("resources", resources),
    ] {
        if !(value.is_null() || value.as_array().is_some_and(Vec::is_empty)) {
            container[key] = value.clone();
        }
    }
    container
}

fn deployment(
    metadata: Value,
    pod_labels: Value,
    replicas: i32,
    strategy: Value,
    pod_spec: Value,
) -> Deployment {
    serde_json::from_value(json!({
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": metadata,
        "spec": {
            "replicas": replicas,
            "strategy": strategy,
            "selector": { "matchLabels": pod_labels },
            "template": { "metadata": { "labels": pod_labels }, "spec": pod_spec },
        },
    }))
    .expect("rendered Deployment is well-formed")
}

pub fn shard_deployment(owner: &Owner, spec: &SearchClusterSpec, index: usize) -> Deployment {
    let shard = &spec.shards[index];
    let pod_labels = json!({
        NAME_LABEL: SERVER_APP,
        CLUSTER_LABEL: owner.cluster,
        SHARD_LABEL: index.to_string(),
    });
    let mut shard_container = container(
        SERVER_APP,
        &spec.shard.image,
        shard_args(spec, shard),
        &json!(spec.shard.env),
        &json!(shard.resources.as_ref().or(spec.shard.resources.as_ref())),
    );
    shard_container["volumeMounts"] = json!([
        { "name": "data", "mountPath": "/data" },
        { "name": "cache", "mountPath": "/cache" },
    ]);
    // Loading every index can take minutes; that is not a hang.
    shard_container["livenessProbe"] = json!({
        "tcpSocket": { "port": "grpc" },
        "initialDelaySeconds": 10,
        "periodSeconds": 30,
    });
    if spec.shard.ipc_lock {
        shard_container["securityContext"]["capabilities"] = json!({ "add": ["IPC_LOCK"] });
    }
    let host_path = |path: String| json!({ "path": path, "type": "DirectoryOrCreate" });
    let mut pod = json!({
        "terminationGracePeriodSeconds": spec
            .shard
            .termination_grace_period_seconds
            .unwrap_or(DEFAULT_GRACE_SECONDS),
        "affinity": { "nodeAffinity": { "requiredDuringSchedulingIgnoredDuringExecution": {
            "nodeSelectorTerms": [{ "matchExpressions": [{
                "key": "kubernetes.io/hostname",
                "operator": "In",
                "values": [shard.node],
            }] }],
        } } },
        "containers": [shard_container],
        "volumes": [
            { "name": "data", "hostPath": host_path(shard.data_path.clone()) },
            { "name": "cache", "hostPath": host_path(cache_path(shard)) },
        ],
    });
    if !spec.shard.tolerations.is_empty() {
        pod["tolerations"] = json!(spec.shard.tolerations);
    }
    deployment(
        owner.metadata(&shard_name(owner.cluster, index), "shard", SERVER_APP),
        pod_labels,
        i32::from(!shard.suspended),
        // The old pod releases the data directory before the new one opens it.
        json!({ "type": "Recreate" }),
        pod,
    )
}

pub fn broker_deployment(owner: &Owner, spec: &SearchClusterSpec, routing: &str) -> Deployment {
    let pod_labels = json!({ NAME_LABEL: BROKER_APP, CLUSTER_LABEL: owner.cluster });
    let mut broker = container(
        BROKER_APP,
        &spec.broker.image,
        broker_args(owner.cluster, owner.namespace, spec, routing),
        &json!(spec.broker.env),
        &json!(spec.broker.resources),
    );
    broker["livenessProbe"] = json!({
        "grpc": { "port": GRPC_PORT },
        "initialDelaySeconds": 10,
        "periodSeconds": 30,
    });
    deployment(
        owner.metadata(&broker_name(owner.cluster), "broker", BROKER_APP),
        pod_labels,
        spec.broker.replicas,
        json!({ "type": "RollingUpdate", "rollingUpdate": { "maxSurge": 1, "maxUnavailable": 0 } }),
        json!({
            "serviceAccountName": spec.broker.service_account_name,
            "terminationGracePeriodSeconds": DEFAULT_GRACE_SECONDS,
            "containers": [broker],
        }),
    )
}

/// The index's address for clients: `<cluster>.<namespace>.svc:50051`.
pub fn broker_service(owner: &Owner) -> Service {
    serde_json::from_value(json!({
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": owner.metadata(owner.cluster, "broker", BROKER_APP),
        "spec": {
            "type": "ClusterIP",
            "selector": { NAME_LABEL: BROKER_APP, CLUSTER_LABEL: owner.cluster },
            "ports": [{ "name": "grpc", "protocol": "TCP", "port": GRPC_PORT, "targetPort": GRPC_PORT }],
        },
    }))
    .expect("rendered Service is well-formed")
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::crd::ShardTemplate;

    pub(crate) fn spec() -> SearchClusterSpec {
        SearchClusterSpec {
            shard: ShardTemplate {
                image: ImageSpec {
                    image: "registry/summa-server:2.2.0".into(),
                    image_pull_policy: None,
                },
                args: vec!["--search-threads".into(), "8".into()],
                ipc_lock: true,
                ..Default::default()
            },
            shards: [(5, "/data2"), (4, "/data3")]
                .into_iter()
                .map(|(weight, root)| ShardSpec {
                    node: "node-a".into(),
                    data_path: format!("{root}/summa/documents_1"),
                    weight: Some(weight),
                    ..Default::default()
                })
                .collect(),
            broker: crate::crd::BrokerSpec {
                image: ImageSpec {
                    image: "registry/summa-broker:2.2.0".into(),
                    image_pull_policy: Some("Always".into()),
                },
                ..Default::default()
            },
            ..Default::default()
        }
    }

    pub(crate) fn owner() -> Owner<'static> {
        Owner {
            cluster: "documents-1",
            namespace: "summa",
            reference: OwnerReference {
                api_version: "summa.spacefrontiers.org/v1alpha1".into(),
                kind: "SearchCluster".into(),
                name: "documents-1".into(),
                uid: "uid".into(),
                controller: Some(true),
                block_owner_deletion: Some(true),
            },
        }
    }

    #[test]
    fn broker_serves_only_its_cluster_with_the_cluster_routing() {
        let mut spec = spec();
        spec.broker.replicas = 3;
        let rule = routing("documents-1", &spec).to_string();
        assert_eq!(rule, "documents_1=0:5,1:4");
        let broker = serde_json::to_value(broker_deployment(&owner(), &spec, &rule)).unwrap();
        let args = &broker["spec"]["template"]["spec"]["containers"][0]["args"];
        assert_eq!(
            args,
            &json!([
                "--addr",
                "0.0.0.0:50051",
                "--discovery",
                "kubernetes",
                "--namespace",
                "summa",
                "--pod-selector",
                "summa.spacefrontiers.org/cluster=documents-1",
                "--placement",
                "documents_1=0:5,1:4",
                "--placement-default",
                "reject",
                "--backend-max-searches",
                "5"
            ])
        );
        assert_eq!(broker["metadata"]["name"], "documents-1-broker");
        let container = &broker["spec"]["template"]["spec"]["containers"][0];
        assert_eq!(container["image"], "registry/summa-broker:2.2.0");
        assert_eq!(container["imagePullPolicy"], "Always");
        assert_eq!(
            broker["metadata"]["ownerReferences"][0]["kind"],
            "SearchCluster"
        );
        let service = serde_json::to_value(broker_service(&owner())).unwrap();
        assert_eq!(service["metadata"]["name"], "documents-1");
        assert_eq!(
            service["spec"]["selector"],
            json!({ NAME_LABEL: BROKER_APP, CLUSTER_LABEL: "documents-1" })
        );
    }

    #[test]
    fn partition_pins_its_node_directories_and_shard_ordinal() {
        let mut spec = spec();
        spec.shards[1].suspended = true;
        let value = serde_json::to_value(shard_deployment(&owner(), &spec, 1)).unwrap();
        assert_eq!(value["metadata"]["name"], "documents-1-1");
        assert_eq!(value["spec"]["replicas"], 0);
        assert_eq!(
            value["spec"]["selector"]["matchLabels"],
            json!({ NAME_LABEL: SERVER_APP, CLUSTER_LABEL: "documents-1", SHARD_LABEL: "1" })
        );
        let pod = &value["spec"]["template"]["spec"];
        assert_eq!(
            pod["volumes"][0]["hostPath"]["path"],
            "/data3/summa/documents_1"
        );
        assert_eq!(
            pod["volumes"][1]["hostPath"]["path"],
            "/data3/summa/documents_1-cache"
        );
        let container = &pod["containers"][0];
        assert_eq!(container["image"], "registry/summa-server:2.2.0");
        assert!(container.get("imagePullPolicy").is_none());
        assert_eq!(
            container["args"],
            json!([
                "--addr",
                "0.0.0.0:50051",
                "--data-dir",
                "/data",
                "--cache-dir",
                "/cache",
                "--max-concurrent-searches",
                "16",
                "--search-threads",
                "8"
            ])
        );
        assert_eq!(
            container["securityContext"]["capabilities"]["add"],
            json!(["IPC_LOCK"])
        );
        assert!(container.get("env").is_none());
        assert!(pod.get("tolerations").is_none());
    }

    #[test]
    fn index_name_follows_the_cluster_name() {
        assert_eq!(index_name("documents-20261101"), "documents_20261101");
        assert_eq!(index_name("social"), "social");
        let mut spec = spec();
        spec.shards.truncate(1);
        spec.shards[0].weight = None;
        assert_eq!(routing("social", &spec).to_string(), "social=0");
    }
}
