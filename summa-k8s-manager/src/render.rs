//! Desired workloads of a `SearchCluster`.
//!
//! Pod templates mirror the hand-written shard and broker manifests they
//! replace field for field, so adopting an existing Deployment under the
//! same name does not change its template (and does not restart it).
//! Ownership labels go on Deployment and Service metadata only.

use k8s_openapi::api::apps::v1::Deployment;
use k8s_openapi::api::core::v1::Service;
use serde_json::{Value, json};
use summa_broker::placement_rules::{PartitionSpec, RuleSpec};

use crate::crd::{PlacementDefault, PlacementRuleSpec, ReplicaSpec, SearchClusterSpec, ShardSpec};

pub const NAME_LABEL: &str = "app.kubernetes.io/name";
pub const COMPONENT_LABEL: &str = "app.kubernetes.io/component";
pub const MANAGED_BY_LABEL: &str = "app.kubernetes.io/managed-by";
pub const MANAGER: &str = "summa-k8s-manager";
pub const CLUSTER_LABEL: &str = "summa.spacefrontiers.org/cluster";
/// Broker discovery label (`summa-broker --shard-label`).
pub const SHARD_LABEL: &str = "summa.spacefrontiers.org/shard-id";
pub const SERVER_APP: &str = "summa-server";
pub const GRPC_PORT: i32 = 50051;
pub const METRICS_PORT: i32 = 9184;
const DEFAULT_GRACE_SECONDS: i64 = 600;

/// Flags the operator renders for every shard replica.
pub const SERVER_RESERVED_FLAGS: &[&str] = &["--addr", "--data-dir", "--cache-dir"];
/// Flags the operator renders for the broker.
pub const BROKER_RESERVED_FLAGS: &[&str] = &[
    "--addr",
    "--discovery",
    "--namespace",
    "--placement",
    "--placement-default",
    "--backend-max-searches",
];

pub fn replica_name(cluster: &str, shard: &ShardSpec, replica: &ReplicaSpec) -> String {
    replica
        .name
        .clone()
        .unwrap_or_else(|| format!("{cluster}-shard-{}", shard.id))
}

pub fn broker_name(cluster: &str, spec: &SearchClusterSpec) -> String {
    spec.broker
        .name
        .clone()
        .unwrap_or_else(|| format!("{cluster}-broker"))
}

/// In-cluster gRPC address of the broker Service.
pub fn broker_address(cluster: &str, namespace: &str, spec: &SearchClusterSpec) -> String {
    format!("{}.{namespace}.svc:{GRPC_PORT}", broker_name(cluster, spec))
}

pub fn rule_spec(rule: &PlacementRuleSpec) -> RuleSpec {
    RuleSpec {
        pattern: rule.pattern.clone(),
        partitions: rule
            .partitions
            .iter()
            .map(|p| PartitionSpec {
                shard: p.shard.clone(),
                weight: p.weight,
            })
            .collect(),
    }
}

/// `--placement` values in rule order.
pub fn placement_args(spec: &SearchClusterSpec) -> Vec<String> {
    spec.placement
        .rules
        .iter()
        .map(|rule| rule_spec(rule).to_string())
        .collect()
}

/// Per-broker share of the per-backend search budget.
pub fn broker_backend_max_searches(spec: &SearchClusterSpec) -> u32 {
    let replicas = u32::try_from(spec.broker.replicas.max(1)).unwrap_or(1);
    (spec.broker.backend_max_searches / replicas).max(1)
}

/// Flag names in an argument list (`--flag value` and `--flag=value`).
pub fn flag_names(args: &[String]) -> Vec<&str> {
    args.iter()
        .filter(|arg| arg.starts_with("--"))
        .map(|arg| arg.split_once('=').map_or(arg.as_str(), |(flag, _)| flag))
        .collect()
}

/// Value of a flag, from `--flag value` or `--flag=value`.
pub fn flag_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.iter().enumerate().find_map(|(i, arg)| {
        if arg == flag {
            args.get(i + 1).map(String::as_str)
        } else {
            arg.strip_prefix(flag)
                .and_then(|rest| rest.strip_prefix('='))
        }
    })
}

/// `--placement` values of a rendered or live broker argument list.
pub fn placement_values(args: &[String]) -> Vec<String> {
    let mut values = Vec::new();
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if arg == "--placement" {
            if let Some(value) = iter.next() {
                values.push(value.clone());
            }
        } else if let Some(value) = arg.strip_prefix("--placement=") {
            values.push(value.to_string());
        }
    }
    values
}

pub fn server_args(spec: &SearchClusterSpec, shard: &ShardSpec) -> Vec<String> {
    let mut args: Vec<String> = [
        "--addr",
        &format!("0.0.0.0:{GRPC_PORT}"),
        "--data-dir",
        "/data",
        "--cache-dir",
        "/cache",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    args.extend(spec.server.args.iter().cloned());
    args.extend(shard.args.iter().cloned());
    args
}

pub fn broker_args(namespace: &str, spec: &SearchClusterSpec) -> Vec<String> {
    let mut args: Vec<String> = vec![
        "--addr".into(),
        format!("0.0.0.0:{GRPC_PORT}"),
        "--discovery".into(),
        "kubernetes".into(),
        "--namespace".into(),
        namespace.into(),
    ];
    for placement in placement_args(spec) {
        args.push("--placement".into());
        args.push(placement);
    }
    if spec.placement.default == PlacementDefault::Reject {
        args.push("--placement-default".into());
        args.push("reject".into());
    }
    args.push("--backend-max-searches".into());
    args.push(broker_backend_max_searches(spec).to_string());
    args.extend(spec.broker.args.iter().cloned());
    args
}

fn managed_labels(cluster: &str, base: Value) -> Value {
    let mut labels = base;
    labels[MANAGED_BY_LABEL] = json!(MANAGER);
    labels[CLUSTER_LABEL] = json!(cluster);
    labels
}

/// Insert optional container fields only when set, so the template stays
/// equal to manifests that omit them.
fn set_optional(container: &mut Value, key: &str, value: Value) {
    let empty = value.is_null() || value.as_array().is_some_and(Vec::is_empty);
    if !empty {
        container[key] = value;
    }
}

pub struct ReplicaWorkload {
    pub shard_id: String,
    pub name: String,
    pub node: String,
    pub data_path: String,
    pub cache_path: String,
    pub deployment: Deployment,
}

pub fn shard_workloads(
    cluster: &str,
    namespace: &str,
    spec: &SearchClusterSpec,
) -> Vec<ReplicaWorkload> {
    spec.shards
        .iter()
        .flat_map(|shard| {
            shard.replicas.iter().map(move |replica| ReplicaWorkload {
                shard_id: shard.id.clone(),
                name: replica_name(cluster, shard, replica),
                node: replica.node.clone(),
                data_path: replica.data_path.clone(),
                cache_path: replica.cache_path.clone(),
                deployment: shard_deployment(cluster, namespace, spec, shard, replica),
            })
        })
        .collect()
}

fn shard_deployment(
    cluster: &str,
    namespace: &str,
    spec: &SearchClusterSpec,
    shard: &ShardSpec,
    replica: &ReplicaSpec,
) -> Deployment {
    let pod_labels = json!({ NAME_LABEL: SERVER_APP, SHARD_LABEL: shard.id });
    let mut container = json!({
        "name": SERVER_APP,
        "image": spec.image,
        "command": [SERVER_APP],
        "args": server_args(spec, shard),
        "ports": [
            { "name": "grpc", "containerPort": GRPC_PORT, "protocol": "TCP" },
            { "name": "metrics", "containerPort": METRICS_PORT, "protocol": "TCP" },
        ],
        "securityContext": { "allowPrivilegeEscalation": false },
        "volumeMounts": [
            { "name": "data", "mountPath": "/data" },
            { "name": "cache", "mountPath": "/cache" },
        ],
        "livenessProbe": {
            "tcpSocket": { "port": "grpc" },
            "initialDelaySeconds": 10,
            "periodSeconds": 30,
        },
        // grpc.health.v1 is NOT_SERVING until every index is open, so the
        // broker never routes to a loading shard. Loading is not a hang:
        // liveness stays on TCP.
        "readinessProbe": {
            "grpc": { "port": GRPC_PORT },
            "initialDelaySeconds": 5,
            "periodSeconds": 10,
        },
    });
    set_optional(
        &mut container,
        "imagePullPolicy",
        json!(spec.image_pull_policy),
    );
    set_optional(&mut container, "env", json!(spec.server.env));
    set_optional(
        &mut container,
        "resources",
        json!(shard.resources.as_ref().or(spec.server.resources.as_ref())),
    );
    if spec.server.ipc_lock {
        container["securityContext"]["capabilities"] = json!({ "add": ["IPC_LOCK"] });
    }
    let host_path = |path: &str| json!({ "path": path, "type": "Directory" });
    let deployment = json!({
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": replica_name(cluster, shard, replica),
            "namespace": namespace,
            "labels": managed_labels(cluster, json!({
                NAME_LABEL: SERVER_APP,
                COMPONENT_LABEL: "server",
                SHARD_LABEL: shard.id,
            })),
        },
        "spec": {
            "replicas": 1,
            // Never two processes on one data directory: the old pod stops
            // before the new one starts.
            "strategy": { "type": "Recreate" },
            "selector": { "matchLabels": pod_labels },
            "template": {
                "metadata": { "labels": pod_labels },
                "spec": {
                    "terminationGracePeriodSeconds": spec
                        .server
                        .termination_grace_period_seconds
                        .unwrap_or(DEFAULT_GRACE_SECONDS),
                    "affinity": {
                        "nodeAffinity": {
                            "requiredDuringSchedulingIgnoredDuringExecution": {
                                "nodeSelectorTerms": [{
                                    "matchExpressions": [{
                                        "key": "kubernetes.io/hostname",
                                        "operator": "In",
                                        "values": [replica.node],
                                    }],
                                }],
                            },
                        },
                    },
                    "containers": [container],
                    "volumes": [
                        { "name": "data", "hostPath": host_path(&replica.data_path) },
                        { "name": "cache", "hostPath": host_path(&replica.cache_path) },
                    ],
                },
            },
        },
    });
    serde_json::from_value(deployment).expect("rendered shard Deployment is well-formed")
}

pub fn broker_deployment(cluster: &str, namespace: &str, spec: &SearchClusterSpec) -> Deployment {
    let name = broker_name(cluster, spec);
    let pod_labels = json!({ NAME_LABEL: name });
    let mut container = json!({
        "name": "summa-broker",
        "image": spec.broker.image.as_ref().unwrap_or(&spec.image),
        "command": ["summa-broker"],
        "args": broker_args(namespace, spec),
        "ports": [
            { "name": "grpc", "containerPort": GRPC_PORT, "protocol": "TCP" },
            { "name": "metrics", "containerPort": METRICS_PORT, "protocol": "TCP" },
        ],
        "securityContext": { "allowPrivilegeEscalation": false },
        "readinessProbe": {
            "grpc": { "port": GRPC_PORT },
            "initialDelaySeconds": 5,
            "periodSeconds": 10,
        },
        "livenessProbe": {
            "grpc": { "port": GRPC_PORT },
            "initialDelaySeconds": 10,
            "periodSeconds": 30,
        },
    });
    set_optional(
        &mut container,
        "imagePullPolicy",
        json!(spec.image_pull_policy),
    );
    set_optional(&mut container, "env", json!(spec.broker.env));
    set_optional(&mut container, "resources", json!(spec.broker.resources));
    let mut pod_spec = json!({
        "terminationGracePeriodSeconds": spec
            .broker
            .termination_grace_period_seconds
            .unwrap_or(DEFAULT_GRACE_SECONDS),
        "containers": [container],
    });
    set_optional(
        &mut pod_spec,
        "serviceAccountName",
        json!(spec.broker.service_account_name),
    );
    let deployment = json!({
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": managed_labels(cluster, json!({
                NAME_LABEL: name,
                COMPONENT_LABEL: "broker",
            })),
        },
        "spec": {
            "replicas": spec.broker.replicas,
            "strategy": {
                "type": "RollingUpdate",
                "rollingUpdate": { "maxSurge": 1, "maxUnavailable": 0 },
            },
            "selector": { "matchLabels": pod_labels },
            "template": {
                "metadata": { "labels": pod_labels },
                "spec": pod_spec,
            },
        },
    });
    serde_json::from_value(deployment).expect("rendered broker Deployment is well-formed")
}

pub fn broker_service(cluster: &str, namespace: &str, spec: &SearchClusterSpec) -> Service {
    let name = broker_name(cluster, spec);
    let service = json!({
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": managed_labels(cluster, json!({ NAME_LABEL: name })),
        },
        "spec": {
            "type": "ClusterIP",
            "selector": { NAME_LABEL: name },
            "ports": [{
                "name": "grpc",
                "protocol": "TCP",
                "port": GRPC_PORT,
                "targetPort": GRPC_PORT,
            }],
        },
    });
    serde_json::from_value(service).expect("rendered broker Service is well-formed")
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::crd::{PartitionRef, PlacementSpec, ServerSpec};

    pub(crate) fn sample_spec() -> SearchClusterSpec {
        SearchClusterSpec {
            image: "registry/summa-server:latest".into(),
            image_pull_policy: Some("Always".into()),
            server: ServerSpec {
                args: vec!["--max-concurrent-searches".into(), "16".into()],
                ipc_lock: true,
                ..Default::default()
            },
            shards: ["2", "3"]
                .iter()
                .map(|id| ShardSpec {
                    id: id.to_string(),
                    args: vec!["--search-threads".into(), "24".into()],
                    resources: None,
                    replicas: vec![ReplicaSpec {
                        name: Some(format!("summa-server-s{id}")),
                        node: "node-a".into(),
                        data_path: format!("/data{id}/summa-server"),
                        cache_path: format!("/data{id}/summa-cache"),
                    }],
                })
                .collect(),
            broker: Default::default(),
            placement: PlacementSpec {
                default: PlacementDefault::Single,
                rules: vec![PlacementRuleSpec {
                    pattern: "documents*".into(),
                    partitions: vec![
                        PartitionRef {
                            shard: "2".into(),
                            weight: Some(5),
                        },
                        PartitionRef {
                            shard: "3".into(),
                            weight: Some(4),
                        },
                    ],
                }],
            },
        }
    }

    #[test]
    fn broker_args_render_placement_in_rule_order_and_split_the_search_budget() {
        let mut spec = sample_spec();
        assert_eq!(
            broker_args("summa", &spec),
            [
                "--addr",
                "0.0.0.0:50051",
                "--discovery",
                "kubernetes",
                "--namespace",
                "summa",
                "--placement",
                "documents*=2:5,3:4",
                "--backend-max-searches",
                "16"
            ]
        );
        spec.broker.replicas = 3;
        assert_eq!(broker_backend_max_searches(&spec), 5);
        let args = broker_args("summa", &spec);
        assert_eq!(placement_values(&args), ["documents*=2:5,3:4"]);
        assert_eq!(flag_value(&args, "--backend-max-searches"), Some("5"));
    }

    #[test]
    fn shard_deployment_pins_its_node_and_host_directories() {
        let spec = sample_spec();
        let workloads = shard_workloads("summa", "summa", &spec);
        assert_eq!(workloads.len(), 2);
        let value = serde_json::to_value(&workloads[0].deployment).unwrap();
        assert_eq!(value["metadata"]["name"], "summa-server-s2");
        assert_eq!(value["metadata"]["labels"][MANAGED_BY_LABEL], MANAGER);
        let template = &value["spec"]["template"];
        // Ownership labels stay off the pod template so adoption does not
        // restart the shard.
        assert_eq!(
            template["metadata"]["labels"],
            json!({ NAME_LABEL: SERVER_APP, SHARD_LABEL: "2" })
        );
        assert_eq!(value["spec"]["strategy"]["type"], "Recreate");
        let pod = &template["spec"];
        assert_eq!(
            pod["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"]
                [0]["matchExpressions"][0]["values"],
            json!(["node-a"])
        );
        assert_eq!(pod["volumes"][0]["hostPath"]["path"], "/data2/summa-server");
        assert_eq!(pod["volumes"][0]["hostPath"]["type"], "Directory");
        let container = &pod["containers"][0];
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
                "24"
            ])
        );
        assert_eq!(
            container["securityContext"]["capabilities"]["add"],
            json!(["IPC_LOCK"])
        );
        assert!(container.get("env").is_none());
        assert!(container.get("resources").is_none());
    }

    #[test]
    fn broker_image_override_leaves_shard_templates_unchanged() {
        let mut spec = sample_spec();
        let shards_before = serde_json::to_value(
            shard_workloads("summa", "summa", &spec)
                .into_iter()
                .map(|w| w.deployment)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        spec.broker.image = Some("registry/summa-server:broker-fix".into());
        let broker = serde_json::to_value(broker_deployment("summa", "summa", &spec)).unwrap();
        assert_eq!(
            broker["spec"]["template"]["spec"]["containers"][0]["image"],
            "registry/summa-server:broker-fix"
        );
        let shards_after = serde_json::to_value(
            shard_workloads("summa", "summa", &spec)
                .into_iter()
                .map(|w| w.deployment)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        assert_eq!(shards_before, shards_after);
    }

    #[test]
    fn default_names_follow_the_cluster_name() {
        let mut spec = sample_spec();
        spec.shards[0].replicas[0].name = None;
        assert_eq!(
            replica_name("summa", &spec.shards[0], &spec.shards[0].replicas[0]),
            "summa-shard-2"
        );
        assert_eq!(broker_name("summa", &spec), "summa-broker");
        assert_eq!(
            broker_address("summa", "search", &spec),
            "summa-broker.search.svc:50051"
        );
    }
}
