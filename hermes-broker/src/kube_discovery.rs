//! Kubernetes discovery: watch Pods carrying the shard label and project
//! them into `DiscoveredEndpoint`s.
//!
//! Pods, not EndpointSlices: shard identity and role are pod labels, and a
//! pod object carries labels, IP, and readiness in one place — no per-shard
//! Service is required. The projection is a pure function so it is unit
//! tested without a cluster; the watch loop is deliberately thin.

use std::collections::BTreeMap;
use std::time::Duration;

use futures::TryStreamExt;
use k8s_openapi::api::core::v1::Pod;
use kube::api::Api;
use kube::runtime::watcher;
use log::{info, warn};
use tokio::sync::watch;

use crate::topology::{BackendId, DiscoveredEndpoint, Role, ShardId};

pub struct KubeDiscoveryConfig {
    pub namespace: String,
    pub shard_label: String,
    pub role_label: String,
    pub port_annotation: String,
    pub default_port: u16,
}

/// Project the watched pod set into endpoints. Pods without a name, without
/// the shard label, or without an IP yet are skipped (a pod with no IP cannot
/// be addressed at all); unready pods are kept but flagged so the admin
/// surface can show them while routing ignores them.
pub fn pods_to_endpoints(
    pods: &BTreeMap<String, Pod>,
    cfg: &KubeDiscoveryConfig,
) -> Vec<DiscoveredEndpoint> {
    let mut endpoints = Vec::new();
    for pod in pods.values() {
        let Some(name) = pod.metadata.name.clone() else {
            continue;
        };
        let labels = pod.metadata.labels.clone().unwrap_or_default();
        let Some(shard) = labels.get(&cfg.shard_label).cloned() else {
            warn!("pod {name} matched the watch selector but lacks the shard label; skipping");
            continue;
        };
        let role = match labels.get(&cfg.role_label).map(String::as_str) {
            None => None,
            Some("master") => Some(Role::Master),
            Some("follower") => Some(Role::Follower),
            Some(other) => {
                warn!("pod {name} has unknown role label '{other}'; treating as unlabeled");
                None
            }
        };
        let Some(ip) = pod.status.as_ref().and_then(|s| s.pod_ip.clone()) else {
            continue;
        };
        let port = pod
            .metadata
            .annotations
            .as_ref()
            .and_then(|a| a.get(&cfg.port_annotation))
            .and_then(|v| v.parse::<u16>().ok())
            .unwrap_or(cfg.default_port);
        let terminating = pod.metadata.deletion_timestamp.is_some();
        let ready = !terminating
            && pod
                .status
                .as_ref()
                .and_then(|s| s.conditions.as_ref())
                .is_some_and(|conditions| {
                    conditions
                        .iter()
                        .any(|c| c.type_ == "Ready" && c.status == "True")
                });
        endpoints.push(DiscoveredEndpoint {
            id: BackendId(name),
            addr: format!("{ip}:{port}"),
            shard: ShardId(shard),
            role,
            ready,
        });
    }
    endpoints.sort_by(|a, b| a.id.cmp(&b.id));
    endpoints
}

/// Long-running pod watch; publishes the full endpoint set on every change.
/// Only exits on shutdown. Watch errors are logged and the stream recovers
/// (kube's watcher re-lists automatically after an error).
pub async fn run(
    cfg: KubeDiscoveryConfig,
    tx: watch::Sender<Vec<DiscoveredEndpoint>>,
    mut shutdown: watch::Receiver<bool>,
) -> anyhow::Result<()> {
    let client = kube::Client::try_default().await.map_err(|e| {
        anyhow::anyhow!("kubernetes discovery requires in-cluster or kubeconfig access: {e}")
    })?;
    let api: Api<Pod> = Api::namespaced(client, &cfg.namespace);
    let watcher_config = watcher::Config::default().labels(&cfg.shard_label);
    info!(
        "watching pods in namespace '{}' with label '{}'",
        cfg.namespace, cfg.shard_label
    );

    let mut store: BTreeMap<String, Pod> = BTreeMap::new();
    let mut pending: BTreeMap<String, Pod> = BTreeMap::new();
    let mut synced = false;
    let mut stream = std::pin::pin!(watcher(api, watcher_config));
    loop {
        tokio::select! {
            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    return Ok(());
                }
            }
            event = stream.try_next() => {
                match event {
                    Ok(Some(watcher::Event::Init)) => {
                        pending.clear();
                        synced = false;
                    }
                    Ok(Some(watcher::Event::InitApply(pod))) => {
                        if let Some(name) = pod.metadata.name.clone() {
                            pending.insert(name, pod);
                        }
                    }
                    Ok(Some(watcher::Event::InitDone)) => {
                        store = std::mem::take(&mut pending);
                        synced = true;
                        let _ = tx.send(pods_to_endpoints(&store, &cfg));
                    }
                    Ok(Some(watcher::Event::Apply(pod))) => {
                        if let Some(name) = pod.metadata.name.clone() {
                            store.insert(name, pod);
                        }
                        if synced {
                            let _ = tx.send(pods_to_endpoints(&store, &cfg));
                        }
                    }
                    Ok(Some(watcher::Event::Delete(pod))) => {
                        if let Some(name) = pod.metadata.name.as_ref() {
                            store.remove(name);
                        }
                        if synced {
                            let _ = tx.send(pods_to_endpoints(&store, &cfg));
                        }
                    }
                    Ok(None) => {
                        warn!("pod watch stream ended; restarting in 5s");
                        tokio::time::sleep(Duration::from_secs(5)).await;
                    }
                    Err(e) => {
                        warn!("pod watch error: {e}; stream will recover");
                        tokio::time::sleep(Duration::from_secs(5)).await;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use k8s_openapi::api::core::v1::{PodCondition, PodStatus};
    use k8s_openapi::apimachinery::pkg::apis::meta::v1::{ObjectMeta, Time};

    fn cfg() -> KubeDiscoveryConfig {
        KubeDiscoveryConfig {
            namespace: "hermes".into(),
            shard_label: "hermes.spacefrontiers.org/shard-id".into(),
            role_label: "hermes.spacefrontiers.org/role".into(),
            port_annotation: "hermes.spacefrontiers.org/grpc-port".into(),
            default_port: 50051,
        }
    }

    fn pod(
        name: &str,
        shard: Option<&str>,
        role: Option<&str>,
        ip: Option<&str>,
        ready: bool,
    ) -> Pod {
        let mut labels = std::collections::BTreeMap::new();
        if let Some(shard) = shard {
            labels.insert(
                "hermes.spacefrontiers.org/shard-id".to_string(),
                shard.to_string(),
            );
        }
        if let Some(role) = role {
            labels.insert(
                "hermes.spacefrontiers.org/role".to_string(),
                role.to_string(),
            );
        }
        Pod {
            metadata: ObjectMeta {
                name: Some(name.to_string()),
                labels: Some(labels),
                ..Default::default()
            },
            status: Some(PodStatus {
                pod_ip: ip.map(String::from),
                conditions: Some(vec![PodCondition {
                    type_: "Ready".to_string(),
                    status: if ready { "True" } else { "False" }.to_string(),
                    ..Default::default()
                }]),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn store(pods: Vec<Pod>) -> BTreeMap<String, Pod> {
        pods.into_iter()
            .map(|p| (p.metadata.name.clone().unwrap(), p))
            .collect()
    }

    #[test]
    fn projects_ready_labeled_pods() {
        let pods = store(vec![
            pod(
                "hermes-a",
                Some("0"),
                Some("master"),
                Some("10.0.0.1"),
                true,
            ),
            pod("hermes-b", Some("1"), None, Some("10.0.0.2"), true),
        ]);
        let endpoints = pods_to_endpoints(&pods, &cfg());
        assert_eq!(endpoints.len(), 2);
        assert_eq!(endpoints[0].id.0, "hermes-a");
        assert_eq!(endpoints[0].addr, "10.0.0.1:50051");
        assert_eq!(endpoints[0].shard.0, "0");
        assert_eq!(endpoints[0].role, Some(Role::Master));
        assert!(endpoints[0].ready);
        assert_eq!(endpoints[1].role, None);
    }

    #[test]
    fn skips_pods_without_ip_or_shard_label() {
        let pods = store(vec![
            pod("no-ip", Some("0"), None, None, true),
            pod("no-shard", None, None, Some("10.0.0.3"), true),
            pod("ok", Some("0"), None, Some("10.0.0.4"), true),
        ]);
        let endpoints = pods_to_endpoints(&pods, &cfg());
        assert_eq!(endpoints.len(), 1);
        assert_eq!(endpoints[0].id.0, "ok");
    }

    #[test]
    fn unready_and_terminating_pods_are_flagged_unready() {
        let mut terminating = pod("dying", Some("0"), None, Some("10.0.0.5"), true);
        terminating.metadata.deletion_timestamp = Some(Time(Default::default()));
        let pods = store(vec![
            pod("starting", Some("0"), None, Some("10.0.0.6"), false),
            terminating,
        ]);
        let endpoints = pods_to_endpoints(&pods, &cfg());
        assert_eq!(endpoints.len(), 2);
        assert!(endpoints.iter().all(|e| !e.ready));
    }

    #[test]
    fn port_annotation_overrides_default() {
        let mut custom = pod("custom", Some("0"), None, Some("10.0.0.7"), true);
        custom.metadata.annotations = Some(
            [(
                "hermes.spacefrontiers.org/grpc-port".to_string(),
                "60051".to_string(),
            )]
            .into_iter()
            .collect(),
        );
        let endpoints = pods_to_endpoints(&store(vec![custom]), &cfg());
        assert_eq!(endpoints[0].addr, "10.0.0.7:60051");
    }

    #[test]
    fn unknown_role_label_is_treated_as_unlabeled() {
        let pods = store(vec![pod(
            "odd",
            Some("0"),
            Some("primary"),
            Some("10.0.0.8"),
            true,
        )]);
        let endpoints = pods_to_endpoints(&pods, &cfg());
        assert_eq!(endpoints[0].role, None);
    }
}
