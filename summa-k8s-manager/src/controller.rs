//! Operator: reconciles each `SearchCluster` into shard replica Deployments,
//! the broker Deployment and the broker Service.
//!
//! Order per pass: validate, prune replicas no longer in the spec (removed
//! shards only when the broker confirms they are empty), apply replicas (a
//! new one waits until no pod on its node holds its directories), apply the
//! broker (a placement change waits until it reroutes no live index), and
//! write status.

use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context as _;
use futures::StreamExt;
use k8s_openapi::api::apps::v1::Deployment;
use k8s_openapi::api::core::v1::{Pod, Service};
use k8s_openapi::apimachinery::pkg::apis::meta::v1::{Condition, Time};
use k8s_openapi::chrono::Utc;
use kube::api::{Api, DeleteParams, ListParams, Patch, PatchParams};
use kube::runtime::controller::{Action, Controller};
use kube::runtime::reflector::ObjectRef;
use kube::runtime::watcher;
use kube::{Client, ResourceExt};
use log::{info, warn};
use serde_json::json;

use crate::broker::{BrokerClient, LiveIndex};
use crate::crd::{BrokerStatus, ReplicaStatus, SearchCluster, SearchClusterStatus, ShardStatus};
use crate::render::{
    CLUSTER_LABEL, COMPONENT_LABEL, MANAGED_BY_LABEL, MANAGER, ReplicaWorkload, SHARD_LABEL,
    broker_address, broker_deployment, broker_name, broker_service, placement_args,
    placement_values, shard_workloads,
};
use crate::safety::{self, parse_rules};
use crate::validate::{normalize_path, paths_overlap, validate};

const SETTLED_REQUEUE: Duration = Duration::from_secs(300);
const PENDING_REQUEUE: Duration = Duration::from_secs(30);
const BLOCKED_REQUEUE: Duration = Duration::from_secs(60);

pub struct Context {
    pub client: Client,
    /// Broker address override (local development); default is the
    /// cluster's broker Service.
    pub broker_override: Option<String>,
}

impl Context {
    pub fn broker(
        &self,
        cluster: &str,
        namespace: &str,
        spec: &crate::crd::SearchClusterSpec,
    ) -> BrokerClient {
        BrokerClient::new(
            self.broker_override
                .clone()
                .unwrap_or_else(|| broker_address(cluster, namespace, spec)),
        )
    }
}

#[derive(Debug)]
pub struct ReconcileError(anyhow::Error);

impl fmt::Display for ReconcileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:#}", self.0)
    }
}

impl std::error::Error for ReconcileError {}

impl From<anyhow::Error> for ReconcileError {
    fn from(error: anyhow::Error) -> Self {
        Self(error)
    }
}

impl From<kube::Error> for ReconcileError {
    fn from(error: kube::Error) -> Self {
        Self(error.into())
    }
}

pub async fn run(ctx: Arc<Context>, namespace: String) {
    let client = ctx.client.clone();
    let clusters: Api<SearchCluster> = Api::namespaced(client.clone(), &namespace);
    let deployments: Api<Deployment> = Api::namespaced(client, &namespace);
    let managed = watcher::Config::default().labels(&format!("{MANAGED_BY_LABEL}={MANAGER}"));
    Controller::new(clusters, watcher::Config::default())
        .watches(deployments, managed, move |deployment| {
            let cluster = deployment.labels().get(CLUSTER_LABEL)?.clone();
            let namespace = deployment.namespace()?;
            Some(ObjectRef::<SearchCluster>::new(&cluster).within(&namespace))
        })
        .shutdown_on_signal()
        .run(reconcile, error_policy, ctx)
        .for_each(|result| async move {
            if let Err(error) = result {
                warn!("reconcile failed: {error}");
            }
        })
        .await;
}

fn error_policy(cluster: Arc<SearchCluster>, error: &ReconcileError, _ctx: Arc<Context>) -> Action {
    warn!("SearchCluster {}: {error}", cluster.name_any());
    metrics::counter!("summa_k8s_manager_reconcile_total", "result" => "error").increment(1);
    Action::requeue(PENDING_REQUEUE)
}

/// Condition outcomes of one pass.
struct Outcome {
    ready: Result<(), (String, String)>,
    placement: Result<(), String>,
    pruned: Result<(), String>,
}

async fn reconcile(
    cluster: Arc<SearchCluster>,
    ctx: Arc<Context>,
) -> Result<Action, ReconcileError> {
    let namespace = cluster.namespace().context("SearchCluster is namespaced")?;
    let name = cluster.name_any();
    let client = ctx.client.clone();
    let clusters: Api<SearchCluster> = Api::namespaced(client.clone(), &namespace);

    // The broker discovers every shard pod in the namespace, so a second
    // cluster there would route to the first one's shards.
    let all = clusters.list(&ListParams::default()).await?;
    if let Some(owner) = namespace_owner(&all.items)
        && owner != name
    {
        let message = format!("namespace {namespace} is managed by SearchCluster {owner}");
        write_invalid(&clusters, &cluster, "NamespaceTaken", &message).await?;
        metrics::counter!("summa_k8s_manager_reconcile_total", "result" => "invalid").increment(1);
        return Ok(Action::requeue(SETTLED_REQUEUE));
    }

    let spec = &cluster.spec;
    let errors = validate(&name, spec);
    if !errors.is_empty() {
        let message = errors
            .iter()
            .map(|e| format!("{}: {}", e.field, e.message))
            .collect::<Vec<_>>()
            .join("; ");
        warn!("SearchCluster {name} is invalid, nothing changed: {message}");
        write_invalid(&clusters, &cluster, "Invalid", &message).await?;
        metrics::counter!("summa_k8s_manager_reconcile_total", "result" => "invalid").increment(1);
        return Ok(Action::requeue(SETTLED_REQUEUE));
    }

    let deployments: Api<Deployment> = Api::namespaced(client.clone(), &namespace);
    let services: Api<Service> = Api::namespaced(client.clone(), &namespace);
    let pods: Api<Pod> = Api::namespaced(client, &namespace);
    let apply = PatchParams::apply(MANAGER).force();
    let broker = ctx.broker(&name, &namespace, spec);
    let workloads = shard_workloads(&name, &namespace, spec);

    let pruned = prune_replicas(&deployments, &broker, &name, spec, &workloads).await?;

    let pod_list = pods.list(&ListParams::default()).await?.items;
    let mut live = Vec::with_capacity(workloads.len());
    for workload in &workloads {
        live.push(deployments.get_opt(&workload.name).await?);
    }
    // A spec change restarts a shard (Recreate) and reopens every index on
    // it. Restart one shard at a time: hold changes while another shard is
    // not ready, so a shared image or argument change never takes every
    // partition down together.
    let mut restarting: Option<String> = workloads
        .iter()
        .zip(&live)
        .find(|(_, deployment)| deployment.as_ref().is_some_and(|d| !deployment_ready(d)))
        .map(|(workload, _)| workload.name.clone());
    let mut waiting = Vec::new();
    for (workload, existing) in workloads.iter().zip(&live) {
        let patch = Patch::Apply(&workload.deployment);
        match existing {
            None => {
                if let Some(conflict) = host_path_conflict(&pod_list, workload) {
                    waiting.push(conflict);
                    continue;
                }
            }
            Some(existing) => {
                let preview = deployments
                    .patch(&workload.name, &apply.clone().dry_run(), &patch)
                    .await
                    .with_context(|| format!("dry-run apply Deployment {}", workload.name))?;
                let restarts = preview.metadata.generation != existing.metadata.generation;
                match &restarting {
                    Some(other) if restarts && other != &workload.name => {
                        waiting.push(format!(
                            "{} waits for {other} to become ready before restarting",
                            workload.name
                        ));
                        continue;
                    }
                    None if restarts => restarting = Some(workload.name.clone()),
                    _ => {}
                }
            }
        }
        deployments
            .patch(&workload.name, &apply, &patch)
            .await
            .with_context(|| format!("apply Deployment {}", workload.name))?;
    }

    let broker_deployment_name = broker_name(&name, spec);
    services
        .patch(
            &broker_deployment_name,
            &apply,
            &Patch::Apply(&broker_service(&name, &namespace, spec)),
        )
        .await
        .with_context(|| format!("apply Service {broker_deployment_name}"))?;
    let existing_broker = deployments.get_opt(&broker_deployment_name).await?;
    let running_placement = existing_broker.as_ref().map(container_placement);
    let desired_placement = placement_args(spec);
    let placement = match &running_placement {
        Some(running) if running != &desired_placement => {
            placement_gate(&broker, running, spec).await
        }
        _ => Ok(()),
    };
    let applied_placement = if placement.is_ok() {
        deployments
            .patch(
                &broker_deployment_name,
                &apply,
                &Patch::Apply(&broker_deployment(&name, &namespace, spec)),
            )
            .await
            .with_context(|| format!("apply Deployment {broker_deployment_name}"))?;
        desired_placement
    } else {
        running_placement.unwrap_or_default()
    };

    // Status from the Deployments as they are now.
    let mut shard_statuses: Vec<ShardStatus> = Vec::new();
    let mut not_ready = Vec::new();
    for workload in &workloads {
        let deployment = deployments.get_opt(&workload.name).await?;
        let message = waiting
            .iter()
            .find(|w| w.starts_with(&format!("{} ", workload.name)))
            .cloned();
        let ready = deployment.as_ref().is_some_and(deployment_ready);
        if !ready {
            not_ready.push(workload.name.clone());
        }
        let replica = ReplicaStatus {
            name: workload.name.clone(),
            node: workload.node.clone(),
            ready,
            message,
        };
        match shard_statuses
            .iter_mut()
            .find(|s| s.id == workload.shard_id)
        {
            Some(status) => status.replicas.push(replica),
            None => shard_statuses.push(ShardStatus {
                id: workload.shard_id.clone(),
                replicas: vec![replica],
            }),
        }
    }
    let broker_deployment = deployments.get_opt(&broker_deployment_name).await?;
    let broker_ready = broker_deployment
        .as_ref()
        .and_then(|d| d.status.as_ref())
        .and_then(|s| s.ready_replicas)
        .unwrap_or(0);
    if broker_deployment
        .as_ref()
        .is_none_or(|d| !deployment_ready(d))
    {
        not_ready.push(broker_deployment_name.clone());
    }

    let ready = if !waiting.is_empty() {
        Err(("Waiting".to_string(), waiting.join("; ")))
    } else if let Err(message) = &placement {
        Err(("PlacementBlocked".to_string(), message.clone()))
    } else if let Err(message) = &pruned {
        Err(("ShardRemovalBlocked".to_string(), message.clone()))
    } else if !not_ready.is_empty() {
        Err((
            "Progressing".to_string(),
            format!("not ready: {}", not_ready.join(", ")),
        ))
    } else {
        Ok(())
    };
    let settled = ready.is_ok();
    let blocked = placement.is_err() || pruned.is_err();
    let outcome = Outcome {
        ready,
        placement,
        pruned,
    };
    let status = SearchClusterStatus {
        observed_generation: cluster.metadata.generation,
        shards: shard_statuses,
        broker: Some(BrokerStatus {
            name: broker_deployment_name,
            replicas: spec.broker.replicas,
            ready_replicas: broker_ready,
        }),
        placement: applied_placement,
        conditions: conditions(&cluster, outcome),
    };
    clusters
        .patch_status(
            &name,
            &PatchParams::default(),
            &Patch::Merge(json!({ "status": status })),
        )
        .await
        .context("write SearchCluster status")?;

    let result = if settled {
        "ok"
    } else if blocked {
        "blocked"
    } else {
        "progressing"
    };
    metrics::counter!("summa_k8s_manager_reconcile_total", "result" => result).increment(1);
    Ok(Action::requeue(if settled {
        SETTLED_REQUEUE
    } else if blocked {
        BLOCKED_REQUEUE
    } else {
        PENDING_REQUEUE
    }))
}

/// Oldest `SearchCluster` in the namespace (name breaks ties).
pub fn namespace_owner(clusters: &[SearchCluster]) -> Option<String> {
    clusters
        .iter()
        .min_by_key(|c| {
            (
                c.metadata.creation_timestamp.clone().map(|t| t.0),
                c.name_any(),
            )
        })
        .map(ResourceExt::name_any)
}

/// Delete replica Deployments that left the spec. A replica whose shard id
/// stays (rename or move) goes immediately; a removed shard goes only when
/// the broker shows it hosts no index.
async fn prune_replicas(
    deployments: &Api<Deployment>,
    broker: &BrokerClient,
    cluster: &str,
    spec: &crate::crd::SearchClusterSpec,
    workloads: &[ReplicaWorkload],
) -> anyhow::Result<Result<(), String>> {
    let selector =
        format!("{MANAGED_BY_LABEL}={MANAGER},{CLUSTER_LABEL}={cluster},{COMPONENT_LABEL}=server");
    let managed = deployments
        .list(&ListParams::default().labels(&selector))
        .await?;
    let desired: BTreeSet<&str> = workloads.iter().map(|w| w.name.as_str()).collect();
    let shard_ids: BTreeSet<&str> = spec.shards.iter().map(|s| s.id.as_str()).collect();
    let stale: Vec<(String, String)> = managed
        .items
        .iter()
        .filter(|d| !desired.contains(d.name_any().as_str()))
        .map(|d| {
            (
                d.name_any(),
                d.labels().get(SHARD_LABEL).cloned().unwrap_or_default(),
            )
        })
        .collect();
    let removed: Vec<String> = stale
        .iter()
        .map(|(_, shard)| shard.clone())
        .filter(|shard| !shard_ids.contains(shard.as_str()))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let removals = if removed.is_empty() {
        Vec::new()
    } else {
        match broker.backends().await {
            Ok(backends) => safety::shard_removals(&backends, &removed),
            Err(error) => removed
                .iter()
                .map(|shard| safety::ShardRemoval {
                    shard: shard.clone(),
                    indexes: Vec::new(),
                    blocked: Some(format!("cannot confirm shard {shard} is empty: {error:#}")),
                })
                .collect(),
        }
    };
    let mut blocked = Vec::new();
    for (deployment, shard) in &stale {
        if let Some(reason) = removals
            .iter()
            .find(|r| &r.shard == shard)
            .and_then(|r| r.blocked.clone())
        {
            blocked.push(format!("kept {deployment}: {reason}"));
            continue;
        }
        info!(
            "deleting Deployment {deployment} (shard {shard}) removed from SearchCluster {cluster}"
        );
        deployments
            .delete(deployment, &DeleteParams::default())
            .await
            .with_context(|| format!("delete Deployment {deployment}"))?;
    }
    Ok(if blocked.is_empty() {
        Ok(())
    } else {
        Err(blocked.join("; "))
    })
}

/// A pod on the workload's node that mounts one of its directories (for
/// example the terminating pod of the replica's previous Deployment).
fn host_path_conflict(pods: &[Pod], workload: &ReplicaWorkload) -> Option<String> {
    let wanted = [
        normalize_path(&workload.data_path),
        normalize_path(&workload.cache_path),
    ];
    pods.iter().find_map(|pod| {
        let spec = pod.spec.as_ref()?;
        if spec.node_name.as_deref() != Some(workload.node.as_str()) {
            return None;
        }
        spec.volumes.iter().flatten().find_map(|volume| {
            let path = normalize_path(&volume.host_path.as_ref()?.path);
            wanted.iter().any(|w| paths_overlap(w, &path)).then(|| {
                format!(
                    "{} waits for pod {} to release {path} on {}",
                    workload.name,
                    pod.name_any(),
                    workload.node
                )
            })
        })
    })
}

/// `--placement` values of the broker container.
pub fn container_placement(deployment: &Deployment) -> Vec<String> {
    deployment
        .spec
        .as_ref()
        .and_then(|s| s.template.spec.as_ref())
        .and_then(|s| s.containers.first())
        .and_then(|c| c.args.as_ref())
        .map(|args| placement_values(args))
        .unwrap_or_default()
}

/// `Err(reason)` when switching the running broker to the spec's rules would
/// reroute a live index, or when that cannot be checked.
pub async fn placement_gate(
    broker: &BrokerClient,
    running: &[String],
    spec: &crate::crd::SearchClusterSpec,
) -> Result<(), String> {
    let current = parse_rules(running)
        .map_err(|e| format!("running broker placement does not parse: {e:#}"))?;
    let proposed = parse_rules(&placement_args(spec))
        .map_err(|e| format!("spec placement does not parse: {e:#}"))?;
    let topology = broker
        .topology()
        .await
        .map_err(|e| format!("cannot verify live indexes before changing placement: {e:#}"))?;
    let live: Vec<LiveIndex> = topology.iter().map(LiveIndex::from).collect();
    let blocked: Vec<String> = safety::placement_changes(&live, &current, &proposed)
        .into_iter()
        .filter_map(|change| change.blocked)
        .collect();
    if blocked.is_empty() {
        Ok(())
    } else {
        Err(blocked.join("; "))
    }
}

fn deployment_ready(deployment: &Deployment) -> bool {
    let Some(status) = &deployment.status else {
        return false;
    };
    let wanted = deployment
        .spec
        .as_ref()
        .and_then(|s| s.replicas)
        .unwrap_or(1);
    status.observed_generation >= deployment.metadata.generation
        && status.updated_replicas.unwrap_or(0) >= wanted
        && status.ready_replicas.unwrap_or(0) >= wanted
}

async fn write_invalid(
    clusters: &Api<SearchCluster>,
    cluster: &SearchCluster,
    reason: &str,
    message: &str,
) -> anyhow::Result<()> {
    let previous = cluster
        .status
        .as_ref()
        .map_or(&[][..], |s| &s.conditions[..]);
    let condition = make_condition(previous, cluster, "Ready", false, reason, message);
    let mut conditions: Vec<Condition> = previous.to_vec();
    match conditions.iter_mut().find(|c| c.type_ == "Ready") {
        Some(ready) => *ready = condition,
        None => conditions.insert(0, condition),
    }
    clusters
        .patch_status(
            &cluster.name_any(),
            &PatchParams::default(),
            &Patch::Merge(json!({
                "status": {
                    "observedGeneration": cluster.metadata.generation,
                    "conditions": conditions,
                }
            })),
        )
        .await
        .context("write SearchCluster status")?;
    Ok(())
}

fn conditions(cluster: &SearchCluster, outcome: Outcome) -> Vec<Condition> {
    let previous = cluster
        .status
        .as_ref()
        .map_or(&[][..], |s| &s.conditions[..]);
    let (ready, ready_reason, ready_message) = match outcome.ready {
        Ok(()) => (
            true,
            "Reconciled".to_string(),
            "all workloads are ready".to_string(),
        ),
        Err((reason, message)) => (false, reason, message),
    };
    let (placement_ok, placement_reason, placement_message) = match outcome.placement {
        Ok(()) => (
            true,
            "Applied",
            "broker runs the spec's placement".to_string(),
        ),
        Err(message) => (false, "Blocked", message),
    };
    let (pruned_ok, pruned_reason, pruned_message) = match outcome.pruned {
        Ok(()) => (true, "Pruned", "no Deployment outside the spec".to_string()),
        Err(message) => (false, "Blocked", message),
    };
    vec![
        make_condition(
            previous,
            cluster,
            "Ready",
            ready,
            &ready_reason,
            &ready_message,
        ),
        make_condition(
            previous,
            cluster,
            "PlacementApplied",
            placement_ok,
            placement_reason,
            &placement_message,
        ),
        make_condition(
            previous,
            cluster,
            "ShardsPruned",
            pruned_ok,
            pruned_reason,
            &pruned_message,
        ),
    ]
}

/// A condition that keeps its transition time while its status is unchanged.
fn make_condition(
    previous: &[Condition],
    cluster: &SearchCluster,
    type_: &str,
    ok: bool,
    reason: &str,
    message: &str,
) -> Condition {
    let status = if ok { "True" } else { "False" };
    let last_transition_time = previous
        .iter()
        .find(|c| c.type_ == type_ && c.status == status)
        .map_or_else(|| Time(Utc::now()), |c| c.last_transition_time.clone());
    Condition {
        type_: type_.to_string(),
        status: status.to_string(),
        reason: reason.to_string(),
        message: message.to_string(),
        observed_generation: cluster.metadata.generation,
        last_transition_time,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use k8s_openapi::api::core::v1::{HostPathVolumeSource, PodSpec, Volume};
    use kube::api::ObjectMeta;

    fn pod(name: &str, node: &str, path: &str) -> Pod {
        Pod {
            metadata: ObjectMeta {
                name: Some(name.into()),
                ..Default::default()
            },
            spec: Some(PodSpec {
                node_name: Some(node.into()),
                volumes: Some(vec![Volume {
                    name: "data".into(),
                    host_path: Some(HostPathVolumeSource {
                        path: path.into(),
                        type_: Some("Directory".into()),
                    }),
                    ..Default::default()
                }]),
                ..Default::default()
            }),
            status: None,
        }
    }

    #[test]
    fn a_new_replica_waits_for_any_pod_holding_its_directories() {
        let spec = crate::render::tests::sample_spec();
        let workloads = shard_workloads("summa", "summa", &spec);
        let old = pod("summa-server-old-abc", "node-a", "/data2/summa-server/");
        let conflict = host_path_conflict(&[old], &workloads[0]).unwrap();
        assert!(conflict.starts_with("summa-server-s2 waits for pod summa-server-old-abc"));
        // Same path on another node, or an unrelated directory, is free.
        let elsewhere = pod("other", "node-b", "/data2/summa-server");
        let unrelated = pod("db", "node-a", "/data2/postgres");
        assert!(host_path_conflict(&[elsewhere, unrelated], &workloads[0]).is_none());
    }

    #[test]
    fn placement_is_read_back_from_the_broker_container() {
        let spec = crate::render::tests::sample_spec();
        let deployment = broker_deployment("summa", "summa", &spec);
        assert_eq!(container_placement(&deployment), placement_args(&spec));
    }
}
