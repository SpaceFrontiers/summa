//! Operator: reconciles each `SearchCluster` into its shard Deployments,
//! broker Deployment and Service, then creates the index.
//!
//! One pass: validate; apply the workloads (a new shard waits until no pod
//! holds its directories); once every shard and the broker serve, create the
//! index or confirm it exists; write status.

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
use kube::runtime::watcher;
use kube::{Client, Resource, ResourceExt};
use log::{info, warn};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::json;

use crate::broker;
use crate::crd::{BrokerStatus, SearchCluster, SearchClusterStatus, ShardSpec, ShardStatus};
use crate::render::{
    CLUSTER_LABEL, COMPONENT_LABEL, MANAGER, Owner, broker_deployment, broker_name, broker_service,
    index_name, routing, shard_deployment, shard_name, shard_paths,
};
use crate::validate::{HostPath, host_paths, normalize_path, paths_overlap, validate};

const SETTLED_REQUEUE: Duration = Duration::from_secs(300);
const PENDING_REQUEUE: Duration = Duration::from_secs(10);
const ERROR_REQUEUE: Duration = Duration::from_secs(30);

const READY: &str = "Ready";
const INDEX_CREATED: &str = "IndexCreated";
/// `IndexCreated` reasons under which the index may exist on some shard, so
/// its routing must not change.
const PINNING_REASONS: &[&str] = &["Exists", "Creating", "PartiallyCreated"];

pub struct Context {
    pub client: Client,
    pub namespace: String,
    /// Broker address of a cluster; `{cluster}` and `{namespace}` are
    /// replaced (`--broker-endpoint`).
    pub broker_endpoint: String,
}

impl Context {
    pub fn broker(&self, cluster: &str) -> String {
        broker::endpoint(&self.broker_endpoint, cluster, &self.namespace)
    }

    pub fn clusters(&self) -> Api<SearchCluster> {
        self.api()
    }

    fn api<K: Resource<Scope = k8s_openapi::NamespaceResourceScope>>(&self) -> Api<K>
    where
        K::DynamicType: Default,
    {
        Api::namespaced(self.client.clone(), &self.namespace)
    }

    /// Directories claimed by clusters created before `cluster` (all others
    /// when it does not exist yet): on an overlap the older cluster keeps
    /// running and the newer one is invalid.
    pub async fn claimed_paths(
        &self,
        cluster: &str,
        created: Option<&Time>,
    ) -> kube::Result<Vec<HostPath>> {
        let key = |time: Option<&Time>, name: String| (time.map(|t| t.0), name);
        let own = created.map(|time| key(Some(time), cluster.to_string()));
        Ok(self
            .clusters()
            .list(&ListParams::default())
            .await?
            .items
            .iter()
            .filter(|other| other.name_any() != cluster)
            .filter(|other| {
                own.as_ref().is_none_or(|own| {
                    &key(other.creation_timestamp().as_ref(), other.name_any()) < own
                })
            })
            .flat_map(|other| host_paths(&other.name_any(), &other.spec))
            .collect())
    }
}

/// Routing fixed because the index may exist on some shard.
pub fn pinned_routing(cluster: &SearchCluster) -> Option<&str> {
    let status = cluster.status.as_ref()?;
    let pinned = status
        .conditions
        .iter()
        .any(|c| c.type_ == INDEX_CREATED && PINNING_REASONS.contains(&c.reason.as_str()));
    pinned.then_some(status.routing.as_deref()).flatten()
}

#[derive(Debug)]
pub struct ReconcileError(anyhow::Error);

impl fmt::Display for ReconcileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:#}", self.0)
    }
}

impl std::error::Error for ReconcileError {}

impl From<kube::Error> for ReconcileError {
    fn from(error: kube::Error) -> Self {
        Self(error.into())
    }
}

pub async fn run(ctx: Arc<Context>) {
    Controller::new(ctx.clusters(), watcher::Config::default())
        .owns(ctx.api::<Deployment>(), watcher::Config::default())
        .owns(ctx.api::<Service>(), watcher::Config::default())
        .shutdown_on_signal()
        .run(reconcile, error_policy, ctx)
        .for_each(|result| async move {
            if let Err(error) = result {
                warn!("reconcile failed: {error}");
            }
        })
        .await;
}

fn error_policy(cluster: Arc<SearchCluster>, error: &ReconcileError, _: Arc<Context>) -> Action {
    warn!("SearchCluster {}: {error}", cluster.name_any());
    count("error");
    Action::requeue(ERROR_REQUEUE)
}

fn count(result: &'static str) {
    metrics::counter!("summa_k8s_manager_reconcile_total", "result" => result).increment(1);
}

type Failure = (&'static str, String);

async fn reconcile(
    cluster: Arc<SearchCluster>,
    ctx: Arc<Context>,
) -> Result<Action, ReconcileError> {
    if cluster.metadata.deletion_timestamp.is_some() {
        return Ok(Action::await_change());
    }
    let name = cluster.name_any();
    let spec = &cluster.spec;
    let mut status = cluster.status.clone().unwrap_or_default();
    status.observed_generation = cluster.metadata.generation;
    let pinned = pinned_routing(&cluster).map(str::to_string);
    let claimed = ctx
        .claimed_paths(&name, cluster.creation_timestamp().as_ref())
        .await?;
    let errors = validate(&name, spec, pinned.as_deref(), &claimed);
    if !errors.is_empty() {
        let message = errors
            .iter()
            .map(|e| format!("{}: {}", e.field, e.message))
            .collect::<Vec<_>>()
            .join("; ");
        warn!("SearchCluster {name} is invalid, nothing changed: {message}");
        set_condition(&mut status, &cluster, READY, Err(("Invalid", message)));
        write_status(&ctx, &cluster, status).await?;
        count("invalid");
        return Ok(Action::requeue(SETTLED_REQUEUE));
    }

    let rule = pinned.unwrap_or_else(|| routing(&name, spec).to_string());
    status.routing = Some(rule.clone());
    let workloads = match apply_workloads(&ctx, &cluster, &rule).await {
        Ok(workloads) => workloads,
        Err(error) => {
            let message = format!("{error:#}");
            warn!("SearchCluster {name}: {message}");
            set_condition(&mut status, &cluster, READY, Err(("ApplyFailed", message)));
            write_status(&ctx, &cluster, status).await?;
            count("error");
            return Ok(Action::requeue(ERROR_REQUEUE));
        }
    };
    status.shards = workloads.shards.clone();
    status.broker = Some(BrokerStatus {
        ready_replicas: workloads.broker_ready_replicas,
    });

    let serving = workloads.serving();
    if serving {
        let index = check_index(&ctx, &cluster, &mut status).await?;
        set_condition(&mut status, &cluster, INDEX_CREATED, index);
    } else if !status.conditions.iter().any(|c| c.type_ == INDEX_CREATED) {
        // Keep an earlier result while shards restart: it may pin the routing.
        set_condition(
            &mut status,
            &cluster,
            INDEX_CREATED,
            Err((
                "Pending",
                "checked once every shard and the broker serve".into(),
            )),
        );
    }
    let ready = ready_outcome(&name, spec.shards.as_slice(), &workloads, &status);
    let settled = match &ready {
        Ok(_) => true,
        Err((reason, _)) => ["Suspended", "MissingSchema", "PartiallyCreated"].contains(reason),
    };
    set_condition(&mut status, &cluster, READY, ready);
    write_status(&ctx, &cluster, status).await?;
    count(if settled { "ok" } else { "progressing" });
    Ok(Action::requeue(if settled {
        SETTLED_REQUEUE
    } else {
        PENDING_REQUEUE
    }))
}

/// Shards and broker as applied in this pass.
struct Workloads {
    shards: Vec<ShardStatus>,
    broker_ready_replicas: i32,
    broker_ready: bool,
}

impl Workloads {
    fn serving(&self) -> bool {
        self.broker_ready && self.shards.iter().all(|s| s.ready)
    }
}

async fn apply_workloads(
    ctx: &Context,
    cluster: &SearchCluster,
    rule: &str,
) -> anyhow::Result<Workloads> {
    let name = cluster.name_any();
    let spec = &cluster.spec;
    let owner = Owner {
        cluster: &name,
        namespace: &ctx.namespace,
        reference: cluster
            .controller_owner_ref(&())
            .context("SearchCluster has a name and uid")?,
    };
    let deployments = ctx.api::<Deployment>();
    if pinned_routing(cluster).is_none() {
        prune_shards(&deployments, &name, spec.shards.len()).await?;
    }
    let mut shards = Vec::with_capacity(spec.shards.len());
    for (i, shard) in spec.shards.iter().enumerate() {
        let shard_name = shard_name(&name, i);
        let existing = deployments.get_opt(&shard_name).await?;
        if existing.is_none()
            && let Some(conflict) = host_path_conflict(ctx, shard).await?
        {
            shards.push(ShardStatus {
                ready: false,
                message: Some(conflict),
            });
            continue;
        }
        let applied = apply_owned(
            &deployments,
            &owner,
            existing,
            &shard_deployment(&owner, spec, i),
        )
        .await?;
        shards.push(ShardStatus {
            ready: deployment_ready(&applied),
            message: shard.suspended.then(|| "suspended".to_string()),
        });
    }
    let services = ctx.api::<Service>();
    let existing = services.get_opt(&name).await?;
    apply_owned(&services, &owner, existing, &broker_service(&owner)).await?;
    let broker = broker_name(&name);
    let existing = deployments.get_opt(&broker).await?;
    let broker = apply_owned(
        &deployments,
        &owner,
        existing,
        &broker_deployment(&owner, spec, rule),
    )
    .await?;
    Ok(Workloads {
        shards,
        broker_ready_replicas: broker
            .status
            .as_ref()
            .and_then(|s| s.ready_replicas)
            .unwrap_or(0),
        broker_ready: deployment_ready(&broker),
    })
}

/// Server-side apply, refusing an existing object this cluster does not
/// control (for example a hand-written Service of the same name).
async fn apply_owned<K>(
    api: &Api<K>,
    owner: &Owner<'_>,
    existing: Option<K>,
    desired: &K,
) -> anyhow::Result<K>
where
    K: Resource + Clone + Serialize + DeserializeOwned + fmt::Debug,
    K::DynamicType: Default,
{
    let name = desired.meta().name.clone().unwrap_or_default();
    if let Some(existing) = existing
        && !existing
            .owner_references()
            .iter()
            .any(|r| r.uid == owner.reference.uid && r.controller == Some(true))
    {
        anyhow::bail!(
            "{} {name} exists and is not owned by SearchCluster {}; delete or rename it",
            K::kind(&K::DynamicType::default()),
            owner.cluster
        );
    }
    api.patch(
        &name,
        &PatchParams::apply(MANAGER).force(),
        &Patch::Apply(desired),
    )
    .await
    .with_context(|| format!("apply {name}"))
}

/// Delete shard Deployments past the spec's shard count. Only runs while
/// the routing is not pinned, before any shard may host the index.
async fn prune_shards(
    deployments: &Api<Deployment>,
    cluster: &str,
    count: usize,
) -> kube::Result<()> {
    let selector = format!("{CLUSTER_LABEL}={cluster},{COMPONENT_LABEL}=shard");
    let wanted: BTreeSet<String> = (0..count).map(|i| shard_name(cluster, i)).collect();
    for deployment in deployments
        .list(&ListParams::default().labels(&selector))
        .await?
    {
        let name = deployment.name_any();
        if !wanted.contains(&name) {
            info!("deleting {name}: shard removed from SearchCluster {cluster}");
            deployments.delete(&name, &DeleteParams::default()).await?;
        }
    }
    Ok(())
}

/// A running pod on the shard's node that mounts one of its directories,
/// for example the terminating pod of a replaced Deployment.
async fn host_path_conflict(ctx: &Context, shard: &ShardSpec) -> kube::Result<Option<String>> {
    let on_node = ListParams::default().fields(&format!("spec.nodeName={}", shard.node));
    let pods = ctx.api::<Pod>().list(&on_node).await?.items;
    Ok(conflicting_pod(&pods, shard))
}

fn conflicting_pod(pods: &[Pod], shard: &ShardSpec) -> Option<String> {
    let wanted = shard_paths(shard).map(|path| normalize_path(&path));
    pods.iter()
        .filter(|pod| {
            // Finished pods (evicted, completed copy jobs) hold no mounts.
            let phase = pod.status.as_ref().and_then(|s| s.phase.as_deref());
            !matches!(phase, Some("Succeeded" | "Failed"))
        })
        .find_map(|pod| {
            let spec = pod.spec.as_ref()?;
            if spec.node_name.as_deref() != Some(shard.node.as_str()) {
                return None;
            }
            spec.volumes.iter().flatten().find_map(|volume| {
                let path = normalize_path(&volume.host_path.as_ref()?.path);
                wanted.iter().any(|w| paths_overlap(w, &path)).then(|| {
                    format!(
                        "waits for pod {} to release {path} on {}",
                        pod.name_any(),
                        shard.node
                    )
                })
            })
        })
}

/// What the index needs, from how many shards host it.
#[derive(Debug, PartialEq, Eq)]
enum IndexDecision {
    Exists,
    Create,
    MissingSchema,
    Partial(usize),
}

fn decide_index(hosting: usize, shards: usize, has_schema: bool) -> IndexDecision {
    match hosting {
        n if n == shards => IndexDecision::Exists,
        0 if has_schema => IndexDecision::Create,
        0 => IndexDecision::MissingSchema,
        n => IndexDecision::Partial(n),
    }
}

/// Create the index when no shard has it. The routing is pinned in status
/// before `CreateIndex` so it can never change under a created partition.
async fn check_index(
    ctx: &Context,
    cluster: &SearchCluster,
    status: &mut SearchClusterStatus,
) -> Result<Result<&'static str, Failure>, ReconcileError> {
    let name = cluster.name_any();
    let index = index_name(&name);
    let address = ctx.broker(&name);
    let shards = cluster.spec.shards.len();
    let hosting = match broker::index_shards(&address, &index).await {
        Ok(hosting) => hosting.len(),
        Err(error) => return Ok(Err(("BrokerUnavailable", format!("{error:#}")))),
    };
    Ok(
        match decide_index(hosting, shards, cluster.spec.schema.is_some()) {
            IndexDecision::Exists => Ok("Exists"),
            IndexDecision::MissingSchema => Err((
                "MissingSchema",
                format!("index {index} does not exist and spec.schema is empty"),
            )),
            IndexDecision::Partial(n) => Err((
                "PartiallyCreated",
                format!(
                    "{n} of {shards} shards host {index}; reads and writes fail until every shard has it"
                ),
            )),
            IndexDecision::Create => {
                set_condition(
                    status,
                    cluster,
                    INDEX_CREATED,
                    Err(("Creating", format!("creating {index} through {address}"))),
                );
                write_status(ctx, cluster, status.clone()).await?;
                info!("creating index {index} through {address}");
                let schema = cluster.spec.schema.as_deref().unwrap_or_default();
                match broker::create_index(&address, &index, schema).await {
                    Ok(()) => Err((
                        "Creating",
                        format!("created {index}; waiting for every shard to report it"),
                    )),
                    Err(error) => Err(("Creating", format!("{error:#}"))),
                }
            }
        },
    )
}

fn ready_outcome(
    cluster: &str,
    shards: &[ShardSpec],
    workloads: &Workloads,
    status: &SearchClusterStatus,
) -> Result<&'static str, Failure> {
    let waiting: Vec<&str> = workloads
        .shards
        .iter()
        .filter(|s| !s.ready)
        .filter_map(|s| s.message.as_deref())
        .filter(|m| m.starts_with("waits"))
        .collect();
    if !waiting.is_empty() {
        return Err(("Waiting", waiting.join("; ")));
    }
    let suspended: Vec<String> = (0..shards.len())
        .filter(|&i| shards[i].suspended)
        .map(|i| shard_name(cluster, i))
        .collect();
    if !suspended.is_empty() {
        return Err(("Suspended", format!("suspended: {}", suspended.join(", "))));
    }
    if !workloads.serving() {
        let not_ready: Vec<String> = workloads
            .shards
            .iter()
            .enumerate()
            .filter(|(_, s)| !s.ready)
            .map(|(i, _)| shard_name(cluster, i))
            .chain((!workloads.broker_ready).then(|| broker_name(cluster)))
            .collect();
        return Err((
            "Progressing",
            format!("not ready: {}", not_ready.join(", ")),
        ));
    }
    match status.conditions.iter().find(|c| c.type_ == INDEX_CREATED) {
        Some(c) if c.status == "True" => Ok("Reconciled"),
        Some(c) => Err((
            PINNING_REASONS
                .iter()
                .chain(&["MissingSchema", "BrokerUnavailable", "Pending"])
                .find(|r| **r == c.reason)
                .copied()
                .unwrap_or("Progressing"),
            c.message.clone(),
        )),
        None => Err(("Progressing", String::new())),
    }
}

/// Every replica runs the current template and is ready.
fn deployment_ready(deployment: &Deployment) -> bool {
    let Some(status) = &deployment.status else {
        return false;
    };
    let wanted = deployment
        .spec
        .as_ref()
        .and_then(|s| s.replicas)
        .unwrap_or(1);
    let at = |count: Option<i32>| count.unwrap_or(0) == wanted;
    wanted > 0
        && status.observed_generation >= deployment.metadata.generation
        && at(status.replicas)
        && at(status.updated_replicas)
        && at(status.ready_replicas)
        && at(status.available_replicas)
}

/// Set a condition, keeping its transition time while its status holds.
fn set_condition(
    status: &mut SearchClusterStatus,
    cluster: &SearchCluster,
    type_: &str,
    outcome: Result<&str, Failure>,
) {
    let (value, reason, message) = match outcome {
        Ok(reason) => ("True", reason.to_string(), String::new()),
        Err((reason, message)) => ("False", reason.to_string(), message),
    };
    let last_transition_time = status
        .conditions
        .iter()
        .find(|c| c.type_ == type_ && c.status == value)
        .map_or_else(|| Time(Utc::now()), |c| c.last_transition_time.clone());
    let condition = Condition {
        type_: type_.to_string(),
        status: value.to_string(),
        reason,
        message,
        observed_generation: cluster.metadata.generation,
        last_transition_time,
    };
    match status.conditions.iter_mut().find(|c| c.type_ == type_) {
        Some(existing) => *existing = condition,
        None => status.conditions.push(condition),
    }
}

/// Server-side apply of the whole status, so fields that become empty are
/// removed rather than left behind by a merge. Skipped when unchanged.
async fn write_status(
    ctx: &Context,
    cluster: &SearchCluster,
    status: SearchClusterStatus,
) -> kube::Result<()> {
    if cluster.status.as_ref() == Some(&status) {
        return Ok(());
    }
    let patch = json!({
        "apiVersion": SearchCluster::api_version(&()),
        "kind": SearchCluster::kind(&()),
        "status": status,
    });
    ctx.clusters()
        .patch_status(
            &cluster.name_any(),
            &PatchParams::apply(MANAGER).force(),
            &Patch::Apply(&patch),
        )
        .await
        .map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;
    use k8s_openapi::api::core::v1::{HostPathVolumeSource, PodSpec, PodStatus, Volume};
    use kube::api::ObjectMeta;

    #[test]
    fn index_is_created_only_when_no_shard_has_it() {
        assert_eq!(decide_index(3, 3, false), IndexDecision::Exists);
        assert_eq!(decide_index(0, 3, true), IndexDecision::Create);
        assert_eq!(decide_index(0, 3, false), IndexDecision::MissingSchema);
        // Creating again would fail on the shards that have it.
        assert_eq!(decide_index(1, 3, true), IndexDecision::Partial(1));
    }

    #[test]
    fn routing_is_pinned_from_the_create_attempt_on() {
        let mut cluster = SearchCluster::new("documents-1", crate::render::tests::spec());
        let mut status = SearchClusterStatus {
            routing: Some("documents_1=0:5,1:4".into()),
            ..Default::default()
        };
        for (reason, pinned) in [
            ("Pending", false),
            ("MissingSchema", false),
            ("Creating", true),
            ("PartiallyCreated", true),
            ("Exists", true),
        ] {
            let outcome = if reason == "Exists" {
                Ok(reason)
            } else {
                Err((reason, String::new()))
            };
            set_condition(&mut status, &cluster, INDEX_CREATED, outcome);
            cluster.status = Some(status.clone());
            assert_eq!(pinned_routing(&cluster).is_some(), pinned, "{reason}");
        }
    }

    #[test]
    fn a_new_shard_waits_only_for_running_pods_holding_its_directories() {
        let pod = |name: &str, node: &str, path: &str, phase: &str| Pod {
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
                        type_: None,
                    }),
                    ..Default::default()
                }]),
                ..Default::default()
            }),
            status: Some(PodStatus {
                phase: Some(phase.into()),
                ..Default::default()
            }),
        };
        let spec = crate::render::tests::spec();
        let shard = &spec.shards[0];
        let old = pod("old", "node-a", "/data2/summa/documents_1/", "Running");
        assert!(conflicting_pod(&[old], shard).unwrap().contains("pod old"));
        let finished = [
            pod("evicted", "node-a", "/data2/summa/documents_1", "Failed"),
            pod("copy", "node-a", "/data2/summa/documents_1", "Succeeded"),
            pod("other", "node-b", "/data2/summa/documents_1", "Running"),
            pod("db", "node-a", "/data2/postgres", "Running"),
        ];
        assert!(conflicting_pod(&finished, shard).is_none());
    }
}
