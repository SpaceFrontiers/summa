//! Console: a single-page UI and JSON API to inspect a `SearchCluster` with
//! live broker topology and to edit its spec after the same validation and
//! safety checks the operator applies.

use std::collections::BTreeSet;
use std::net::SocketAddr;
use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use k8s_openapi::api::apps::v1::Deployment;
use k8s_openapi::api::core::v1::Node;
use kube::ResourceExt;
use kube::api::{Api, ListParams, PostParams};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::broker::LiveIndex;
use crate::controller::{Context, container_placement, namespace_owner};
use crate::crd::{SearchCluster, SearchClusterSpec};
use crate::render::{broker_name, placement_args};
use crate::safety::{self, PlacementChange, ShardRemoval, parse_rules};
use crate::validate::{FieldError, validate};

const PAGE: &str = include_str!("console.html");

pub struct ConsoleState {
    pub ctx: Arc<Context>,
    pub namespace: String,
    /// Fixed cluster name; default is the namespace's owning cluster.
    pub cluster: Option<String>,
}

type Shared = Arc<ConsoleState>;

pub async fn serve(
    state: ConsoleState,
    addr: SocketAddr,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> anyhow::Result<()> {
    let app = Router::new()
        .route("/", get(|| async { Html(PAGE) }))
        .route("/api/cluster", get(get_cluster).put(put_cluster))
        .route("/api/cluster/check", post(check_cluster))
        .route("/api/broker/refresh", post(refresh_broker))
        .with_state(Arc::new(state));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    log::info!("console listening on http://{addr}");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await?;
    Ok(())
}

struct ApiError(StatusCode, Value);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(self.1)).into_response()
    }
}

impl From<kube::Error> for ApiError {
    fn from(error: kube::Error) -> Self {
        let status = match &error {
            kube::Error::Api(response) => {
                StatusCode::from_u16(response.code).unwrap_or(StatusCode::BAD_GATEWAY)
            }
            _ => StatusCode::BAD_GATEWAY,
        };
        ApiError(status, json!({ "error": error.to_string() }))
    }
}

async fn find_cluster(state: &ConsoleState) -> Result<SearchCluster, ApiError> {
    let api: Api<SearchCluster> = Api::namespaced(state.ctx.client.clone(), &state.namespace);
    let name = match &state.cluster {
        Some(name) => name.clone(),
        None => namespace_owner(&api.list(&ListParams::default()).await?.items).ok_or_else(|| {
            ApiError(
                StatusCode::NOT_FOUND,
                json!({ "error": format!("no SearchCluster in namespace {}", state.namespace) }),
            )
        })?,
    };
    Ok(api.get(&name).await?)
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct NodeView {
    name: String,
    ready: bool,
    schedulable: bool,
    cpu: Option<String>,
    memory: Option<String>,
}

async fn get_cluster(State(state): State<Shared>) -> Result<Json<Value>, ApiError> {
    let cluster = find_cluster(&state).await?;
    let name = cluster.name_any();
    let broker = state.ctx.broker(&name, &state.namespace, &cluster.spec);
    let nodes_api: Api<Node> = Api::all(state.ctx.client.clone());
    let list_params = ListParams::default();
    let (topology, backends, nodes) = tokio::join!(
        broker.topology(),
        broker.backends(),
        nodes_api.list(&list_params)
    );
    let mut errors = Vec::new();
    let mut keep = |result: Result<Value, String>| match result {
        Ok(value) => value,
        Err(error) => {
            errors.push(error);
            Value::Null
        }
    };
    let topology = keep(topology.map(|t| json!(t)).map_err(|e| format!("{e:#}")));
    let backends = keep(backends.map(|b| json!(b)).map_err(|e| format!("{e:#}")));
    let nodes = keep(
        nodes
            .map(|list| json!(list.items.iter().map(node_view).collect::<Vec<_>>()))
            .map_err(|e| format!("list nodes: {e}")),
    );
    Ok(Json(json!({
        "namespace": state.namespace,
        "cluster": {
            "name": name,
            "resourceVersion": cluster.resource_version(),
            "generation": cluster.metadata.generation,
            "spec": cluster.spec,
            "status": cluster.status,
        },
        "broker": {
            "address": broker.address(),
            "topology": topology,
            "backends": backends,
        },
        "nodes": nodes,
        "errors": errors,
    })))
}

fn node_view(node: &Node) -> NodeView {
    let status = node.status.as_ref();
    let ready = status
        .and_then(|s| s.conditions.as_ref())
        .and_then(|c| c.iter().find(|c| c.type_ == "Ready"))
        .is_some_and(|c| c.status == "True");
    let allocatable = status.and_then(|s| s.allocatable.as_ref());
    NodeView {
        name: node.name_any(),
        ready,
        schedulable: !node
            .spec
            .as_ref()
            .and_then(|s| s.unschedulable)
            .unwrap_or(false),
        cpu: allocatable.and_then(|a| a.get("cpu")).map(|q| q.0.clone()),
        memory: allocatable
            .and_then(|a| a.get("memory"))
            .map(|q| q.0.clone()),
    }
}

#[derive(Serialize, Default)]
#[serde(rename_all = "camelCase")]
struct CheckReport {
    ok: bool,
    errors: Vec<FieldError>,
    placement: Vec<PlacementChange>,
    removed_shards: Vec<ShardRemoval>,
    /// Live checks that could not run.
    live_errors: Vec<String>,
}

/// Validation plus the live checks the operator will gate on.
async fn check(
    state: &ConsoleState,
    current: &SearchCluster,
    proposed: &SearchClusterSpec,
) -> CheckReport {
    let name = current.name_any();
    let mut report = CheckReport {
        errors: validate(&name, proposed),
        ..Default::default()
    };
    let deployments: Api<Deployment> = Api::namespaced(state.ctx.client.clone(), &state.namespace);
    let broker = state.ctx.broker(&name, &state.namespace, &current.spec);
    let running = match deployments
        .get_opt(&broker_name(&name, &current.spec))
        .await
    {
        Ok(deployment) => deployment.as_ref().map(container_placement),
        Err(error) => {
            report
                .live_errors
                .push(format!("read broker Deployment: {error}"));
            None
        }
    };

    if let Some(running) = running {
        let desired = placement_args(proposed);
        if running != desired
            && let (Ok(current_rules), Ok(proposed_rules)) =
                (parse_rules(&running), parse_rules(&desired))
        {
            match broker.topology().await {
                Ok(topology) => {
                    let live: Vec<LiveIndex> = topology.iter().map(LiveIndex::from).collect();
                    report.placement =
                        safety::placement_changes(&live, &current_rules, &proposed_rules);
                }
                Err(error) => report.live_errors.push(format!(
                    "cannot verify live indexes for the placement change: {error:#}"
                )),
            }
        }
        let kept: BTreeSet<&str> = proposed.shards.iter().map(|s| s.id.as_str()).collect();
        let removed: Vec<String> = current
            .spec
            .shards
            .iter()
            .map(|s| s.id.clone())
            .filter(|id| !kept.contains(id.as_str()))
            .collect();
        if !removed.is_empty() {
            match broker.backends().await {
                Ok(backends) => report.removed_shards = safety::shard_removals(&backends, &removed),
                Err(error) => report.live_errors.push(format!(
                    "cannot confirm removed shards are empty: {error:#}"
                )),
            }
        }
    }
    report.ok = report.errors.is_empty()
        && report.live_errors.is_empty()
        && report.placement.iter().all(|c| c.blocked.is_none())
        && report.removed_shards.iter().all(|r| r.blocked.is_none());
    report
}

#[derive(Deserialize)]
struct CheckRequest {
    spec: SearchClusterSpec,
}

async fn check_cluster(
    State(state): State<Shared>,
    Json(request): Json<CheckRequest>,
) -> Result<Json<Value>, ApiError> {
    let cluster = find_cluster(&state).await?;
    Ok(Json(json!(check(&state, &cluster, &request.spec).await)))
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct UpdateRequest {
    resource_version: String,
    spec: SearchClusterSpec,
}

async fn put_cluster(
    State(state): State<Shared>,
    Json(request): Json<UpdateRequest>,
) -> Result<Json<Value>, ApiError> {
    let mut cluster = find_cluster(&state).await?;
    if cluster.resource_version().as_deref() != Some(request.resource_version.as_str()) {
        return Err(ApiError(
            StatusCode::CONFLICT,
            json!({ "error": "the SearchCluster changed since it was loaded; reload and reapply" }),
        ));
    }
    let report = check(&state, &cluster, &request.spec).await;
    if !report.ok {
        return Err(ApiError(StatusCode::UNPROCESSABLE_ENTITY, json!(report)));
    }
    cluster.spec = request.spec;
    // The resource version makes the replace conditional: a concurrent edit
    // fails with 409 instead of being overwritten.
    let api: Api<SearchCluster> = Api::namespaced(state.ctx.client.clone(), &state.namespace);
    let updated = api
        .replace(&cluster.name_any(), &PostParams::default(), &cluster)
        .await?;
    log::info!(
        "console updated SearchCluster {} to generation {:?}",
        updated.name_any(),
        updated.metadata.generation
    );
    Ok(Json(json!({
        "resourceVersion": updated.resource_version(),
        "generation": updated.metadata.generation,
        "report": report,
    })))
}

async fn refresh_broker(State(state): State<Shared>) -> Result<Json<Value>, ApiError> {
    let cluster = find_cluster(&state).await?;
    let broker = state
        .ctx
        .broker(&cluster.name_any(), &state.namespace, &cluster.spec);
    let polled = broker.refresh().await.map_err(|e| {
        ApiError(
            StatusCode::BAD_GATEWAY,
            json!({ "error": format!("{e:#}") }),
        )
    })?;
    Ok(Json(json!({ "backendsPolled": polled })))
}
