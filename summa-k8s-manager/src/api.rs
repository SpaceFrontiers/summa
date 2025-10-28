//! JSON API and console: list clusters with live index statistics, and
//! create, edit and delete them after the operator's validation.
//!
//! Two listeners serve the same routes. The console listener binds loopback
//! for `kubectl port-forward` and only accepts `localhost` Host headers (a
//! web page cannot rebind its name onto it); the API listener requires the
//! bearer token.

use std::sync::Arc;

use axum::extract::{Path, Query, Request, State};
use axum::http::{StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post, put};
use axum::{Json, Router};
use k8s_openapi::api::core::v1::Node;
use kube::ResourceExt;
use kube::api::{Api, DeleteParams, ListParams, PostParams};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::broker;
use crate::controller::{Context, pinned_routing};
use crate::crd::{SearchCluster, SearchClusterSpec};
use crate::render::index_name;
use crate::validate::{FieldError, validate};

const PAGE: &str = include_str!("console.html");
/// Annotation that makes the API refuse to delete a cluster.
pub const DELETION_PROTECTION: &str = "summa.spacefrontiers.org/deletion-protection";

/// Who a listener serves.
#[derive(Clone)]
pub enum Access {
    /// Loopback console: requests must name `localhost` as their host.
    Console,
    /// In-cluster API: requests must carry this bearer token.
    Token(Arc<str>),
}

type Shared = Arc<Context>;

pub fn router(ctx: Arc<Context>, access: Access) -> Router {
    Router::new()
        .route("/", get(|| async { Html(PAGE) }))
        .route("/api/clusters", get(list).post(create))
        .route("/api/clusters/check", post(check))
        .route("/api/clusters/{name}", put(update).delete(remove))
        .layer(middleware::from_fn_with_state(access, authorize))
        .with_state(ctx)
}

async fn authorize(State(access): State<Access>, request: Request, next: Next) -> Response {
    if allowed(&access, request.headers()) {
        next.run(request).await
    } else {
        ApiError::new(StatusCode::UNAUTHORIZED, "not authorized").into_response()
    }
}

fn allowed(access: &Access, headers: &axum::http::HeaderMap) -> bool {
    let value = |name| headers.get(name).and_then(|value| value.to_str().ok());
    match access {
        Access::Console => value(header::HOST).is_some_and(is_local_host),
        Access::Token(token) => value(header::AUTHORIZATION)
            .and_then(|value| value.strip_prefix("Bearer "))
            .is_some_and(|bearer| constant_time_eq(token.as_bytes(), bearer.as_bytes())),
    }
}

fn is_local_host(host: &str) -> bool {
    let name = host.rsplit_once(':').map_or(host, |(name, port)| {
        if port.bytes().all(|b| b.is_ascii_digit()) {
            name
        } else {
            host
        }
    });
    matches!(name, "localhost" | "127.0.0.1" | "[::1]")
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

struct ApiError(StatusCode, Value);

impl ApiError {
    fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self(status, json!({ "error": message.into() }))
    }

    fn invalid(errors: Vec<FieldError>) -> Self {
        Self(
            StatusCode::UNPROCESSABLE_ENTITY,
            json!({ "errors": errors }),
        )
    }
}

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
        Self::new(status, error.to_string())
    }
}

type ApiResult = Result<Json<Value>, ApiError>;

async fn list(State(ctx): State<Shared>) -> ApiResult {
    let mut clusters = ctx.clusters().list(&ListParams::default()).await?.items;
    clusters.sort_by_key(ResourceExt::name_any);
    let views = futures::future::join_all(clusters.iter().map(|cluster| view(&ctx, cluster))).await;
    let nodes: Api<Node> = Api::all(ctx.client.clone());
    let nodes = match nodes.list(&ListParams::default()).await {
        Ok(list) => json!(list.items.iter().map(node_view).collect::<Vec<_>>()),
        Err(error) => json!({ "error": error.to_string() }),
    };
    Ok(Json(
        json!({ "namespace": ctx.namespace, "clusters": views, "nodes": nodes }),
    ))
}

/// A cluster with the live statistics of its index.
async fn view(ctx: &Context, cluster: &SearchCluster) -> Value {
    let name = cluster.name_any();
    let address = ctx.broker(&name);
    let live = broker::index_shards(&address, &index_name(&name)).await;
    json!({
        "name": name,
        "index": index_name(&name),
        "endpoint": address,
        "generation": cluster.metadata.generation,
        "protected": protected(cluster),
        "spec": cluster.spec,
        "status": cluster.status,
        "shards": live.as_ref().ok(),
        "liveError": live.err().map(|e| format!("{e:#}")),
    })
}

fn protected(cluster: &SearchCluster) -> bool {
    cluster
        .annotations()
        .get(DELETION_PROTECTION)
        .is_some_and(|v| v == "true")
}

fn node_view(node: &Node) -> Value {
    let status = node.status.as_ref();
    let ready = status
        .and_then(|s| s.conditions.as_ref())
        .and_then(|c| c.iter().find(|c| c.type_ == "Ready"))
        .is_some_and(|c| c.status == "True");
    let allocatable = |key: &str| {
        status
            .and_then(|s| s.allocatable.as_ref())
            .and_then(|a| a.get(key))
            .map(|q| q.0.clone())
    };
    json!({
        "name": node.name_any(),
        "ready": ready,
        "schedulable": !node.spec.as_ref().and_then(|s| s.unschedulable).unwrap_or(false),
        "cpu": allocatable("cpu"),
        "memory": allocatable("memory"),
    })
}

/// The spec in a request, refusing fields the resource does not have (a
/// misspelled `suspend` must not silently do nothing).
fn parse_spec(spec: Value) -> Result<SearchClusterSpec, ApiError> {
    let typed: SearchClusterSpec = serde_json::from_value(spec.clone()).map_err(|error| {
        ApiError::invalid(vec![FieldError {
            field: "spec".into(),
            message: error.to_string(),
        }])
    })?;
    let mut unknown = Vec::new();
    unknown_fields(&spec, &json!(typed), "spec", &mut unknown);
    if !unknown.is_empty() {
        return Err(ApiError::invalid(
            unknown
                .into_iter()
                .map(|field| FieldError {
                    field,
                    message: "unknown field".into(),
                })
                .collect(),
        ));
    }
    Ok(typed)
}

/// Paths present in the request but absent after a round trip through the
/// typed spec.
fn unknown_fields(request: &Value, typed: &Value, path: &str, out: &mut Vec<String>) {
    match (request, typed) {
        (Value::Object(request), Value::Object(typed)) => {
            for (key, value) in request {
                let field = format!("{path}.{key}");
                match typed.get(key) {
                    Some(typed) => unknown_fields(value, typed, &field, out),
                    // Empty values are omitted when serialized.
                    None if is_empty(value) => {}
                    None => out.push(field),
                }
            }
        }
        (Value::Array(request), Value::Array(typed)) => {
            for (i, (value, typed)) in request.iter().zip(typed).enumerate() {
                unknown_fields(value, typed, &format!("{path}[{i}]"), out);
            }
        }
        _ => {}
    }
}

fn is_empty(value: &Value) -> bool {
    match value {
        Value::Null | Value::Bool(false) => true,
        Value::Array(items) => items.is_empty(),
        Value::Object(fields) => fields.is_empty(),
        _ => false,
    }
}

/// Validation the operator applies: the routing of an existing index and
/// the directories of older clusters.
async fn errors(
    ctx: &Context,
    name: &str,
    spec: &SearchClusterSpec,
    current: Option<&SearchCluster>,
) -> Result<Vec<FieldError>, ApiError> {
    let created = current.and_then(|cluster| cluster.creation_timestamp());
    let claimed = ctx.claimed_paths(name, created.as_ref()).await?;
    Ok(validate(
        name,
        spec,
        current.and_then(pinned_routing),
        &claimed,
    ))
}

#[derive(Deserialize)]
struct ClusterRequest {
    name: String,
    spec: Value,
}

async fn check(State(ctx): State<Shared>, Json(request): Json<ClusterRequest>) -> ApiResult {
    let spec = parse_spec(request.spec)?;
    let current = ctx.clusters().get_opt(&request.name).await?;
    let errors = errors(&ctx, &request.name, &spec, current.as_ref()).await?;
    Ok(Json(json!({ "errors": errors })))
}

async fn create(
    State(ctx): State<Shared>,
    Json(request): Json<ClusterRequest>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    let spec = parse_spec(request.spec)?;
    let errors = errors(&ctx, &request.name, &spec, None).await?;
    if !errors.is_empty() {
        return Err(ApiError::invalid(errors));
    }
    let mut cluster = SearchCluster::new(&request.name, spec);
    cluster.metadata.namespace = Some(ctx.namespace.clone());
    let created = ctx
        .clusters()
        .create(&PostParams::default(), &cluster)
        .await?;
    log::info!("API created SearchCluster {}", created.name_any());
    Ok((StatusCode::CREATED, Json(view(&ctx, &created).await)))
}

#[derive(Deserialize)]
struct UpdateRequest {
    /// `metadata.generation` the edit is based on: a spec change since then
    /// is a conflict, a status update is not.
    generation: i64,
    spec: Value,
}

async fn update(
    State(ctx): State<Shared>,
    Path(name): Path<String>,
    Json(request): Json<UpdateRequest>,
) -> ApiResult {
    let spec = parse_spec(request.spec)?;
    let mut cluster = ctx.clusters().get(&name).await?;
    if cluster.metadata.generation != Some(request.generation) {
        return Err(ApiError::new(
            StatusCode::CONFLICT,
            format!("{name} changed since it was loaded; reload and reapply"),
        ));
    }
    let errors = errors(&ctx, &name, &spec, Some(&cluster)).await?;
    if !errors.is_empty() {
        return Err(ApiError::invalid(errors));
    }
    // Replace with the resource version just read: an edit landing in
    // between fails with 409 instead of being overwritten.
    cluster.spec = spec;
    let updated = ctx
        .clusters()
        .replace(&name, &PostParams::default(), &cluster)
        .await?;
    log::info!(
        "API updated SearchCluster {name} to generation {:?}",
        updated.metadata.generation
    );
    Ok(Json(view(&ctx, &updated).await))
}

#[derive(Deserialize)]
struct DeleteQuery {
    confirm: Option<String>,
}

async fn remove(
    State(ctx): State<Shared>,
    Path(name): Path<String>,
    Query(query): Query<DeleteQuery>,
) -> ApiResult {
    if query.confirm.as_deref() != Some(name.as_str()) {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            format!("confirm deletion with ?confirm={name}"),
        ));
    }
    let cluster = ctx.clusters().get(&name).await?;
    if protected(&cluster) {
        return Err(ApiError::new(
            StatusCode::CONFLICT,
            format!("{name} has {DELETION_PROTECTION}: \"true\"; remove the annotation first"),
        ));
    }
    ctx.clusters()
        .delete(&name, &DeleteParams::default())
        .await?;
    log::info!("API deleted SearchCluster {name}");
    Ok(Json(json!({ "deleted": name })))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn console_accepts_only_local_host_names() {
        for host in [
            "localhost",
            "localhost:8080",
            "127.0.0.1:8080",
            "[::1]:8080",
        ] {
            assert!(is_local_host(host), "{host}");
        }
        for host in [
            "evil.example:8080",
            "localhost.evil.example",
            "10.0.0.5:8080",
        ] {
            assert!(!is_local_host(host), "{host}");
        }
    }

    #[test]
    fn misspelled_fields_are_refused() {
        let mut spec = json!(crate::render::tests::spec());
        spec["shards"][0]["suspend"] = json!(true);
        spec["broker"]["replica"] = json!(2);
        spec["shard"]["args"] = json!([]);
        let Err(ApiError(status, body)) = parse_spec(spec) else {
            panic!("unknown fields accepted");
        };
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        let fields: Vec<&str> = body["errors"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["field"].as_str().unwrap())
            .collect();
        assert_eq!(fields, ["spec.broker.replica", "spec.shards[0].suspend"]);
    }
}
