//! IndexService routing: every write RPC goes whole to the master of the
//! shard hosting the index; responses are forwarded verbatim so client-side
//! contracts (duplicate-primary-key detection, backpressure partial-batch
//! semantics, `DocumentError.index` positions) are byte-faithful.
//!
//! A partitioned index (a placement rule listing several shards) fans every
//! write out: CreateIndex, Commit, ForceMerge, Reorder, DeleteIndex, Retrain
//! and Alter go to every partition master and all must succeed; documents
//! go to the partition their primary key hashes to (`partition::route_documents`),
//! with error positions mapped back to the request.
//!
//! CreateIndex is the one placement decision the broker owns: a glob rule
//! (or the default policy) picks the shard(s) for a new index, which is how
//! dated full-build names land on the right host.

use std::sync::Arc;
use std::time::{Duration, Instant};

use log::info;
use prost::Message;
use tonic::{Request, Response, Status, Streaming};

use crate::client::forward_timeout;
use crate::context::BrokerContext;
use crate::metrics as m;
use crate::partition;
use crate::proto::hermes::index_service_server::IndexService;
use crate::proto::hermes::{
    AlterVectorIndexRequest, AlterVectorIndexResponse, BatchIndexDocumentsRequest,
    BatchIndexDocumentsResponse, CommitRequest, CommitResponse, CreateIndexRequest,
    CreateIndexResponse, DeleteIndexRequest, DeleteIndexResponse, ForceMergeRequest,
    ForceMergeResponse, IndexDocumentRequest, IndexDocumentsResponse, ListIndexesRequest,
    ListIndexesResponse, NamedDocument, ReorderRequest, ReorderResponse, RetrainVectorIndexRequest,
    RetrainVectorIndexResponse,
};
use crate::routes::{Route, Target, record_backend};

/// Streaming IndexDocuments is buffered per index and forwarded as
/// BatchIndexDocuments once either threshold is reached (mirrors the
/// server's own internal 512-message stream batching).
const STREAM_FLUSH_DOCS: usize = 512;
const STREAM_FLUSH_BYTES: usize = 4 * 1024 * 1024;

pub struct BrokerIndexService {
    pub ctx: Arc<BrokerContext>,
}

/// One unary write RPC sent whole to every target of the route (one for an
/// unpartitioned index); all must succeed, responses are folded by `merge`.
macro_rules! forward_write {
    ($self:ident, $request:ident, $method:ident, $rpc_name:literal, $merge:expr) => {{
        $self.ctx.check_admission()?;
        let timeout = forward_timeout($request.metadata());
        let req = $request.into_inner();
        let index_name = req.index_name.clone();
        let route = $self.ctx.write_route(&index_name)?;
        let calls = route.targets().iter().map(|target| {
            let mut outbound = Request::new(req.clone());
            if let Some(t) = timeout {
                outbound.set_timeout(t);
            }
            let mut client = target.channels.index.clone();
            let target = target.clone();
            async move {
                let started = Instant::now();
                let result = client.$method(outbound).await;
                (target, started, result)
            }
        });
        let mut responses = Vec::with_capacity(route.targets().len());
        for (target, started, result) in futures::future::join_all(calls).await {
            let code = result
                .as_ref()
                .map(|_| tonic::Code::Ok)
                .unwrap_or_else(|s| s.code());
            record_backend(&target.backend_id, $rpc_name, started, code);
            match result {
                Ok(response) => responses.push(response.into_inner()),
                Err(status) if route.is_partitioned() => {
                    return Err(partition::partition_failure(
                        &index_name,
                        &target.shard,
                        status,
                    ));
                }
                Err(status) => return Err(status),
            }
        }
        let merge = $merge;
        Ok(Response::new(merge(responses)))
    }};
}

impl BrokerIndexService {
    /// Send one batch to the route: whole to a single master, or split by
    /// primary key across the partitions.
    async fn write_batch(
        &self,
        index_name: &str,
        documents: Vec<NamedDocument>,
        timeout: Option<Duration>,
    ) -> Result<BatchIndexDocumentsResponse, Status> {
        let route = self.ctx.write_route(index_name)?;
        match route {
            Route::Single(target) => {
                let mut outbound = Request::new(BatchIndexDocumentsRequest {
                    index_name: index_name.to_string(),
                    documents,
                });
                if let Some(t) = timeout {
                    outbound.set_timeout(t);
                }
                let started = Instant::now();
                let result = target
                    .channels
                    .index
                    .clone()
                    .batch_index_documents(outbound)
                    .await;
                let code = result
                    .as_ref()
                    .map(|_| tonic::Code::Ok)
                    .unwrap_or_else(|s| s.code());
                record_backend(&target.backend_id, "batch_index_documents", started, code);
                Ok(result?.into_inner())
            }
            Route::Partitioned(targets) => {
                let primary_key = self.ctx.primary_key_field(index_name, &targets[0]).await?;
                let routed = partition::route_documents(documents, &primary_key, targets.len());
                let calls = routed
                    .groups
                    .into_iter()
                    .zip(targets.iter())
                    .filter(|((_, documents), _)| !documents.is_empty())
                    .map(|((positions, documents), target)| {
                        let mut outbound = Request::new(BatchIndexDocumentsRequest {
                            index_name: index_name.to_string(),
                            documents,
                        });
                        if let Some(t) = timeout {
                            outbound.set_timeout(t);
                        }
                        let mut client = target.channels.index.clone();
                        let target: Target = target.clone();
                        async move {
                            let started = Instant::now();
                            let result = client.batch_index_documents(outbound).await;
                            (target, positions, started, result)
                        }
                    });
                let mut responses = Vec::with_capacity(targets.len());
                for (target, positions, started, result) in futures::future::join_all(calls).await {
                    let code = result
                        .as_ref()
                        .map(|_| tonic::Code::Ok)
                        .unwrap_or_else(|s| s.code());
                    record_backend(&target.backend_id, "batch_index_documents", started, code);
                    match result {
                        Ok(response) => responses.push((positions, response.into_inner())),
                        Err(status) => {
                            return Err(partition::partition_failure(
                                index_name,
                                &target.shard,
                                status,
                            ));
                        }
                    }
                }
                Ok(partition::merge_batch_responses(
                    responses,
                    routed.unroutable,
                ))
            }
        }
    }
}

#[tonic::async_trait]
impl IndexService for BrokerIndexService {
    async fn create_index(
        &self,
        request: Request<CreateIndexRequest>,
    ) -> Result<Response<CreateIndexResponse>, Status> {
        self.ctx.check_admission()?;
        let timeout = forward_timeout(request.metadata());
        let req = request.into_inner();
        let index_name = req.index_name.clone();
        let targets: Vec<(String, String, String)> = {
            let snapshot = self.ctx.snapshot.load();
            snapshot
                .select_create_backends(&index_name, &self.ctx.placement)
                .inspect_err(|status| {
                    metrics::counter!(
                        m::WRITE_REJECTED,
                        "index" => index_name.clone(),
                        "reason" => crate::context::code_label(status.code()),
                    )
                    .increment(1);
                })?
                .into_iter()
                .map(|backend| {
                    (
                        backend.endpoint.addr.clone(),
                        backend.endpoint.id.0.clone(),
                        backend.endpoint.shard.0.clone(),
                    )
                })
                .collect()
        };
        let partitioned = targets.len() > 1;
        for (_, backend_id, shard) in &targets {
            info!(
                "creating index '{}' on shard '{}' (backend {}){}",
                index_name,
                shard,
                backend_id,
                if partitioned {
                    format!(", partition of {}", targets.len())
                } else {
                    String::new()
                }
            );
        }
        let mut success = true;
        for (addr, backend_id, shard) in &targets {
            let channels = self.ctx.pool.get(addr)?;
            let mut outbound = Request::new(req.clone());
            if let Some(t) = timeout {
                outbound.set_timeout(t);
            }
            let started = Instant::now();
            let result = channels.index.clone().create_index(outbound).await;
            let code = result
                .as_ref()
                .map(|_| tonic::Code::Ok)
                .unwrap_or_else(|s| s.code());
            record_backend(backend_id, "create_index", started, code);
            match result {
                Ok(response) => success &= response.into_inner().success,
                Err(status) => {
                    info!(
                        "create_index '{index_name}' failed on shard '{shard}': {}",
                        status.code()
                    );
                    // Make any partition that did get created routable, so a
                    // retry sees AlreadyExists there instead of a gap.
                    self.ctx.refresh.notify_one();
                    return Err(if partitioned {
                        partition::partition_failure(&index_name, shard, status)
                    } else {
                        status
                    });
                }
            }
        }
        // Make the new index routable without waiting a poll interval.
        self.ctx.refresh.notify_one();
        Ok(Response::new(CreateIndexResponse { success }))
    }

    async fn index_documents(
        &self,
        request: Request<Streaming<IndexDocumentRequest>>,
    ) -> Result<Response<IndexDocumentsResponse>, Status> {
        self.ctx.check_admission()?;
        let budget = forward_timeout(request.metadata());
        let started = Instant::now();
        let mut stream = request.into_inner();
        let mut current_index: Option<String> = None;
        let mut buffer: Vec<NamedDocument> = Vec::new();
        let mut buffered_bytes = 0usize;
        let mut indexed_count = 0u32;
        let mut errors = Vec::new();
        // Forward one buffered run as a BatchIndexDocuments to the index's
        // current master (or partition masters). NOTE: like hermes-server's
        // own stream handling, DocumentError.index positions are relative to
        // the flushed batch, not the whole stream.
        async fn flush(
            service: &BrokerIndexService,
            index_name: &str,
            documents: Vec<NamedDocument>,
            budget: Option<Duration>,
            started: Instant,
            indexed_count: &mut u32,
            errors: &mut Vec<crate::proto::hermes::DocumentError>,
        ) -> Result<(), Status> {
            if documents.is_empty() {
                return Ok(());
            }
            metrics::counter!(m::STREAM_FLUSHES, "index" => index_name.to_string()).increment(1);
            let timeout = match budget {
                Some(total) => {
                    let remaining = total.saturating_sub(started.elapsed());
                    if remaining.is_zero() {
                        return Err(Status::deadline_exceeded(
                            "client deadline exhausted mid-stream",
                        ));
                    }
                    Some(remaining)
                }
                None => None,
            };
            let response = service.write_batch(index_name, documents, timeout).await?;
            *indexed_count += response.indexed_count;
            errors.extend(response.errors);
            Ok(())
        }
        while let Some(message) = stream.message().await? {
            if current_index.as_deref() != Some(message.index_name.as_str()) {
                if let Some(previous) = current_index.take() {
                    flush(
                        self,
                        &previous,
                        std::mem::take(&mut buffer),
                        budget,
                        started,
                        &mut indexed_count,
                        &mut errors,
                    )
                    .await?;
                    buffered_bytes = 0;
                }
                current_index = Some(message.index_name.clone());
            }
            let document = NamedDocument {
                fields: message.fields,
            };
            buffered_bytes += document.encoded_len();
            buffer.push(document);
            if buffer.len() >= STREAM_FLUSH_DOCS || buffered_bytes >= STREAM_FLUSH_BYTES {
                let index_name = current_index.clone().expect("set above");
                flush(
                    self,
                    &index_name,
                    std::mem::take(&mut buffer),
                    budget,
                    started,
                    &mut indexed_count,
                    &mut errors,
                )
                .await?;
                buffered_bytes = 0;
            }
        }
        if let Some(index_name) = current_index {
            flush(
                self,
                &index_name,
                std::mem::take(&mut buffer),
                budget,
                started,
                &mut indexed_count,
                &mut errors,
            )
            .await?;
        }
        Ok(Response::new(IndexDocumentsResponse {
            indexed_count,
            errors,
        }))
    }

    async fn batch_index_documents(
        &self,
        request: Request<BatchIndexDocumentsRequest>,
    ) -> Result<Response<BatchIndexDocumentsResponse>, Status> {
        self.ctx.check_admission()?;
        let timeout = forward_timeout(request.metadata());
        let req = request.into_inner();
        let response = self
            .write_batch(&req.index_name, req.documents, timeout)
            .await?;
        Ok(Response::new(response))
    }

    async fn commit(
        &self,
        request: Request<CommitRequest>,
    ) -> Result<Response<CommitResponse>, Status> {
        forward_write!(self, request, commit, "commit", |parts: Vec<
            CommitResponse,
        >| {
            CommitResponse {
                success: parts.iter().all(|p| p.success),
                num_docs: parts.iter().fold(0u32, |n, p| n.saturating_add(p.num_docs)),
            }
        })
    }

    async fn force_merge(
        &self,
        request: Request<ForceMergeRequest>,
    ) -> Result<Response<ForceMergeResponse>, Status> {
        forward_write!(self, request, force_merge, "force_merge", |parts: Vec<
            ForceMergeResponse,
        >| {
            ForceMergeResponse {
                success: parts.iter().all(|p| p.success),
                num_segments: parts
                    .iter()
                    .fold(0u32, |n, p| n.saturating_add(p.num_segments)),
            }
        })
    }

    async fn reorder(
        &self,
        request: Request<ReorderRequest>,
    ) -> Result<Response<ReorderResponse>, Status> {
        forward_write!(self, request, reorder, "reorder", |parts: Vec<
            ReorderResponse,
        >| {
            ReorderResponse {
                success: parts.iter().all(|p| p.success),
                num_segments: parts
                    .iter()
                    .fold(0u32, |n, p| n.saturating_add(p.num_segments)),
            }
        })
    }

    async fn delete_index(
        &self,
        request: Request<DeleteIndexRequest>,
    ) -> Result<Response<DeleteIndexResponse>, Status> {
        let index_name = request.get_ref().index_name.clone();
        let result = forward_write!(self, request, delete_index, "delete_index", |parts: Vec<
            DeleteIndexResponse,
        >| {
            DeleteIndexResponse {
                success: parts.iter().all(|p| p.success),
            }
        });
        if result.is_ok() {
            self.ctx.forget_index(&index_name);
            self.ctx.refresh.notify_one();
        }
        result
    }

    async fn list_indexes(
        &self,
        _request: Request<ListIndexesRequest>,
    ) -> Result<Response<ListIndexesResponse>, Status> {
        self.ctx.check_admission()?;
        // Served from the cached topology: this is the health probe of every
        // hermes client (search-api gates readiness on it with a 2s timeout),
        // so it must never fan out to slow backends.
        let snapshot = self.ctx.snapshot.load();
        Ok(Response::new(ListIndexesResponse {
            index_names: snapshot.all_index_names(),
        }))
    }

    async fn retrain_vector_index(
        &self,
        request: Request<RetrainVectorIndexRequest>,
    ) -> Result<Response<RetrainVectorIndexResponse>, Status> {
        forward_write!(
            self,
            request,
            retrain_vector_index,
            "retrain_vector_index",
            |parts: Vec<RetrainVectorIndexResponse>| RetrainVectorIndexResponse {
                success: parts.iter().all(|p| p.success),
            }
        )
    }

    async fn alter_vector_index(
        &self,
        request: Request<AlterVectorIndexRequest>,
    ) -> Result<Response<AlterVectorIndexResponse>, Status> {
        forward_write!(
            self,
            request,
            alter_vector_index,
            "alter_vector_index",
            |parts: Vec<AlterVectorIndexResponse>| AlterVectorIndexResponse {
                publication_generation: parts
                    .iter()
                    .map(|p| p.publication_generation)
                    .max()
                    .unwrap_or_default(),
                state: parts.first().map(|p| p.state).unwrap_or_default(),
            }
        )
    }
}
