//! IndexService routing: every write RPC goes whole to the master of the
//! shard hosting the index; responses are forwarded verbatim so client-side
//! contracts (duplicate-primary-key detection, backpressure partial-batch
//! semantics, `DocumentError.index` positions) are byte-faithful.
//!
//! CreateIndex is the one placement decision the broker owns: a glob rule
//! (or the default policy) picks the shard for a new index, which is how
//! dated full-build names land on the right host.

use std::sync::Arc;
use std::time::{Duration, Instant};

use log::info;
use prost::Message;
use tonic::{Request, Response, Status, Streaming};

use crate::client::forward_timeout;
use crate::context::{BrokerContext, code_label};
use crate::metrics as m;
use crate::proto::hermes::index_service_server::IndexService;
use crate::proto::hermes::{
    BatchIndexDocumentsRequest, BatchIndexDocumentsResponse, CommitRequest, CommitResponse,
    CreateIndexRequest, CreateIndexResponse, DeleteIndexRequest, DeleteIndexResponse,
    ForceMergeRequest, ForceMergeResponse, IndexDocumentRequest, IndexDocumentsResponse,
    ListIndexesRequest, ListIndexesResponse, NamedDocument, ReorderRequest, ReorderResponse,
    RetrainVectorIndexRequest, RetrainVectorIndexResponse,
};

/// Streaming IndexDocuments is buffered per index and forwarded as
/// BatchIndexDocuments once either threshold is reached (mirrors the
/// server's own internal 512-message stream batching).
const STREAM_FLUSH_DOCS: usize = 512;
const STREAM_FLUSH_BYTES: usize = 4 * 1024 * 1024;

pub struct BrokerIndexService {
    pub ctx: Arc<BrokerContext>,
}

fn record_backend(backend: &str, rpc: &'static str, started: Instant, code: tonic::Code) {
    metrics::histogram!(
        m::BACKEND_DURATION,
        "backend" => backend.to_string(),
        "rpc" => rpc,
    )
    .record(started.elapsed().as_secs_f64());
    metrics::counter!(
        m::BACKEND_REQUESTS,
        "backend" => backend.to_string(),
        "rpc" => rpc,
        "code" => code_label(code),
    )
    .increment(1);
}

impl BrokerIndexService {
    /// Resolve the master channel for a write against an existing index,
    /// counting refusals.
    fn write_route(
        &self,
        index_name: &str,
    ) -> Result<(crate::client::BackendChannels, String), Status> {
        let snapshot = self.ctx.snapshot.load();
        let backend = snapshot
            .select_write_backend(index_name)
            .inspect_err(|status| {
                metrics::counter!(
                    m::WRITE_REJECTED,
                    "index" => index_name.to_string(),
                    "reason" => code_label(status.code()),
                )
                .increment(1);
            })?;
        let channels = self.ctx.pool.get(&backend.endpoint.addr)?;
        Ok((channels, backend.endpoint.id.0.clone()))
    }
}

/// One unary write RPC forwarded whole to the resolved master.
macro_rules! forward_write {
    ($self:ident, $request:ident, $method:ident, $rpc_name:literal) => {{
        $self.ctx.check_admission()?;
        let timeout = forward_timeout($request.metadata());
        let req = $request.into_inner();
        let (channels, backend_id) = $self.write_route(&req.index_name)?;
        let mut outbound = Request::new(req);
        if let Some(t) = timeout {
            outbound.set_timeout(t);
        }
        let started = Instant::now();
        let result = channels.index.clone().$method(outbound).await;
        let code = result
            .as_ref()
            .map(|_| tonic::Code::Ok)
            .unwrap_or_else(|s| s.code());
        record_backend(&backend_id, $rpc_name, started, code);
        result.map(|r| Response::new(r.into_inner()))
    }};
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
        let (addr, backend_id, shard) = {
            let snapshot = self.ctx.snapshot.load();
            let backend = snapshot
                .select_create_backend(&req.index_name, &self.ctx.placement)
                .inspect_err(|status| {
                    metrics::counter!(
                        m::WRITE_REJECTED,
                        "index" => req.index_name.clone(),
                        "reason" => code_label(status.code()),
                    )
                    .increment(1);
                })?;
            (
                backend.endpoint.addr.clone(),
                backend.endpoint.id.0.clone(),
                backend.endpoint.shard.0.clone(),
            )
        };
        info!(
            "creating index '{}' on shard '{}' (backend {})",
            req.index_name, shard, backend_id
        );
        let channels = self.ctx.pool.get(&addr)?;
        let index_name = req.index_name.clone();
        let mut outbound = Request::new(req);
        if let Some(t) = timeout {
            outbound.set_timeout(t);
        }
        let started = Instant::now();
        let result = channels.index.clone().create_index(outbound).await;
        let code = result
            .as_ref()
            .map(|_| tonic::Code::Ok)
            .unwrap_or_else(|s| s.code());
        record_backend(&backend_id, "create_index", started, code);
        if result.is_ok() {
            // Make the new index routable without waiting a poll interval.
            self.ctx.refresh.notify_one();
        }
        result
            .map(|r| Response::new(r.into_inner()))
            .inspect_err(|status| {
                info!("create_index '{index_name}' failed: {}", status.code());
            })
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
        // current master. NOTE: like hermes-server's own stream handling,
        // DocumentError.index positions are relative to the flushed batch,
        // not the whole stream.
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
            let (channels, backend_id) = service.write_route(index_name)?;
            metrics::counter!(m::STREAM_FLUSHES, "index" => index_name.to_string()).increment(1);
            let mut outbound = Request::new(BatchIndexDocumentsRequest {
                index_name: index_name.to_string(),
                documents,
            });
            if let Some(total) = budget {
                let remaining = total.saturating_sub(started.elapsed());
                if remaining.is_zero() {
                    return Err(Status::deadline_exceeded(
                        "client deadline exhausted mid-stream",
                    ));
                }
                outbound.set_timeout(remaining);
            }
            let call_started = Instant::now();
            let result = channels.index.clone().batch_index_documents(outbound).await;
            let code = result
                .as_ref()
                .map(|_| tonic::Code::Ok)
                .unwrap_or_else(|s| s.code());
            record_backend(&backend_id, "batch_index_documents", call_started, code);
            let response = result?.into_inner();
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
        forward_write!(
            self,
            request,
            batch_index_documents,
            "batch_index_documents"
        )
    }

    async fn commit(
        &self,
        request: Request<CommitRequest>,
    ) -> Result<Response<CommitResponse>, Status> {
        forward_write!(self, request, commit, "commit")
    }

    async fn force_merge(
        &self,
        request: Request<ForceMergeRequest>,
    ) -> Result<Response<ForceMergeResponse>, Status> {
        forward_write!(self, request, force_merge, "force_merge")
    }

    async fn reorder(
        &self,
        request: Request<ReorderRequest>,
    ) -> Result<Response<ReorderResponse>, Status> {
        forward_write!(self, request, reorder, "reorder")
    }

    async fn delete_index(
        &self,
        request: Request<DeleteIndexRequest>,
    ) -> Result<Response<DeleteIndexResponse>, Status> {
        let result = forward_write!(self, request, delete_index, "delete_index");
        if result.is_ok() {
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
        forward_write!(self, request, retrain_vector_index, "retrain_vector_index")
    }
}
