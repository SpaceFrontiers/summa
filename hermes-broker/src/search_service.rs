//! SearchService pass-through: exact index → its backend, request and
//! response forwarded verbatim. The broker adds only admission (never more
//! in-flight searches per backend than the backend itself would admit),
//! deadline propagation, and routing metrics.

use std::sync::Arc;
use std::time::Instant;

use tonic::{Request, Response, Status};

use crate::client::{capacity_exhausted, forward_timeout};
use crate::context::{BrokerContext, code_label};
use crate::metrics as m;
use crate::proto::hermes::search_service_server::SearchService;
use crate::proto::hermes::{
    GetDocumentRequest, GetDocumentResponse, GetIndexInfoRequest, GetIndexInfoResponse,
    GetTextStatsRequest, GetTextStatsResponse, SearchRequest, SearchResponse,
};

pub struct BrokerSearchService {
    pub ctx: Arc<BrokerContext>,
}

impl BrokerSearchService {
    /// Resolve the read backend for an index, recording routing metrics.
    fn read_route(
        &self,
        index_name: &str,
    ) -> Result<(crate::client::BackendChannels, String), Status> {
        let snapshot = self.ctx.snapshot.load();
        let selection = snapshot.select_read_backend(index_name, self.ctx.next_rotation())?;
        if selection.ambiguous {
            metrics::counter!(m::AMBIGUOUS_INDEX, "index" => index_name.to_string()).increment(1);
        }
        if selection.stale {
            metrics::counter!(
                m::STALE_TOPOLOGY_SERVES,
                "backend" => selection.backend.endpoint.id.0.clone()
            )
            .increment(1);
        }
        let channels = self.ctx.pool.get(&selection.backend.endpoint.addr)?;
        Ok((channels, selection.backend.endpoint.id.0.clone()))
    }
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

#[tonic::async_trait]
impl SearchService for BrokerSearchService {
    async fn search(
        &self,
        request: Request<SearchRequest>,
    ) -> Result<Response<SearchResponse>, Status> {
        self.ctx.check_admission()?;
        let timeout = forward_timeout(request.metadata());
        let req = request.into_inner();
        let index_name = req.index_name.clone();
        let started = Instant::now();

        let result: Result<SearchResponse, Status> = async {
            let (channels, backend_id) = self.read_route(&req.index_name)?;

            // try_acquire (never queue): overload is reported immediately with
            // the same message hermes-server uses, so client backoff/retry
            // logic cannot tell the broker and the backend apart.
            let _global = match &self.ctx.global_search_permits {
                Some(permits) => Some(permits.clone().try_acquire_owned().map_err(|_| {
                    metrics::counter!(
                        m::ADMISSION_REJECTED,
                        "index" => index_name.clone(), "scope" => "global"
                    )
                    .increment(1);
                    capacity_exhausted()
                })?),
                None => None,
            };
            let _backend = channels
                .search_permits
                .clone()
                .try_acquire_owned()
                .map_err(|_| {
                    metrics::counter!(
                        m::ADMISSION_REJECTED,
                        "index" => index_name.clone(), "scope" => "backend"
                    )
                    .increment(1);
                    capacity_exhausted()
                })?;

            let mut outbound = Request::new(req);
            if let Some(t) = timeout {
                outbound.set_timeout(t);
            }
            let call_started = Instant::now();
            let result = channels.search.clone().search(outbound).await;
            let code = result
                .as_ref()
                .map(|_| tonic::Code::Ok)
                .unwrap_or_else(|s| s.code());
            record_backend(&backend_id, "search", call_started, code);
            result.map(|r| r.into_inner())
        }
        .await;

        let status_label = match &result {
            Ok(_) => "ok",
            Err(status) => code_label(status.code()),
        };
        metrics::histogram!(
            m::SEARCH_DURATION,
            "index" => index_name.clone(), "status" => status_label,
        )
        .record(started.elapsed().as_secs_f64());
        metrics::counter!(
            m::SEARCH_REQUESTS,
            "index" => index_name, "status" => status_label,
        )
        .increment(1);

        result.map(Response::new)
    }

    async fn get_document(
        &self,
        request: Request<GetDocumentRequest>,
    ) -> Result<Response<GetDocumentResponse>, Status> {
        self.ctx.check_admission()?;
        let timeout = forward_timeout(request.metadata());
        let req = request.into_inner();
        let (channels, backend_id) = self.read_route(&req.index_name)?;
        let mut outbound = Request::new(req);
        if let Some(t) = timeout {
            outbound.set_timeout(t);
        }
        let started = Instant::now();
        let result = channels.search.clone().get_document(outbound).await;
        let code = result
            .as_ref()
            .map(|_| tonic::Code::Ok)
            .unwrap_or_else(|s| s.code());
        record_backend(&backend_id, "get_document", started, code);
        result.map(|r| Response::new(r.into_inner()))
    }

    /// One index lives on one shard today, so the statistics of that shard
    /// are the whole; a scatter-gather broker sums this per shard and sends
    /// the total back as `SearchRequest.text_stats`.
    async fn get_text_stats(
        &self,
        request: Request<GetTextStatsRequest>,
    ) -> Result<Response<GetTextStatsResponse>, Status> {
        self.ctx.check_admission()?;
        let timeout = forward_timeout(request.metadata());
        let req = request.into_inner();
        let (channels, backend_id) = self.read_route(&req.index_name)?;
        let mut outbound = Request::new(req);
        if let Some(t) = timeout {
            outbound.set_timeout(t);
        }
        let started = Instant::now();
        let result = channels.search.clone().get_text_stats(outbound).await;
        let code = result
            .as_ref()
            .map(|_| tonic::Code::Ok)
            .unwrap_or_else(|s| s.code());
        record_backend(&backend_id, "get_text_stats", started, code);
        result.map(|r| Response::new(r.into_inner()))
    }

    async fn get_index_info(
        &self,
        request: Request<GetIndexInfoRequest>,
    ) -> Result<Response<GetIndexInfoResponse>, Status> {
        self.ctx.check_admission()?;
        let timeout = forward_timeout(request.metadata());
        let req = request.into_inner();
        let (channels, backend_id) = self.read_route(&req.index_name)?;
        let mut outbound = Request::new(req);
        if let Some(t) = timeout {
            outbound.set_timeout(t);
        }
        let started = Instant::now();
        let result = channels.search.clone().get_index_info(outbound).await;
        let code = result
            .as_ref()
            .map(|_| tonic::Code::Ok)
            .unwrap_or_else(|s| s.code());
        record_backend(&backend_id, "get_index_info", started, code);
        result.map(|r| Response::new(r.into_inner()))
    }
}
