//! SearchService routing: reads go to a routable replica of the shard
//! hosting the index and are forwarded verbatim.
//!
//! A partitioned index fans reads out: `Search` first collects the text
//! statistics of the query's terms from every partition and sends the sum
//! back with each partition's request (so BM25 scores are comparable across
//! partitions), then merges the per-partition windows by score
//! (`partition::merge_search_responses`). `GetDocument` asks every
//! partition, `GetIndexInfo` and `GetTextStats` aggregate.

use std::sync::Arc;
use std::time::Instant;

use tonic::{Request, Response, Status};

use crate::client::{capacity_exhausted, forward_timeout};
use crate::context::{BrokerContext, code_label};
use crate::metrics as m;
use crate::partition;
use crate::proto::hermes::search_service_server::SearchService;
use crate::proto::hermes::{
    GetDocumentRequest, GetDocumentResponse, GetIndexInfoRequest, GetIndexInfoResponse,
    GetTextStatsRequest, GetTextStatsResponse, SearchRequest, SearchResponse,
};
use crate::routes::{Route, Target, record_backend};

/// hermes-server refuses result windows above this; a partitioned search
/// widens every partition's window to `offset + limit`, which must fit.
const MAX_PARTITION_WINDOW: u32 = 10_000;

pub struct BrokerSearchService {
    pub ctx: Arc<BrokerContext>,
}

/// One unary read RPC sent whole to every target of the route; all must
/// succeed (a partitioned read with a missing partition is a wrong answer).
macro_rules! forward_read {
    ($self:ident, $req:expr, $timeout:expr, $route:expr, $method:ident, $rpc_name:literal) => {{
        let req = $req;
        let index_name = req.index_name.clone();
        let route: &Route = $route;
        let calls = route.targets().iter().map(|target| {
            let mut outbound = Request::new(req.clone());
            if let Some(t) = $timeout {
                outbound.set_timeout(t);
            }
            let mut client = target.channels.search.clone();
            let target: Target = target.clone();
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
        Ok::<_, Status>(responses)
    }};
}

impl BrokerSearchService {
    /// Admission for one search on every target of the route: try_acquire
    /// (never queue), so overload is reported immediately with the same
    /// message hermes-server uses.
    fn admit(
        &self,
        index_name: &str,
        route: &Route,
    ) -> Result<Vec<tokio::sync::OwnedSemaphorePermit>, Status> {
        let mut permits = Vec::with_capacity(route.targets().len() + 1);
        if let Some(global) = &self.ctx.global_search_permits {
            permits.push(global.clone().try_acquire_owned().map_err(|_| {
                metrics::counter!(
                    m::ADMISSION_REJECTED,
                    "index" => index_name.to_string(), "scope" => "global"
                )
                .increment(1);
                capacity_exhausted()
            })?);
        }
        for target in route.targets() {
            permits.push(
                target
                    .channels
                    .search_permits
                    .clone()
                    .try_acquire_owned()
                    .map_err(|_| {
                        metrics::counter!(
                            m::ADMISSION_REJECTED,
                            "index" => index_name.to_string(), "scope" => "backend"
                        )
                        .increment(1);
                        capacity_exhausted()
                    })?,
            );
        }
        Ok(permits)
    }

    async fn attach_text_stats(
        &self,
        req: &mut SearchRequest,
        timeout: Option<std::time::Duration>,
        route: &Route,
    ) -> Result<(), Status> {
        // Shared BM25 statistics: every partition scores with the sum.
        if req.text_stats.is_none()
            && let Some(query) = req.query.as_ref().and_then(partition::text_stats_query)
        {
            let stats = forward_read!(
                self,
                GetTextStatsRequest {
                    index_name: req.index_name.clone(),
                    query: Some(query),
                },
                timeout,
                route,
                get_text_stats,
                "get_text_stats"
            )?;
            req.text_stats = Some(partition::merge_text_stats(
                stats.into_iter().filter_map(|s| s.stats).collect(),
            ));
        }
        Ok(())
    }

    async fn search_coordinated(
        &self,
        req: SearchRequest,
        timeout: Option<std::time::Duration>,
        route: &Route,
        permits: Vec<tokio::sync::OwnedSemaphorePermit>,
    ) -> Result<SearchResponse, Status> {
        let started = Instant::now();
        let mut plan = crate::ranking::CoordinatorPlan::new(req, route.targets().len())?;
        self.attach_text_stats(&mut plan.shard_request, timeout, route)
            .await?;
        let remaining = || {
            timeout
                .map(|budget| {
                    budget
                        .checked_sub(started.elapsed())
                        .filter(|duration| !duration.is_zero())
                        .ok_or_else(|| {
                            Status::deadline_exceeded("coordinator search deadline expired")
                        })
                })
                .transpose()
        };
        let rpc_timeout = remaining()?;
        // Bound the sum of concurrently decoded responses, including compressed
        // transports. No unbounded per-shard allowance is multiplied by fan-out.
        let decode_limit = crate::ranking::MAX_TRANSFER_BYTES / route.targets().len();
        let calls = route.targets().iter().map(|target| {
            let mut outbound = Request::new(plan.shard_request.clone());
            let index_name = plan.shard_request.index_name.clone();
            if let Some(timeout) = rpc_timeout {
                outbound.set_timeout(timeout);
            }
            let mut client = target
                .channels
                .search
                .clone()
                .max_decoding_message_size(decode_limit);
            async move {
                let call_started = Instant::now();
                let result = client.search(outbound).await;
                let code = result
                    .as_ref()
                    .map(|_| tonic::Code::Ok)
                    .unwrap_or_else(|status| status.code());
                record_backend(&target.backend_id, "search", call_started, code);
                result
                    .map(|response| response.into_inner())
                    .map_err(|status| {
                        if route.is_partitioned() {
                            partition::partition_failure(&index_name, &target.shard, status)
                        } else {
                            status
                        }
                    })
            }
        });
        let responses = futures::future::try_join_all(calls).await?;
        let timeout = remaining()?;
        let selection = tokio::task::spawn_blocking(move || {
            // Cancellation must not release admission while ranking still runs.
            let _permits = permits;
            let scoring_started = Instant::now();
            let mut response = plan.finish(responses)?;
            let timings = response.timings.get_or_insert_with(Default::default);
            timings.candidate_scoring_us = timings
                .candidate_scoring_us
                .saturating_add(scoring_started.elapsed().as_micros() as u64);
            timings.total_us = started.elapsed().as_micros() as u64;
            response.took_ms = timings.total_us / 1000;
            Ok::<_, Status>(response)
        });
        let selected = if let Some(timeout) = timeout {
            tokio::time::timeout(timeout, selection)
                .await
                .map_err(|_| Status::deadline_exceeded("coordinator ranking deadline expired"))?
        } else {
            selection.await
        };
        selected.map_err(|_| Status::internal("coordinator ranking worker failed"))?
    }

    async fn search_partitioned(
        &self,
        mut req: SearchRequest,
        timeout: Option<std::time::Duration>,
        route: &Route,
    ) -> Result<SearchResponse, Status> {
        let offset = req.offset;
        let limit = req.limit;
        let window = offset.saturating_add(limit);
        if window > MAX_PARTITION_WINDOW {
            return Err(Status::invalid_argument(format!(
                "offset + limit = {window} exceeds the {MAX_PARTITION_WINDOW} window a partitioned index can merge"
            )));
        }
        self.attach_text_stats(&mut req, timeout, route).await?;
        req.offset = 0;
        req.limit = window;
        let expected_method = crate::ranking::expected_export_method(&req);
        let responses = forward_read!(self, req, timeout, route, search, "search")?;
        if expected_method.is_some_and(|expected| {
            responses
                .iter()
                .any(|response| response.ranking_method != expected)
        }) {
            return Err(Status::failed_precondition(
                "a shard does not support the requested candidate scoring contract; complete the Hermes rollout",
            ));
        }
        partition::merge_search_responses(responses, offset as usize, limit as usize)
    }
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
            let route = self.ctx.read_route(&req.index_name)?;
            let permits = self.admit(&index_name, &route)?;
            if crate::ranking::handles(&req) {
                return self.search_coordinated(req, timeout, &route, permits).await;
            }
            let _permits = permits;
            match &route {
                Route::Single(target) => {
                    let expected_method = crate::ranking::expected_export_method(&req);
                    let mut outbound = Request::new(req);
                    if let Some(t) = timeout {
                        outbound.set_timeout(t);
                    }
                    let call_started = Instant::now();
                    let result = target.channels.search.clone().search(outbound).await;
                    let code = result
                        .as_ref()
                        .map(|_| tonic::Code::Ok)
                        .unwrap_or_else(|s| s.code());
                    record_backend(&target.backend_id, "search", call_started, code);
                    result.and_then(|r| {
                        let response = r.into_inner();
                        if expected_method
                            .is_some_and(|expected| response.ranking_method != expected)
                        {
                            return Err(Status::failed_precondition(
                                "backend lacks requested candidate scoring contract",
                            ));
                        }
                        Ok(response)
                    })
                }
                Route::Partitioned(_) => self.search_partitioned(req, timeout, &route).await,
            }
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
        let route = self.ctx.read_route(&req.index_name)?;
        // A document address names a segment, which lives on exactly one
        // partition: ask every partition, the one that has it answers.
        let calls = route.targets().iter().map(|target| {
            let mut outbound = Request::new(req.clone());
            if let Some(t) = timeout {
                outbound.set_timeout(t);
            }
            let mut client = target.channels.search.clone();
            let target: Target = target.clone();
            async move {
                let started = Instant::now();
                let result = client.get_document(outbound).await;
                (target, started, result)
            }
        });
        let mut not_found: Option<Status> = None;
        let mut failure: Option<Status> = None;
        for (target, started, result) in futures::future::join_all(calls).await {
            let code = result
                .as_ref()
                .map(|_| tonic::Code::Ok)
                .unwrap_or_else(|s| s.code());
            record_backend(&target.backend_id, "get_document", started, code);
            match result {
                Ok(response) => return Ok(Response::new(response.into_inner())),
                Err(status) if status.code() == tonic::Code::NotFound => {
                    not_found.get_or_insert(status);
                }
                Err(status) => {
                    failure.get_or_insert(if route.is_partitioned() {
                        partition::partition_failure(&req.index_name, &target.shard, status)
                    } else {
                        status
                    });
                }
            }
        }
        Err(failure
            .or(not_found)
            .unwrap_or_else(|| Status::not_found("document not found")))
    }

    async fn get_text_stats(
        &self,
        request: Request<GetTextStatsRequest>,
    ) -> Result<Response<GetTextStatsResponse>, Status> {
        self.ctx.check_admission()?;
        let timeout = forward_timeout(request.metadata());
        let req = request.into_inner();
        let route = self.ctx.read_route(&req.index_name)?;
        let responses =
            forward_read!(self, req, timeout, &route, get_text_stats, "get_text_stats")?;
        let stats = if route.is_partitioned() {
            Some(partition::merge_text_stats(
                responses.into_iter().filter_map(|r| r.stats).collect(),
            ))
        } else {
            responses.into_iter().next().and_then(|r| r.stats)
        };
        Ok(Response::new(GetTextStatsResponse { stats }))
    }

    async fn get_index_info(
        &self,
        request: Request<GetIndexInfoRequest>,
    ) -> Result<Response<GetIndexInfoResponse>, Status> {
        self.ctx.check_admission()?;
        let timeout = forward_timeout(request.metadata());
        let req = request.into_inner();
        let route = self.ctx.read_route(&req.index_name)?;
        let responses =
            forward_read!(self, req, timeout, &route, get_index_info, "get_index_info")?;
        Ok(Response::new(partition::merge_index_info(responses)))
    }
}
