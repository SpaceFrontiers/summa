use crate::apis::consumer::ConsumerApiImpl;
use crate::apis::index::IndexApiImpl;
use crate::apis::reflection::ReflectionApiImpl;
use crate::apis::search::SearchApiImpl;
use crate::configs::ApplicationConfigHolder;
use crate::errors::SummaResult;
use crate::proto::consumer_api_server::ConsumerApiServer;
use crate::proto::index_api_server::IndexApiServer;
use crate::proto::reflection_api_server::ReflectionApiServer;
use crate::proto::search_api_server::SearchApiServer;
use crate::services::IndexService;
use crate::utils::random::generate_request_id;
use crate::utils::signal_channel::signal_channel;
use futures::FutureExt;
use std::time::Duration;
use tonic::metadata::MetadataValue;
use tonic::service::interceptor;
use tonic::transport::Server;
use tonic::Request;
use tonic::Status;
use tower::ServiceBuilder;
use tower_http::classify::GrpcFailureClass;
use tower_http::trace::TraceLayer;
use tracing::{info, info_span, instrument, warn, Span};

/// GRPC server exposing [API](crate::apis)
pub struct GrpcServer {
    application_config: ApplicationConfigHolder,
    index_service: IndexService,
}

impl GrpcServer {
    fn on_request(request: &hyper::Request<hyper::Body>, _span: &Span) {
        info!(path = ?request.uri().path());
    }
    fn on_response<T>(response: &hyper::Response<T>, duration: Duration, _span: &Span) {
        info!(duration = ?duration, status = ?response.status().as_u16());
    }
    fn on_failure(error: GrpcFailureClass, duration: Duration, _span: &Span) {
        warn!(error = ?error, duration = ?duration);
    }
    fn set_ids(mut request: Request<()>) -> Result<Request<()>, Status> {
        let metadata = request.metadata_mut();
        match metadata.get("request-id").map(|v| v.as_bytes()) {
            None | Some(b"") => metadata.insert("request-id", MetadataValue::from_str(&generate_request_id()).unwrap()),
            _ => None,
        };
        match metadata.get("session-id").map(|v| v.as_bytes()) {
            None | Some(b"") => metadata.insert("session-id", MetadataValue::from_str(&generate_request_id()).unwrap()),
            _ => None,
        };
        Ok(request)
    }

    fn set_span(request: &hyper::Request<hyper::Body>) -> tracing::Span {
        info_span!(
            "request",
            request_id = ?request.headers().get("request-id").unwrap(),
            session_id = ?request.headers().get("session-id").unwrap(),
        )
    }

    /// New GRPC server
    pub async fn new(application_config: &ApplicationConfigHolder, index_service: &IndexService) -> SummaResult<GrpcServer> {
        Ok(GrpcServer {
            application_config: application_config.clone(),
            index_service: index_service.clone(),
        })
    }

    /// Starts all nested services and start serving requests
    #[instrument("lifecycle", skip_all)]
    pub async fn start(self) -> SummaResult<()> {
        let consumer_api = ConsumerApiImpl::new(&self.index_service)?;
        let index_api = IndexApiImpl::new(&self.application_config, &self.index_service)?;
        let reflection_api = ReflectionApiImpl::new(&self.index_service)?;
        let search_api = SearchApiImpl::new(&self.index_service)?;
        let grpc_config = self.application_config.read().grpc.clone();

        let layer = ServiceBuilder::new()
            .timeout(Duration::from_secs(grpc_config.timeout_seconds))
            .layer(interceptor(GrpcServer::set_ids))
            .layer(
                TraceLayer::new_for_grpc()
                    .make_span_with(GrpcServer::set_span)
                    .on_request(GrpcServer::on_request)
                    .on_response(GrpcServer::on_response)
                    .on_failure(GrpcServer::on_failure),
            )
            .into_inner();

        let router = Server::builder()
            .layer(layer)
            .add_service(ConsumerApiServer::new(consumer_api))
            .add_service(IndexApiServer::new(index_api))
            .add_service(ReflectionApiServer::new(reflection_api))
            .add_service(SearchApiServer::new(search_api));

        let rx = signal_channel();
        let addr = grpc_config.endpoint.parse()?;
        info!(action = "starting", addr = ?addr);
        router
            .serve_with_shutdown(addr, async move {
                rx.map(drop).await;
                info!(action = "sigterm_received");
            })
            .await?;
        self.index_service.stop().await?;
        info!(action = "terminated");
        Ok(())
    }
}
