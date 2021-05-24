use super::base::BaseServer;
use crate::apis::consumer::ConsumerApiImpl;
use crate::apis::index::IndexApiImpl;
use crate::apis::reflection::ReflectionApiImpl;
use crate::apis::search::SearchApiImpl;
use crate::configs::ApplicationConfigHolder;
use crate::errors::SummaResult;
use crate::proto;
use crate::proto::consumer_api_server::ConsumerApiServer;
use crate::proto::index_api_server::IndexApiServer;
use crate::proto::reflection_api_server::ReflectionApiServer;
use crate::proto::search_api_server::SearchApiServer;
use crate::services::IndexService;
use crate::utils::random::generate_request_id;
use crate::utils::thread_handler::ControlMessage;
use async_broadcast::Receiver;
use hyper::header::{HeaderName, HeaderValue};
use std::future::Future;
use std::time::Duration;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::transport::Server;
use tower::ServiceBuilder;
use tower_http::classify::GrpcFailureClass;
use tower_http::set_header::SetRequestHeaderLayer;
use tower_http::trace::TraceLayer;
use tracing::{info, info_span, instrument, warn, Span};

/// GRPC server exposing [API](crate::apis)
pub struct GrpcServer {
    application_config: ApplicationConfigHolder,
    index_service: IndexService,
}

impl BaseServer for GrpcServer {}

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

    fn set_span(request: &hyper::Request<hyper::Body>) -> Span {
        info_span!(
            "request",
            request_id = ?request.headers().get("request-id").unwrap(),
            session_id = ?request.headers().get("session-id").unwrap(),
        )
    }

    /// New GRPC server
    pub fn new(application_config: &ApplicationConfigHolder, index_service: &IndexService) -> SummaResult<GrpcServer> {
        Ok(GrpcServer {
            application_config: application_config.clone(),
            index_service: index_service.clone(),
        })
    }

    /// Starts all nested services and start serving requests
    #[instrument("lifecycle", skip_all)]
    pub fn start(self, mut terminator: Receiver<ControlMessage>) -> SummaResult<impl Future<Output = SummaResult<()>>> {
        let consumer_api = ConsumerApiImpl::new(&self.index_service)?;
        let index_api = IndexApiImpl::new(&self.application_config, &self.index_service)?;
        let reflection_api = ReflectionApiImpl::new(&self.index_service)?;
        let search_api = SearchApiImpl::new(&self.index_service)?;
        let grpc_reflection_service = tonic_reflection::server::Builder::configure()
            .include_reflection_service(false)
            .register_encoded_file_descriptor_set(proto::FILE_DESCRIPTOR_SET)
            .build()
            .unwrap();
        let grpc_config = self.application_config.read().grpc.clone();

        let layer = ServiceBuilder::new()
            .timeout(Duration::from_secs(grpc_config.timeout_seconds))
            .layer(SetRequestHeaderLayer::if_not_present(HeaderName::from_static("request-id"), |_: &_| {
                Some(HeaderValue::from_str(&generate_request_id()).unwrap())
            }))
            .layer(SetRequestHeaderLayer::if_not_present(HeaderName::from_static("session-id"), |_: &_| {
                Some(HeaderValue::from_str(&generate_request_id()).unwrap())
            }))
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
            .add_service(grpc_reflection_service)
            .add_service(ConsumerApiServer::new(consumer_api))
            .add_service(IndexApiServer::new(index_api))
            .add_service(ReflectionApiServer::new(reflection_api))
            .add_service(SearchApiServer::new(search_api));

        let listener = GrpcServer::set_listener(&grpc_config.endpoint)?;
        info!(action = "binded", endpoint = ?grpc_config.endpoint);
        let server = async move {
            router
                .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async move {
                    terminator.recv().await.unwrap();
                    info!(action = "sigterm_received");
                    self.index_service.stop().await.unwrap();
                    info!(action = "terminated");
                })
                .await
                .unwrap();
            Ok(())
        };

        Ok(server)
    }
}
