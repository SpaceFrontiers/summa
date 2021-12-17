use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use async_broadcast::Receiver;
use hyper::header::{HeaderName, HeaderValue};
use proto::beacon_api_server::BeaconApiServer;
use proto::consumer_api_server::ConsumerApiServer;
use proto::index_api_server::IndexApiServer;
use proto::reflection_api_server::ReflectionApiServer;
use proto::search_api_server::SearchApiServer;
use summa_core::configs::ConfigProxy;
use summa_core::utils::random::generate_request_id;
use summa_core::utils::thread_handler::ControlMessage;
use summa_proto::proto;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::transport::Server;
use tower::ServiceBuilder;
use tower_http::classify::GrpcFailureClass;
use tower_http::set_header::SetRequestHeaderLayer;
use tower_http::trace::TraceLayer;
use tracing::{info, info_span, instrument, warn, Instrument, Span};

use crate::apis::beacon::BeaconApiImpl;
use crate::apis::consumer::ConsumerApiImpl;
use crate::apis::index::IndexApiImpl;
use crate::apis::reflection::ReflectionApiImpl;
use crate::apis::search::SearchApiImpl;
use crate::errors::SummaServerResult;
use crate::services::{Beacon, Index};

/// GRPC server exposing [API](crate::apis)
pub struct Grpc {
    server_config: Arc<dyn ConfigProxy<crate::configs::server::Config>>,
    beacon_service: Beacon,
    index_service: Index,
}

impl Grpc {
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
            request_id = ?request.headers().get("request-id").expect("request-id must be set"),
            session_id = ?request.headers().get("session-id").expect("session-id must be set"),
        )
    }

    fn set_listener(endpoint: &str) -> SummaServerResult<tokio::net::TcpListener> {
        let std_listener = std::net::TcpListener::bind(endpoint)?;
        std_listener.set_nonblocking(true)?;
        Ok(tokio::net::TcpListener::from_std(std_listener)?)
    }

    /// New GRPC server
    pub fn new(
        server_config: &Arc<dyn ConfigProxy<crate::configs::server::Config>>,
        beacon_service: &Beacon,
        index_service: &Index,
    ) -> SummaServerResult<Grpc> {
        Ok(Grpc {
            server_config: server_config.clone(),
            beacon_service: beacon_service.clone(),
            index_service: index_service.clone(),
        })
    }

    /// Starts all nested services and start serving requests
    #[instrument("lifecycle", skip_all)]
    pub async fn start(self, mut terminator: Receiver<ControlMessage>) -> SummaServerResult<impl Future<Output = SummaServerResult<()>>> {
        let consumer_api = ConsumerApiImpl::new(&self.index_service)?;
        let index_api = IndexApiImpl::new(&self.server_config, &self.index_service)?;
        let reflection_api = ReflectionApiImpl::new(&self.index_service)?;
        let search_api = SearchApiImpl::new(&self.index_service)?;
        let grpc_reflection_service = tonic_reflection::server::Builder::configure()
            .include_reflection_service(false)
            .register_encoded_file_descriptor_set(proto::FILE_DESCRIPTOR_SET)
            .build()
            .expect("cannot build grpc server");
        let grpc_config = self.server_config.read().await.get().grpc.clone();

        let layer = ServiceBuilder::new()
            .layer(SetRequestHeaderLayer::if_not_present(HeaderName::from_static("request-id"), |_: &_| {
                Some(HeaderValue::from_str(&generate_request_id()).expect("invalid generated request id"))
            }))
            .layer(SetRequestHeaderLayer::if_not_present(HeaderName::from_static("session-id"), |_: &_| {
                Some(HeaderValue::from_str(&generate_request_id()).expect("invalid generated session id"))
            }))
            .concurrency_limit(10)
            .buffer(100)
            .layer(
                TraceLayer::new_for_grpc()
                    .make_span_with(Grpc::set_span)
                    .on_request(Grpc::on_request)
                    .on_response(Grpc::on_response)
                    .on_failure(Grpc::on_failure),
            )
            .into_inner();

        let mut router = Server::builder()
            .layer(layer)
            .max_frame_size(grpc_config.max_frame_size_bytes.map(|x| x / 256))
            .add_service(grpc_reflection_service)
            .add_service(ConsumerApiServer::new(consumer_api))
            .add_service(IndexApiServer::new(index_api))
            .add_service(ReflectionApiServer::new(reflection_api))
            .add_service(SearchApiServer::new(search_api));

        let beacon_api = BeaconApiImpl::new(&self.beacon_service, &self.index_service)?;
        router = router.add_service(BeaconApiServer::new(beacon_api));

        let listener = Grpc::set_listener(&grpc_config.endpoint)?;
        info!(action = "binded", endpoint = ?grpc_config.endpoint);
        let server = async move {
            router
                .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async move {
                    let signal_result = terminator.recv().await;
                    info!(action = "sigterm_received", received = ?signal_result);
                    let service_result = self.index_service.stop(false).await;
                    info!(action = "terminated", result = ?service_result);
                })
                .instrument(info_span!(parent: None, "lifecycle"))
                .await?;
            Ok(())
        };

        Ok(server)
    }
}
