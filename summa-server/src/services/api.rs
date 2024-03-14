use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use async_broadcast::Receiver;
use futures_util::future::try_join_all;
use http_body_util::combinators::UnsyncBoxBody;
use hyper::header::{HeaderName, HeaderValue};
use proto::consumer_api_server::ConsumerApiServer;
use proto::index_api_server::IndexApiServer;
use proto::public_api_server::PublicApiServer;
use proto::reflection_api_server::ReflectionApiServer;
use proto::search_api_server::SearchApiServer;
use summa_core::configs::ConfigProxy;
use summa_proto::proto;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::codec::CompressionEncoding;
use tonic::transport::Server;
use tracing::{info, info_span, instrument, Instrument, Span};

use crate::apis::consumer::ConsumerApiImpl;
use crate::apis::index::IndexApiImpl;
use crate::apis::public::PublicApiImpl;
use crate::apis::reflection::ReflectionApiImpl;
use crate::apis::search::SearchApiImpl;
use crate::errors::SummaServerResult;
use crate::services::Index;
use crate::utils::thread_handler::ControlMessage;

/// GRPC server exposing [API](crate::apis)
pub struct Api {
    server_config_holder: Arc<dyn ConfigProxy<crate::configs::server::Config>>,
    index_service: Index,
}

impl Api {
    fn set_span(request: &hyper::Request<UnsyncBoxBody<bytes::Bytes, tonic::Status>>) -> Span {
        info_span!("request",)
    }

    fn set_listener(endpoint: &str) -> SummaServerResult<tokio::net::TcpListener> {
        let std_listener = std::net::TcpListener::bind(endpoint)?;
        std_listener.set_nonblocking(true)?;
        Ok(tokio::net::TcpListener::from_std(std_listener)?)
    }

    /// New GRPC server
    pub fn new(server_config_holder: &Arc<dyn ConfigProxy<crate::configs::server::Config>>, index_service: &Index) -> SummaServerResult<Api> {
        Ok(Api {
            server_config_holder: server_config_holder.clone(),
            index_service: index_service.clone(),
        })
    }

    /// Starts all nested services and start serving requests
    #[instrument("lifecycle", skip_all)]
    pub async fn prepare_serving_future(&self, terminator: Receiver<ControlMessage>) -> SummaServerResult<impl Future<Output = SummaServerResult<()>>> {
        let mut futures: Vec<Box<dyn Future<Output = SummaServerResult<()>> + Send>> = vec![];
        let index_service = self.index_service.clone();
        let consumer_api = ConsumerApiImpl::new(&index_service)?;
        let index_api = IndexApiImpl::new(&self.server_config_holder, &index_service)?;
        let reflection_api = ReflectionApiImpl::new(&index_service)?;
        let search_api = SearchApiImpl::new(&index_service)?;
        let public_api = PublicApiImpl::new(&index_service)?;

        let grpc_reflection_service = tonic_reflection::server::Builder::configure()
            .include_reflection_service(false)
            .register_encoded_file_descriptor_set(proto::FILE_DESCRIPTOR_SET)
            .build_v1()
            .expect("cannot build grpc server");
        let api_config = self.server_config_holder.read().await.get().api.clone();

        let consumer_service = ConsumerApiServer::new(consumer_api);
        let reflection_service = ReflectionApiServer::new(reflection_api);

        let mut index_service = IndexApiServer::new(index_api)
            .accept_compressed(CompressionEncoding::Gzip)
            .send_compressed(CompressionEncoding::Gzip);
        if let Some(max_from_size_bytes) = api_config.max_frame_size_bytes {
            index_service = index_service
                .max_decoding_message_size(max_from_size_bytes as usize)
                .max_encoding_message_size(max_from_size_bytes as usize);
        }

        let mut search_service = SearchApiServer::new(search_api)
            .accept_compressed(CompressionEncoding::Gzip)
            .send_compressed(CompressionEncoding::Gzip);
        if let Some(max_from_size_bytes) = api_config.max_frame_size_bytes {
            search_service = search_service
                .max_decoding_message_size(max_from_size_bytes as usize)
                .max_encoding_message_size(max_from_size_bytes as usize);
        }

        let public_service = PublicApiServer::new(public_api)
            .accept_compressed(CompressionEncoding::Gzip)
            .send_compressed(CompressionEncoding::Gzip);
        if let Some(max_from_size_bytes) = api_config.max_frame_size_bytes {
            search_service = search_service
                .max_decoding_message_size(max_from_size_bytes as usize)
                .max_encoding_message_size(max_from_size_bytes as usize);
        }

        let grpc_router = Server::builder()
            .max_frame_size(api_config.max_frame_size_bytes.map(|x| x / 256))
            .http2_keepalive_interval(Some(Duration::from_secs(api_config.keep_alive_timeout_seconds)))
            .max_connection_age(Duration::from_secs(api_config.max_connection_age_seconds))
            .add_service(grpc_reflection_service)
            .add_service(consumer_service.clone())
            .add_service(index_service.clone())
            .add_service(public_service.clone())
            .add_service(reflection_service.clone())
            .add_service(search_service.clone());
        let grpc_listener = Api::set_listener(&api_config.grpc_endpoint)?;
        let mut grpc_terminator = terminator.clone();
        futures.push(Box::new(async move {
            grpc_router
                .serve_with_incoming_shutdown(TcpListenerStream::new(grpc_listener), async move {
                    info!(action = "binded", endpoint = ?api_config.grpc_endpoint);
                    let signal_result = grpc_terminator.recv().await;
                    info!(action = "sigterm_received", received = ?signal_result);
                })
                .instrument(info_span!(parent: None, "lifecycle", mode = "grpc"))
                .await?;
            info!(action = "terminated", mode = "grpc");
            Ok(())
        }));

        if let Some(http_endpoint) = api_config.http_endpoint {
            let http_router = Server::builder()
                .accept_http1(true)
                .add_service(tonic_web::enable(search_service))
                .add_service(tonic_web::enable(public_service));
            let http_listener = Api::set_listener(&http_endpoint)?;
            let mut http_terminator = terminator.clone();
            futures.push(Box::new(async move {
                http_router
                    .serve_with_incoming_shutdown(TcpListenerStream::new(http_listener), async move {
                        info!(action = "binded", endpoint = ?http_endpoint);
                        let signal_result = http_terminator.recv().await;
                        info!(action = "sigterm_received", received = ?signal_result);
                    })
                    .instrument(info_span!(parent: None, "lifecycle", mode = "http"))
                    .await?;
                info!(action = "terminated", mode = "http");
                Ok(())
            }));
        }
        Ok(async move {
            try_join_all(futures.into_iter().map(Box::into_pin)).await?;
            Ok(())
        })
    }
}
