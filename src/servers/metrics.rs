//! HTTP server exposing metrics in Prometheus format

use super::base::BaseServer;
use crate::configs::ApplicationConfigHolder;
use crate::errors::SummaResult;
use crate::utils::thread_handler::ControlMessage;
use async_broadcast::Receiver;
use hyper::{
    header::{HeaderValue, CONTENT_TYPE},
    service::{make_service_fn, service_fn},
    Body, Method, Request, Response, Server,
};
use opentelemetry_prometheus::PrometheusExporter;
use prometheus::{Encoder, TextEncoder};
use std::convert::Infallible;
use std::future::Future;
use std::sync::Arc;
use tracing::{info, info_span, instrument};

lazy_static! {
    static ref EMPTY_HEADER_VALUE: HeaderValue = HeaderValue::from_static("");
}

pub struct MetricsServer {
    application_config: ApplicationConfigHolder,
    exporter: PrometheusExporter,
}

impl BaseServer for MetricsServer {}

struct AppState {
    exporter: PrometheusExporter,
}

impl MetricsServer {
    pub fn new(application_config: &ApplicationConfigHolder) -> SummaResult<MetricsServer> {
        let exporter = opentelemetry_prometheus::exporter()
            .with_default_histogram_boundaries(vec![0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
            .init();
        Ok(MetricsServer {
            application_config: application_config.clone(),
            exporter,
        })
    }
    async fn serve_request(request: Request<Body>, state: Arc<AppState>) -> Result<Response<Body>, hyper::Error> {
        let _span = info_span!(
            "request",
            request_id = ?request.headers().get("request-id").unwrap_or(&EMPTY_HEADER_VALUE),
            session_id = ?request.headers().get("session-id").unwrap_or(&EMPTY_HEADER_VALUE),
        );
        info!(path = ?request.uri().path());
        let response = match request.method() {
            &Method::GET => {
                let mut buffer = vec![];
                let encoder = TextEncoder::new();
                let metric_families = state.exporter.registry().gather();
                encoder.encode(&metric_families[..], &mut buffer).unwrap();
                Response::builder()
                    .status(200)
                    .header(CONTENT_TYPE, encoder.format_type())
                    .body(Body::from(buffer))
                    .unwrap()
            }
            _ => Response::builder().status(404).body(Body::from("Missing Page")).unwrap(),
        };
        Ok(response)
    }

    #[instrument("lifecycle", skip_all)]
    pub async fn start(&self, mut terminator: Receiver<ControlMessage>) -> SummaResult<impl Future<Output = SummaResult<()>>> {
        let metrics_config = self.application_config.read().await.metrics.clone();
        let state = Arc::new(AppState {
            exporter: self.exporter.clone(),
        });
        let service = make_service_fn(move |_conn| {
            let state = state.clone();
            async move { Ok::<_, Infallible>(service_fn(move |request| MetricsServer::serve_request(request, state.clone()))) }
        });

        let server = Server::bind(&metrics_config.endpoint.parse()?).serve(service);
        info!(action = "binded", endpoint = ?metrics_config.endpoint);
        let graceful = server.with_graceful_shutdown(async move {
            terminator.recv().await.unwrap();
            info!(action = "sigterm_received");
        });

        Ok(async move {
            graceful.await.unwrap();
            info!(action = "terminated");
            Ok(())
        })
    }
}
