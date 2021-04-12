//! HTTP server exposing metrics in Prometheus format

use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;

use crate::configs::ApplicationConfigHolder;
use crate::errors::SummaResult;
use crate::utils::signal_channel::signal_channel;
use hyper::{
    header::{HeaderValue, CONTENT_TYPE},
    service::{make_service_fn, service_fn},
    Body, Method, Request, Response, Server,
};
use opentelemetry::global;
use opentelemetry_prometheus::PrometheusExporter;
use prometheus::{Encoder, TextEncoder};
use tracing::{info, info_span, instrument};

lazy_static! {
    static ref EMPTY_HEADER_VALUE: HeaderValue = HeaderValue::from_static("");
}

pub struct MetricsServer {
    addr: SocketAddr,
    exporter: PrometheusExporter,
}

struct AppState {
    exporter: PrometheusExporter,
}

impl MetricsServer {
    pub fn new(application_config: &ApplicationConfigHolder) -> SummaResult<MetricsServer> {
        let addr: SocketAddr = application_config.read().metrics.endpoint.parse()?;
        let exporter = opentelemetry_prometheus::exporter().with_host(addr.ip().to_string()).with_port(addr.port()).init();
        global::meter("summa");
        Ok(MetricsServer { addr, exporter })
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

    #[instrument("lifecycle", skip(self))]
    pub async fn start(&self) -> SummaResult<()> {
        let state = Arc::new(AppState { exporter: self.exporter.clone() });
        let service = make_service_fn(move |_conn| {
            let state = state.clone();
            async move { Ok::<_, Infallible>(service_fn(move |request| MetricsServer::serve_request(request, state.clone()))) }
        });

        let rx = signal_channel();
        let server = Server::bind(&self.addr).serve(service);
        info!(action = "starting", addr = ?self.addr);
        let graceful = server.with_graceful_shutdown(async {
            rx.await.ok();
            info!(action = "sigterm_received");
        });
        graceful.await?;
        info!(action = "terminated");
        Ok(())
    }
}
