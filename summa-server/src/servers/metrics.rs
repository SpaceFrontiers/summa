//! HTTP server exposing metrics in Prometheus format

use std::convert::Infallible;
use std::fmt::Debug;
use std::future::Future;
use std::sync::Arc;

use async_broadcast::Receiver;
use hyper::{
    header::{HeaderValue, CONTENT_TYPE},
    service::{make_service_fn, service_fn},
    Body, Method, Request, Response, Server,
};
use opentelemetry::sdk::export::metrics::{aggregation, AggregatorSelector};
use opentelemetry::sdk::metrics::aggregators::Aggregator;
use opentelemetry::sdk::metrics::sdk_api::{Descriptor, InstrumentKind};
use opentelemetry::sdk::metrics::{aggregators, controllers, processors};
use opentelemetry_prometheus::PrometheusExporter;
use prometheus::{Encoder, TextEncoder};
use summa_core::configs::ApplicationConfigHolder;
use summa_core::utils::thread_handler::ControlMessage;
use tracing::{info, info_span, instrument};

use super::base::BaseServer;
use crate::errors::SummaServerResult;
use crate::search_engine::IndexMeter;
use crate::services::IndexService;

lazy_static! {
    static ref EMPTY_HEADER_VALUE: HeaderValue = HeaderValue::from_static("");
}

#[derive(Clone)]
pub struct MetricsServer {
    application_config: ApplicationConfigHolder,
    exporter: PrometheusExporter,
}

impl BaseServer for MetricsServer {}

struct AppState {
    exporter: PrometheusExporter,
    index_service: IndexService,
    index_meter: IndexMeter,
}

#[derive(Debug)]
struct CustomAgg;

impl AggregatorSelector for CustomAgg {
    fn aggregator_for(&self, descriptor: &Descriptor) -> Option<Arc<dyn Aggregator + Send + Sync>> {
        match descriptor.instrument_kind() {
            InstrumentKind::Counter => Some(Arc::new(aggregators::sum())),
            _ => match descriptor.unit() {
                Some("bytes") => Some(Arc::new(aggregators::last_value())),
                Some("seconds") => Some(Arc::new(aggregators::histogram(&[
                    0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0,
                ]))),
                _ => Some(Arc::new(aggregators::last_value())),
            },
        }
    }
}

impl MetricsServer {
    pub fn new(application_config: &ApplicationConfigHolder) -> SummaServerResult<MetricsServer> {
        let controller = controllers::basic(processors::factory(CustomAgg, aggregation::cumulative_temporality_selector()).with_memory(true)).build();
        let exporter = opentelemetry_prometheus::exporter(controller).init();
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
                for index_holder in state.index_service.index_holders().await.values() {
                    state.index_meter.record_metrics(index_holder)
                }

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
    pub async fn start(
        &self,
        index_service: &IndexService,
        mut terminator: Receiver<ControlMessage>,
    ) -> SummaServerResult<impl Future<Output = SummaServerResult<()>>> {
        let metrics_config = self.application_config.read().await.metrics.clone();
        let state = Arc::new(AppState {
            exporter: self.exporter.clone(),
            index_service: index_service.clone(),
            index_meter: IndexMeter::new(),
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
