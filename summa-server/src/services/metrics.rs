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
use tracing::{info, info_span, instrument};
use tracing_futures::Instrument;

use crate::components::IndexMeter;
use crate::errors::{Error, SummaServerResult};
use crate::services::Index;
use crate::utils::thread_handler::ControlMessage;

#[derive(Clone)]
pub struct Metrics {
    config: crate::configs::metrics::Config,
    exporter: PrometheusExporter,
}

struct AppState {
    exporter: PrometheusExporter,
    index_service: Index,
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

impl Metrics {
    pub fn new(config: &crate::configs::metrics::Config) -> SummaServerResult<Metrics> {
        let controller = controllers::basic(processors::factory(CustomAgg, aggregation::cumulative_temporality_selector())).build();
        let exporter = opentelemetry_prometheus::exporter(controller).init();
        Ok(Metrics {
            config: config.clone(),
            exporter,
        })
    }

    async fn serve_request(request: Request<Body>, state: Arc<AppState>) -> Result<Response<Body>, hyper::Error> {
        let empty_header_value = HeaderValue::from_static("");
        let _span = info_span!(
            "request",
            request_id = ?request.headers().get("request-id").unwrap_or(&empty_header_value),
            session_id = ?request.headers().get("session-id").unwrap_or(&empty_header_value),
        );
        info!(path = ?request.uri().path());
        let response = match request.method() {
            &Method::GET => {
                for index_holder in state.index_service.index_registry().index_holders().read().await.values() {
                    state
                        .index_meter
                        .record_metrics(index_holder)
                        .map_err(Error::from)
                        .expect("cannot record meters")
                }

                let mut buffer = vec![];
                let encoder = TextEncoder::new();
                let metric_families = state.exporter.registry().gather();
                encoder.encode(&metric_families[..], &mut buffer).expect("prometheus failed");
                Response::builder()
                    .status(200)
                    .header(CONTENT_TYPE, encoder.format_type())
                    .body(Body::from(buffer))
                    .expect("encoding body failed")
            }
            _ => Response::builder().status(404).body(Body::from("Missing Page")).expect("encoding body failed"),
        };
        Ok(response)
    }

    #[instrument("lifecycle", skip_all)]
    pub async fn prepare_serving_future(
        &self,
        index_service: &Index,
        mut terminator: Receiver<ControlMessage>,
    ) -> SummaServerResult<impl Future<Output = SummaServerResult<()>>> {
        let state = Arc::new(AppState {
            exporter: self.exporter.clone(),
            index_service: index_service.clone(),
            index_meter: IndexMeter::new(),
        });
        let service = make_service_fn(move |_conn| {
            let state = state.clone();
            async move { Ok::<_, Infallible>(service_fn(move |request| Metrics::serve_request(request, state.clone()))) }
        });

        let server = Server::bind(&self.config.endpoint.parse()?).serve(service);
        info!(action = "binded", endpoint = ?self.config.endpoint);
        let graceful = server.with_graceful_shutdown(async move {
            let signal_result = terminator.recv().await;
            info!(action = "sigterm_received", received = ?signal_result);
        });

        Ok(async move {
            match graceful.await {
                Ok(_) => info!(action = "terminated"),
                Err(e) => info!(action = "terminated", error = ?e),
            }
            Ok(())
        }
        .instrument(info_span!(parent: None, "lifecycle")))
    }
}
