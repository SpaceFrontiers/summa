//! HTTP server exposing metrics in Prometheus format

use std::fmt::Debug;
use std::future::Future;
use std::sync::Arc;

use async_broadcast::Receiver;
use http_body_util::Full;
use hyper::server::conn::http1;
use hyper::{
    header::{HeaderValue, CONTENT_TYPE},
    service::service_fn,
    Method, Request, Response,
};
use hyper_util::rt::TokioIo;
use opentelemetry::metrics::MeterProvider;
use opentelemetry::sdk::metrics::reader::AggregationSelector;
use opentelemetry::sdk::metrics::{Aggregation, InstrumentKind};
use prometheus::{Encoder, Registry, TextEncoder};
use tokio::net::TcpListener;
use tracing::{error, info, info_span, instrument};
use tracing_futures::Instrument;

use crate::components::IndexMeter;
use crate::errors::{Error, SummaServerResult};
use crate::services::Index;
use crate::utils::thread_handler::ControlMessage;

#[derive(Clone)]
pub struct Metrics {
    config: crate::configs::metrics::Config,
    registry: Registry,
}

struct AppState {
    index_service: Index,
    index_meter: IndexMeter,
    registry: Registry,
}

#[derive(Debug)]
struct CustomAgg;

impl AggregationSelector for CustomAgg {
    fn aggregation(&self, kind: InstrumentKind) -> Aggregation {
        match kind {
            InstrumentKind::Counter => Aggregation::Sum,
            InstrumentKind::Histogram => Aggregation::ExplicitBucketHistogram {
                boundaries: vec![0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
                record_min_max: false,
            },
            _ => Aggregation::LastValue,
        }
    }
}

impl Metrics {
    pub fn new(config: &crate::configs::metrics::Config) -> SummaServerResult<Metrics> {
        let registry = Registry::new();
        Ok(Metrics {
            config: config.clone(),
            registry,
        })
    }

    async fn serve_request(request: Request<hyper::body::Incoming>, state: Arc<AppState>) -> Result<Response<Full<hyper::body::Bytes>>, hyper::Error> {
        let empty_header_value = HeaderValue::from_static("");
        let _span = info_span!(
            "request",
            request_id = ?request.headers().get("x-request-id").unwrap_or(&empty_header_value),
            session_id = ?request.headers().get("x-session-id").unwrap_or(&empty_header_value),
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
                let metric_families = state.registry.gather();
                encoder.encode(&metric_families[..], &mut buffer).expect("prometheus failed");
                Response::builder()
                    .status(200)
                    .header(CONTENT_TYPE, encoder.format_type())
                    .body(Full::new(hyper::body::Bytes::from(buffer)))
                    .expect("encoding body failed")
            }
            _ => Response::builder()
                .status(404)
                .body(Full::new(hyper::body::Bytes::from("Missing Page")))
                .expect("encoding body failed"),
        };
        Ok(response)
    }

    #[instrument("lifecycle", skip_all)]
    pub async fn prepare_serving_future(
        &self,
        index_service: &Index,
        mut terminator: Receiver<ControlMessage>,
    ) -> SummaServerResult<impl Future<Output = SummaServerResult<()>>> {
        let exporter = opentelemetry_prometheus::exporter()
            .with_registry(self.registry.clone())
            .with_aggregation_selector(CustomAgg)
            .build()
            .expect("internal error");
        let provider = opentelemetry::sdk::metrics::MeterProvider::builder().with_reader(exporter).build();
        let meter = provider.meter("summa");

        let state = Arc::new(AppState {
            index_service: index_service.clone(),
            index_meter: IndexMeter::new(meter),
            registry: self.registry.clone(),
        });
        let listener = TcpListener::bind(&self.config.endpoint).await?;
        let http = http1::Builder::new();
        let svc = service_fn(move |request| {
            let cloned_state = state.clone();
            Metrics::serve_request(request, cloned_state)
        });

        Ok(async move {
            loop {
                tokio::select! {
                    Ok((stream, _addr)) = listener.accept() => {
                        let io = TokioIo::new(stream);
                        let conn = http.serve_connection(io, svc.clone());
                        tokio::spawn(async move {
                            if let Err(e) = conn.await {
                                error!(action = "error_serving_connection", error = ?e);
                            }
                        });
                    },
                    signal_result = &mut terminator.recv() => {
                        info!(action = "sigterm_received", received = ?signal_result);
                        break;
                    }
                }
            }
            Ok(())
        }
        .instrument(info_span!(parent: None, "lifecycle")))
    }
}
