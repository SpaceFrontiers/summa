use std::path::PathBuf;
use tracing_appender::non_blocking::{NonBlocking, WorkerGuard};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{fmt, EnvFilter, Layer};

const ENV_FILTER: &str = "librdkafka=trace,\
    rdkafka::client=debug,\
    summa::services=info,\
    summa::search_engine=info,\
    summa::servers[lifecycle]=info,\
    summa::consumers=info,\
    tantivy=info";

const REQUEST_ENV_FILTER: &str = "summa::servers::grpc[request]=info,summa::servers::metrics[request]=info";

fn create_writer(log_path: &PathBuf, name: &str, guards: &mut Vec<WorkerGuard>) -> NonBlocking {
    let file_appender = tracing_appender::rolling::daily(log_path, name);
    let (file_writer, guard) = tracing_appender::non_blocking(file_appender);
    guards.push(guard);
    file_writer
}

pub fn default() -> Vec<WorkerGuard> {
    let default_layer = fmt::layer()
        .with_level(true)
        .with_target(true)
        .with_thread_names(true)
        .with_filter(EnvFilter::new(format!("{},{}", ENV_FILTER, REQUEST_ENV_FILTER)));
    tracing_subscriber::registry().with(default_layer).init();
    vec![]
}

pub fn file(log_path: &PathBuf) -> Vec<WorkerGuard> {
    let mut guards = Vec::new();
    let file_writer_request = create_writer(log_path, "request.log", &mut guards);
    let file_writer_query = create_writer(log_path, "query.log", &mut guards);
    let file_writer_summa = create_writer(log_path, "summa.log", &mut guards);

    let filter_layer_request = EnvFilter::new(REQUEST_ENV_FILTER);
    let filter_layer_query = EnvFilter::new("query");

    let request_layer = fmt::layer()
        .with_thread_names(false)
        .with_target(false)
        .with_writer(file_writer_request)
        .with_filter(filter_layer_request);
    let query_layer = fmt::layer()
        .with_thread_names(false)
        .with_target(false)
        .with_writer(file_writer_query)
        .with_filter(filter_layer_query);
    let default_layer = fmt::layer()
        .with_thread_names(false)
        .with_target(true)
        .with_level(true)
        .with_writer(file_writer_summa)
        .with_filter(EnvFilter::new(ENV_FILTER));
    tracing_subscriber::registry().with(request_layer).with(query_layer).with(default_layer).init();

    guards
}

#[cfg(test)]
pub mod tests {
    use std::sync::Once;

    static INIT: Once = Once::new();
    pub fn initialize_default_once() {
        INIT.call_once(|| {
            super::default();
        });
    }
}
