use std::fs::{File, OpenOptions};
use std::io;
use std::io::Write;
use std::path::{Path, PathBuf};

use tracing_appender::non_blocking::{NonBlocking, WorkerGuard};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{fmt, EnvFilter, Layer};

const ENV_FILTER: &str = "librdkafka=warn,\
    quinn_proto::connection=error,\
    rdkafka::client=warn,\
    summa_core::components=info,\
    summa_core::directories=info,\
    summa_core=warn,\
    summa_server::services=info,\
    summa_server::services::api=warn,\
    summa_server::components=info,\
    summa_server::server[lifecycle]=info,\
    summa_server=warn,\
    tantivy::store::compressors=error,\
    warn";

const REQUEST_ENV_FILTER: &str = "summa_server::server::grpc[request]=info,summa_server::server::metrics[request]=info";

struct WatchedWriter {
    file: File,
    file_name: PathBuf,
}

impl WatchedWriter {
    pub fn new(file_name: impl AsRef<Path>) -> io::Result<WatchedWriter> {
        Ok(WatchedWriter {
            file: OpenOptions::new().append(true).create(true).open(&file_name)?,
            file_name: file_name.as_ref().to_path_buf(),
        })
    }
    fn reopen(&mut self) -> io::Result<()> {
        self.file = OpenOptions::new().append(true).create(true).open(&self.file_name)?;
        Ok(())
    }
}

impl Write for WatchedWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.file.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()?;
        if !self.file_name.exists() {
            self.reopen()?;
        }
        Ok(())
    }
}

fn create_writer(log_path: &Path, name: &str, guards: &mut Vec<WorkerGuard>) -> io::Result<NonBlocking> {
    let file = WatchedWriter::new(log_path.join(name).with_extension("log"))?;
    let (file_writer, guard) = tracing_appender::non_blocking(file);
    guards.push(guard);
    Ok(file_writer)
}

pub fn default() -> Vec<WorkerGuard> {
    let default_layer = fmt::layer()
        .with_level(true)
        .with_target(true)
        .with_thread_names(true)
        .with_filter(EnvFilter::new(format!("{ENV_FILTER},{REQUEST_ENV_FILTER}")));
    tracing_subscriber::registry().with(default_layer).init();
    vec![]
}

pub fn file(log_path: &Path) -> io::Result<Vec<WorkerGuard>> {
    let mut guards = Vec::new();

    std::fs::create_dir_all(log_path)?;
    let file_writer_request = create_writer(log_path, "request", &mut guards)?;
    let file_writer_query = create_writer(log_path, "query", &mut guards)?;
    let file_writer_summa = create_writer(log_path, "summa", &mut guards)?;

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

    Ok(guards)
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
