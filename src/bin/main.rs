use futures::try_join;

use clap::{arg, command};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use summa::configs::ApplicationConfig;
use summa::configs::GlobalConfig;
use summa::errors::SummaResult;
use summa::servers::GrpcServer;
use summa::servers::MetricsServer;
use tokio::runtime;
use tracing_appender::non_blocking::{NonBlocking, WorkerGuard};
use tracing_subscriber::fmt;
use tracing_subscriber::prelude::*;
use tracing_subscriber::EnvFilter;

struct Application {
    configurator: GlobalConfig,
}

const LONG_ABOUT: &str = "
Fast full-text search server with following features:

- Simple GRPC API for managing multiple indices and for search
- Indexing documents through Kafka or directly
- Tracing with OpenTelemetry and exposing metrics in Prometheus format
- Various configurable tokenizers (including CJK)
- Fine CLI and asynchronous client library written in Python
";

impl Application {
    pub fn new(configurator: GlobalConfig) -> SummaResult<Application> {
        Ok(Application { configurator })
    }

    pub fn create_runtime(&self) -> SummaResult<runtime::Runtime> {
        Ok(runtime::Builder::new_multi_thread()
            .enable_all()
            .thread_name_fn(|| {
                static ATOMIC_ID: AtomicUsize = AtomicUsize::new(0);
                let id = ATOMIC_ID.fetch_add(1, Ordering::SeqCst);
                format!("tokio-runtime-workers-{}", id)
            })
            .build()?)
    }

    pub fn run(&self) -> SummaResult<()> {
        let rt = self.create_runtime()?;
        rt.block_on(async {
            let application_config = self.configurator.application_config.read();
            let metrics_server = MetricsServer::new(application_config.metrics.endpoint.parse()?)?;
            let grpc_server = GrpcServer::new(application_config.grpc.endpoint.parse()?, &application_config.data_path, &self.configurator.runtime_config).await?;
            drop(application_config);

            try_join!(metrics_server.start(), grpc_server.start())?;
            Ok(())
        })
    }
}

fn create_writer(log_path: &PathBuf, name: &str, guards: &mut Vec<WorkerGuard>) -> NonBlocking {
    let file_appender = tracing_appender::rolling::daily(log_path, name);
    let (file_writer, guard) = tracing_appender::non_blocking(file_appender);
    guards.push(guard);
    file_writer
}

pub fn setup_tracing(log_path: &PathBuf, debug: bool) -> Vec<WorkerGuard> {
    let mut guards = Vec::new();
    let file_writer_request = create_writer(log_path, "request.log", &mut guards);
    let file_writer_query = create_writer(log_path, "query.log", &mut guards);
    let file_writer_summa = create_writer(log_path, "summa.log", &mut guards);

    let filter_layer_request = EnvFilter::new("summa::servers::grpc[request]=info,summa::servers::metrics[request]=info");
    let filter_layer_query = EnvFilter::new("query");
    let filter_layer_summa = EnvFilter::new("summa::services=info,summa::search_engine=info,summa::consumers=info");

    if debug {
        let default_layer = fmt::layer().with_level(true).with_target(true).with_thread_names(true).with_filter(filter_layer_summa);
        tracing_subscriber::registry().with(default_layer).init();
    } else {
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
            .with_target(false)
            .with_level(true)
            .with_writer(file_writer_summa)
            .with_filter(filter_layer_summa);
        tracing_subscriber::registry().with(request_layer).with(query_layer).with(default_layer).init();
    };
    guards
}

fn proceed_args() -> clap::ArgMatches {
    command!()
        .name("summa")
        .override_usage("summa-server [OPTIONS] <SUBCOMMAND>")
        .subcommand_required(true)
        .arg_required_else_help(true)
        .about(LONG_ABOUT)
        .version("2.0.0")
        .author("Pasha Podolsky")
        .arg(arg!(-v --verbose ... "Level of verbosity"))
        .subcommand(command!("generate-config").about("Generate default config file"))
        .subcommand(
            command!("serve")
                .about("Launch search server")
                .arg(arg!(<CONFIG> "Search engine config file").required(true).takes_value(true)),
        )
        .get_matches()
}

fn main() -> SummaResult<()> {
    let matches = proceed_args();
    match matches.subcommand() {
        Some(("generate-config", _)) => {
            let default_config = ApplicationConfig::default();
            println!("{}", serde_yaml::to_string(&default_config).unwrap());
            Ok(())
        }
        Some(("serve", submatches)) => {
            let config_path = submatches.value_of("CONFIG").map(Path::new).unwrap();
            let configurator = GlobalConfig::new(config_path)?;
            let application_config = configurator.application_config.read();
            let _log_guard = setup_tracing(&application_config.log_path, application_config.debug);
            drop(application_config);
            let app = Application::new(configurator)?;
            app.run()
        }
        _ => unreachable!("Exhausted list of subcommands and subcommand_required prevents `None`"),
    }
}
