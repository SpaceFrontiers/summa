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
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::filter::EnvFilter;
use tracing_subscriber::fmt;
use tracing_subscriber::prelude::*;

struct Application {
    configurator: GlobalConfig,
}

const LONG_ABOUT: &str = "
Fast full-text search server with following features:

- Simple GRPC API
- Indexing documents through Kafka";

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

pub fn setup_tracing(log_path: &PathBuf, debug: bool) -> WorkerGuard {
    let file_appender = tracing_appender::rolling::daily(log_path, "summa.log");
    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);
    let filter_layer = EnvFilter::try_from_default_env().or_else(|_| EnvFilter::try_new("info")).unwrap();

    if debug {
        tracing_subscriber::registry()
            .with(filter_layer)
            .with(
                fmt::layer()
                    .with_level(true) // don't include levels in formatted output
                    .with_target(true) // don't include targets
                    .with_thread_names(true),
            )
            .init();
    } else {
        tracing_subscriber::registry()
            .with(filter_layer)
            .with(
                fmt::layer()
                    .with_level(true) // don't include levels in formatted output
                    .with_target(true) // don't include targets
                    .with_thread_names(true)
                    .with_writer(non_blocking),
            )
            .init();
    };
    guard
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
