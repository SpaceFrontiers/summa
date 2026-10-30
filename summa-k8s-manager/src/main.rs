//! summa-k8s-manager: Kubernetes operator, API and console for Summa
//! indexes, one `SearchCluster` each. See `docs/summa-k8s-manager.md`.

mod api;
mod broker;
mod controller;
mod crd;
mod render;
mod validate;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context as _;
use clap::{Parser, Subcommand};
use kube::CustomResourceExt;
use log::info;

use crate::api::Access;
use crate::controller::Context;
use crate::crd::SearchCluster;

/// Summa index operator, API and console
#[derive(Parser, Debug)]
#[command(name = "summa-k8s-manager")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Print the SearchCluster CustomResourceDefinition as JSON.
    Crd,
    /// Run the operator, the API and the console.
    Run(RunArgs),
}

#[derive(clap::Args, Debug)]
struct RunArgs {
    /// Namespace whose SearchClusters this process manages.
    #[arg(long, default_value = "summa")]
    namespace: String,
    /// Console address for `kubectl port-forward`; no token, `localhost`
    /// Host headers only.
    #[arg(long, default_value = "127.0.0.1:8080")]
    console_addr: SocketAddr,
    /// API address for in-cluster clients; served only with --api-token-file.
    #[arg(long, default_value = "0.0.0.0:8081")]
    api_addr: SocketAddr,
    /// File holding the bearer token the API requires.
    #[arg(long)]
    api_token_file: Option<PathBuf>,
    /// Prometheus exporter address.
    #[arg(long, default_value = "0.0.0.0:9184")]
    metrics_addr: SocketAddr,
    /// Broker address of a cluster; `{cluster}` and `{namespace}` are replaced.
    #[arg(long, default_value = "{cluster}.{namespace}.svc:50051")]
    broker_endpoint: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info,kube_runtime=warn,kube_client=warn"),
    )
    .init();
    match Args::parse().command {
        Command::Crd => {
            println!("{}", serde_json::to_string_pretty(&SearchCluster::crd())?);
            Ok(())
        }
        Command::Run(args) => run(args).await,
    }
}

async fn run(args: RunArgs) -> anyhow::Result<()> {
    // kube's rustls-tls needs a process-level crypto provider.
    let _ = rustls::crypto::ring::default_provider().install_default();
    metrics_exporter_prometheus::PrometheusBuilder::new()
        .with_http_listener(args.metrics_addr)
        .install()
        .context("install Prometheus exporter")?;
    let token = args
        .api_token_file
        .as_ref()
        .map(|path| {
            let token = std::fs::read_to_string(path)
                .with_context(|| format!("read --api-token-file {}", path.display()))?;
            anyhow::ensure!(!token.trim().is_empty(), "--api-token-file is empty");
            Ok(Arc::<str>::from(token.trim()))
        })
        .transpose()?;
    let ctx = Arc::new(Context {
        client: kube::Client::try_default()
            .await
            .context("Kubernetes client configuration")?,
        namespace: args.namespace,
        broker_endpoint: args.broker_endpoint,
    });
    info!(
        "managing SearchClusters in namespace {}; console on {}; API on {}; metrics on {}",
        ctx.namespace,
        args.console_addr,
        if token.is_some() {
            args.api_addr.to_string()
        } else {
            "off (no --api-token-file)".into()
        },
        args.metrics_addr
    );
    let console = serve(
        args.console_addr,
        api::router(Arc::clone(&ctx), Access::Console),
    );
    let api = async {
        match token {
            Some(token) => {
                serve(
                    args.api_addr,
                    api::router(Arc::clone(&ctx), Access::Token(token)),
                )
                .await
            }
            None => Ok(()),
        }
    };
    // The controller stops on SIGTERM/SIGINT by itself.
    let (console, api, ()) = tokio::join!(console, api, controller::run(Arc::clone(&ctx)));
    console.and(api)
}

async fn serve(addr: SocketAddr, app: axum::Router) -> anyhow::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("bind {addr}"))?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .with_context(|| format!("serve {addr}"))
}

async fn shutdown_signal() {
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                signal.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {}
        () = terminate => {}
    }
}
