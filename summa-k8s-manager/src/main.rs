//! summa-k8s-manager: Kubernetes operator and console for `SearchCluster`
//! resources. See `docs/summa-k8s-manager.md`.

mod broker;
mod console;
mod controller;
mod crd;
mod render;
mod safety;
mod validate;

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::Context as _;
use clap::{Parser, Subcommand};
use kube::CustomResourceExt;
use log::info;

use crate::console::ConsoleState;
use crate::controller::Context;
use crate::crd::SearchCluster;

/// Summa cluster operator and console
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
    /// Run the operator and the console.
    Run(RunArgs),
}

#[derive(clap::Args, Debug)]
struct RunArgs {
    /// Namespace whose SearchCluster this process manages.
    #[arg(long, default_value = "summa")]
    namespace: String,
    /// SearchCluster the console edits; default is the namespace's oldest.
    #[arg(long)]
    cluster: Option<String>,
    /// Console address. The default is reachable only through
    /// `kubectl port-forward`; the console has no authentication.
    #[arg(long, default_value = "127.0.0.1:8080")]
    console_addr: SocketAddr,
    /// Prometheus exporter address.
    #[arg(long, default_value = "0.0.0.0:9184")]
    metrics_addr: SocketAddr,
    /// Broker gRPC address override; default is the cluster's broker Service.
    #[arg(long)]
    broker_addr: Option<String>,
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
    // kube's rustls-tls needs a process-level crypto provider selected
    // explicitly (rustls 0.23 panics at first TLS use otherwise).
    let _ = rustls::crypto::ring::default_provider().install_default();
    metrics_exporter_prometheus::PrometheusBuilder::new()
        .with_http_listener(args.metrics_addr)
        .install()
        .context("install Prometheus exporter")?;
    let client = kube::Client::try_default()
        .await
        .context("Kubernetes client configuration")?;
    let ctx = Arc::new(Context {
        client,
        broker_override: args.broker_addr,
    });
    info!(
        "managing SearchCluster resources in namespace {}; metrics on {}",
        args.namespace, args.metrics_addr
    );
    let console = console::serve(
        ConsoleState {
            ctx: Arc::clone(&ctx),
            namespace: args.namespace.clone(),
            cluster: args.cluster,
        },
        args.console_addr,
        shutdown_signal(),
    );
    let operator = controller::run(ctx, args.namespace);
    // The controller stops on SIGTERM/SIGINT by itself; the console stops on
    // the same signals.
    let (console, ()) = tokio::join!(console, operator);
    console
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
