//! Hermes gRPC broker: one address that fronts many hermes-server instances.
//!
//! Serves the exact hermes.proto SearchService/IndexService so existing
//! clients switch by re-pointing their endpoint, plus the broker-only
//! hermes.broker.BrokerService and grpc.health.v1. Backends are discovered
//! in Kubernetes (pod labels) or configured statically; per-index routing is
//! learned by polling each backend's ListIndexes.

mod admin_service;
mod client;
mod context;
mod discovery;
mod index_service;
mod kube_discovery;
mod metrics;
mod placement;
mod poller;
mod proto;
mod search_service;
mod topology;

use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use anyhow::Result;
use arc_swap::ArcSwap;
use clap::Parser;
use log::{info, warn};
use tokio::sync::{Semaphore, watch};
use tonic::codec::CompressionEncoding;
use tonic::transport::Server;

use crate::client::ClientPool;
use crate::context::BrokerContext;
use crate::placement::{PlacementDefault, PlacementRules};
use crate::proto::broker::broker_service_server::BrokerServiceServer;
use crate::proto::hermes::index_service_server::IndexServiceServer;
use crate::proto::hermes::search_service_server::SearchServiceServer;
use crate::topology::TopologySnapshot;

#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum DiscoveryMode {
    Kubernetes,
    Static,
}

/// Hermes gRPC broker
#[derive(Parser, Debug)]
#[command(name = "hermes-broker")]
#[command(about = "Routes hermes.proto RPCs across multiple Hermes server instances")]
struct Args {
    /// Address to bind to
    #[arg(short, long, default_value = "0.0.0.0:50051")]
    addr: String,

    /// Address for the Prometheus /metrics HTTP endpoint.
    /// Set to "off" to disable the exporter.
    #[arg(long, default_value = "0.0.0.0:9184")]
    metrics_addr: String,

    /// How backends are discovered
    #[arg(long, value_enum, default_value = "kubernetes")]
    discovery: DiscoveryMode,

    /// Kubernetes namespace to watch for hermes-server pods
    #[arg(long, default_value = "hermes")]
    namespace: String,

    /// Pod label carrying the shard id; its presence selects the pods to watch
    #[arg(long, default_value = "hermes.spacefrontiers.org/shard-id")]
    shard_label: String,

    /// Pod label carrying the replica role (master|follower); a shard's sole
    /// unlabeled member is the implicit master
    #[arg(long, default_value = "hermes.spacefrontiers.org/role")]
    role_label: String,

    /// Pod annotation overriding the backend gRPC port
    #[arg(long, default_value = "hermes.spacefrontiers.org/grpc-port")]
    port_annotation: String,

    /// Backend gRPC port when the annotation is absent
    #[arg(long, default_value = "50051")]
    backend_port: u16,

    /// Static backend (repeatable): "id=hs-a,addr=127.0.0.1:50051,shard=0[,role=master]"
    #[arg(long = "backend")]
    backends: Vec<String>,

    /// CreateIndex placement rule (repeatable, first match wins): "pattern=shard",
    /// e.g. "documents*=0". Also pins reads/writes when an index name appears
    /// on several shards during a migration.
    #[arg(long = "placement")]
    placements: Vec<String>,

    /// Where CreateIndex lands when no placement rule matches
    #[arg(long, value_enum, default_value = "single")]
    placement_default: PlacementDefault,

    /// Per-backend in-flight Search cap; mirror of the backend's own
    /// --max-concurrent-searches so the broker never over-admits one backend
    #[arg(long, default_value = "16")]
    backend_max_searches: usize,

    /// Optional broker-global in-flight Search cap on top of the per-backend
    /// caps (unset = per-backend caps only)
    #[arg(long)]
    max_concurrent_searches: Option<usize>,

    /// Steady-state seconds between ListIndexes polls of a healthy backend
    #[arg(long, default_value = "15")]
    index_poll_interval_secs: u64,

    /// Probe seconds for suspect/evicted backends (half-open recovery)
    #[arg(long, default_value = "5")]
    probe_interval_secs: u64,

    /// Seconds a backend may stay suspect (serving off its last-known index
    /// map) before eviction removes it from rotation
    #[arg(long, default_value = "60")]
    backend_unreachable_grace_secs: u64,

    /// Per-poll ListIndexes deadline in seconds
    #[arg(long, default_value = "3")]
    list_timeout_secs: u64,

    /// Maximum number of tokio worker threads (default: min(cpus, 16))
    #[arg(long)]
    worker_threads: Option<usize>,

    /// Maximum decoded (received) gRPC message size in MiB for client-facing
    /// SearchService requests. Mirror the hermes-server flag of the same name
    /// so the broker stays transparent to clients.
    #[arg(long, default_value = "4")]
    search_max_decode_mb: usize,

    /// Maximum encoded (sent) gRPC message size in MiB for client-facing
    /// SearchService responses; must cover the backends'
    /// --max-search-response-mb hydration budget.
    #[arg(long, default_value = "256")]
    search_max_encode_mb: usize,

    /// Maximum decoded (received) gRPC message size in MiB for client-facing
    /// IndexService requests (bounds one indexing batch).
    #[arg(long, default_value = "256")]
    index_max_decode_mb: usize,

    /// Maximum encoded (sent) gRPC message size in MiB for client-facing
    /// IndexService responses.
    #[arg(long, default_value = "64")]
    index_max_encode_mb: usize,

    /// Maximum decoded message size in MiB on broker→backend channels; must
    /// cover the backends' --search-max-encode-mb / --index-max-encode-mb.
    #[arg(long, default_value = "256")]
    backend_max_decode_mb: usize,

    /// Maximum encoded message size in MiB on broker→backend channels; must
    /// cover forwarded index batches (>= --index-max-decode-mb).
    #[arg(long, default_value = "256")]
    backend_max_encode_mb: usize,
}

/// gRPC message caps resolved from CLI flags. Inconsistent combinations that
/// would strand traffic inside the broker are refused or warned about at
/// startup, never discovered per-request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MessageCaps {
    search_max_decode: usize,
    search_max_encode: usize,
    index_max_decode: usize,
    index_max_encode: usize,
    backend_max_decode: usize,
    backend_max_encode: usize,
}

fn nonzero_mb_to_bytes(flag: &str, mb: usize) -> Result<usize> {
    if mb == 0 {
        return Err(anyhow::anyhow!("{flag} must be greater than zero"));
    }
    mb.checked_mul(1024 * 1024)
        .ok_or_else(|| anyhow::anyhow!("{flag} is too large"))
}

fn resolve_message_caps(args: &Args) -> Result<MessageCaps> {
    let caps = MessageCaps {
        search_max_decode: nonzero_mb_to_bytes(
            "--search-max-decode-mb",
            args.search_max_decode_mb,
        )?,
        search_max_encode: nonzero_mb_to_bytes(
            "--search-max-encode-mb",
            args.search_max_encode_mb,
        )?,
        index_max_decode: nonzero_mb_to_bytes("--index-max-decode-mb", args.index_max_decode_mb)?,
        index_max_encode: nonzero_mb_to_bytes("--index-max-encode-mb", args.index_max_encode_mb)?,
        backend_max_decode: nonzero_mb_to_bytes(
            "--backend-max-decode-mb",
            args.backend_max_decode_mb,
        )?,
        backend_max_encode: nonzero_mb_to_bytes(
            "--backend-max-encode-mb",
            args.backend_max_encode_mb,
        )?,
    };
    // Warnings, not errors: a fleet may intentionally run asymmetric caps
    // (e.g. while rolling out a raise), but a silent mismatch strands
    // responses inside the broker with a confusing per-request error.
    if caps.search_max_encode < caps.backend_max_decode {
        warn!(
            "--search-max-encode-mb ({}) is below --backend-max-decode-mb ({}): backend \
             search responses in between will fail to re-encode at the broker edge",
            args.search_max_encode_mb, args.backend_max_decode_mb,
        );
    }
    if caps.backend_max_encode < caps.index_max_decode {
        warn!(
            "--backend-max-encode-mb ({}) is below --index-max-decode-mb ({}): index \
             batches in between are accepted from clients but cannot be forwarded",
            args.backend_max_encode_mb, args.index_max_decode_mb,
        );
    }
    Ok(caps)
}

fn main() -> Result<()> {
    // Install panic hook that logs backtrace: catches panics in spawned tasks
    // that would otherwise be silent.
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let bt = std::backtrace::Backtrace::force_capture();
        eprintln!("=== PANIC ===\n{info}\n{bt}");
        default_hook(info);
    }));

    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("hermes_broker=info"),
    )
    .init();

    let args = Args::parse();

    let worker_threads = args
        .worker_threads
        .unwrap_or_else(|| num_cpus::get().min(16));

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(worker_threads)
        .thread_name("hermes-broker")
        .enable_all()
        .build()?;

    runtime.block_on(async_main(args))
}

async fn async_main(args: Args) -> Result<()> {
    // Prometheus exporter. Fail loud: a bad address or bind failure aborts
    // startup rather than silently serving without metrics.
    if args.metrics_addr != "off" {
        let metrics_addr: SocketAddr = args.metrics_addr.parse().map_err(|e| {
            anyhow::anyhow!("invalid --metrics-addr '{}': {}", args.metrics_addr, e)
        })?;
        metrics_exporter_prometheus::PrometheusBuilder::new()
            .with_http_listener(metrics_addr)
            .install()
            .map_err(|e| {
                anyhow::anyhow!("failed to start metrics exporter on {metrics_addr}: {e}")
            })?;
        info!("Prometheus metrics on http://{metrics_addr}/metrics");
    } else {
        warn!("Prometheus metrics exporter disabled (--metrics-addr off)");
    }

    let addr: SocketAddr = args.addr.parse()?;
    let message_caps = resolve_message_caps(&args)?;
    if args.backend_max_searches == 0 {
        return Err(anyhow::anyhow!(
            "--backend-max-searches must be greater than zero"
        ));
    }
    if args.max_concurrent_searches == Some(0) {
        return Err(anyhow::anyhow!(
            "--max-concurrent-searches must be greater than zero when set"
        ));
    }

    let placement_rules = args
        .placements
        .iter()
        .map(|s| placement::parse_placement(s))
        .collect::<Result<Vec<_>>>()?;
    for rule in &placement_rules {
        info!(
            "placement rule: {} -> shard '{}'",
            rule.pattern, rule.shard.0
        );
    }
    let placement = Arc::new(PlacementRules::new(placement_rules, args.placement_default));

    // Discovery feeds a watch channel of full endpoint sets; the poller owns
    // everything downstream of it.
    let (endpoints_tx, endpoints_rx) = watch::channel(Vec::new());
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let discovery_handle = match args.discovery {
        DiscoveryMode::Static => {
            if args.backends.is_empty() {
                return Err(anyhow::anyhow!(
                    "--discovery static requires at least one --backend"
                ));
            }
            let endpoints = args
                .backends
                .iter()
                .map(|s| discovery::parse_static_backend(s))
                .collect::<Result<Vec<_>>>()?;
            for ep in &endpoints {
                info!(
                    "static backend {} at {} (shard '{}')",
                    ep.id.0, ep.addr, ep.shard.0
                );
            }
            endpoints_tx.send(endpoints)?;
            // Keep the sender alive for the process lifetime so the poller's
            // watch channel never closes.
            tokio::spawn(async move {
                let _guard = endpoints_tx;
                std::future::pending::<()>().await;
            })
        }
        DiscoveryMode::Kubernetes => {
            if !args.backends.is_empty() {
                return Err(anyhow::anyhow!(
                    "--backend is only valid with --discovery static"
                ));
            }
            let cfg = kube_discovery::KubeDiscoveryConfig {
                namespace: args.namespace.clone(),
                shard_label: args.shard_label.clone(),
                role_label: args.role_label.clone(),
                port_annotation: args.port_annotation.clone(),
                default_port: args.backend_port,
            };
            // kube's rustls-tls requires a process-level crypto provider;
            // installing twice is fine (subsequent installs error, ignored).
            let _ = rustls::crypto::ring::default_provider().install_default();
            let shutdown = shutdown_rx.clone();
            let fail_shutdown = shutdown_tx.clone();
            // Fail loud: a broker with no discovery routes nothing. The
            // double-spawn also converts a PANIC inside the discovery task
            // (JoinError) into shutdown instead of a zombie broker that
            // forever answers NOT_SERVING.
            tokio::spawn(async move {
                let outcome = tokio::spawn(kube_discovery::run(cfg, endpoints_tx, shutdown)).await;
                match outcome {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => {
                        log::error!("kubernetes discovery failed: {e:#}; shutting down");
                        let _ = fail_shutdown.send(true);
                    }
                    Err(join_error) => {
                        log::error!("kubernetes discovery crashed: {join_error}; shutting down");
                        let _ = fail_shutdown.send(true);
                    }
                }
            })
        }
    };

    let pool = Arc::new(ClientPool::new(
        args.backend_max_searches,
        message_caps.backend_max_decode,
        message_caps.backend_max_encode,
    ));
    let snapshot = Arc::new(ArcSwap::from_pointee(TopologySnapshot::default()));
    let (health_reporter, health_service) = tonic_health::server::health_reporter();
    health_reporter
        .set_service_status("", tonic_health::ServingStatus::NotServing)
        .await;

    let (poller_handle, refresh) = poller::spawn_poller(
        endpoints_rx,
        Arc::clone(&pool),
        Arc::clone(&placement),
        Arc::clone(&snapshot),
        health_reporter.clone(),
        poller::PollerConfig {
            poll_interval: Duration::from_secs(args.index_poll_interval_secs),
            probe_interval: Duration::from_secs(args.probe_interval_secs),
            grace: Duration::from_secs(args.backend_unreachable_grace_secs),
            list_timeout: Duration::from_secs(args.list_timeout_secs),
        },
        shutdown_rx.clone(),
    );

    let shutting_down = Arc::new(AtomicBool::new(false));
    let ctx = Arc::new(BrokerContext {
        snapshot,
        pool,
        placement,
        refresh,
        global_search_permits: args
            .max_concurrent_searches
            .map(|n| Arc::new(Semaphore::new(n))),
        read_rotation: AtomicUsize::new(0),
        shutting_down: Arc::clone(&shutting_down),
    });

    let search_service = search_service::BrokerSearchService {
        ctx: Arc::clone(&ctx),
    };
    let index_service = index_service::BrokerIndexService {
        ctx: Arc::clone(&ctx),
    };
    let admin_service = admin_service::BrokerAdminService {
        ctx: Arc::clone(&ctx),
    };

    // Server reflection so grpcurl and friends can drive the broker (and the
    // hermes.proto surface behind it) without local .proto files — the
    // migration runbook's verification steps depend on this.
    let reflection_service = tonic_reflection::server::Builder::configure()
        .register_encoded_file_descriptor_set(proto::HERMES_DESCRIPTOR)
        .register_encoded_file_descriptor_set(proto::BROKER_DESCRIPTOR)
        .build_v1()
        .map_err(|e| anyhow::anyhow!("failed to build reflection service: {e}"))?;

    info!("Hermes broker v{}", env!("CARGO_PKG_VERSION"));
    info!("Starting Hermes broker on {addr}");
    info!("Discovery: {:?}", args.discovery);
    info!(
        "Per-backend search admission: {}",
        args.backend_max_searches
    );
    match args.max_concurrent_searches {
        Some(n) => info!("Broker-global search admission: {n}"),
        None => info!("Broker-global search admission: per-backend caps only"),
    }

    // Defaults mirror hermes-server so the broker is transparent to clients
    // written against the server's envelope.
    info!(
        "gRPC message caps: search {}/{} MiB decode/encode, index {}/{} MiB decode/encode, \
         backend channels {}/{} MiB decode/encode",
        args.search_max_decode_mb,
        args.search_max_encode_mb,
        args.index_max_decode_mb,
        args.index_max_encode_mb,
        args.backend_max_decode_mb,
        args.backend_max_encode_mb,
    );

    let signal_flag = Arc::clone(&shutting_down);
    let signal_shutdown = shutdown_tx.clone();
    let drain_health = health_reporter.clone();
    let internal_shutdown_rx = shutdown_rx.clone();
    let serve_result = Server::builder()
        .tcp_keepalive(Some(Duration::from_secs(60)))
        .http2_keepalive_interval(Some(Duration::from_secs(30)))
        .http2_keepalive_timeout(Some(Duration::from_secs(10)))
        .http2_adaptive_window(Some(true))
        .initial_connection_window_size(Some(4 * 1024 * 1024))
        .initial_stream_window_size(Some(2 * 1024 * 1024))
        .max_concurrent_streams(Some(256))
        .concurrency_limit_per_connection(128)
        .add_service(health_service)
        .add_service(reflection_service)
        .add_service(
            SearchServiceServer::new(search_service)
                .max_decoding_message_size(message_caps.search_max_decode)
                .max_encoding_message_size(message_caps.search_max_encode)
                .accept_compressed(CompressionEncoding::Gzip)
                .accept_compressed(CompressionEncoding::Zstd)
                .send_compressed(CompressionEncoding::Zstd),
        )
        .add_service(
            IndexServiceServer::new(index_service)
                .max_decoding_message_size(message_caps.index_max_decode)
                .max_encoding_message_size(message_caps.index_max_encode)
                .accept_compressed(CompressionEncoding::Gzip)
                .accept_compressed(CompressionEncoding::Zstd)
                .send_compressed(CompressionEncoding::Zstd),
        )
        .add_service(BrokerServiceServer::new(admin_service))
        .serve_with_shutdown(addr, async move {
            // OS signal, or internal failure (e.g. discovery crash flips the
            // shutdown channel): either way exit rather than linger as a
            // routing-dead process.
            let mut internal_shutdown = internal_shutdown_rx;
            tokio::select! {
                _ = shutdown_signal() => {}
                _ = async {
                    while internal_shutdown.changed().await.is_ok() {
                        if *internal_shutdown.borrow() {
                            break;
                        }
                    }
                } => {
                    warn!("internal shutdown requested; draining");
                }
            }
            // Refuse new RPCs while tonic drains in-flight ones, and flip the
            // health probe so k8s stops routing to this pod.
            signal_flag.store(true, Ordering::Relaxed);
            drain_health
                .set_service_status("", tonic_health::ServingStatus::NotServing)
                .await;
            let _ = signal_shutdown.send(true);
        })
        .await;

    // A transport failure also takes the process through ordered shutdown.
    shutting_down.store(true, Ordering::Relaxed);
    let _ = shutdown_tx.send(true);

    info!("[shutdown] gRPC server drained; waiting for background tasks");
    let _ = poller_handle.await;
    discovery_handle.abort();
    serve_result?;
    info!("Hermes broker shut down gracefully");
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install ctrl+c handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            warn!("Received ctrl+c, starting graceful shutdown...");
        }
        _ = terminate => {
            warn!("Received SIGTERM, starting graceful shutdown...");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_defaults() {
        let args = Args::try_parse_from(["hermes-broker"]).unwrap();
        assert_eq!(args.discovery, DiscoveryMode::Kubernetes);
        assert_eq!(args.namespace, "hermes");
        assert_eq!(args.backend_max_searches, 16);
        assert_eq!(args.max_concurrent_searches, None);
        assert_eq!(args.placement_default, PlacementDefault::Single);

        // Message caps default to the historical hard-coded envelope.
        let caps = resolve_message_caps(&args).unwrap();
        assert_eq!(
            caps,
            MessageCaps {
                search_max_decode: 4 * 1024 * 1024,
                search_max_encode: 256 * 1024 * 1024,
                index_max_decode: 256 * 1024 * 1024,
                index_max_encode: 64 * 1024 * 1024,
                backend_max_decode: 256 * 1024 * 1024,
                backend_max_encode: 256 * 1024 * 1024,
            }
        );
    }

    #[test]
    fn message_cap_flags_override_and_reject_zero() {
        let args = Args::try_parse_from([
            "hermes-broker",
            "--search-max-encode-mb",
            "512",
            "--backend-max-decode-mb",
            "512",
        ])
        .unwrap();
        let caps = resolve_message_caps(&args).unwrap();
        assert_eq!(caps.search_max_encode, 512 * 1024 * 1024);
        assert_eq!(caps.backend_max_decode, 512 * 1024 * 1024);

        let zero = Args::try_parse_from(["hermes-broker", "--index-max-decode-mb", "0"]).unwrap();
        assert!(resolve_message_caps(&zero).is_err());
    }

    #[test]
    fn cli_static_mode_with_backends_and_placements() {
        let args = Args::try_parse_from([
            "hermes-broker",
            "--discovery",
            "static",
            "--backend",
            "id=a,addr=127.0.0.1:50051,shard=0",
            "--backend",
            "id=b,addr=127.0.0.1:50052,shard=1,role=master",
            "--placement",
            "documents*=0",
            "--placement",
            "social*=1",
        ])
        .unwrap();
        assert_eq!(args.backends.len(), 2);
        assert_eq!(args.placements.len(), 2);
    }
}
