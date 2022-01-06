use std::future::Future;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;

use async_broadcast::Receiver;
use clap::{arg, command};
use futures_util::future::try_join_all;
use iroh_unixfs::content_loader::GatewayUrl;
use summa_core::configs::ConfigProxy;
use summa_core::utils::thread_handler::ControlMessage;
use tracing::{info, info_span, Instrument};

use crate::configs::server::ConfigHolder;
use crate::errors::{Error, SummaServerResult};
use crate::logging;
use crate::services::gateway::Gateway;
use crate::services::store::Store;
use crate::services::{Api, Index, Metrics, P2p};
use crate::utils::signal_channel;

pub struct Server {
    server_config_holder: Arc<dyn ConfigProxy<crate::configs::server::Config>>,
}

const LONG_ABOUT: &str = "
Fast full-text search server.

Documentation: https://izihawa.github.io/summa
";

impl Server {
    pub fn from_server_config(config: Arc<dyn ConfigProxy<crate::configs::server::Config>>) -> Server {
        Server { server_config_holder: config }
    }

    pub async fn proceed_args() -> SummaServerResult<()> {
        let matches = command!()
            .name("summa")
            .override_usage("summa-server [OPTIONS] <SUBCOMMAND>")
            .subcommand_required(true)
            .arg_required_else_help(true)
            .about(LONG_ABOUT)
            .version(option_env!("CARGO_PKG_VERSION").unwrap_or("master"))
            .arg(arg!(-v --verbose ... "Level of verbosity"))
            .subcommand(
                command!("generate-config")
                    .about("Generate default config file")
                    .arg(arg!(-d <DATA_PATH> "Path for storing configs and data").default_value("data").num_args(1))
                    .arg(arg!(-g <API_GRPC_ENDPOINT> "API GRPC endpoint").default_value("127.0.0.1:8082").num_args(1))
                    .arg(arg!(-g <API_HTTP_ENDPOINT> "API HTTP endpoint").default_value("127.0.0.1:8081").num_args(1))
                    .arg(
                        arg!(-m <METRICS_ENDPOINT> "Metrics listen endpoint")
                            .default_value("127.0.0.1:8084")
                            .num_args(1),
                    ),
            )
            .subcommand(
                command!("serve")
                    .about("Launch search server")
                    .arg(arg!(<CONFIG> "Search engine config file").num_args(1)),
            )
            .get_matches();

        match matches.subcommand() {
            Some(("generate-config", submatches)) => {
                let data_path = PathBuf::from(submatches.try_get_one::<String>("DATA_PATH")?.expect("no value"));
                let api_grpc_endpoint = submatches.try_get_one::<String>("API_GRPC_ENDPOINT")?.expect("no value");
                let api_http_endpoint = submatches.try_get_one::<String>("API_HTTP_ENDPOINT")?;
                let metrics_endpoint = submatches.try_get_one::<String>("METRICS_ENDPOINT")?.expect("no value");
                let server_config = crate::configs::server::ConfigBuilder::default()
                    .data_path(data_path.join("bin"))
                    .logs_path(data_path.join("logs"))
                    .api(
                        crate::configs::api::ConfigBuilder::default()
                            .grpc_endpoint(api_grpc_endpoint.to_string())
                            .http_endpoint(api_http_endpoint.cloned())
                            .build()
                            .map_err(summa_core::Error::from)?,
                    )
                    .metrics(
                        crate::configs::metrics::ConfigBuilder::default()
                            .endpoint(metrics_endpoint.to_string())
                            .build()
                            .map_err(summa_core::Error::from)?,
                    )
                    .p2p(Some(
                        crate::configs::p2p::ConfigBuilder::default()
                            .key_store_path(data_path.join("ks"))
                            .build()
                            .map_err(summa_core::Error::from)?,
                    ))
                    .store(
                        crate::configs::store::ConfigBuilder::default()
                            .path(data_path.join("store"))
                            .build()
                            .map_err(summa_core::Error::from)?,
                    )
                    .build()
                    .map_err(summa_core::Error::from)?;
                println!("{}", serde_yaml::to_string(&server_config).expect("cannot serialize config"));
                Ok(())
            }
            Some(("serve", submatches)) => {
                let config_path = PathBuf::from(submatches.try_get_one::<String>("CONFIG")?.expect("no value"));
                let server_config_holder = ConfigHolder::from_path(config_path)?;
                let _guards = {
                    let server_config = server_config_holder.read().await;
                    let log_guards = if server_config.get().debug {
                        logging::default()
                    } else {
                        logging::file(&server_config.get().log_path)?
                    };
                    tokio::fs::create_dir_all(&server_config.get().data_path)
                        .await
                        .map_err(|e| Error::IO((e, Some(server_config.get().data_path.clone()))))?;
                    log_guards
                };
                let app = Server::from_server_config(server_config_holder);
                app.run().await
            }
            _ => unreachable!("Exhausted list of subcommands and subcommand_required prevents `None`"),
        }
    }

    pub async fn serve(&self, terminator: Receiver<ControlMessage>) -> SummaServerResult<impl Future<Output = SummaServerResult<()>>> {
        let mut futures: Vec<Box<dyn Future<Output = SummaServerResult<()>> + Send>> = vec![];
        let server_config = self.server_config_holder.read().await.get().clone();
        let mut iroh_config = iroh_rpc_client::Config::default_network();

        let iroh_rpc_client = iroh_rpc_client::Client::new(iroh_config.clone()).await?;
        let content_loader = iroh_unixfs::content_loader::FullLoader::new(
            iroh_rpc_client,
            iroh_unixfs::content_loader::FullLoaderConfig {
                http_gateways: server_config
                    .p2p
                    .as_ref()
                    .map(|p2p| p2p.http_gateways.iter().map(|x| GatewayUrl::from_str(x)).collect::<Result<_, _>>())
                    .transpose()?
                    .unwrap_or_default(),
                indexer: None,
            },
        )?;

        let p2p_service = if let Some(p2p_config) = server_config.p2p.clone() {
            let p2p_service = P2p::new(p2p_config).await?;
            iroh_config.p2p_addr = Some(p2p_service.rpc_addr().clone());
            let p2p_future = p2p_service.prepare_serving_future(terminator.clone()).await?;
            futures.push(Box::new(p2p_future));
            Some(p2p_service)
        } else {
            None
        };

        let store_service = Store::new(server_config.store.clone(), content_loader).await?;
        iroh_config.store_addr = Some(store_service.rpc_addr().clone());
        futures.push(Box::new(store_service.prepare_serving_future(terminator.clone()).await?));

        if let Some(gateway_config) = server_config.gateway.clone() {
            futures.push(Box::new(
                Gateway::new(gateway_config, &store_service, p2p_service.as_ref())
                    .await?
                    .prepare_serving_future(terminator.clone())
                    .await?,
            ));
        }

        let index_service = Index::new(&self.server_config_holder, store_service).await?;
        futures.push(Box::new(index_service.prepare_serving_future(terminator.clone()).await?));

        let metrics_service = Metrics::new(&server_config.metrics)?;
        futures.push(Box::new(metrics_service.prepare_serving_future(&index_service, terminator.clone()).await?));

        let api_service = Api::new(&self.server_config_holder, &index_service)?;
        futures.push(Box::new(api_service.prepare_serving_future(terminator.clone()).await?));

        Ok(async move {
            try_join_all(futures.into_iter().map(Box::into_pin)).await?;
            info!(action = "all_systems_down");
            Ok(())
        }
        .instrument(info_span!("lifecycle")))
    }

    async fn run(&self) -> SummaServerResult<()> {
        let server = self.serve(signal_channel()?).await?;
        server.await
    }
}

#[cfg(test)]
mod tests {
    use std::default::Default;
    use std::path::Path;

    use async_broadcast::broadcast;
    use summa_core::utils::thread_handler::{ControlMessage, ThreadHandler};
    use summa_proto::proto;
    use summa_proto::proto::index_api_client::IndexApiClient;
    use tonic::transport::Channel;

    use super::*;
    use crate::configs::server::tests::create_test_server_config;
    use crate::services::index::tests::create_test_schema;

    async fn create_index_api_client(endpoint: &str) -> IndexApiClient<Channel> {
        IndexApiClient::connect(endpoint.to_owned()).await.unwrap()
    }

    async fn create_client_server(root_path: &Path) -> SummaServerResult<(ThreadHandler<SummaServerResult<()>>, IndexApiClient<Channel>)> {
        let server_config_holder = ConfigHolder::from_path_or(root_path.join("summa.yaml"), || create_test_server_config(&root_path.join("data")))?;
        let server_config = server_config_holder.read().await.get().clone();
        tokio::fs::create_dir_all(&server_config.data_path)
            .await
            .map_err(|e| Error::IO((e, Some(server_config.data_path.clone()))))?;
        let api_grpc_endpoint = server_config.api.grpc_endpoint.clone();
        let (server_terminator, receiver) = broadcast::<ControlMessage>(1);
        let thread_handler = ThreadHandler::new(
            tokio::spawn(Server::from_server_config(server_config_holder).serve(&receiver).await?),
            server_terminator,
        );
        let client = create_index_api_client(&format!("http://{}", &api_grpc_endpoint)).await;
        Ok((thread_handler, client))
    }

    async fn create_index(
        index_api_client: &mut IndexApiClient<Channel>,
        index_name: &str,
        schema: &str,
    ) -> Result<tonic::Response<proto::CreateIndexResponse>, tonic::Status> {
        index_api_client
            .create_index(tonic::Request::new(proto::CreateIndexRequest {
                index_name: index_name.to_owned(),
                index_engine: proto::CreateIndexEngineRequest::File.into(),
                schema: schema.to_owned(),
                ..Default::default()
            }))
            .await
    }

    async fn create_default_index(index_api_client: &mut IndexApiClient<Channel>) -> Result<tonic::Response<proto::CreateIndexResponse>, tonic::Status> {
        let schema = create_test_schema();
        let schema_str = serde_yaml::to_string(&schema).unwrap();
        create_index(index_api_client, "test_index", &schema_str).await
    }

    #[tokio::test]
    async fn test_application() {
        logging::tests::initialize_default_once();
        let root_path = tempdir::TempDir::new("summa_test").unwrap();
        let (thread_handler, mut index_api_client) = create_client_server(root_path.path()).await.unwrap();

        let schema = create_test_schema();
        let schema_str = serde_yaml::to_string(&schema).unwrap();

        let response = create_index(&mut index_api_client, "test_index", &schema_str).await.unwrap();
        assert_eq!(
            response.into_inner(),
            proto::CreateIndexResponse {
                index: Some(proto::IndexDescription {
                    index_name: "test_index".to_owned(),
                    index_engine: Some(proto::IndexEngineConfig {
                        config: Some(proto::index_engine_config::Config::File(proto::FileEngineConfig {
                            path: root_path.into_path().join("data").join("test_index").to_string_lossy().to_string()
                        }))
                    }),
                    ..Default::default()
                }),
            }
        );
        thread_handler.stop().await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_persistence() {
        logging::tests::initialize_default_once();
        let root_path = tempdir::TempDir::new("summa_test").unwrap();

        let (thread_handler_1, mut index_api_client_1) = create_client_server(root_path.path()).await.unwrap();
        assert!(create_default_index(&mut index_api_client_1).await.is_ok());
        thread_handler_1.stop().await.unwrap().unwrap();

        let (thread_handler_2, mut index_api_client_2) = create_client_server(root_path.path()).await.unwrap();
        assert_eq!(
            index_api_client_2
                .get_indices(tonic::Request::new(proto::GetIndicesRequest {}))
                .await
                .unwrap()
                .into_inner(),
            proto::GetIndicesResponse {
                indices: vec![proto::IndexDescription {
                    index_name: "test_index".to_owned(),
                    index_engine: Some(proto::IndexEngineConfig {
                        config: Some(proto::index_engine_config::Config::File(proto::FileEngineConfig {
                            path: root_path.into_path().join("data").join("test_index").to_string_lossy().to_string()
                        }))
                    }),
                    ..Default::default()
                }]
            }
        );
        thread_handler_2.stop().await.unwrap().unwrap();
    }
}
