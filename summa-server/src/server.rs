use std::future::Future;
use std::path::PathBuf;
use std::sync::Arc;

use async_broadcast::Receiver;
use clap::{arg, command};
use futures_util::future::try_join_all;
use summa_core::configs::ConfigProxy;
use tracing::{info, info_span, Instrument};

use crate::configs::server::ConfigHolder;
use crate::errors::{Error, SummaServerResult};
use crate::logging;
#[cfg(feature = "metrics")]
use crate::services::Metrics;
use crate::services::{Api, Index};
use crate::utils::thread_handler::ControlMessage;
use crate::utils::{increase_fd_limit, signal_channel};

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
                    .arg(arg!(-a <API_GRPC_ENDPOINT> "API GRPC endpoint").default_value("127.0.0.1:8082").num_args(1)),
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
                let server_config = crate::configs::server::ConfigBuilder::default()
                    .data_path(data_path.join("bin"))
                    .logs_path(data_path.join("logs"))
                    .api(
                        crate::configs::api::ConfigBuilder::default()
                            .grpc_endpoint(api_grpc_endpoint.to_string())
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
        #[cfg(unix)]
        match increase_fd_limit() {
            Ok(soft) => tracing::debug!("NOFILE limit: soft = {}", soft),
            Err(err) => tracing::error!("Error increasing NOFILE limit: {}", err),
        }

        let mut futures: Vec<Box<dyn Future<Output = SummaServerResult<()>> + Send>> = vec![];

        let index_service = Index::new(&self.server_config_holder)?;
        futures.push(Box::new(index_service.prepare_serving_future(terminator.clone()).await?));

        #[cfg(feature = "metrics")]
        if let Some(metrics_config) = &self.server_config_holder.read().await.get().metrics.clone() {
            let metrics_service = Metrics::new(metrics_config)?;
            futures.push(Box::new(metrics_service.prepare_serving_future(&index_service, terminator.clone()).await?));
        }

        let api_service = Api::new(&self.server_config_holder, &index_service)?;
        futures.push(Box::new(api_service.prepare_serving_future(terminator.clone()).await?));

        Ok(async move {
            try_join_all(futures.into_iter().map(Box::into_pin)).await?;
            info!(action = "all_systems_down");
            Ok(())
        }
        .instrument(info_span!("lifecycle")))
    }

    pub async fn run(&self) -> SummaServerResult<()> {
        let server = self.serve(signal_channel()?).await?;
        server.await
    }
}

#[cfg(test)]
mod tests {
    use std::default::Default;
    use std::path::Path;

    use async_broadcast::broadcast;
    use serde_json::json;
    use summa_core::components::test_utils::create_test_schema;
    use summa_proto::proto;
    use summa_proto::proto::index_api_client::IndexApiClient;
    use summa_proto::proto::score::Score::F64Score;
    use summa_proto::proto::search_api_client::SearchApiClient;
    use tonic::transport::Channel;

    use super::*;
    use crate::configs::server::tests::create_test_server_config;
    use crate::utils::thread_handler::{ControlMessage, ThreadHandler};

    async fn create_index_api_client(endpoint: &str) -> IndexApiClient<Channel> {
        IndexApiClient::connect(endpoint.to_owned()).await.unwrap()
    }

    async fn create_search_api_client(endpoint: &str) -> SearchApiClient<Channel> {
        SearchApiClient::connect(endpoint.to_owned()).await.unwrap()
    }

    async fn create_client_server(
        root_path: &Path,
    ) -> SummaServerResult<(ThreadHandler<SummaServerResult<()>>, IndexApiClient<Channel>, SearchApiClient<Channel>)> {
        let server_config_holder = ConfigHolder::from_path_or(root_path.join("summa.yaml"), || create_test_server_config(&root_path.join("data")))?;
        let server_config = server_config_holder.read().await.get().clone();
        tokio::fs::create_dir_all(&server_config.data_path)
            .await
            .map_err(|e| Error::IO((e, Some(server_config.data_path.clone()))))?;
        let api_grpc_endpoint = server_config.api.grpc_endpoint.clone();
        let (server_terminator, receiver) = broadcast::<ControlMessage>(1);
        let thread_handler = ThreadHandler::new(
            tokio::spawn(Server::from_server_config(server_config_holder).serve(receiver).await?),
            server_terminator,
        );
        let index_client = create_index_api_client(&format!("http://{api_grpc_endpoint}")).await;
        let search_client = create_search_api_client(&format!("http://{api_grpc_endpoint}")).await;
        Ok((thread_handler, index_client, search_client))
    }

    async fn create_index(
        index_api_client: &mut IndexApiClient<Channel>,
        index_name: &str,
        schema: &str,
    ) -> Result<tonic::Response<proto::CreateIndexResponse>, tonic::Status> {
        let r = index_api_client
            .create_index(tonic::Request::new(proto::CreateIndexRequest {
                index_name: index_name.to_owned(),
                index_engine: Some(proto::create_index_request::IndexEngine::File(proto::CreateFileEngineRequest {})),
                index_attributes: Some(proto::IndexAttributes { ..Default::default() }),
                query_parser_config: Some(proto::QueryParserConfig {
                    default_fields: vec!["title".to_string(), "body".to_string()],
                    ..Default::default()
                }),
                schema: schema.to_owned(),
                ..Default::default()
            }))
            .await?;
        index_api_client
            .index_document(proto::IndexDocumentRequest {
                index_name: "test_index".to_string(),
                document: json!({"title": "title1", "body": "body1"}).to_string().as_bytes().to_vec(),
            })
            .await?;
        index_api_client
            .index_document(proto::IndexDocumentRequest {
                index_name: "test_index".to_string(),
                document: json!({"title": "title2", "body": "body2"}).to_string().as_bytes().to_vec(),
            })
            .await?;
        index_api_client
            .commit_index(proto::CommitIndexRequest {
                index_name: "test_index".to_string(),
            })
            .await?;
        index_api_client
            .index_document(proto::IndexDocumentRequest {
                index_name: "test_index".to_string(),
                document: json!({"title": "title3", "body": "body3"}).to_string().as_bytes().to_vec(),
            })
            .await?;
        index_api_client
            .commit_index(proto::CommitIndexRequest {
                index_name: "test_index".to_string(),
            })
            .await?;
        Ok(r)
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
        let (thread_handler, mut index_api_client, _) = create_client_server(root_path.path()).await.unwrap();

        let schema = create_test_schema();
        let schema_str = serde_yaml::to_string(&schema).unwrap();

        let response = create_index(&mut index_api_client, "test_index", &schema_str).await.unwrap();
        assert_eq!(response.into_inner().index.unwrap().index_name, "test_index");
        thread_handler.stop().await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_persistence() {
        logging::tests::initialize_default_once();
        let root_path = tempdir::TempDir::new("summa_test").unwrap();

        let (thread_handler_1, mut index_api_client_1, mut search_api_client_1) = create_client_server(root_path.path()).await.unwrap();
        assert!(create_default_index(&mut index_api_client_1).await.is_ok());
        let search_response = search_api_client_1
            .search(tonic::Request::new(proto::SearchRequest {
                index_queries: vec![proto::IndexQuery {
                    index_alias: "test_index".to_string(),
                    query: Some(proto::Query {
                        query: Some(proto::query::Query::Match(proto::MatchQuery {
                            value: "title3".to_string(),
                            query_parser_config: Some(proto::QueryParserConfig {
                                default_fields: vec!["title".to_string(), "body".to_string()],
                                ..Default::default()
                            }),
                            ..Default::default()
                        })),
                    }),
                    collectors: vec![proto::Collector {
                        collector: Some(proto::collector::Collector::TopDocs(proto::TopDocsCollector {
                            limit: 1,
                            offset: 0,
                            scorer: None,
                            snippet_configs: Default::default(),
                            explain: false,
                            fields: vec![],
                        })),
                    }],
                    is_fieldnorms_scoring_enabled: None,
                }],
                tags: Default::default(),
            }))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(
            search_response.collector_outputs[0],
            proto::CollectorOutput {
                collector_output: Some(proto::collector_output::CollectorOutput::Documents(proto::DocumentsCollectorOutput {
                    scored_documents: vec![proto::ScoredDocument {
                        document: "{\"body\":\"body3\",\"title\":\"title3\"}".to_string(),
                        score: Some(proto::Score {
                            score: Some(F64Score(0.9808291792869568))
                        }),
                        index_alias: "test_index".to_string(),
                        ..Default::default()
                    }],
                    has_next: false,
                })),
            }
        );
        thread_handler_1.stop().await.unwrap().unwrap();
        let (thread_handler_2, _, mut search_api_client_2) = create_client_server(root_path.path()).await.unwrap();
        let search_response = search_api_client_2
            .search(tonic::Request::new(proto::SearchRequest {
                index_queries: vec![proto::IndexQuery {
                    index_alias: "test_index".to_string(),
                    query: Some(proto::Query {
                        query: Some(proto::query::Query::Match(proto::MatchQuery {
                            value: "title3".to_string(),
                            query_parser_config: Some(proto::QueryParserConfig {
                                default_fields: vec!["title".to_string(), "body".to_string()],
                                ..Default::default()
                            }),
                            ..Default::default()
                        })),
                    }),
                    collectors: vec![proto::Collector {
                        collector: Some(proto::collector::Collector::TopDocs(proto::TopDocsCollector {
                            limit: 1,
                            offset: 0,
                            ..Default::default()
                        })),
                    }],
                    is_fieldnorms_scoring_enabled: None,
                }],
                tags: Default::default(),
            }))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(
            search_response.collector_outputs[0],
            proto::CollectorOutput {
                collector_output: Some(proto::collector_output::CollectorOutput::Documents(proto::DocumentsCollectorOutput {
                    scored_documents: vec![proto::ScoredDocument {
                        document: "{\"body\":\"body3\",\"title\":\"title3\"}".to_string(),
                        score: Some(proto::Score {
                            score: Some(F64Score(0.9808291792869568))
                        }),
                        index_alias: "test_index".to_string(),
                        ..Default::default()
                    }],
                    has_next: false,
                })),
            }
        );
        thread_handler_2.stop().await.unwrap().unwrap();
    }
}
