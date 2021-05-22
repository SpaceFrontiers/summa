use crate::configs::{ApplicationConfig, ApplicationConfigHolder};
use crate::errors::SummaResult;
use crate::logging;
use crate::metrics::register_meter;
use crate::servers::GrpcServer;
use crate::servers::MetricsServer;
use crate::services::IndexService;
use crate::utils::signal_channel::signal_channel;
use crate::utils::thread_handler::ControlMessage;
use async_broadcast::Receiver;
use clap::{arg, command};
use futures::try_join;
use std::future::Future;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::runtime;

pub struct Application {
    config: ApplicationConfigHolder,
}

const LONG_ABOUT: &str = "
Fast full-text search server with following features:

- Fine CLI and asynchronous client library written in Python
- Simple GRPC API for managing multiple indices and for search
- Extendable query parsing on Python client side
- Ranking documents with custom and/or user-defined scoring functions
- Indexing documents through Kafka or directly
- Tracing with OpenTelemetry and exposing metrics in Prometheus format
- Reflection API for deep insights about text data
- Configurable tokenizers (including CJK)
";

impl Application {
    pub fn from_config(config: ApplicationConfig) -> Application {
        Application {
            config: ApplicationConfigHolder::from_config(config),
        }
    }

    pub fn from_application_config_holder(config: ApplicationConfigHolder) -> Application {
        Application { config }
    }

    pub fn proceed_args() -> SummaResult<()> {
        let matches = command!()
            .name("summa")
            .override_usage("summa-server [OPTIONS] <SUBCOMMAND>")
            .subcommand_required(true)
            .arg_required_else_help(true)
            .about(LONG_ABOUT)
            .version("0.5.0")
            .author("Pasha Podolsky")
            .arg(arg!(-v --verbose ... "Level of verbosity"))
            .subcommand(command!("generate-config").about("Generate default config file"))
            .subcommand(
                command!("serve")
                    .about("Launch search server")
                    .arg(arg!(<CONFIG> "Search engine config file").required(true).takes_value(true)),
            )
            .get_matches();

        match matches.subcommand() {
            Some(("generate-config", _)) => {
                let default_config = ApplicationConfig::new("data");
                println!("{}", serde_yaml::to_string(&default_config).unwrap());
                Ok(())
            }
            Some(("serve", submatches)) => {
                let config_path = submatches.value_of("CONFIG").map(Path::new).unwrap();
                let application_config_holder = ApplicationConfigHolder::from_path(config_path)?;
                let _guards = {
                    let application_config = application_config_holder.read();
                    if application_config.debug {
                        logging::default();
                        vec![]
                    } else {
                        logging::file(&application_config.log_path)
                    }
                };
                let app = Application::from_application_config_holder(application_config_holder);
                app.run()
            }
            _ => unreachable!("Exhausted list of subcommands and subcommand_required prevents `None`"),
        }
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

    pub fn serve(&self, terminator: &Receiver<ControlMessage>) -> SummaResult<impl Future<Output = SummaResult<()>>> {
        let metrics_server_future = MetricsServer::new(&self.config)?.start(terminator.clone())?;

        let index_service = IndexService::new(&self.config);
        let grpc_server_future = GrpcServer::new(&self.config, &index_service)?.start(terminator.clone())?;

        Ok(async move {
            index_service.setup_index_holders().await?;
            register_meter(&index_service);
            try_join!(metrics_server_future, grpc_server_future)?;
            Ok(())
        })
    }

    fn run(&self) -> SummaResult<()> {
        let runtime = self.create_runtime()?;
        runtime.block_on(async move {
            let receiver = signal_channel();
            let server = self.serve(&receiver)?;
            server.await
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::configs::application_config::tests::create_test_application_config;
    use crate::proto;
    use crate::proto::index_api_client::IndexApiClient;
    use crate::search_engine::index_holder::tests::create_test_schema;
    use crate::utils::thread_handler::{ControlMessage, ThreadHandler};
    use async_broadcast::broadcast;
    use std::default::Default;
    use tonic::transport::Channel;

    async fn create_index_api_client(endpoint: &str) -> IndexApiClient<Channel> {
        IndexApiClient::connect(endpoint.to_owned()).await.unwrap()
    }

    async fn create_client_server(root_path: &Path) -> SummaResult<(ThreadHandler, IndexApiClient<Channel>)> {
        let config_holder = ApplicationConfigHolder::from_path_or(root_path.join("summa.yaml"), || create_test_application_config(&root_path.join("data")))?;
        let grpc_endpoint = config_holder.read().grpc.endpoint.clone();
        let (server_terminator, receiver) = broadcast::<ControlMessage>(1);
        let thread_handler = ThreadHandler::new(
            tokio::spawn(Application::from_application_config_holder(config_holder).serve(&receiver)?),
            server_terminator,
        );
        let client = create_index_api_client(&format!("http://{}", &grpc_endpoint)).await;
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
                index_engine: proto::IndexEngine::File.into(),
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
    async fn test_application() -> SummaResult<()> {
        logging::tests::initialize_default_once();
        let root_path = tempdir::TempDir::new("summa_test").unwrap();
        let (thread_handler, mut index_api_client) = create_client_server(root_path.path()).await?;

        let schema = create_test_schema();
        let schema_str = serde_yaml::to_string(&schema).unwrap();

        let response = create_index(&mut index_api_client, "test_index", &schema_str).await.unwrap();
        assert_eq!(
            response.into_inner(),
            proto::CreateIndexResponse {
                index: Some(proto::Index {
                    index_name: "test_index".to_owned(),
                    index_engine: "File".to_owned(),
                    ..Default::default()
                }),
            }
        );
        thread_handler.stop().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_persistence() -> SummaResult<()> {
        logging::tests::initialize_default_once();
        let root_path = tempdir::TempDir::new("summa_test").unwrap();

        let (thread_handler_1, mut index_api_client_1) = create_client_server(root_path.path()).await?;
        assert!(create_default_index(&mut index_api_client_1).await.is_ok());
        thread_handler_1.stop().await?;

        let (thread_handler_2, mut index_api_client_2) = create_client_server(root_path.path()).await?;
        assert_eq!(
            index_api_client_2
                .get_indices(tonic::Request::new(proto::GetIndicesRequest {}))
                .await
                .unwrap()
                .into_inner(),
            proto::GetIndicesResponse {
                indices: vec![proto::Index {
                    index_name: "test_index".to_owned(),
                    index_engine: "File".to_owned(),
                    ..Default::default()
                }]
            }
        );
        thread_handler_2.stop().await?;

        Ok(())
    }
}
