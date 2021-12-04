use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::sync::Arc;

use futures_util::future::join_all;
use summa_core::components::{IndexHolder, IndexRegistry};
use summa_core::configs::{
    ApplicationConfigHolder, ConfigProxy, ConsumerConfig, IndexConfig, IndexConfigBuilder, IndexConfigFilePartProxy, IndexEngine, Persistable,
};
use summa_core::directories::DefaultExternalRequestGenerator;
use summa_core::errors::SummaResult;
use summa_core::utils::sync::{Handler, OwningHandler};
use summa_proto::proto;
use tantivy::IndexSettings;
use tokio::sync::{RwLockReadGuard, RwLockWriteGuard};
use tracing::{info, instrument};

use crate::errors::SummaServerResult;
use crate::errors::{Error, ValidationError};
use crate::hyper_external_request::HyperExternalRequest;
use crate::requests::{validators, AlterIndexRequest, CreateConsumerRequest, CreateIndexRequest, DeleteConsumerRequest, DeleteIndexRequest};

/// The main struct responsible for indices lifecycle. Here lives indices creation and deletion as well as committing and indexing new documents.
#[derive(Clone, Default, Debug)]
pub struct IndexService {
    application_config: ApplicationConfigHolder,
    index_registry: IndexRegistry,
}

#[derive(Default)]
pub struct DeleteIndexResult {}

impl From<DeleteIndexResult> for proto::DeleteIndexResponse {
    fn from(_delete_index_request: DeleteIndexResult) -> Self {
        proto::DeleteIndexResponse {}
    }
}

/// The main entry point for managing Summa indices
impl IndexService {
    /// Read-locked `HashMap` of all indices
    pub async fn index_holders(&self) -> RwLockReadGuard<'_, HashMap<String, OwningHandler<IndexHolder>>> {
        self.index_registry.index_holders().await
    }

    /// Write-locked `HashMap` of all indices
    ///
    /// Taking this lock means locking metadata modification
    pub async fn index_holders_mut(&self) -> RwLockWriteGuard<'_, HashMap<String, OwningHandler<IndexHolder>>> {
        self.index_registry.index_holders_mut().await
    }

    /// Returns `Handler` to `IndexHolder`.
    ///
    /// It is safe to keep `Handler<IndexHolder>` cause `Index` won't be deleted until `Handler` is alive.
    /// Though, `IndexHolder` can be removed from the registry of `IndexHolder`s to prevent new queries
    pub async fn get_index_holder(&self, index_alias: &str) -> SummaServerResult<Handler<IndexHolder>> {
        Ok(self
            .index_registry
            .get_index_holder_by_name(
                self.application_config
                    .read()
                    .await
                    .resolve_index_alias(index_alias)
                    .as_deref()
                    .unwrap_or(index_alias),
            )
            .await?)
    }

    /// Creates new `IndexService` with `ApplicationConfigHolder`
    pub fn new(application_config_holder: &ApplicationConfigHolder) -> IndexService {
        IndexService {
            application_config: application_config_holder.clone(),
            index_registry: IndexRegistry::default(),
        }
    }

    /// Create `IndexHolder`s from config
    pub(crate) async fn setup_index_holders(&self) -> SummaServerResult<()> {
        let mut index_holders = HashMap::new();
        for index_name in self.application_config.read().await.indices.keys() {
            let index_config_proxy = Arc::new(IndexConfigFilePartProxy::new(&self.application_config, index_name));
            let index = IndexHolder::open::<HyperExternalRequest, DefaultExternalRequestGenerator<HyperExternalRequest>>(
                &index_config_proxy.read().await.get().index_engine,
            )
            .await?;
            index_holders.insert(
                index_name.clone(),
                OwningHandler::new(IndexHolder::setup(index_name, index, index_config_proxy).await?),
            );
        }
        info!(action = "index_holders", index_holders = ?index_holders);
        *self.index_registry.index_holders_mut().await = index_holders;
        Ok(())
    }

    pub(crate) async fn insert_config(&self, index_name: &str, index_config: &IndexConfig) -> SummaServerResult<()> {
        let mut application_config = self.application_config.write().await;
        match application_config.indices.entry(index_name.to_owned()) {
            Entry::Occupied(o) => Err(ValidationError::ExistingIndex(o.key().to_owned())),
            Entry::Vacant(v) => {
                v.insert(index_config.clone());
                Ok(())
            }
        }?;
        application_config.save()?;
        Ok(())
    }

    /// Create consumer and insert it into the consumer registry. Add it to the `IndexHolder` afterwards.
    #[instrument(skip_all, fields(index_name = index_name))]
    pub async fn attach_index(&self, index_name: &str) -> SummaServerResult<Handler<IndexHolder>> {
        let index_path = self.application_config.read().await.get_path_for_index_data(index_name);
        if !index_path.exists() {
            return Err(ValidationError::MissingIndex(index_name.to_string()).into());
        }
        let index_engine = IndexEngine::File(index_path);

        let index_config = IndexConfigBuilder::default()
            .index_engine(index_engine.clone())
            .build()
            .map_err(summa_core::Error::from)?;
        self.insert_config(index_name, &index_config).await?;

        let index_config_proxy = Arc::new(IndexConfigFilePartProxy::new(&self.application_config, index_name));
        let index = IndexHolder::open::<HyperExternalRequest, DefaultExternalRequestGenerator<HyperExternalRequest>>(
            &index_config_proxy.read().await.get().index_engine,
        ).await?;
        Ok(self.index_registry.add(IndexHolder::setup(index_name, index, index_config_proxy).await?).await)
    }

    /// Create consumer and insert it into the consumer registry. Add it to the `IndexHolder` afterwards.
    #[instrument(skip_all, fields(index_name = ?create_index_request.index_name))]
    pub async fn create_index(&self, create_index_request: CreateIndexRequest) -> SummaServerResult<Handler<IndexHolder>> {
        let index_engine = match create_index_request.index_engine {
            proto::IndexEngine::Memory => IndexEngine::Memory(create_index_request.schema.clone()),
            proto::IndexEngine::File => IndexEngine::File(self.application_config.read().await.get_path_for_index_data(&create_index_request.index_name)),
        };
        let mut index_config_builder = IndexConfigBuilder::default();
        index_config_builder
            .index_engine(index_engine.clone())
            .primary_key(create_index_request.primary_key.to_owned())
            .default_fields(create_index_request.default_fields.clone())
            .multi_fields(HashSet::from_iter(create_index_request.multi_fields.clone().into_iter()))
            .autocommit_interval_ms(create_index_request.autocommit_interval_ms);
        if let Some(writer_threads) = create_index_request.writer_threads {
            index_config_builder.writer_threads(writer_threads);
        }
        if let Some(writer_heap_size_bytes) = create_index_request.writer_heap_size_bytes {
            index_config_builder.writer_heap_size_bytes(writer_heap_size_bytes);
        }

        let index_config = index_config_builder.build().map_err(summa_core::Error::from)?;
        self.insert_config(&create_index_request.index_name, &index_config).await?;

        let index_settings = IndexSettings {
            docstore_compression: create_index_request.compression,
            docstore_blocksize: create_index_request.blocksize.unwrap_or(16384),
            sort_by_field: create_index_request.sort_by_field.clone(),
            ..Default::default()
        };
        let index_config_proxy = Arc::new(IndexConfigFilePartProxy::new(&self.application_config, &create_index_request.index_name));
        let index = IndexHolder::create::<HyperExternalRequest, DefaultExternalRequestGenerator<HyperExternalRequest>>(
            &create_index_request.schema,
            index_settings,
            &index_config_proxy.read().await.get().index_engine,
        )
        .await?;
        Ok(self
            .index_registry
            .add(IndexHolder::setup(&create_index_request.index_name, index, index_config_proxy).await?)
            .await)
    }

    async fn alter_index_settings(&self, alter_index_request: AlterIndexRequest, index_holder: &IndexHolder) -> SummaServerResult<()> {
        let index_updater = index_holder.index_updater();
        let mut index_updater = index_updater.write().await;
        if let Some(compression) = alter_index_request.compression {
            index_updater.index_mut().settings_mut().docstore_compression = compression
        }
        if let Some(blocksize) = alter_index_request.blocksize {
            index_updater.index_mut().settings_mut().docstore_blocksize = blocksize
        }
        if let Some(sort_by_field) = alter_index_request.sort_by_field {
            index_updater.index_mut().settings_mut().sort_by_field = match sort_by_field.field.as_str() {
                "" => None,
                _ => Some(sort_by_field),
            }
        }
        if let Some(default_fields) = alter_index_request.default_fields {
            let parsed_default_fields = validators::parse_default_fields(index_holder.schema(), &default_fields)?;
            let mut index_config_proxy = index_holder.index_config_proxy().write().await;
            index_config_proxy.get_mut().default_fields = parsed_default_fields;
            index_config_proxy.commit()?;
        }
        if let Some(multi_fields) = alter_index_request.multi_fields {
            let parsed_multi_fields = validators::parse_multi_fields(index_holder.schema(), &multi_fields)?;
            let mut index_config_proxy = index_holder.index_config_proxy().write().await;
            index_config_proxy.get_mut().multi_fields = HashSet::from_iter(parsed_multi_fields.into_iter());
            index_config_proxy.commit()?;
        }
        Ok(index_updater.commit(None).await?)
    }

    /// Alters index
    #[instrument(skip_all, fields(index_name = ?alter_index_request.index_name))]
    pub async fn alter_index(&self, alter_index_request: AlterIndexRequest) -> SummaServerResult<Handler<IndexHolder>> {
        let index_holder = self.index_registry.get_index_holder_by_name(&alter_index_request.index_name).await?;
        self.alter_index_settings(alter_index_request, &index_holder).await?;
        index_holder.reload_query_parser().await?;
        Ok(index_holder)
    }

    /// Delete index, optionally with all its aliases and consumers
    #[instrument(skip_all, fields(index_name = ?delete_index_request.index_name))]
    pub async fn delete_index(&self, delete_index_request: DeleteIndexRequest) -> SummaServerResult<DeleteIndexResult> {
        let index_holder = {
            let application_config = self.application_config.write().await;
            match self.index_registry.index_holders_mut().await.entry(delete_index_request.index_name.to_owned()) {
                Entry::Occupied(entry) => {
                    let index_holder = entry.get();
                    let aliases = application_config.get_index_aliases_for_index(&delete_index_request.index_name);
                    let consumer_names = index_holder.index_updater().read().await.consumer_names();

                    if !aliases.is_empty() {
                        return Err(ValidationError::Aliased(aliases.join(", ")).into());
                    }
                    if !consumer_names.is_empty() {
                        return Err(Error::Core(
                            summa_core::errors::ValidationError::ExistingConsumer(consumer_names.join(", ")).into(),
                        ));
                    }
                    entry.remove()
                }
                Entry::Vacant(_) => return Err(ValidationError::MissingIndex(delete_index_request.index_name.to_owned()).into()),
            }
        };
        index_holder.into_inner().await.delete().await?;
        Ok(DeleteIndexResult::default())
    }

    /// Returns all existent consumers for all indices
    pub async fn get_consumers(&self) -> SummaServerResult<Vec<(String, String)>> {
        let application_config = self.application_config.read().await;
        let mut result = Vec::new();
        for (index_name, index_config) in &application_config.indices {
            result.extend(
                index_config
                    .consumer_configs
                    .keys()
                    .map(|consumer_name| (index_name.to_owned(), consumer_name.to_owned())),
            )
        }
        Ok(result)
    }

    /// Create consumer and insert it into the consumer registry. Add it to the `IndexHolder` afterwards.
    #[instrument(skip_all, fields(consumer_name = ?create_consumer_request.consumer_name))]
    pub async fn create_consumer(&self, create_consumer_request: &CreateConsumerRequest) -> SummaServerResult<String> {
        let index_holder = self.get_index_holder(&create_consumer_request.index_alias).await?;
        index_holder
            .index_updater()
            .write()
            .await
            .create_consumer(&create_consumer_request.consumer_name, &create_consumer_request.consumer_config)
            .await?;
        Ok(index_holder.index_name().to_owned())
    }

    /// Delete consumer from the consumer registry and from `IndexHolder` afterwards.
    #[instrument(skip_all, fields(consumer_name = ?delete_consumer_request.consumer_name))]
    pub async fn delete_consumer(&self, delete_consumer_request: DeleteConsumerRequest) -> SummaServerResult<()> {
        Ok(self
            .get_index_holder(&delete_consumer_request.index_alias)
            .await?
            .index_updater()
            .write()
            .await
            .delete_consumer(&delete_consumer_request.consumer_name)
            .await?)
    }

    /// Stopping index holders
    pub async fn stop(self) -> SummaServerResult<()> {
        join_all(
            self.index_registry
                .index_holders_mut()
                .await
                .drain()
                .map(|(_, index_holder)| async move { index_holder.into_inner().await.stop().await }),
        )
        .await
        .into_iter()
        .collect::<SummaResult<Vec<_>>>()
        .map_err(crate::errors::Error::from)?;
        Ok(())
    }

    /// Returns `ConsumerConfig`
    pub(crate) async fn get_consumer_config(&self, index_alias: &str, consumer_name: &str) -> SummaServerResult<ConsumerConfig> {
        Ok(self.get_index_holder(index_alias).await?.get_consumer_config(consumer_name).await?)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::path::Path;

    use summa_core::components::SummaDocument;
    use summa_core::configs::application_config::tests::create_test_application_config_holder;
    use summa_proto::proto_traits::collector::shortcuts::{scored_doc, top_docs_collector, top_docs_collector_output, top_docs_collector_with_eval_expr};
    use summa_proto::proto_traits::query::shortcuts::match_query;
    use tantivy::doc;
    use tantivy::schema::{IndexRecordOption, Schema, TextFieldIndexing, TextOptions, FAST, INDEXED, STORED};

    use super::*;
    use crate::logging;
    use crate::requests::{CreateIndexRequestBuilder, DeleteIndexRequestBuilder};

    pub async fn create_test_index_service(data_path: &Path) -> IndexService {
        IndexService::new(&create_test_application_config_holder(&data_path))
    }
    pub async fn create_test_index_holder(index_service: &IndexService, schema: &Schema) -> SummaServerResult<Handler<IndexHolder>> {
        index_service
            .create_index(
                CreateIndexRequestBuilder::default()
                    .index_name("test_index".to_owned())
                    .default_fields(vec!["title".to_owned(), "body".to_owned()])
                    .index_engine(proto::IndexEngine::Memory)
                    .schema(schema.clone())
                    .build()
                    .unwrap(),
            )
            .await
    }

    pub fn create_test_schema() -> Schema {
        let mut schema_builder = Schema::builder();

        schema_builder.add_i64_field("id", FAST | INDEXED | STORED);
        schema_builder.add_i64_field("issued_at", FAST | INDEXED | STORED);
        schema_builder.add_text_field(
            "title",
            TextOptions::default().set_stored().set_indexing_options(
                TextFieldIndexing::default()
                    .set_tokenizer("summa")
                    .set_index_option(IndexRecordOption::WithFreqsAndPositions),
            ),
        );
        schema_builder.add_text_field(
            "body",
            TextOptions::default().set_stored().set_indexing_options(
                TextFieldIndexing::default()
                    .set_tokenizer("summa")
                    .set_index_option(IndexRecordOption::WithFreqsAndPositions),
            ),
        );

        schema_builder.build()
    }

    #[tokio::test]
    async fn test_same_name_index() {
        logging::tests::initialize_default_once();
        let index_service = IndexService::default();
        assert!(index_service
            .create_index(
                CreateIndexRequestBuilder::default()
                    .index_name("test_index".to_owned())
                    .index_engine(proto::IndexEngine::Memory)
                    .schema(create_test_schema())
                    .build()
                    .unwrap(),
            )
            .await
            .is_ok());

        assert!(index_service
            .create_index(
                CreateIndexRequestBuilder::default()
                    .index_name("test_index".to_owned())
                    .index_engine(proto::IndexEngine::Memory)
                    .schema(create_test_schema())
                    .build()
                    .unwrap(),
            )
            .await
            .is_err());
    }

    #[tokio::test]
    async fn test_index_create_delete() -> SummaServerResult<()> {
        logging::tests::initialize_default_once();
        let root_path = tempdir::TempDir::new("summa_test").unwrap();
        let data_path = root_path.path().join("data");
        let application_config_holder = create_test_application_config_holder(&data_path);

        let index_service = IndexService::new(&application_config_holder);
        index_service
            .create_index(
                CreateIndexRequestBuilder::default()
                    .index_name("test_index".to_owned())
                    .index_engine(proto::IndexEngine::Memory)
                    .schema(create_test_schema())
                    .build()
                    .unwrap(),
            )
            .await?;
        assert!(index_service
            .delete_index(DeleteIndexRequestBuilder::default().index_name("test_index".to_owned()).build().unwrap())
            .await
            .is_ok());
        assert!(index_service
            .create_index(
                CreateIndexRequestBuilder::default()
                    .index_name("test_index".to_owned())
                    .index_engine(proto::IndexEngine::Memory)
                    .schema(create_test_schema())
                    .build()
                    .unwrap()
            )
            .await
            .is_ok());
        Ok(())
    }

    #[tokio::test]
    async fn test_search() -> SummaServerResult<()> {
        logging::tests::initialize_default_once();
        let schema = create_test_schema();

        let root_path = tempdir::TempDir::new("summa_test").unwrap();
        let data_path = root_path.path().join("data");

        let index_service = create_test_index_service(&data_path).await;
        let index_holder = create_test_index_holder(&index_service, &schema).await?;

        index_holder.index_document(SummaDocument::TantivyDocument(doc!(
            schema.get_field("id").unwrap() => 1i64,
            schema.get_field("title").unwrap() => "Headcrab",
            schema.get_field("body").unwrap() => "Physically, headcrabs are frail: a few bullets or a single strike from the player's melee weapon being sufficient to dispatch them. \
            They are also relatively slow-moving and their attacks inflict very little damage. However, they can leap long distances and heights. \
            Headcrabs seek out larger human hosts, which are converted into zombie-like mutants that attack any living lifeform nearby. \
            The converted humans are more resilient than an ordinary human would be and inherit the headcrab's resilience toward toxic and radioactive materials. \
            Headcrabs and headcrab zombies die slowly when they catch fire. \
            The games also establish that while headcrabs are parasites that prey on humans, they are also the prey of the creatures of their homeworld. \
            Bullsquids, Vortigaunts, barnacles and antlions will all eat headcrabs and Vortigaunts can be seen cooking them in several locations in-game.",
            schema.get_field("issued_at").unwrap() => 1652986134i64
        ))).await?;
        index_holder.index_updater().write().await.commit(None).await?;
        index_holder.index_reader().reload().unwrap();
        assert_eq!(
            index_holder.search("index", &match_query("headcrabs"), &vec![top_docs_collector(10)]).await?,
            vec![top_docs_collector_output(
                vec![scored_doc(
                    "{\
                        \"body\":\"Physically, headcrabs are frail: a few bullets or a single strike from the player's melee weapon being sufficient \
                        to dispatch them. They are also relatively slow-moving and their attacks inflict very little damage. However, they can leap long distances \
                        and heights. Headcrabs seek out larger human hosts, which are converted into zombie-like mutants that attack any living lifeform nearby. \
                        The converted humans are more resilient than an ordinary human would be and inherit the headcrab's resilience toward toxic and radioactive materials. \
                        Headcrabs and headcrab zombies die slowly when they catch fire. \
                        The games also establish that while headcrabs are parasites that prey on humans, they are also the prey of the creatures of their homeworld. \
                        Bullsquids, Vortigaunts, barnacles and antlions will all eat headcrabs and Vortigaunts can be seen cooking them in several locations in-game.\",\
                        \"id\":1,\
                        \"issued_at\":1652986134,\
                        \"title\":\"Headcrab\"}",
                    0.5126588344573975,
                    0
                )],
                false
            )]
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_custom_ranking() -> SummaServerResult<()> {
        logging::tests::initialize_default_once();
        let schema = create_test_schema();

        let root_path = tempdir::TempDir::new("summa_test").unwrap();
        let data_path = root_path.path().join("data");

        let index_service = create_test_index_service(&data_path).await;
        let index_holder = create_test_index_holder(&index_service, &schema).await?;

        index_holder
            .index_document(SummaDocument::TantivyDocument(doc!(
                schema.get_field("id").unwrap() => 1i64,
                schema.get_field("title").unwrap() => "term1 term2",
                schema.get_field("body").unwrap() => "term3 term4 term5 term6",
                schema.get_field("issued_at").unwrap() => 100i64
            )))
            .await?;
        index_holder
            .index_document(SummaDocument::TantivyDocument(doc!(
                schema.get_field("id").unwrap() => 2i64,
                schema.get_field("title").unwrap() => "term2 term3",
                schema.get_field("body").unwrap() => "term1 term7 term8 term9 term10",
                schema.get_field("issued_at").unwrap() => 110i64
            )))
            .await?;
        index_holder.index_updater().write().await.commit(None).await?;
        index_holder.index_reader().reload().unwrap();
        assert_eq!(
            index_holder
                .search("index", &match_query("term1"), &vec![top_docs_collector_with_eval_expr(10, "issued_at")])
                .await?,
            vec![top_docs_collector_output(
                vec![
                    scored_doc(
                        "{\"body\":\"term1 term7 term8 term9 term10\",\"id\":2,\"issued_at\":110,\"title\":\"term2 term3\"}",
                        110.0,
                        0
                    ),
                    scored_doc(
                        "{\"body\":\"term3 term4 term5 term6\",\"id\":1,\"issued_at\":100,\"title\":\"term1 term2\"}",
                        100.0,
                        1
                    )
                ],
                false
            )]
        );
        assert_eq!(
            index_holder
                .search("index", &match_query("term1"), &vec![top_docs_collector_with_eval_expr(10, "-issued_at")])
                .await?,
            vec![top_docs_collector_output(
                vec![
                    scored_doc(
                        "{\"body\":\"term3 term4 term5 term6\",\"id\":1,\"issued_at\":100,\"title\":\"term1 term2\"}",
                        -100.0,
                        0
                    ),
                    scored_doc(
                        "{\"body\":\"term1 term7 term8 term9 term10\",\"id\":2,\"issued_at\":110,\"title\":\"term2 term3\"}",
                        -110.0,
                        1
                    ),
                ],
                false
            )]
        );
        Ok(())
    }
}
