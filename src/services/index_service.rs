use crate::configs::{ApplicationConfig, ApplicationConfigHolder, IndexEngine, KafkaConsumerConfig, Persistable};

use crate::errors::{Error, SummaResult, ValidationError};
use crate::requests::{CreateConsumerRequest, CreateIndexRequest, DeleteConsumerRequest, DeleteIndexRequest};
use crate::search_engine::IndexHolder;
use crate::utils::sync::{Handler, OwningHandler};
use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::sync::Arc;
use tantivy::directory::GarbageCollectionResult;
use tokio::fs::remove_dir_all;
use tracing::{info, instrument};

/// The main struct responsible for indices lifecycle. Here lives indices creation and deletion as well as committing and indexing new documents.
#[derive(Clone, Debug)]
pub struct IndexService {
    application_config: ApplicationConfigHolder,
    /// RwLock for index modification operations like creating new ones or deleting old ones
    index_holders: Arc<RwLock<HashMap<String, OwningHandler<IndexHolder>>>>,
}

#[derive(Default)]
pub struct DeleteIndexResult {
    pub deleted_aliases: Vec<String>,
    pub deleted_consumers: Vec<String>,
}

///
impl IndexService {
    pub async fn new(application_config: &ApplicationConfigHolder) -> SummaResult<IndexService> {
        let mut index_holders: HashMap<String, OwningHandler<IndexHolder>> = HashMap::new();
        {
            let application_config = application_config.read();
            for (index_name, index_config) in &application_config.indices {
                let index_holder = IndexHolder::open(index_name, &application_config, index_config).await?;
                index_holders.insert(index_name.clone(), OwningHandler::new(index_holder));
            }
        }
        let index_service = IndexService {
            application_config: application_config.clone(),
            index_holders: Arc::new(RwLock::new(index_holders)),
        };
        Ok(index_service)
    }

    pub fn index_holders(&self) -> RwLockReadGuard<'_, HashMap<String, OwningHandler<IndexHolder>>> {
        self.index_holders.read()
    }

    pub fn index_holders_mut(&self) -> RwLockWriteGuard<'_, HashMap<String, OwningHandler<IndexHolder>>> {
        self.index_holders.write()
    }

    pub fn get_index_holder(&self, index_alias_or_name: &str) -> SummaResult<Handler<IndexHolder>> {
        let index_name = self.application_config.read().resolve_index_alias(index_alias_or_name);
        let index_name = index_name.as_deref().unwrap_or(index_alias_or_name);
        let index_holder = self
            .index_holders
            .read()
            .get(index_name)
            .ok_or_else(|| Error::ValidationError(ValidationError::MissingIndexError(index_alias_or_name.to_owned())))?
            .handler();
        Ok(index_holder)
    }

    /// Create consumer and insert it into the consumer registry. Add it to the `IndexHolder` afterwards.
    pub async fn create_index(&self, create_index_request: CreateIndexRequest) -> SummaResult<()> {
        let mut application_config = self.application_config.write();
        let mut index_holders = self.index_holders.write();

        let index_path = application_config.get_path_for_index_data(&create_index_request.index_name);
        if index_path.exists() {
            Err(ValidationError::ExistingPathError(index_path.to_owned()))?;
        }
        std::fs::create_dir_all(&index_path)?;

        let index_holder = IndexHolder::create(
            &create_index_request.index_name,
            &index_path,
            &create_index_request.index_config,
            &create_index_request.schema,
        )
        .await?;
        index_holders.insert(create_index_request.index_name.to_owned(), OwningHandler::new(index_holder));
        application_config
            .autosave()
            .indices
            .insert(create_index_request.index_name.to_owned(), create_index_request.index_config);
        Ok(())
    }

    fn check_delete_conditions(&self, aliases: &Vec<String>, index_holder: &IndexHolder) -> SummaResult<()> {
        if aliases.len() > 0 {
            Err(ValidationError::AliasedError(aliases.join(", ")))?;
        }
        if index_holder.has_consumers() {
            Err(ValidationError::ExistingConsumersError(index_holder.index_name().to_owned()))?;
        }
        Ok(())
    }

    async fn prepare_delete(
        &self,
        application_config: &mut ApplicationConfig,
        delete_index_request: &DeleteIndexRequest,
        index_holder: &IndexHolder,
    ) -> SummaResult<DeleteIndexResult> {
        let mut delete_index_result = DeleteIndexResult::default();
        let aliases = application_config.get_index_aliases_for_index(&delete_index_request.index_name);

        if delete_index_request.cascade {
            application_config.delete_index_aliases(&aliases);
            delete_index_result.deleted_aliases.extend(aliases);
            index_holder.delete_all_consumers().await?;
        } else {
            self.check_delete_conditions(&aliases, index_holder)?;
        }

        Ok(delete_index_result)
    }

    #[instrument(skip_all)]
    pub async fn delete_index(&self, delete_index_request: DeleteIndexRequest) -> SummaResult<DeleteIndexResult> {
        let mut application_config = self.application_config.write();
        let mut index_holders = self.index_holders.write();
        let index_holder = match index_holders.get(&delete_index_request.index_name) {
            Some(index_holder) => index_holder,
            None => Err(ValidationError::MissingIndexError(delete_index_request.index_name.to_owned()))?,
        };

        let delete_index_result = self.prepare_delete(&mut application_config, &delete_index_request, index_holder).await?;
        match application_config.indices.remove(&delete_index_request.index_name) {
            Some(index_config) => match index_config.index_engine {
                IndexEngine::Memory(_) => (),
                IndexEngine::File => {
                    info!(action = "delete_directory");
                    let index_path = application_config.get_path_for_index_data(&delete_index_request.index_name);
                    remove_dir_all(&index_path).await.map_err(|e| Error::IOError((e, Some(index_path.to_path_buf()))))?;
                }
            },
            None => Err(ValidationError::MissingIndexError(delete_index_request.index_name.to_owned()))?,
        };
        application_config.save()?;

        match index_holders.remove(&delete_index_request.index_name) {
            Some(index_holder) => {
                let inner = tokio::task::spawn_blocking(move || index_holder.into_inner()).await?;
                inner.stop().await?;
            }
            None => Err(ValidationError::MissingIndexError(delete_index_request.index_name.to_owned()))?,
        }

        Ok(delete_index_result)
    }

    #[instrument(skip_all)]
    pub async fn vacuum_index(&self, index_name: &str) -> SummaResult<GarbageCollectionResult> {
        let index_holders = self.index_holders.read();
        let index_holder = match index_holders.get(index_name) {
            Some(index_holder) => index_holder,
            None => Err(ValidationError::MissingIndexError(index_name.to_owned()))?,
        };
        index_holder.vacuum().await
    }

    /// Create consumer and insert it into the consumer registry. Add it to the `IndexHolder` afterwards.
    pub fn get_consumers(&self) -> SummaResult<Vec<(String, String)>> {
        let application_config = self.application_config.read();
        let mut result = Vec::new();
        for (index_name, index_config) in &application_config.indices {
            result.extend(index_config.consumer_configs.keys().map(|consumer_name| (index_name.to_owned(), consumer_name.to_owned())))
        }
        Ok(result)
    }

    /// Create consumer and insert it into the consumer registry. Add it to the `IndexHolder` afterwards.
    pub async fn create_consumer(&self, create_consumer_request: CreateConsumerRequest) -> SummaResult<()> {
        match self
            .application_config
            .write()
            .autosave()
            .indices
            .get_mut(&create_consumer_request.index_name)
            .ok_or(ValidationError::MissingIndexError(create_consumer_request.index_name.to_owned()))?
            .consumer_configs
            .entry(create_consumer_request.consumer_name.to_owned())
        {
            Entry::Occupied(o) => Err(ValidationError::ExistingConsumerError(o.key().to_owned())),
            Entry::Vacant(v) => {
                v.insert(create_consumer_request.consumer_config.clone());
                Ok(())
            }
        }?;

        let index_holder = self.get_index_holder(&create_consumer_request.index_name)?;
        index_holder
            .create_consumer(&create_consumer_request.consumer_name, create_consumer_request.consumer_config)
            .await?;

        Ok(())
    }

    /// Delete consumer from the consumer registry and from `IndexHolder` afterwards.
    pub async fn delete_consumer(&self, delete_consumer_request: DeleteConsumerRequest) -> SummaResult<()> {
        self.application_config
            .write()
            .autosave()
            .indices
            .get_mut(&delete_consumer_request.index_name)
            .ok_or(ValidationError::MissingIndexError(delete_consumer_request.index_name.to_owned()))?
            .consumer_configs
            .remove(&delete_consumer_request.consumer_name)
            .ok_or(ValidationError::MissingConsumerError(delete_consumer_request.consumer_name.to_owned()))?;

        let index_holder = self.get_index_holder(&delete_consumer_request.index_name)?;
        index_holder.delete_consumer(&delete_consumer_request.consumer_name).await?;

        Ok(())
    }

    /// Stopping index holders
    pub async fn stop(self) -> SummaResult<()> {
        for (_, index_holder) in self.index_holders_mut().drain() {
            index_holder.into_inner().stop().await?;
        }
        Ok(())
    }

    pub(crate) fn get_consumer_config(&self, index_name: &str, consumer_name: &str) -> SummaResult<KafkaConsumerConfig> {
        Ok(self
            .application_config
            .read()
            .indices
            .get(index_name)
            .ok_or(ValidationError::MissingIndexError(index_name.to_owned()))?
            .consumer_configs
            .get(consumer_name)
            .ok_or(ValidationError::MissingConsumerError(consumer_name.to_owned()))?
            .clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tantivy::doc;

    use crate::configs::IndexConfig;
    use crate::proto;
    use crate::search_engine::SummaDocument;
    use tantivy::schema::{SchemaBuilder, TEXT};

    #[tokio::test]
    async fn test_index_service() -> SummaResult<()> {
        let application_config = ApplicationConfigHolder::default();
        let index_service = IndexService::new(&application_config).await?;

        let mut schema_builder = SchemaBuilder::new();
        let title_field = schema_builder.add_text_field("title", TEXT);
        let body_field = schema_builder.add_text_field("body", TEXT);
        let schema = schema_builder.build();
        let create_index_request = CreateIndexRequest {
            index_name: "test-index".to_owned(),
            index_config: IndexConfig {
                default_fields: vec![schema.get_field("title").unwrap(), schema.get_field("body").unwrap()],
                index_engine: IndexEngine::Memory(schema.clone()),
                ..Default::default()
            },
            schema,
        };
        index_service.create_index(create_index_request).await?;
        let index_holder = index_service.get_index_holder("test-index")?;
        index_holder.index_document(
            SummaDocument::TantivyDocument(doc!(
                title_field => "Test title",
                body_field => "Test body",
            )),
            false,
        )?;
        index_holder.commit().await?;
        index_holder.reload()?;
        let search_result = index_holder.search("body", 1, 0).await?;
        assert_eq!(
            search_result,
            proto::SearchResponse {
                index_name: "test-index".into(),
                scored_documents: vec![proto::ScoredDocument {
                    document: "{}".to_owned(),
                    position: 0,
                    score: 0.28768212,
                }],
                has_next: false,
            }
        );
        Ok(())
    }
}
