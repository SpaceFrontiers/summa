use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::future::Future;
use std::sync::Arc;

use futures::future::join_all;
use itertools::Itertools;
use summa_proto::proto;
use summa_proto::proto::collector_output::CollectorOutput;
use summa_proto::proto::Score;
use tantivy::DocAddress;
use tokio::sync::RwLock;
use tracing::info;

use super::IndexHolder;
use crate::components::custom_serializer::NamedFieldDocument;
use crate::components::fruit_extractors::{IntermediateExtractionResult, ReadyCollectorOutput, ScoredDocAddress};
use crate::components::Driver;
use crate::configs::{ConfigProxy, DirectProxy};
use crate::errors::{SummaResult, ValidationError};
use crate::proto_traits::Wrapper;
use crate::utils::sync::{Handler, OwningHandler};
use crate::utils::transpose;
use crate::Error;

/// The main struct responsible for combining different indices and managing their lifetime.
#[derive(Clone)]
pub struct IndexRegistry {
    core_config_holder: Arc<dyn ConfigProxy<crate::configs::core::Config>>,
    index_holders: Arc<RwLock<HashMap<String, OwningHandler<IndexHolder>>>>,
}

impl Default for IndexRegistry {
    fn default() -> Self {
        IndexRegistry {
            core_config_holder: Arc::new(DirectProxy::default()),
            index_holders: Arc::default(),
        }
    }
}

impl Debug for IndexRegistry {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IndexRegistry").field("index_holders", &self.index_holders).finish()
    }
}

struct ScoredDocAddressRefWithAlias<'a> {
    index_alias: &'a str,
    scored_doc_address: &'a ScoredDocAddress,
}

impl ScoredDocAddressRefWithAlias<'_> {
    pub fn doc_address(&self) -> DocAddress {
        self.scored_doc_address.doc_address
    }

    pub fn score(&self) -> &Option<Score> {
        &self.scored_doc_address.score
    }
}

impl IndexRegistry {
    pub fn new(core_config: &Arc<dyn ConfigProxy<crate::configs::core::Config>>) -> IndexRegistry {
        IndexRegistry {
            core_config_holder: core_config.clone(),
            index_holders: Arc::default(),
        }
    }

    /// Read-locked `HashMap` of all indices
    pub fn index_holders(&self) -> &Arc<RwLock<HashMap<String, OwningHandler<IndexHolder>>>> {
        &self.index_holders
    }

    pub async fn index_holders_cloned(&self) -> HashMap<String, Handler<IndexHolder>> {
        self.index_holders()
            .read()
            .await
            .iter()
            .map(|(index_name, handler)| (index_name.to_string(), handler.handler()))
            .collect()
    }

    pub async fn clear(self) {
        self.index_holders().write().await.clear();
    }

    /// Returns `Handler` to `IndexHolder`.
    ///
    /// It is safe to keep `Handler<IndexHolder>` cause `Index` won't be deleted until `Handler` is alive.
    /// Though, `IndexHolder` can be removed from the registry of `IndexHolder`s to prevent new queries
    pub async fn get_index_holder(&self, index_alias: &str) -> SummaResult<Handler<IndexHolder>> {
        self.get_index_holder_by_name(
            self.core_config_holder
                .read()
                .await
                .get()
                .resolve_index_alias(index_alias)
                .as_deref()
                .unwrap_or(index_alias),
        )
        .await
    }

    /// Retrieve `IndexHolder` by its name
    pub async fn get_index_holder_by_name(&self, index_name: &str) -> SummaResult<Handler<IndexHolder>> {
        Ok(self
            .index_holders()
            .read()
            .await
            .get(index_name)
            .ok_or_else(|| Error::Validation(Box::new(ValidationError::MissingIndex(index_name.to_owned()))))?
            .handler())
    }

    /// Add new index to `IndexRegistry`
    pub async fn add(&self, index_holder: IndexHolder) -> SummaResult<Handler<IndexHolder>> {
        let index_holder = OwningHandler::new(index_holder);
        let index_holder_handler = index_holder.handler();
        self.index_holders().write().await.insert(index_holder.index_name().to_string(), index_holder);
        info!(action = "added", index_name = ?index_holder_handler.index_name());
        Ok(index_holder_handler)
    }

    /// Deletes index from `IndexRegistry`
    pub async fn delete(&self, index_name: &str) {
        self.index_holders().write().await.remove(index_name);
    }

    /// Searches in several indices simultaneously and merges results
    pub async fn search_futures(
        &self,
        index_queries: Vec<proto::IndexQuery>,
    ) -> SummaResult<Vec<impl Future<Output = SummaResult<Vec<IntermediateExtractionResult>>>>> {
        index_queries
            .into_iter()
            .map(|index_query| {
                let this = self.clone();
                Ok(async move {
                    let index_holder = this.get_index_holder(&index_query.index_alias).await?;
                    index_holder
                        .search(
                            &index_query.index_alias,
                            index_query
                                .query
                                .and_then(|query| query.query)
                                .as_ref()
                                .unwrap_or_else(|| &proto::query::Query::All(proto::AllQuery {})),
                            index_query.collectors,
                        )
                        .await
                })
            })
            .collect::<SummaResult<Vec<_>>>()
    }

    /// Merges several `proto::CollectorOutput`
    pub async fn finalize_extraction(
        &self,
        intermediate_extractions_results: Vec<Vec<IntermediateExtractionResult>>,
        driver: Driver,
    ) -> SummaResult<Vec<proto::CollectorOutput>> {
        let extractions_results = transpose(intermediate_extractions_results);
        let mut merged_response = vec![];

        for intermediate_extraction_results in extractions_results.into_iter() {
            merged_response.push(proto::CollectorOutput {
                collector_output: Some(match intermediate_extraction_results.first() {
                    None => return Err(Error::Validation(Box::new(ValidationError::EmptyArgument("collectors".to_string())))),
                    Some(IntermediateExtractionResult::Ready(ReadyCollectorOutput::Aggregation(_))) => todo!(),
                    Some(IntermediateExtractionResult::Ready(ReadyCollectorOutput::Count(_))) => {
                        let counts = intermediate_extraction_results
                            .into_iter()
                            .map(|extraction_result| extraction_result.as_count().expect("expected count collector").count);
                        CollectorOutput::Count(proto::CountCollectorOutput { count: counts.sum() })
                    }
                    Some(IntermediateExtractionResult::Ready(ReadyCollectorOutput::Facet(_))) => todo!(),
                    Some(IntermediateExtractionResult::PreparedDocumentReferences(first_prepared_document_references)) => {
                        let offset = first_prepared_document_references.offset as usize;
                        let limit = first_prepared_document_references.limit as usize;
                        let prepared_documents_references: Vec<_> = intermediate_extraction_results
                            .into_iter()
                            .map(|extraction_result| extraction_result.as_document_references().expect("expected top_docs collector"))
                            .collect();
                        let mut extraction_toolings = HashMap::new();
                        let mut has_next = false;

                        let snippet_generators: HashMap<_, _> = join_all(prepared_documents_references.iter().map(|prepared_document_references| async move {
                            (
                                prepared_document_references.index_alias.as_str(),
                                if let Some(snippet_generator_config) = prepared_document_references.snippet_generator_config.as_ref() {
                                    snippet_generator_config.as_tantivy_async().await
                                } else {
                                    Vec::default()
                                },
                            )
                        }))
                        .await
                        .into_iter()
                        .collect();

                        for prepared_document_references in &prepared_documents_references {
                            has_next |= prepared_document_references.has_next;
                            extraction_toolings.insert(
                                prepared_document_references.index_alias.as_str(),
                                &prepared_document_references.extraction_tooling,
                            );
                        }
                        let merged_scored_doc_address_refs: Vec<_> = prepared_documents_references
                            .iter()
                            .map(|top_docs| {
                                top_docs.scored_doc_addresses.iter().map(|scored_doc_address| ScoredDocAddressRefWithAlias {
                                    index_alias: top_docs.index_alias.as_str(),
                                    scored_doc_address,
                                })
                            })
                            .kmerge_by(|a, b| a.score() > b.score())
                            .collect();

                        let extraction_toolings_ref = &extraction_toolings;
                        let snippet_generators_ref = &snippet_generators;
                        let driver_ref = &driver;

                        let scored_documents = join_all(merged_scored_doc_address_refs.into_iter().enumerate().skip(offset).take(limit).map(
                            |(position, scored_doc_address_ref)| {
                                let doc_address = scored_doc_address_ref.doc_address();
                                let extraction_tooling = extraction_toolings_ref[scored_doc_address_ref.index_alias];
                                let searcher = extraction_tooling.searcher.clone();
                                async move {
                                    let document = driver_ref.execute_blocking(move || searcher.doc(doc_address)).await??;
                                    Ok(proto::ScoredDocument {
                                        document: NamedFieldDocument::from_document(
                                            extraction_tooling.searcher.schema(),
                                            &extraction_tooling.query_fields,
                                            &extraction_tooling.multi_fields,
                                            &document,
                                        )
                                        .to_json_string(),
                                        score: scored_doc_address_ref.score().clone(),
                                        position: position as u32,
                                        snippets: snippet_generators_ref[scored_doc_address_ref.index_alias]
                                            .iter()
                                            .map(|(field_name, snippet_generator)| {
                                                (
                                                    field_name.to_string(),
                                                    Wrapper::from(snippet_generator.snippet_from_doc(&document)).into_inner(),
                                                )
                                            })
                                            .collect(),
                                        index_alias: scored_doc_address_ref.index_alias.to_string(),
                                    })
                                }
                            },
                        ))
                        .await
                        .into_iter()
                        .collect::<SummaResult<Vec<_>>>()?;
                        CollectorOutput::TopDocs(proto::TopDocsCollectorOutput { has_next, scored_documents })
                    }
                }),
            });
        }
        Ok(merged_response)
    }
}
