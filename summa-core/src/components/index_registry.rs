use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::future::Future;
use std::sync::Arc;

use futures::future::join_all;
use itertools::Itertools;
use summa_proto::proto;
use summa_proto::proto::collector_output::CollectorOutput;
use summa_proto::proto::Score;
use tantivy::{DocAddress, SnippetGenerator};
use tokio::sync::RwLock;
use tracing::{info, trace};

use super::IndexHolder;
use crate::components::custom_serializer::NamedFieldDocument;
use crate::components::fruit_extractors::{IntermediateExtractionResult, ReadyCollectorOutput, ScoredDocAddress};
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
        info!(action = "added");
        Ok(index_holder_handler)
    }

    /// Deletes index from `IndexRegistry`
    pub async fn delete(&self, index_name: &str) {
        self.index_holders().write().await.remove(index_name);
    }

    /// Searches in several indices simultaneously and merges results
    pub fn search_futures(
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
                                .unwrap_or_else(|| proto::query::Query::All(proto::AllQuery {})),
                            index_query.collectors,
                            index_query.is_fieldnorms_scoring_enabled,
                        )
                        .await
                })
            })
            .collect::<SummaResult<Vec<_>>>()
    }

    /// Merges several `IntermediateExtractionResult`
    pub async fn finalize_extraction(&self, ie_results: Vec<Vec<IntermediateExtractionResult>>) -> SummaResult<Vec<proto::CollectorOutput>> {
        if ie_results.is_empty() {
            return Ok(vec![]);
        }
        let ie_results = transpose(ie_results);
        let mut collector_outputs = vec![];

        for ie_result in ie_results.into_iter() {
            collector_outputs.push(proto::CollectorOutput {
                collector_output: Some(match ie_result.first() {
                    None => return Err(Error::Validation(Box::new(ValidationError::EmptyArgument("collectors".to_string())))),
                    Some(IntermediateExtractionResult::Ready(ReadyCollectorOutput::Aggregation(aggregation_collector_output))) => {
                        if ie_result.len() == 1 {
                            CollectorOutput::Aggregation(aggregation_collector_output.clone())
                        } else {
                            todo!();
                        }
                    }
                    Some(IntermediateExtractionResult::Ready(ReadyCollectorOutput::Count(_))) => {
                        trace!(action = "count_collector_finalization");
                        let counts = ie_result
                            .into_iter()
                            .map(|ie_result| ie_result.as_count().expect("expected count collector").count);
                        CollectorOutput::Count(proto::CountCollectorOutput { count: counts.sum() })
                    }
                    Some(IntermediateExtractionResult::Ready(ReadyCollectorOutput::Facet(facet_collector_output))) => {
                        if ie_result.len() == 1 {
                            CollectorOutput::Facet(facet_collector_output.clone())
                        } else {
                            todo!();
                        }
                    }
                    Some(IntermediateExtractionResult::PreparedDocumentReferences(first_prepared_document_references)) => {
                        trace!(action = "prepared_documents_finalization");

                        let offset = first_prepared_document_references.offset as usize;
                        let limit = first_prepared_document_references.limit as usize;

                        let docs_references: Vec<_> = ie_result
                            .into_iter()
                            .map(|ie_result| ie_result.as_document_references().expect("expected top_docs collector"))
                            .collect();

                        let mut has_next = false;
                        let mut total_documents = 0;
                        let mut extraction_toolings = HashMap::new();
                        let mut snippet_generators_futures = vec![];

                        for doc_references in &docs_references {
                            has_next |= doc_references.has_next;
                            total_documents += doc_references.scored_doc_addresses.len();
                            extraction_toolings.insert(doc_references.index_alias.as_str(), &doc_references.extraction_tooling);
                            if let Some(sg_config) = &doc_references.snippet_generator_config {
                                snippet_generators_futures.push(async move { (doc_references.index_alias.as_str(), sg_config.as_tantivy_async().await) });
                            }
                        }
                        has_next |= total_documents > limit + offset;

                        trace!(action = "generate_snippets");
                        let snippet_generators: HashMap<&str, Vec<(String, SnippetGenerator)>> =
                            join_all(snippet_generators_futures).await.into_iter().collect();

                        info!(
                            action = "retrieving_documents",
                            limit = limit,
                            offset = offset,
                            total_documents = total_documents
                        );
                        let merged_scored_doc_address_refs: Vec<_> = docs_references
                            .iter()
                            .map(|doc_references| {
                                doc_references
                                    .scored_doc_addresses
                                    .iter()
                                    .map(|scored_doc_address| ScoredDocAddressRefWithAlias {
                                        index_alias: doc_references.index_alias.as_str(),
                                        scored_doc_address,
                                    })
                            })
                            .kmerge_by(|a, b| a.score() > b.score())
                            .skip(offset)
                            .take(limit)
                            .collect();

                        let extraction_toolings_ref = &extraction_toolings;
                        let snippet_generators_ref = &snippet_generators;

                        let scored_documents = join_all(
                            merged_scored_doc_address_refs
                                .into_iter()
                                .enumerate()
                                .map(|(position, scored_doc_address_ref)| {
                                    let doc_address = scored_doc_address_ref.doc_address();
                                    let extraction_tooling = extraction_toolings_ref[scored_doc_address_ref.index_alias];
                                    let searcher = extraction_tooling.searcher.clone();
                                    async move {
                                        #[cfg(feature = "tokio-rt")]
                                        let document = tokio::task::spawn_blocking(move || searcher.doc(doc_address)).await??;
                                        #[cfg(not(feature = "tokio-rt"))]
                                        let document = searcher.doc_async(doc_address).await?;
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
                                            snippets: snippet_generators_ref
                                                .get(scored_doc_address_ref.index_alias)
                                                .map(|snippet_generator| {
                                                    snippet_generator
                                                        .iter()
                                                        .map(|(field_name, snippet_generator)| {
                                                            (
                                                                field_name.to_string(),
                                                                Wrapper::from(snippet_generator.snippet_from_doc(&document)).into_inner(),
                                                            )
                                                        })
                                                        .collect()
                                                })
                                                .unwrap_or_default(),
                                            index_alias: scored_doc_address_ref.index_alias.to_string(),
                                        })
                                    }
                                }),
                        )
                        .await
                        .into_iter()
                        .collect::<SummaResult<Vec<_>>>()?;
                        CollectorOutput::Documents(proto::DocumentsCollectorOutput { has_next, scored_documents })
                    }
                }),
            });
        }
        Ok(collector_outputs)
    }
}
