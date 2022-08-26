use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use futures::future::join_all;
use summa_proto::proto;
use summa_proto::proto::collector_output::CollectorOutput;
use summa_proto::proto::Score;
use tantivy::DocAddress;
use tokio::sync::RwLock;
use tracing::{info, trace};

use super::IndexHolder;
use crate::components::custom_serializer::NamedFieldDocument;
use crate::components::fruit_extractors::{IntermediateExtractionResult, ReadyCollectorOutput, ScoredDocAddress};
use crate::configs::{ConfigProxy, DirectProxy};
use crate::errors::{SummaResult, ValidationError};
use crate::proto_traits::Wrapper;
use crate::utils::sync::{Handler, OwningHandler};
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

    /// Merges several `IntermediateExtractionResult`
    pub async fn finalize_extraction(&self, ie_results: Vec<IntermediateExtractionResult>) -> SummaResult<Vec<proto::CollectorOutput>> {
        if ie_results.is_empty() {
            return Ok(vec![]);
        }
        let mut collector_outputs = vec![];

        for ie_result in ie_results.into_iter() {
            collector_outputs.push(proto::CollectorOutput {
                collector_output: Some(match ie_result {
                    IntermediateExtractionResult::Ready(ReadyCollectorOutput::Aggregation(aggregation_collector_output)) => {
                        CollectorOutput::Aggregation(aggregation_collector_output)
                    }
                    IntermediateExtractionResult::Ready(ReadyCollectorOutput::Count(count_collector_output)) => CollectorOutput::Count(count_collector_output),
                    IntermediateExtractionResult::Ready(ReadyCollectorOutput::Facet(facet_collector_output)) => CollectorOutput::Facet(facet_collector_output),
                    IntermediateExtractionResult::PreparedDocumentReferences(prepared_document_references) => {
                        trace!(action = "prepared_documents_finalization");
                        let extraction_tooling = &prepared_document_references.extraction_tooling;
                        let snippet_generator = if let Some(snippet_generator_config) = prepared_document_references.snippet_generator_config {
                            Some(snippet_generator_config.as_tantivy_async().await)
                        } else {
                            None
                        };

                        let scored_doc_address_refs =
                            prepared_document_references
                                .scored_doc_addresses
                                .iter()
                                .map(|scored_doc_address| ScoredDocAddressRefWithAlias {
                                    index_alias: prepared_document_references.index_alias.as_str(),
                                    scored_doc_address,
                                });
                        let extraction_tooling_ref = &extraction_tooling;
                        let snippet_generator_ref = &snippet_generator;

                        let scored_documents = join_all(scored_doc_address_refs.into_iter().enumerate().map(|(position, scored_doc_address_ref)| {
                            let doc_address = scored_doc_address_ref.doc_address();
                            let searcher = extraction_tooling_ref.searcher.clone();
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
                                    snippets: snippet_generator_ref
                                        .as_ref()
                                        .map(|snippet_generator_ref| {
                                            snippet_generator_ref
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
                        }))
                        .await
                        .into_iter()
                        .collect::<SummaResult<Vec<_>>>()?;
                        CollectorOutput::Documents(proto::DocumentsCollectorOutput {
                            has_next: prepared_document_references.has_next,
                            scored_documents,
                        })
                    }
                }),
            });
        }
        Ok(collector_outputs)
    }
}
