use std::collections::HashMap;
use std::sync::Arc;

use futures::future::join_all;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use summa_proto::proto;
use tokio::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use super::IndexHolder;
use crate::errors::{SummaResult, ValidationError};
use crate::Error;

/// Packed query to a single index
#[derive(Deserialize, Serialize)]
pub struct IndexQuery {
    index_name: String,
    query: proto::Query,
    collectors: Vec<proto::Collector>,
}

/// The main struct responsible for combining different indices and managing their lifetime.
#[derive(Clone, Default, Debug)]
pub struct IndexRegistry {
    index_holders: Arc<RwLock<HashMap<String, Arc<IndexHolder>>>>,
}

impl IndexRegistry {
    /// Read-locked `HashMap` of all indices
    pub async fn index_holders(&self) -> RwLockReadGuard<'_, HashMap<String, Arc<IndexHolder>>> {
        self.index_holders.read().await
    }

    /// Write-locked `HashMap` of all indices
    ///
    /// Taking this lock means locking metadata modification
    pub async fn index_holders_mut(&self) -> RwLockWriteGuard<'_, HashMap<String, Arc<IndexHolder>>> {
        self.index_holders.write().await
    }

    /// Retrieve `IndexHolder` by its name
    pub async fn get_index_holder_by_name(&self, index_name: &str) -> SummaResult<Arc<IndexHolder>> {
        Ok(self
            .index_holders()
            .await
            .get(index_name)
            .ok_or_else(|| Error::Validation(ValidationError::MissingIndex(index_name.to_owned())))?
            .clone())
    }

    /// Add new index to `IndexRegistry`
    pub async fn add(&self, index_holder: IndexHolder) -> Arc<IndexHolder> {
        let index_holder = Arc::new(index_holder);
        self.index_holders_mut()
            .await
            .insert(index_holder.index_name().to_string(), index_holder.clone());
        index_holder
    }

    /// Deletes index from `IndexRegistry`
    pub async fn delete(&self, index_name: &str) {
        self.index_holders_mut().await.remove(index_name);
    }

    /// Searches in several indices simultaneously and merges results
    pub async fn search(&self, index_queries: &[IndexQuery]) -> SummaResult<Vec<proto::CollectorOutput>> {
        let index_holders = self.index_holders().await;
        let futures = index_queries.iter().map(|index_query| {
            let index_holder = index_holders.get(&index_query.index_name).unwrap();
            async move {
                index_holder
                    .search(&index_query.index_name, &index_query.query, &index_query.collectors)
                    .await
                    .unwrap()
            }
        });
        self.merge_responses(&join_all(futures).await)
    }

    /// Merges several `proto::CollectorOutput`
    fn merge_responses(&self, collectors_outputs: &[Vec<proto::CollectorOutput>]) -> SummaResult<Vec<proto::CollectorOutput>> {
        let mut merged_response = vec![];
        for (ix, collector_output) in collectors_outputs.get(0).unwrap().iter().enumerate() {
            merged_response.push(proto::CollectorOutput {
                collector_output: match &collector_output.collector_output {
                    Some(proto::collector_output::CollectorOutput::Aggregation(_)) => todo!(),
                    Some(proto::collector_output::CollectorOutput::Count(_)) => {
                        let counts = collectors_outputs
                            .iter()
                            .map(|collectors_outputs| collectors_outputs[ix].as_count().expect("Expected count collector").count);
                        Some(proto::collector_output::CollectorOutput::Count(proto::CountCollectorOutput {
                            count: counts.sum(),
                        }))
                    }
                    Some(proto::collector_output::CollectorOutput::Facet(_)) => todo!(),
                    Some(proto::collector_output::CollectorOutput::ReservoirSampling(_)) => todo!(),
                    Some(proto::collector_output::CollectorOutput::TopDocs(_)) => {
                        let top_docs = collectors_outputs.iter().map(|collectors_output| collectors_output[ix].as_top_docs().unwrap());
                        let has_next = top_docs.clone().any(|top_docs| top_docs.has_next);
                        let mut scored_documents: Vec<_> = top_docs
                            .map(|top_docs| top_docs.scored_documents.iter())
                            .kmerge_by(|a, b| a > b)
                            .cloned()
                            .collect();
                        for (position, scored_document) in scored_documents.iter_mut().enumerate() {
                            scored_document.position = position as u32;
                        }
                        Some(proto::collector_output::CollectorOutput::TopDocs(proto::TopDocsCollectorOutput {
                            has_next,
                            scored_documents,
                        }))
                    }
                    None => todo!(),
                },
            });
        }
        Ok(merged_response)
    }
}
