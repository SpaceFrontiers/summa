use super::IndexHolder;
use crate::errors::{SummaResult, ValidationError};
use crate::Error;
use futures::future::join_all;
use itertools::Itertools;
use std::collections::HashMap;
use std::sync::Arc;
use summa_proto::proto;
use tokio::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

/// The main struct responsible for indices lifecycle. Here lives indices creation and deletion as well as committing and indexing new documents.
#[derive(Clone, Default, Debug)]
pub struct IndexRegistry {
    index_holders: Arc<RwLock<HashMap<String, Arc<IndexHolder>>>>,
}

impl IndexRegistry {
    /// Read-locked `HashMap` of all indices
    async fn index_holders(&self) -> RwLockReadGuard<'_, HashMap<String, Arc<IndexHolder>>> {
        self.index_holders.read().await
    }

    /// Write-locked `HashMap` of all indices
    ///
    /// Taking this lock means locking metadata modification
    async fn index_holders_mut(&self) -> RwLockWriteGuard<'_, HashMap<String, Arc<IndexHolder>>> {
        self.index_holders.write().await
    }

    pub async fn get_index_holder_by_name(&self, index_name: &str) -> SummaResult<Arc<IndexHolder>> {
        Ok(self
            .index_holders()
            .await
            .get(index_name)
            .ok_or_else(|| Error::Validation(ValidationError::MissingIndex(index_name.to_owned())))?
            .clone())
    }

    pub async fn add(&mut self, index_holder: IndexHolder) {
        self.index_holders
            .write()
            .await
            .insert(index_holder.index_name().to_string(), Arc::new(index_holder));
    }

    pub async fn delete(&mut self, index_name: &str) {
        self.index_holders.write().await.remove(index_name);
    }

    pub async fn search<TIndexName: AsRef<str>>(
        &self,
        index_names: &[TIndexName],
        query: &proto::Query,
        collectors: Vec<proto::Collector>,
    ) -> SummaResult<Vec<summa_proto::proto::CollectorOutput>> {
        let index_holders = self.index_holders.read().await;
        let search_futures = index_names
            .iter()
            .map(|index_name| {
                index_holders
                    .get(index_name.as_ref())
                    .ok_or_else(|| ValidationError::MissingIndex(index_name.as_ref().to_string()).into())
                    .map(|index| async { index.search(index_name.as_ref(), query, collectors.clone()).await.unwrap() })
            })
            .collect::<SummaResult<Vec<_>>>()?;
        Ok(self.merge_responses(&join_all(search_futures).await)?)
    }

    fn merge_responses(&self, collectors_outputs: &Vec<Vec<proto::CollectorOutput>>) -> SummaResult<Vec<proto::CollectorOutput>> {
        let mut merged_response = vec![];
        for (ix, collector_output) in collectors_outputs.iter().nth(0).unwrap().iter().enumerate() {
            merged_response.push(proto::CollectorOutput {
                collector_output: match &collector_output.collector_output {
                    Some(proto::collector_output::CollectorOutput::Aggregation(_)) => todo!(),
                    Some(proto::collector_output::CollectorOutput::Count(_)) => {
                        let counts = collectors_outputs
                            .iter()
                            .map(|collectors_outputs| collectors_outputs[ix].as_count().unwrap().count);
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
                            (*scored_document).position = position as u32;
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
