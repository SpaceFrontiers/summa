use izihawa_ttl_cache::TtlCache;
use summa_proto::proto;

use crate::components::IntermediateExtractionResult;
use crate::configs::core::CollectorCacheConfig;

const BLOCK_SIZE: u32 = 100;

pub struct CollectorCache {
    cache: TtlCache<String, IntermediateExtractionResult>,
    ttl_interval_ms: instant::Duration,
}

impl CollectorCache {
    pub fn new(config: &CollectorCacheConfig) -> Self {
        CollectorCache {
            ttl_interval_ms: instant::Duration::from_millis(config.ttl_interval_ms.unwrap_or(120000)),
            cache: TtlCache::new(config.size),
        }
    }

    pub fn remove_expired(&mut self) {
        self.cache.remove_expired();
    }

    pub fn caching_key(&self, caching_key: &str, collector: &proto::Collector) -> String {
        match collector {
            proto::Collector {
                collector: Some(proto::collector::Collector::TopDocs(top_docs)),
            } => {
                format!(
                    "{}|TopDocsCollector {{ limit={}, offset={}, scorer={:?} }}|",
                    caching_key, top_docs.limit, top_docs.offset, top_docs.scorer
                )
            }
            other => format!("{}|{:?}", caching_key, other),
        }
    }

    pub fn is_caching_enabled(collector: &proto::Collector) -> bool {
        match collector {
            proto::Collector {
                collector: Some(proto::collector::Collector::TopDocs(top_docs)),
            } => {
                let left_bound = top_docs.offset;
                let right_bound = left_bound + top_docs.limit;

                let left_block_bound = top_docs.offset - top_docs.offset % BLOCK_SIZE;
                let right_block_bound = left_block_bound + BLOCK_SIZE;

                left_block_bound <= left_bound && right_bound <= right_block_bound
            }
            proto::Collector {
                collector: Some(proto::collector::Collector::ReservoirSampling(_)),
            } => false,
            _ => true,
        }
    }

    pub fn adjust_collector(collector: &proto::Collector) -> proto::Collector {
        match collector {
            proto::Collector {
                collector: Some(proto::collector::Collector::TopDocs(top_docs)),
            } => {
                let mut top_docs = top_docs.clone();
                top_docs.offset -= top_docs.offset % BLOCK_SIZE;
                top_docs.limit = BLOCK_SIZE;
                proto::Collector {
                    collector: Some(proto::collector::Collector::TopDocs(top_docs)),
                }
            }
            other => other.clone(),
        }
    }

    pub fn adjust_result(result: &IntermediateExtractionResult, collector: &proto::Collector) -> IntermediateExtractionResult {
        match result {
            IntermediateExtractionResult::Ready(r) => IntermediateExtractionResult::Ready(r.clone()),
            IntermediateExtractionResult::PreparedDocumentReferences(prepared_document_references) => {
                let mut prepared_document_references = prepared_document_references.clone();
                match &collector.collector {
                    Some(proto::collector::Collector::TopDocs(top_docs_collector)) => {
                        prepared_document_references.has_next = prepared_document_references.has_next
                            || prepared_document_references.scored_doc_addresses.len()
                                > ((top_docs_collector.offset + top_docs_collector.limit) % BLOCK_SIZE) as usize;
                        prepared_document_references.scored_doc_addresses = prepared_document_references
                            .scored_doc_addresses
                            .into_iter()
                            .skip((top_docs_collector.offset % BLOCK_SIZE) as usize)
                            .take((((top_docs_collector.limit - 1) % BLOCK_SIZE) + 1) as usize)
                            .collect()
                    }
                    _other => {}
                }
                IntermediateExtractionResult::PreparedDocumentReferences(prepared_document_references)
            }
        }
    }

    pub fn get(&mut self, caching_key: &str, adjusted_collector: &proto::Collector, collector: &proto::Collector) -> Option<IntermediateExtractionResult> {
        let caching_key = self.caching_key(caching_key, adjusted_collector);
        self.cache.get(&caching_key).map(|cache_value| Self::adjust_result(cache_value, collector))
    }

    pub fn put(&mut self, caching_key: &str, collector: &proto::Collector, value: IntermediateExtractionResult) {
        let caching_key = self.caching_key(caching_key, collector);
        self.cache.insert(caching_key.clone(), value.clone(), self.ttl_interval_ms);
    }
}
