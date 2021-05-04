use crate::proto;
use crate::search_engine::custom_serializer::NamedFieldDocument;
use std::collections::HashSet;
use tantivy::collector::{FruitHandle, MultiFruit};
use tantivy::schema::Field;
use tantivy::{DocAddress, LeasedItem, Searcher};

pub trait FruitExtractor: Sync + Send {
    fn extract(self: Box<Self>, mf: &mut MultiFruit, searcher: &LeasedItem<Searcher>, multi_fields: &HashSet<Field>) -> proto::CollectorResult;
}

pub struct TopDocs {
    handle: FruitHandle<Vec<(f32, DocAddress)>>,
    limit: usize,
}

impl TopDocs {
    pub fn new(handle: FruitHandle<Vec<(f32, DocAddress)>>, limit: usize) -> TopDocs {
        TopDocs { handle, limit }
    }
}

impl FruitExtractor for TopDocs {
    fn extract(self: Box<Self>, mf: &mut MultiFruit, searcher: &LeasedItem<Searcher>, multi_fields: &HashSet<Field>) -> proto::CollectorResult {
        let schema = searcher.schema();
        let fruit = self.handle.extract(mf);
        let scored_documents_iter = fruit.iter().enumerate().map(|(position, (score, doc_address))| {
            let document = searcher.doc(*doc_address).unwrap();
            proto::ScoredDocument {
                document: NamedFieldDocument::from_document(schema, multi_fields, &document).to_json(),
                score: *score,
                position: position.try_into().unwrap(),
            }
        });
        let len = scored_documents_iter.len();
        let right_bound = std::cmp::min(self.limit, len);
        let scored_documents = scored_documents_iter.take(right_bound).collect();
        let has_next = len > self.limit;
        proto::CollectorResult {
            collector_result: Some(proto::collector_result::CollectorResult::TopDocs(proto::TopDocsCollectorResult {
                scored_documents,
                has_next,
            })),
        }
    }
}

pub struct ReservoirSampling(pub FruitHandle<Vec<DocAddress>>);

impl FruitExtractor for ReservoirSampling {
    fn extract(self: Box<Self>, mf: &mut MultiFruit, searcher: &LeasedItem<Searcher>, multi_fields: &HashSet<Field>) -> proto::CollectorResult {
        let schema = searcher.schema();
        proto::CollectorResult {
            collector_result: Some(proto::collector_result::CollectorResult::ReservoirSampling(proto::ReservoirSamplingCollectorResult {
                documents: self
                    .0
                    .extract(mf)
                    .iter()
                    .map(|doc_address| NamedFieldDocument::from_document(schema, multi_fields, &searcher.doc(*doc_address).unwrap()).to_json())
                    .collect(),
            })),
        }
    }
}

pub struct Count(pub FruitHandle<usize>);

impl FruitExtractor for Count {
    fn extract(self: Box<Self>, mf: &mut MultiFruit, _searcher: &LeasedItem<Searcher>, _multi_fields: &HashSet<Field>) -> proto::CollectorResult {
        proto::CollectorResult {
            collector_result: Some(proto::collector_result::CollectorResult::Count(proto::CountCollectorResult {
                count: self.0.extract(mf) as u32,
            })),
        }
    }
}
