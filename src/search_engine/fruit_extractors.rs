use crate::errors::{Error, SummaResult};
use crate::proto;
use crate::search_engine::custom_serializer::NamedFieldDocument;
use crate::search_engine::scorers::EvalScorer;
use std::collections::HashSet;
use tantivy::collector::{FruitHandle, MultiCollector, MultiFruit};
use tantivy::schema::{Field, Schema};
use tantivy::{DocAddress, DocId, LeasedItem, Score, Searcher, SegmentReader};

/// Extracts data from `MultiFruit` and moving it to the `proto::CollectorResult`
pub trait FruitExtractor: Sync + Send {
    fn extract(self: Box<Self>, multi_fruit: &mut MultiFruit, searcher: &LeasedItem<Searcher>, multi_fields: &HashSet<Field>) -> proto::CollectorResult;
}

pub fn build_fruit_extractor(
    collector_proto: &proto::Collector,
    schema: &Schema,
    multi_collector: &mut MultiCollector,
) -> SummaResult<Box<dyn FruitExtractor>> {
    match &collector_proto.collector {
        Some(proto::collector::Collector::TopDocs(top_docs_collector_proto)) => Ok(match top_docs_collector_proto.scorer {
            None | Some(proto::Scorer { scorer: None }) => Box::new(TopDocs::new(
                multi_collector.add_collector(
                    tantivy::collector::TopDocs::with_limit(top_docs_collector_proto.limit.try_into().unwrap())
                        .and_offset(top_docs_collector_proto.offset.try_into().unwrap()),
                ),
                top_docs_collector_proto.limit.try_into().unwrap(),
            )) as Box<dyn FruitExtractor>,
            Some(proto::Scorer {
                scorer: Some(proto::scorer::Scorer::EvalExpr(ref eval_expr)),
            }) => {
                let eval_scorer_seed = EvalScorer::new(eval_expr, &schema)?;
                let top_docs_collector = tantivy::collector::TopDocs::with_limit(top_docs_collector_proto.limit.try_into().unwrap())
                    .and_offset(top_docs_collector_proto.offset.try_into().unwrap())
                    .tweak_score(move |segment_reader: &SegmentReader| {
                        let mut eval_scorer = eval_scorer_seed.get_for_segment_reader(segment_reader).unwrap();
                        move |doc_id: DocId, original_score: Score| eval_scorer.score(doc_id, original_score)
                    });
                Box::new(TopDocs::new(
                    multi_collector.add_collector(top_docs_collector),
                    top_docs_collector_proto.limit.try_into().unwrap(),
                )) as Box<dyn FruitExtractor>
            }
            Some(proto::Scorer {
                scorer: Some(proto::scorer::Scorer::OrderBy(ref field_name)),
            }) => {
                let field = schema.get_field(field_name).ok_or(Error::FieldDoesNotExistError(field_name.to_owned()))?;
                let top_docs_collector = tantivy::collector::TopDocs::with_limit(top_docs_collector_proto.limit.try_into().unwrap())
                    .and_offset(top_docs_collector_proto.offset.try_into().unwrap())
                    .order_by_u64_field(field);
                Box::new(TopDocs::new(
                    multi_collector.add_collector(top_docs_collector),
                    top_docs_collector_proto.limit.try_into().unwrap(),
                )) as Box<dyn FruitExtractor>
            }
        }),
        Some(proto::collector::Collector::ReservoirSampling(reservoir_sampling_collector_proto)) => {
            let reservoir_sampling_collector: crate::search_engine::collectors::ReservoirSampling = reservoir_sampling_collector_proto.into();
            Ok(Box::new(ReservoirSampling(multi_collector.add_collector(reservoir_sampling_collector))) as Box<dyn FruitExtractor>)
        }
        Some(proto::collector::Collector::Count(count_collector_proto)) => {
            let count_collector: tantivy::collector::Count = count_collector_proto.into();
            Ok(Box::new(Count(multi_collector.add_collector(count_collector))) as Box<dyn FruitExtractor>)
        }
        None => Ok(Box::new(Count(multi_collector.add_collector(tantivy::collector::Count))) as Box<dyn FruitExtractor>),
    }
}

pub struct TopDocs<T: 'static + Copy + Into<proto::Score> + Sync + Send> {
    handle: FruitHandle<Vec<(T, DocAddress)>>,
    limit: usize,
}

impl<T: 'static + Copy + Into<proto::Score> + Sync + Send> TopDocs<T> {
    pub fn new(handle: FruitHandle<Vec<(T, DocAddress)>>, limit: usize) -> TopDocs<T> {
        TopDocs { handle, limit }
    }
}

impl<T: 'static + Copy + Into<proto::Score> + Sync + Send> FruitExtractor for TopDocs<T> {
    fn extract(self: Box<Self>, multi_fruit: &mut MultiFruit, searcher: &LeasedItem<Searcher>, multi_fields: &HashSet<Field>) -> proto::CollectorResult {
        let schema = searcher.schema();
        let fruit = self.handle.extract(multi_fruit);
        let scored_documents_iter = fruit.iter().enumerate().map(|(position, (score, doc_address))| {
            let document = searcher.doc(*doc_address).unwrap();
            proto::ScoredDocument {
                document: NamedFieldDocument::from_document(schema, multi_fields, &document).to_json(),
                score: Some((*score).into()),
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
    fn extract(self: Box<Self>, multi_fruit: &mut MultiFruit, searcher: &LeasedItem<Searcher>, multi_fields: &HashSet<Field>) -> proto::CollectorResult {
        let schema = searcher.schema();
        proto::CollectorResult {
            collector_result: Some(proto::collector_result::CollectorResult::ReservoirSampling(
                proto::ReservoirSamplingCollectorResult {
                    documents: self
                        .0
                        .extract(multi_fruit)
                        .iter()
                        .map(|doc_address| NamedFieldDocument::from_document(schema, multi_fields, &searcher.doc(*doc_address).unwrap()).to_json())
                        .collect(),
                },
            )),
        }
    }
}

pub struct Count(pub FruitHandle<usize>);

impl FruitExtractor for Count {
    fn extract(self: Box<Self>, multi_fruit: &mut MultiFruit, _searcher: &LeasedItem<Searcher>, _multi_fields: &HashSet<Field>) -> proto::CollectorResult {
        proto::CollectorResult {
            collector_result: Some(proto::collector_result::CollectorResult::Count(proto::CountCollectorResult {
                count: self.0.extract(multi_fruit) as u32,
            })),
        }
    }
}
