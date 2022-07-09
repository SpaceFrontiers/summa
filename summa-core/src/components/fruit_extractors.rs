use std::collections::{HashMap, HashSet};

use rustc_hash::FxHashMap;
use summa_proto::proto;
use tantivy::aggregation::agg_result::AggregationResults;
use tantivy::collector::{FacetCounts, FruitHandle, MultiCollector, MultiFruit};
use tantivy::query::Query;
use tantivy::schema::{Field, FieldType};
use tantivy::{Order, Searcher};

use crate::components::snippet_generator::SnippetGeneratorConfig;
use crate::components::IndexHolder;
use crate::errors::{BuilderError, Error, SummaResult};
use crate::proto_traits::Wrapper;
use crate::scorers::eval_scorer_tweaker::EvalScorerTweaker;
use crate::scorers::EvalScorer;
use crate::{collectors, validators};

pub struct ScoredDocAddress {
    pub doc_address: tantivy::DocAddress,
    pub score: Option<proto::Score>,
}

#[derive(Clone)]
pub struct ExtractionTooling {
    pub searcher: Searcher,
    pub query_fields: Option<HashSet<Field>>,
    pub multi_fields: HashSet<Field>,
}

impl ExtractionTooling {
    pub fn new(searcher: Searcher, query_fields: Option<HashSet<Field>>, multi_fields: HashSet<Field>) -> ExtractionTooling {
        ExtractionTooling {
            searcher,
            query_fields,
            multi_fields,
        }
    }
}

pub struct PreparedDocumentReferences {
    pub index_alias: String,
    pub extraction_tooling: ExtractionTooling,
    pub snippet_generator_config: Option<SnippetGeneratorConfig>,
    pub scored_doc_addresses: Vec<ScoredDocAddress>,
    pub has_next: bool,
    pub limit: u32,
    pub offset: u32,
}

pub enum ReadyCollectorOutput {
    Aggregation(proto::AggregationCollectorOutput),
    Count(proto::CountCollectorOutput),
    Facet(proto::FacetCollectorOutput),
}

pub enum IntermediateExtractionResult {
    Ready(ReadyCollectorOutput),
    PreparedDocumentReferences(PreparedDocumentReferences),
}

impl IntermediateExtractionResult {
    pub fn as_document_references(self) -> Option<PreparedDocumentReferences> {
        if let IntermediateExtractionResult::PreparedDocumentReferences(document_references) = self {
            Some(document_references)
        } else {
            None
        }
    }

    pub fn as_count(&self) -> Option<&proto::CountCollectorOutput> {
        if let IntermediateExtractionResult::Ready(ReadyCollectorOutput::Count(count)) = &self {
            Some(count)
        } else {
            None
        }
    }
}

/// Extracts data from `MultiFruit` and moving it to the `proto::CollectorOutput`
#[async_trait]
pub trait FruitExtractor: Sync + Send {
    fn extract(self: Box<Self>, multi_fruit: &mut MultiFruit) -> SummaResult<IntermediateExtractionResult>;
}

pub fn parse_aggregations(aggregations: HashMap<String, proto::Aggregation>) -> SummaResult<HashMap<String, tantivy::aggregation::agg_req::Aggregation>> {
    aggregations
        .into_iter()
        .map(|(name, aggregation)| {
            let aggregation = match aggregation.aggregation {
                None => Err(Error::InvalidAggregation),
                Some(aggregation) => Wrapper::from(aggregation).try_into(),
            }?;
            Ok((name, aggregation))
        })
        .collect::<SummaResult<HashMap<_, _>>>()
}

fn parse_aggregation_results(
    aggregation_results: FxHashMap<String, tantivy::aggregation::agg_result::AggregationResult>,
) -> HashMap<String, proto::AggregationResult> {
    aggregation_results
        .into_iter()
        .map(|(name, aggregation_result)| (name, Wrapper::from(aggregation_result).into_inner()))
        .collect()
}

pub fn build_fruit_extractor(
    index_holder: &IndexHolder,
    index_alias: &str,
    searcher: Searcher,
    collector_proto: proto::Collector,
    query: &dyn Query,
    multi_collector: &mut MultiCollector,
) -> SummaResult<Box<dyn FruitExtractor>> {
    match collector_proto.collector {
        Some(proto::collector::Collector::TopDocs(top_docs_collector_proto)) => {
            let query_fields = validators::parse_fields(searcher.schema(), &top_docs_collector_proto.fields)?;
            let query_fields = (!query_fields.is_empty()).then(|| HashSet::from_iter(query_fields.into_iter().map(|x| x.0)));
            Ok(match top_docs_collector_proto.scorer {
                None | Some(proto::Scorer { scorer: None }) => Box::new(
                    TopDocsBuilder::default()
                        .handle(multi_collector.add_collector(tantivy::collector::TopDocs::with_limit(
                            (top_docs_collector_proto.offset + top_docs_collector_proto.limit + 1) as usize,
                        )))
                        .index_alias(index_alias.to_string())
                        .searcher(searcher)
                        .query(query.box_clone())
                        .limit(top_docs_collector_proto.limit)
                        .offset(top_docs_collector_proto.offset)
                        .snippet_configs(top_docs_collector_proto.snippet_configs)
                        .multi_fields(index_holder.multi_fields().clone())
                        .query_fields(query_fields)
                        .build()?,
                ) as Box<dyn FruitExtractor>,
                Some(proto::Scorer {
                    scorer: Some(proto::scorer::Scorer::EvalExpr(ref eval_expr)),
                }) => {
                    let eval_scorer_seed = EvalScorer::new(eval_expr, searcher.schema())?;
                    let top_docs_collector =
                        tantivy::collector::TopDocs::with_limit((top_docs_collector_proto.offset + top_docs_collector_proto.limit + 1) as usize)
                            .tweak_score(EvalScorerTweaker::new(eval_scorer_seed));
                    Box::new(
                        TopDocsBuilder::default()
                            .handle(multi_collector.add_collector(top_docs_collector))
                            .index_alias(index_alias.to_string())
                            .searcher(searcher)
                            .query(query.box_clone())
                            .limit(top_docs_collector_proto.limit)
                            .offset(top_docs_collector_proto.offset)
                            .snippet_configs(top_docs_collector_proto.snippet_configs)
                            .multi_fields(index_holder.multi_fields().clone())
                            .query_fields(query_fields)
                            .build()?,
                    ) as Box<dyn FruitExtractor>
                }
                Some(proto::Scorer {
                    scorer: Some(proto::scorer::Scorer::OrderBy(field_name)),
                }) => {
                    let top_docs_collector =
                        tantivy::collector::TopDocs::with_limit((top_docs_collector_proto.offset + top_docs_collector_proto.limit + 1) as usize)
                            .order_by_fast_field(field_name, Order::Desc);
                    Box::<TopDocs<u64>>::new(
                        TopDocsBuilder::default()
                            .handle(multi_collector.add_collector(top_docs_collector))
                            .index_alias(index_alias.to_string())
                            .searcher(searcher)
                            .query(query.box_clone())
                            .limit(top_docs_collector_proto.limit)
                            .offset(top_docs_collector_proto.offset)
                            .snippet_configs(top_docs_collector_proto.snippet_configs)
                            .multi_fields(index_holder.multi_fields().clone())
                            .query_fields(query_fields)
                            .build()?,
                    ) as Box<dyn FruitExtractor>
                }
            })
        }
        Some(proto::collector::Collector::ReservoirSampling(reservoir_sampling_collector_proto)) => {
            let query_fields = validators::parse_fields(searcher.schema(), &reservoir_sampling_collector_proto.fields)?;
            let query_fields = (!query_fields.is_empty()).then(|| HashSet::from_iter(query_fields.into_iter().map(|x| x.0)));
            let reservoir_sampling_collector = collectors::ReservoirSampling::with_limit(reservoir_sampling_collector_proto.limit as usize);
            Ok(Box::new(
                ReservoirSamplingBuilder::default()
                    .handle(multi_collector.add_collector(reservoir_sampling_collector))
                    .index_alias(index_alias.to_string())
                    .searcher(searcher)
                    .multi_fields(index_holder.multi_fields().clone())
                    .query_fields(query_fields)
                    .limit(reservoir_sampling_collector_proto.limit)
                    .build()?,
            ) as Box<dyn FruitExtractor>)
        }
        Some(proto::collector::Collector::Count(_)) => Ok(Box::new(Count(multi_collector.add_collector(tantivy::collector::Count))) as Box<dyn FruitExtractor>),
        Some(proto::collector::Collector::Facet(facet_collector_proto)) => {
            let mut facet_collector = tantivy::collector::FacetCollector::for_field(facet_collector_proto.field);
            for facet in &facet_collector_proto.facets {
                facet_collector.add_facet(facet);
            }
            Ok(Box::new(Facet(multi_collector.add_collector(facet_collector))) as Box<dyn FruitExtractor>)
        }
        Some(proto::collector::Collector::Aggregation(aggregation_collector_proto)) => {
            let aggregation_collector =
                tantivy::aggregation::AggregationCollector::from_aggs(parse_aggregations(aggregation_collector_proto.aggregations)?, Default::default());
            Ok(Box::new(Aggregation(multi_collector.add_collector(aggregation_collector))) as Box<dyn FruitExtractor>)
        }
        None => Ok(Box::new(Count(multi_collector.add_collector(tantivy::collector::Count))) as Box<dyn FruitExtractor>),
    }
}

#[derive(Builder)]
#[builder(pattern = "owned", build_fn(error = "BuilderError"))]
pub struct TopDocs<T: 'static + Copy + Into<proto::Score> + Sync + Send> {
    searcher: Searcher,
    index_alias: String,
    handle: FruitHandle<Vec<(T, tantivy::DocAddress)>>,
    limit: u32,
    offset: u32,
    snippet_configs: HashMap<String, u32>,
    query: Box<dyn Query>,
    #[builder(default = "None")]
    query_fields: Option<HashSet<Field>>,
    multi_fields: HashSet<Field>,
}

#[async_trait]
impl<T: 'static + Copy + Into<proto::Score> + Sync + Send> FruitExtractor for TopDocs<T> {
    fn extract(self: Box<Self>, multi_fruit: &mut MultiFruit) -> SummaResult<IntermediateExtractionResult> {
        let fruit = self.handle.extract(multi_fruit);
        let length = fruit.len();
        let doc_addresses = fruit
            .into_iter()
            .take(std::cmp::min((self.offset + self.limit) as usize, length))
            .map(|(score, doc_address)| ScoredDocAddress {
                doc_address,
                score: Some(score.into()),
            })
            .collect();
        Ok(IntermediateExtractionResult::PreparedDocumentReferences(PreparedDocumentReferences {
            index_alias: self.index_alias,
            extraction_tooling: ExtractionTooling::new(self.searcher.clone(), self.query_fields, self.multi_fields),
            snippet_generator_config: Some(SnippetGeneratorConfig::new(self.searcher, self.query, self.snippet_configs)),
            scored_doc_addresses: doc_addresses,
            has_next: length > (self.offset + self.limit) as usize,
            limit: self.limit,
            offset: self.offset,
        }))
    }
}

#[derive(Builder)]
#[builder(pattern = "owned", build_fn(error = "BuilderError"))]
pub struct ReservoirSampling {
    searcher: Searcher,
    index_alias: String,
    handle: FruitHandle<Vec<tantivy::DocAddress>>,
    #[builder(default = "None")]
    query_fields: Option<HashSet<Field>>,
    multi_fields: HashSet<Field>,
    limit: u32,
}

impl FruitExtractor for ReservoirSampling {
    fn extract(self: Box<Self>, multi_fruit: &mut MultiFruit) -> SummaResult<IntermediateExtractionResult> {
        Ok(IntermediateExtractionResult::PreparedDocumentReferences(PreparedDocumentReferences {
            scored_doc_addresses: self
                .handle
                .extract(multi_fruit)
                .into_iter()
                .map(|doc_address| ScoredDocAddress { doc_address, score: None })
                .collect(),
            index_alias: self.index_alias,
            has_next: false,
            limit: self.limit,
            extraction_tooling: ExtractionTooling::new(self.searcher, self.query_fields, self.multi_fields),
            snippet_generator_config: None,
            offset: 0,
        }))
    }
}

pub struct Count(pub FruitHandle<usize>);

impl FruitExtractor for Count {
    fn extract(self: Box<Self>, multi_fruit: &mut MultiFruit) -> SummaResult<IntermediateExtractionResult> {
        Ok(IntermediateExtractionResult::Ready(ReadyCollectorOutput::Count(proto::CountCollectorOutput {
            count: self.0.extract(multi_fruit) as u32,
        })))
    }
}

pub struct Facet(pub FruitHandle<FacetCounts>);

impl FruitExtractor for Facet {
    fn extract(self: Box<Self>, multi_fruit: &mut MultiFruit) -> SummaResult<IntermediateExtractionResult> {
        Ok(IntermediateExtractionResult::Ready(ReadyCollectorOutput::Facet(proto::FacetCollectorOutput {
            facet_counts: self.0.extract(multi_fruit).get("").map(|(facet, count)| (facet.to_string(), count)).collect(),
        })))
    }
}

pub struct Aggregation(pub FruitHandle<AggregationResults>);

impl FruitExtractor for Aggregation {
    fn extract(self: Box<Self>, multi_fruit: &mut MultiFruit) -> SummaResult<IntermediateExtractionResult> {
        Ok(IntermediateExtractionResult::Ready(ReadyCollectorOutput::Aggregation(
            proto::AggregationCollectorOutput {
                aggregation_results: parse_aggregation_results(self.0.extract(multi_fruit).0),
            },
        )))
    }
}
