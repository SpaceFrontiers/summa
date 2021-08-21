use crate::errors::ValidationError::InvalidAggregation;
use crate::errors::{Error, SummaResult, ValidationError};
use crate::proto;
use crate::search_engine::custom_serializer::NamedFieldDocument;
use crate::search_engine::scorers::EvalScorer;
use std::collections::{HashMap, HashSet};
use tantivy::aggregation::agg_result::AggregationResults;
use tantivy::collector::{FacetCounts, FruitHandle, MultiCollector, MultiFruit};
use tantivy::query::Query;
use tantivy::schema::{Field, Schema as Fields};
use tantivy::{DocAddress, DocId, Score, Searcher, SegmentReader, SnippetGenerator};

/// Extracts data from `MultiFruit` and moving it to the `proto::CollectorOutput`
pub trait FruitExtractor: Sync + Send {
    fn extract(
        self: Box<Self>,
        external_index_alias: &str,
        multi_fruit: &mut MultiFruit,
        searcher: &Searcher,
        multi_fields: &HashSet<Field>,
    ) -> proto::CollectorOutput;
}

pub fn parse_aggregations(aggregations: HashMap<String, proto::Aggregation>) -> SummaResult<HashMap<String, tantivy::aggregation::agg_req::Aggregation>> {
    let aggregations = aggregations
        .into_iter()
        .map(|(name, aggregation)| {
            let aggregation = match aggregation.aggregation {
                None => Err(Error::Validation(InvalidAggregation)),
                Some(a) => a.try_into(),
            }?;
            Ok((name, aggregation))
        })
        .collect::<SummaResult<Vec<_>>>()?;
    Ok(HashMap::from_iter(aggregations.into_iter()))
}

fn parse_aggregation_results(
    aggregation_results: HashMap<String, tantivy::aggregation::agg_result::AggregationResult>,
) -> HashMap<String, proto::AggregationResult> {
    aggregation_results
        .into_iter()
        .map(|(name, aggregation_result)| (name, aggregation_result.into()))
        .collect()
}

fn check_snippets(fields: &Fields, snippets: &HashMap<String, u32>) -> SummaResult<()> {
    for snippet_field in snippets.keys() {
        if fields.get_field(snippet_field).is_none() {
            return Err(ValidationError::MissingSnippetField(snippet_field.to_string()).into());
        }
    }
    Ok(())
}

pub fn build_fruit_extractor(
    collector_proto: proto::Collector,
    query: &dyn Query,
    fields: &Fields,
    multi_collector: &mut MultiCollector,
) -> SummaResult<Box<dyn FruitExtractor>> {
    match collector_proto.collector {
        Some(proto::collector::Collector::TopDocs(top_docs_collector_proto)) => {
            check_snippets(fields, &top_docs_collector_proto.snippets)?;
            Ok(match top_docs_collector_proto.scorer {
                None | Some(proto::Scorer { scorer: None }) => Box::new(
                    TopDocsBuilder::default()
                        .handle(
                            multi_collector.add_collector(
                                tantivy::collector::TopDocs::with_limit(top_docs_collector_proto.limit.try_into().unwrap())
                                    .and_offset(top_docs_collector_proto.offset.try_into().unwrap()),
                            ),
                        )
                        .query(query.box_clone())
                        .explain(top_docs_collector_proto.explain)
                        .limit(top_docs_collector_proto.limit)
                        .snippets(top_docs_collector_proto.snippets)
                        .build()?,
                ) as Box<dyn FruitExtractor>,
                Some(proto::Scorer {
                    scorer: Some(proto::scorer::Scorer::EvalExpr(ref eval_expr)),
                }) => {
                    let eval_scorer_seed = EvalScorer::new(eval_expr, fields)?;
                    let top_docs_collector = tantivy::collector::TopDocs::with_limit(top_docs_collector_proto.limit.try_into().unwrap())
                        .and_offset(top_docs_collector_proto.offset.try_into().unwrap())
                        .tweak_score(move |segment_reader: &SegmentReader| {
                            let mut eval_scorer = eval_scorer_seed.get_for_segment_reader(segment_reader).unwrap();
                            move |doc_id: DocId, original_score: Score| eval_scorer.score(doc_id, original_score)
                        });
                    Box::new(
                        TopDocsBuilder::default()
                            .handle(multi_collector.add_collector(top_docs_collector))
                            .query(query.box_clone())
                            .explain(top_docs_collector_proto.explain)
                            .limit(top_docs_collector_proto.limit)
                            .snippets(top_docs_collector_proto.snippets)
                            .build()?,
                    ) as Box<dyn FruitExtractor>
                }
                Some(proto::Scorer {
                    scorer: Some(proto::scorer::Scorer::OrderBy(ref field_name)),
                }) => {
                    let field = fields.get_field(field_name).ok_or_else(|| Error::FieldDoesNotExist(field_name.to_owned()))?;
                    let top_docs_collector = tantivy::collector::TopDocs::with_limit(top_docs_collector_proto.limit.try_into().unwrap())
                        .and_offset(top_docs_collector_proto.offset.try_into().unwrap())
                        .order_by_u64_field(field);
                    Box::new(
                        TopDocsBuilder::default()
                            .handle(multi_collector.add_collector(top_docs_collector))
                            .query(query.box_clone())
                            .explain(top_docs_collector_proto.explain)
                            .limit(top_docs_collector_proto.limit)
                            .snippets(top_docs_collector_proto.snippets)
                            .build()?,
                    ) as Box<dyn FruitExtractor>
                }
            })
        }
        Some(proto::collector::Collector::ReservoirSampling(reservoir_sampling_collector_proto)) => {
            let reservoir_sampling_collector: crate::search_engine::collectors::ReservoirSampling = reservoir_sampling_collector_proto.into();
            Ok(Box::new(ReservoirSampling(multi_collector.add_collector(reservoir_sampling_collector))) as Box<dyn FruitExtractor>)
        }
        Some(proto::collector::Collector::Count(_)) => Ok(Box::new(Count(multi_collector.add_collector(tantivy::collector::Count))) as Box<dyn FruitExtractor>),
        Some(proto::collector::Collector::Facet(facet_collector_proto)) => {
            let field = fields
                .get_field(&facet_collector_proto.field)
                .ok_or_else(|| Error::FieldDoesNotExist(facet_collector_proto.field.to_owned()))?;
            let mut facet_collector = tantivy::collector::FacetCollector::for_field(field);
            for facet in &facet_collector_proto.facets {
                facet_collector.add_facet(facet);
            }
            Ok(Box::new(Facet(multi_collector.add_collector(facet_collector))) as Box<dyn FruitExtractor>)
        }
        Some(proto::collector::Collector::Aggregation(aggregation_collector_proto)) => {
            let aggregation_collector =
                tantivy::aggregation::AggregationCollector::from_aggs(parse_aggregations(aggregation_collector_proto.aggregations)?, None);
            Ok(Box::new(Aggregation(multi_collector.add_collector(aggregation_collector))) as Box<dyn FruitExtractor>)
        }
        None => Ok(Box::new(Count(multi_collector.add_collector(tantivy::collector::Count))) as Box<dyn FruitExtractor>),
    }
}

#[derive(Builder)]
#[builder(pattern = "owned", build_fn(error = "ValidationError"))]
pub struct TopDocs<T: 'static + Copy + Into<proto::Score> + Sync + Send> {
    handle: FruitHandle<Vec<(T, DocAddress)>>,
    limit: u32,
    snippets: HashMap<String, u32>,
    query: Box<dyn Query>,
    #[builder(default = "false")]
    explain: bool,
}

impl<T: 'static + Copy + Into<proto::Score> + Sync + Send> TopDocs<T> {
    fn snippet_generators(&self, searcher: &Searcher) -> HashMap<String, SnippetGenerator> {
        self.snippets
            .iter()
            .map(|(field_name, max_num_chars)| {
                let snippet_field = searcher.schema().get_field(field_name).unwrap();
                let mut snippet_generator = SnippetGenerator::create(searcher, &*self.query, snippet_field).unwrap();
                snippet_generator.set_max_num_chars(*max_num_chars as usize);
                (field_name.to_string(), snippet_generator)
            })
            .collect()
    }
}

impl<T: 'static + Copy + Into<proto::Score> + Sync + Send> FruitExtractor for TopDocs<T> {
    fn extract(
        self: Box<Self>,
        external_index_alias: &str,
        multi_fruit: &mut MultiFruit,
        searcher: &Searcher,
        multi_fields: &HashSet<Field>,
    ) -> proto::CollectorOutput {
        let fields = searcher.schema();
        let snippet_generators = self.snippet_generators(searcher);
        let fruit = self.handle.extract(multi_fruit);
        let scored_documents_iter = fruit.iter().enumerate().map(|(position, (score, doc_address))| {
            let document = searcher.doc(*doc_address).unwrap();
            proto::ScoredDocument {
                document: NamedFieldDocument::from_document(fields, multi_fields, &document).to_json(),
                score: Some((*score).into()),
                position: position as u32,
                snippets: snippet_generators
                    .iter()
                    .map(|(field_name, snippet_generator)| (field_name.to_string(), snippet_generator.snippet_from_doc(&document).into()))
                    .collect(),
                index_alias: external_index_alias.to_string(),
            }
        });
        let len = scored_documents_iter.len();
        let scored_documents = scored_documents_iter.take(std::cmp::min(self.limit as usize, len)).collect();
        let has_next = len > self.limit as usize;
        proto::CollectorOutput {
            collector_output: Some(proto::collector_output::CollectorOutput::TopDocs(proto::TopDocsCollectorOutput {
                scored_documents,
                has_next,
            })),
        }
    }
}

pub struct ReservoirSampling(pub FruitHandle<Vec<DocAddress>>);

impl FruitExtractor for ReservoirSampling {
    fn extract(
        self: Box<Self>,
        external_index_alias: &str,
        multi_fruit: &mut MultiFruit,
        searcher: &Searcher,
        multi_fields: &HashSet<Field>,
    ) -> proto::CollectorOutput {
        let fields = searcher.schema();
        proto::CollectorOutput {
            collector_output: Some(proto::collector_output::CollectorOutput::ReservoirSampling(
                proto::ReservoirSamplingCollectorOutput {
                    random_documents: self
                        .0
                        .extract(multi_fruit)
                        .iter()
                        .map(|doc_address| proto::RandomDocument {
                            index_alias: external_index_alias.to_string(),
                            document: NamedFieldDocument::from_document(fields, multi_fields, &searcher.doc(*doc_address).unwrap()).to_json(),
                            score: Some((1.0).into()),
                        })
                        .collect(),
                },
            )),
        }
    }
}

pub struct Count(pub FruitHandle<usize>);

impl FruitExtractor for Count {
    fn extract(
        self: Box<Self>,
        _external_index_alias: &str,
        multi_fruit: &mut MultiFruit,
        _searcher: &Searcher,
        _multi_fields: &HashSet<Field>,
    ) -> proto::CollectorOutput {
        proto::CollectorOutput {
            collector_output: Some(proto::collector_output::CollectorOutput::Count(proto::CountCollectorOutput {
                count: self.0.extract(multi_fruit) as u32,
            })),
        }
    }
}

pub struct Facet(pub FruitHandle<FacetCounts>);

impl FruitExtractor for Facet {
    fn extract(
        self: Box<Self>,
        _external_index_alias: &str,
        multi_fruit: &mut MultiFruit,
        _searcher: &Searcher,
        _multi_fields: &HashSet<Field>,
    ) -> proto::CollectorOutput {
        proto::CollectorOutput {
            collector_output: Some(proto::collector_output::CollectorOutput::Facet(proto::FacetCollectorOutput {
                facet_counts: self.0.extract(multi_fruit).get("").map(|(facet, count)| (facet.to_string(), count)).collect(),
            })),
        }
    }
}

pub struct Aggregation(pub FruitHandle<AggregationResults>);

impl FruitExtractor for Aggregation {
    fn extract(
        self: Box<Self>,
        _external_index_alias: &str,
        multi_fruit: &mut MultiFruit,
        _searcher: &Searcher,
        _multi_fields: &HashSet<Field>,
    ) -> proto::CollectorOutput {
        proto::CollectorOutput {
            collector_output: Some(proto::collector_output::CollectorOutput::Aggregation(proto::AggregationCollectorOutput {
                aggregation_results: parse_aggregation_results(self.0.extract(multi_fruit).0),
            })),
        }
    }
}
