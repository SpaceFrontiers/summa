use crate::collectors;
use crate::custom_serializer::NamedFieldDocument;
use crate::errors::{Error, SummaResult, ValidationError};
use crate::scorers::EvalScorer;
use futures::future::join_all;
use std::collections::{HashMap, HashSet};
use summa_proto::proto;
use tantivy::aggregation::agg_result::AggregationResults;
use tantivy::collector::{FacetCounts, FruitHandle, MultiCollector, MultiFruit};
use tantivy::query::Query;
use tantivy::schema::{Field, Schema};
use tantivy::{DocAddress, DocId, Score, Searcher, SegmentReader, SnippetGenerator};

/// Extracts data from `MultiFruit` and moving it to the `proto::CollectorOutput`
#[async_trait]
pub trait FruitExtractor: Sync + Send {
    fn extract(
        self: Box<Self>,
        external_index_alias: &str,
        multi_fruit: &mut MultiFruit,
        searcher: &Searcher,
        multi_fields: &HashSet<Field>,
    ) -> proto::CollectorOutput;

    async fn extract_async(
        self: Box<Self>,
        external_index_alias: &str,
        multi_fruit: &mut MultiFruit,
        searcher: &Searcher,
        multi_fields: &HashSet<Field>,
    ) -> proto::CollectorOutput {
        self.extract(external_index_alias, multi_fruit, searcher, multi_fields)
    }
}

pub fn parse_aggregations(aggregations: HashMap<String, proto::Aggregation>) -> SummaResult<HashMap<String, tantivy::aggregation::agg_req::Aggregation>> {
    let aggregations = aggregations
        .into_iter()
        .map(|(name, aggregation)| {
            let aggregation = match aggregation.aggregation {
                None => Err(Error::Proto(summa_proto::errors::Error::InvalidAggregation)),
                Some(aggregation) => aggregation.try_into().map_err(Error::Proto),
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

fn get_fields<'a, F: Iterator<Item = &'a String>>(schema: &Schema, fields: F) -> SummaResult<Option<HashSet<Field>>> {
    let fields = fields
        .map(|field| {
            schema
                .get_field(field)
                .ok_or_else(|| Error::Validation(ValidationError::MissingField(field.to_owned())))
        })
        .collect::<SummaResult<HashSet<Field>>>()?;
    Ok((!fields.is_empty()).then_some(fields))
}

pub fn build_fruit_extractor(
    collector_proto: proto::Collector,
    query: &dyn Query,
    schema: &Schema,
    multi_collector: &mut MultiCollector,
) -> SummaResult<Box<dyn FruitExtractor>> {
    match collector_proto.collector {
        Some(proto::collector::Collector::TopDocs(top_docs_collector_proto)) => {
            get_fields(schema, top_docs_collector_proto.snippets.keys())?;
            let fields = get_fields(schema, top_docs_collector_proto.fields.iter())?;
            Ok(match top_docs_collector_proto.scorer {
                None | Some(proto::Scorer { scorer: None }) => Box::new(
                    TopDocsBuilder::default()
                        .handle(
                            multi_collector.add_collector(
                                tantivy::collector::TopDocs::with_limit((top_docs_collector_proto.limit + 1) as usize)
                                    .and_offset(top_docs_collector_proto.offset as usize),
                            ),
                        )
                        .query(query.box_clone())
                        .explain(top_docs_collector_proto.explain)
                        .limit(top_docs_collector_proto.limit)
                        .snippets(top_docs_collector_proto.snippets)
                        .fields(fields)
                        .build()?,
                ) as Box<dyn FruitExtractor>,
                Some(proto::Scorer {
                    scorer: Some(proto::scorer::Scorer::EvalExpr(ref eval_expr)),
                }) => {
                    let eval_scorer_seed = EvalScorer::new(eval_expr, schema)?;
                    let top_docs_collector = tantivy::collector::TopDocs::with_limit((top_docs_collector_proto.limit + 1) as usize)
                        .and_offset(top_docs_collector_proto.offset as usize)
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
                            .fields(fields)
                            .build()?,
                    ) as Box<dyn FruitExtractor>
                }
                Some(proto::Scorer {
                    scorer: Some(proto::scorer::Scorer::OrderBy(ref field_name)),
                }) => {
                    let order_by_field = schema
                        .get_field(field_name)
                        .ok_or_else(|| ValidationError::MissingField(field_name.to_owned()))?;
                    let top_docs_collector = tantivy::collector::TopDocs::with_limit((top_docs_collector_proto.limit + 1) as usize)
                        .and_offset(top_docs_collector_proto.offset as usize)
                        .order_by_u64_field(order_by_field);
                    Box::new(
                        TopDocsBuilder::default()
                            .handle(multi_collector.add_collector(top_docs_collector))
                            .query(query.box_clone())
                            .explain(top_docs_collector_proto.explain)
                            .limit(top_docs_collector_proto.limit)
                            .snippets(top_docs_collector_proto.snippets)
                            .fields(fields)
                            .build()?,
                    ) as Box<dyn FruitExtractor>
                }
            })
        }
        Some(proto::collector::Collector::ReservoirSampling(reservoir_sampling_collector_proto)) => {
            let fields = get_fields(schema, reservoir_sampling_collector_proto.fields.iter())?;
            let reservoir_sampling_collector = collectors::ReservoirSampling::with_limit(reservoir_sampling_collector_proto.limit as usize);
            Ok(Box::new(
                ReservoirSamplingBuilder::default()
                    .handle(multi_collector.add_collector(reservoir_sampling_collector))
                    .fields(fields)
                    .build()?,
            ) as Box<dyn FruitExtractor>)
        }
        Some(proto::collector::Collector::Count(_)) => Ok(Box::new(Count(multi_collector.add_collector(tantivy::collector::Count))) as Box<dyn FruitExtractor>),
        Some(proto::collector::Collector::Facet(facet_collector_proto)) => {
            let field = schema
                .get_field(&facet_collector_proto.field)
                .ok_or_else(|| ValidationError::MissingField(facet_collector_proto.field.to_owned()))?;
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
    #[builder(default = "None")]
    fields: Option<HashSet<Field>>,
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

#[async_trait]
impl<T: 'static + Copy + Into<proto::Score> + Sync + Send> FruitExtractor for TopDocs<T> {
    fn extract(
        self: Box<Self>,
        external_index_alias: &str,
        multi_fruit: &mut MultiFruit,
        searcher: &Searcher,
        multi_fields: &HashSet<Field>,
    ) -> proto::CollectorOutput {
        let schema = searcher.schema();
        let snippet_generators = self.snippet_generators(searcher);
        let fruit = self.handle.extract(multi_fruit);
        let scored_documents_iter = fruit.iter().enumerate().map(|(position, (score, doc_address))| {
            let document = searcher.doc(*doc_address).unwrap();
            proto::ScoredDocument {
                document: NamedFieldDocument::from_document(schema, &self.fields, multi_fields, &document).to_json(),
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

    // ToDo: deduplicate!
    async fn extract_async(
        self: Box<Self>,
        external_index_alias: &str,
        multi_fruit: &mut MultiFruit,
        searcher: &Searcher,
        multi_fields: &HashSet<Field>,
    ) -> proto::CollectorOutput {
        let schema = searcher.schema();
        let snippet_generators = self.snippet_generators(searcher);
        let fruit = self.handle.extract(multi_fruit);
        let length = fruit.len();
        let limit = std::cmp::min(self.limit as usize, length);
        let has_next = length > self.limit as usize;

        let scored_documents = join_all(
            fruit
                .iter()
                .take(limit)
                .map(|(score, doc_address)| async move { (score, searcher.doc_async(*doc_address).await.unwrap()) }),
        )
        .await
        .into_iter()
        .enumerate()
        .map(|(position, (score, document))| proto::ScoredDocument {
            document: NamedFieldDocument::from_document(schema, &self.fields, multi_fields, &document).to_json(),
            score: Some((*score).into()),
            position: position as u32,
            snippets: snippet_generators
                .iter()
                .map(|(field_name, snippet_generator)| (field_name.to_string(), snippet_generator.snippet_from_doc(&document).into()))
                .collect(),
            index_alias: external_index_alias.to_string(),
        })
        .collect();
        proto::CollectorOutput {
            collector_output: Some(proto::collector_output::CollectorOutput::TopDocs(proto::TopDocsCollectorOutput {
                scored_documents,
                has_next,
            })),
        }
    }
}

#[derive(Builder)]
#[builder(pattern = "owned", build_fn(error = "ValidationError"))]
pub struct ReservoirSampling {
    handle: FruitHandle<Vec<DocAddress>>,
    #[builder(default = "None")]
    fields: Option<HashSet<Field>>,
}

impl FruitExtractor for ReservoirSampling {
    fn extract(
        self: Box<Self>,
        external_index_alias: &str,
        multi_fruit: &mut MultiFruit,
        searcher: &Searcher,
        multi_fields: &HashSet<Field>,
    ) -> proto::CollectorOutput {
        let schema = searcher.schema();
        proto::CollectorOutput {
            collector_output: Some(proto::collector_output::CollectorOutput::ReservoirSampling(
                proto::ReservoirSamplingCollectorOutput {
                    random_documents: self
                        .handle
                        .extract(multi_fruit)
                        .iter()
                        .map(|doc_address| proto::RandomDocument {
                            index_alias: external_index_alias.to_string(),
                            document: NamedFieldDocument::from_document(schema, &self.fields, multi_fields, &searcher.doc(*doc_address).unwrap()).to_json(),
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
