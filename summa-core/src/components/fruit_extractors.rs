use std::collections::{HashMap, HashSet};

use futures::future::join_all;
use rustc_hash::FxHashMap;
use summa_proto::proto;
use tantivy::aggregation::agg_result::AggregationResults;
use tantivy::collector::{FacetCounts, FruitHandle, MultiCollector, MultiFruit};
use tantivy::query::Query;
use tantivy::schema::{Field, Schema};
use tantivy::{DocAddress, DocId, Document, Score, Searcher, SegmentReader, SnippetGenerator};

use super::custom_serializer::NamedFieldDocument;
use crate::collectors;
use crate::errors::{Error, SummaResult, ValidationError};
use crate::scorers::EvalScorer;

/// Extracts data from `MultiFruit` and moving it to the `proto::CollectorOutput`
#[async_trait]
pub trait FruitExtractor: Sync + Send {
    fn extract(
        self: Box<Self>,
        external_index_alias: &str,
        multi_fruit: &mut MultiFruit,
        searcher: &Searcher,
        multi_fields: &HashSet<Field>,
    ) -> SummaResult<proto::CollectorOutput>;

    async fn extract_async(
        self: Box<Self>,
        external_index_alias: &str,
        multi_fruit: &mut MultiFruit,
        searcher: &Searcher,
        multi_fields: &HashSet<Field>,
    ) -> SummaResult<proto::CollectorOutput> {
        self.extract(external_index_alias, multi_fruit, searcher, multi_fields)
    }
}

pub fn parse_aggregations(aggregations: &HashMap<String, proto::Aggregation>) -> SummaResult<HashMap<String, tantivy::aggregation::agg_req::Aggregation>> {
    let aggregations = aggregations
        .into_iter()
        .map(|(name, aggregation)| {
            let aggregation = match &aggregation.aggregation {
                None => Err(Error::Proto(summa_proto::errors::Error::InvalidAggregation)),
                Some(aggregation) => aggregation.clone().try_into().map_err(Error::Proto),
            }?;
            Ok((name.clone(), aggregation))
        })
        .collect::<SummaResult<Vec<_>>>()?;
    Ok(HashMap::from_iter(aggregations.into_iter()))
}

fn parse_aggregation_results(
    aggregation_results: FxHashMap<String, tantivy::aggregation::agg_result::AggregationResult>,
) -> HashMap<String, proto::AggregationResult> {
    aggregation_results
        .into_iter()
        .map(|(name, aggregation_result)| (name, aggregation_result.into()))
        .collect()
}

fn get_strict_fields<'a, F: Iterator<Item = &'a String>>(schema: &Schema, fields: F) -> SummaResult<Option<HashSet<Field>>> {
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
    collector_proto: &proto::Collector,
    query: &dyn Query,
    schema: &Schema,
    multi_collector: &mut MultiCollector,
) -> SummaResult<Box<dyn FruitExtractor>> {
    match &collector_proto.collector {
        Some(proto::collector::Collector::TopDocs(top_docs_collector_proto)) => {
            let fields = get_strict_fields(schema, top_docs_collector_proto.fields.iter())?;
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
                        .limit(top_docs_collector_proto.limit)
                        .snippets(top_docs_collector_proto.snippets.clone())
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
                            let mut eval_scorer = eval_scorer_seed.get_for_segment_reader(segment_reader).expect("Wrong eval expression");
                            move |doc_id: DocId, original_score: Score| eval_scorer.score(doc_id, original_score)
                        });
                    Box::new(
                        TopDocsBuilder::default()
                            .handle(multi_collector.add_collector(top_docs_collector))
                            .query(query.box_clone())
                            .limit(top_docs_collector_proto.limit)
                            .snippets(top_docs_collector_proto.snippets.clone())
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
                            .limit(top_docs_collector_proto.limit)
                            .snippets(top_docs_collector_proto.snippets.clone())
                            .fields(fields)
                            .build()?,
                    ) as Box<dyn FruitExtractor>
                }
            })
        }
        Some(proto::collector::Collector::ReservoirSampling(reservoir_sampling_collector_proto)) => {
            let fields = get_strict_fields(schema, reservoir_sampling_collector_proto.fields.iter())?;
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
                tantivy::aggregation::AggregationCollector::from_aggs(parse_aggregations(&aggregation_collector_proto.aggregations)?, None);
            Ok(Box::new(Aggregation(multi_collector.add_collector(aggregation_collector))) as Box<dyn FruitExtractor>)
        }
        None => Ok(Box::new(Count(multi_collector.add_collector(tantivy::collector::Count))) as Box<dyn FruitExtractor>),
    }
}

fn extract_base<T: 'static + Copy + Into<proto::Score> + Sync + Send>(
    scored_documents_iter: impl Iterator<Item = (T, Document)>,
    external_index_alias: &str,
    schema: &Schema,
    has_next: bool,
    fields: &Option<HashSet<Field>>,
    multi_fields: &HashSet<Field>,
    snippet_generators: HashMap<String, SnippetGenerator>,
) -> SummaResult<proto::CollectorOutput> {
    let scored_documents = scored_documents_iter
        .enumerate()
        .map(|(position, (score, document))| {
            Ok(proto::ScoredDocument {
                document: NamedFieldDocument::from_document(schema, fields, multi_fields, &document).to_json()?,
                score: Some(score.into()),
                position: position as u32,
                snippets: snippet_generators
                    .iter()
                    .map(|(field_name, snippet_generator)| (field_name.to_string(), snippet_generator.snippet_from_doc(&document).into()))
                    .collect(),
                index_alias: external_index_alias.to_string(),
            })
        })
        .collect::<SummaResult<_>>()?;
    Ok(proto::CollectorOutput {
        collector_output: Some(proto::collector_output::CollectorOutput::TopDocs(proto::TopDocsCollectorOutput {
            scored_documents,
            has_next,
        })),
    })
}

#[derive(Builder)]
#[builder(pattern = "owned", build_fn(error = "ValidationError"))]
pub struct TopDocs<T: 'static + Copy + Into<proto::Score> + Sync + Send> {
    handle: FruitHandle<Vec<(T, DocAddress)>>,
    limit: u32,
    snippets: HashMap<String, u32>,
    query: Box<dyn Query>,
    #[builder(default = "None")]
    fields: Option<HashSet<Field>>,
}

impl<T: 'static + Copy + Into<proto::Score> + Sync + Send> TopDocs<T> {
    fn snippet_generators(&self, searcher: &Searcher) -> HashMap<String, SnippetGenerator> {
        self.snippets
            .iter()
            .filter_map(|(field_name, max_num_chars)| {
                searcher.schema().get_field(field_name).map(|snippet_field| {
                    let mut snippet_generator = SnippetGenerator::create(searcher, &*self.query, snippet_field).expect("Snippet generator cannot be created");
                    snippet_generator.set_max_num_chars(*max_num_chars as usize);
                    (field_name.to_string(), snippet_generator)
                })
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
    ) -> SummaResult<proto::CollectorOutput> {
        let snippet_generators = self.snippet_generators(searcher);
        let fruit = self.handle.extract(multi_fruit);
        let length = fruit.len();

        let scored_documents = fruit
            .iter()
            .take(std::cmp::min(self.limit as usize, length))
            .map(|(score, doc_address)| (*score, searcher.doc(*doc_address).expect("Document retrieving failed")));
        extract_base(
            scored_documents,
            external_index_alias,
            searcher.schema(),
            length > self.limit as usize,
            &self.fields,
            multi_fields,
            snippet_generators,
        )
    }

    async fn extract_async(
        self: Box<Self>,
        external_index_alias: &str,
        multi_fruit: &mut MultiFruit,
        searcher: &Searcher,
        multi_fields: &HashSet<Field>,
    ) -> SummaResult<proto::CollectorOutput> {
        let snippet_generators = self.snippet_generators(searcher);
        let fruit = self.handle.extract(multi_fruit);
        let length = fruit.len();

        let scored_documents = join_all(
            fruit
                .iter()
                .take(std::cmp::min(self.limit as usize, length))
                .map(|(score, doc_address)| async move { (*score, searcher.doc_async(*doc_address).await.expect("Document retrieving failed")) }),
        )
        .await;
        extract_base(
            scored_documents.into_iter(),
            external_index_alias,
            searcher.schema(),
            length > self.limit as usize,
            &self.fields,
            multi_fields,
            snippet_generators,
        )
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
    ) -> SummaResult<proto::CollectorOutput> {
        let schema = searcher.schema();
        Ok(proto::CollectorOutput {
            collector_output: Some(proto::collector_output::CollectorOutput::ReservoirSampling(
                proto::ReservoirSamplingCollectorOutput {
                    random_documents: self
                        .handle
                        .extract(multi_fruit)
                        .iter()
                        .map(|doc_address| {
                            Ok(proto::RandomDocument {
                                index_alias: external_index_alias.to_string(),
                                document: NamedFieldDocument::from_document(schema, &self.fields, multi_fields, &searcher.doc(*doc_address)?).to_json()?,
                                score: Some((1.0).into()),
                            })
                        })
                        .collect::<SummaResult<_>>()?,
                },
            )),
        })
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
    ) -> SummaResult<proto::CollectorOutput> {
        Ok(proto::CollectorOutput {
            collector_output: Some(proto::collector_output::CollectorOutput::Count(proto::CountCollectorOutput {
                count: self.0.extract(multi_fruit) as u32,
            })),
        })
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
    ) -> SummaResult<proto::CollectorOutput> {
        Ok(proto::CollectorOutput {
            collector_output: Some(proto::collector_output::CollectorOutput::Facet(proto::FacetCollectorOutput {
                facet_counts: self.0.extract(multi_fruit).get("").map(|(facet, count)| (facet.to_string(), count)).collect(),
            })),
        })
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
    ) -> SummaResult<proto::CollectorOutput> {
        Ok(proto::CollectorOutput {
            collector_output: Some(proto::collector_output::CollectorOutput::Aggregation(proto::AggregationCollectorOutput {
                aggregation_results: parse_aggregation_results(self.0.extract(multi_fruit).0),
            })),
        })
    }
}
