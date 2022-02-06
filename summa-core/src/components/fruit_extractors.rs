use std::collections::{HashMap, HashSet};

use futures::future::join_all;
use rustc_hash::FxHashMap;
use summa_proto::proto;
use tantivy::aggregation::agg_result::AggregationResults;
use tantivy::collector::{FacetCounts, FruitHandle, MultiCollector, MultiFruit, ScoreSegmentTweaker, ScoreTweaker};
use tantivy::query::{Query, QueryClone};
use tantivy::schema::{Field, Schema};
use tantivy::{DocAddress, DocId, Searcher, SegmentReader, SnippetGenerator};

use super::custom_serializer::NamedFieldDocument;
use crate::collectors;
use crate::errors::{BuilderError, Error, SummaResult};
use crate::proto_traits::Wrapper;
use crate::scorers::{EvalScorer, SegmentEvalScorer};

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

fn get_strict_fields<'a, F: Iterator<Item = &'a String>>(schema: &Schema, fields: F) -> SummaResult<Option<HashSet<Field>>> {
    let fields = fields.map(|field| schema.get_field(field)).collect::<Result<HashSet<Field>, _>>()?;
    Ok((!fields.is_empty()).then_some(fields))
}

struct EvalScorerSegmentScoreTweaker {
    segment_eval_scorer: SegmentEvalScorer,
}

impl EvalScorerSegmentScoreTweaker {
    pub fn new(segment_eval_scorer: SegmentEvalScorer) -> Self {
        EvalScorerSegmentScoreTweaker { segment_eval_scorer }
    }
}

impl ScoreSegmentTweaker<f64> for EvalScorerSegmentScoreTweaker {
    fn score(&mut self, doc: DocId, score: f32) -> f64 {
        self.segment_eval_scorer.score(doc, score)
    }
}

struct EvalScorerTweaker {
    eval_scorer_seed: EvalScorer,
}

impl EvalScorerTweaker {
    pub fn new(eval_scorer_seed: EvalScorer) -> Self {
        EvalScorerTweaker { eval_scorer_seed }
    }
}

#[async_trait]
impl ScoreTweaker<f64> for EvalScorerTweaker {
    type Child = EvalScorerSegmentScoreTweaker;

    fn segment_tweaker(&self, segment_reader: &SegmentReader) -> tantivy::Result<Self::Child> {
        Ok(EvalScorerSegmentScoreTweaker::new(
            self.eval_scorer_seed.get_for_segment_reader(segment_reader).expect("Wrong eval expression"),
        ))
    }

    async fn segment_tweaker_async(&self, segment_reader: &SegmentReader) -> tantivy::Result<Self::Child> {
        Ok(EvalScorerSegmentScoreTweaker::new(
            self.eval_scorer_seed
                .get_for_segment_reader_async(segment_reader)
                .await
                .expect("Wrong eval expression"),
        ))
    }
}

pub fn build_fruit_extractor(
    collector_proto: proto::Collector,
    query: &dyn Query,
    schema: &Schema,
    multi_collector: &mut MultiCollector,
) -> SummaResult<Box<dyn FruitExtractor>> {
    match collector_proto.collector {
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
                        .tweak_score(EvalScorerTweaker::new(eval_scorer_seed));
                    Box::new(
                        TopDocsBuilder::default()
                            .handle(multi_collector.add_collector(top_docs_collector))
                            .query(query.box_clone())
                            .limit(top_docs_collector_proto.limit)
                            .snippets(top_docs_collector_proto.snippets)
                            .fields(fields)
                            .build()?,
                    ) as Box<dyn FruitExtractor>
                }
                Some(proto::Scorer {
                    scorer: Some(proto::scorer::Scorer::OrderBy(ref field_name)),
                }) => {
                    let order_by_field = schema.get_field(field_name)?;
                    let top_docs_collector = tantivy::collector::TopDocs::with_limit((top_docs_collector_proto.limit + 1) as usize)
                        .and_offset(top_docs_collector_proto.offset as usize)
                        .order_by_u64_field(order_by_field);
                    Box::new(
                        TopDocsBuilder::default()
                            .handle(multi_collector.add_collector(top_docs_collector))
                            .query(query.box_clone())
                            .limit(top_docs_collector_proto.limit)
                            .snippets(top_docs_collector_proto.snippets)
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
            let field = schema.get_field(&facet_collector_proto.field)?;
            let mut facet_collector = tantivy::collector::FacetCollector::for_field(field);
            for facet in &facet_collector_proto.facets {
                facet_collector.add_facet(facet);
            }
            Ok(Box::new(Facet(multi_collector.add_collector(facet_collector))) as Box<dyn FruitExtractor>)
        }
        Some(proto::collector::Collector::Aggregation(aggregation_collector_proto)) => {
            let aggregation_collector =
                tantivy::aggregation::AggregationCollector::from_aggs(parse_aggregations(aggregation_collector_proto.aggregations)?, None, schema.clone());
            Ok(Box::new(Aggregation(multi_collector.add_collector(aggregation_collector))) as Box<dyn FruitExtractor>)
        }
        None => Ok(Box::new(Count(multi_collector.add_collector(tantivy::collector::Count))) as Box<dyn FruitExtractor>),
    }
}

#[derive(Builder)]
#[builder(pattern = "owned", build_fn(error = "BuilderError"))]
pub struct TopDocs<T: 'static + Copy + Into<proto::Score> + Sync + Send> {
    handle: FruitHandle<Vec<(T, DocAddress)>>,
    limit: u32,
    snippets: HashMap<String, u32>,
    query: Box<dyn Query>,
    #[builder(default = "None")]
    fields: Option<HashSet<Field>>,
}

fn snippet_generators(snippets: &HashMap<String, u32>, query: &dyn Query, searcher: &Searcher) -> HashMap<String, SnippetGenerator> {
    snippets
        .iter()
        .filter_map(|(field_name, max_num_chars)| {
            searcher.schema().get_field(field_name).ok().map(|snippet_field| {
                let mut snippet_generator = SnippetGenerator::create(searcher, query, snippet_field).expect("Snippet generator cannot be created");
                snippet_generator.set_max_num_chars(*max_num_chars as usize);
                (field_name.to_string(), snippet_generator)
            })
        })
        .collect()
}

async fn snippet_generators_async(snippets: &HashMap<String, u32>, query: Box<dyn Query>, searcher: &Searcher) -> HashMap<String, SnippetGenerator> {
    let futures = snippets.iter().filter_map(|(field_name, max_num_chars)| {
        let query = query.box_clone();
        searcher.schema().get_field(field_name).ok().map(|snippet_field| async move {
            let mut snippet_generator = SnippetGenerator::create_async(searcher, &query, snippet_field)
                .await
                .expect("Snippet generator cannot be created");
            snippet_generator.set_max_num_chars(*max_num_chars as usize);
            (field_name.to_string(), snippet_generator)
        })
    });
    join_all(futures).await.into_iter().collect()
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
        let snippet_generators = snippet_generators(&self.snippets, &self.query, searcher);
        let fruit = self.handle.extract(multi_fruit);
        let length = fruit.len();

        let scored_documents = fruit.iter().take(std::cmp::min(self.limit as usize, length));
        let scored_documents = searcher.index().search_executor().map(
            |(position, (score, doc_address))| {
                let document = searcher.doc(*doc_address).expect("Document retrieving failed");
                Ok(proto::ScoredDocument {
                    document: NamedFieldDocument::from_document(searcher.schema(), &self.fields, multi_fields, &document).to_json(),
                    score: Some((*score).into()),
                    position: position as u32,
                    snippets: snippet_generators
                        .iter()
                        .map(|(field_name, snippet_generator)| {
                            (
                                field_name.to_string(),
                                Wrapper::from(snippet_generator.snippet_from_doc(&document)).into_inner(),
                            )
                        })
                        .collect(),
                    index_alias: external_index_alias.to_string(),
                })
            },
            scored_documents.enumerate(),
        )?;
        Ok(proto::CollectorOutput {
            collector_output: Some(proto::collector_output::CollectorOutput::TopDocs(proto::TopDocsCollectorOutput {
                scored_documents,
                has_next: length > self.limit as usize,
            })),
        })
    }

    async fn extract_async(
        self: Box<Self>,
        external_index_alias: &str,
        multi_fruit: &mut MultiFruit,
        searcher: &Searcher,
        multi_fields: &HashSet<Field>,
    ) -> SummaResult<proto::CollectorOutput> {
        let snippet_generators = snippet_generators_async(&self.snippets, self.query, searcher).await;
        let fruit = self.handle.extract(multi_fruit);
        let length = fruit.len();

        let scored_documents = fruit.into_iter().take(std::cmp::min(self.limit as usize, length));
        let document_futures = scored_documents.enumerate().map(|(position, (score, doc_address))| {
            let snippet_generators = &snippet_generators;
            let fields = &self.fields;
            async move {
                // Calling `doc` may be computationally expensive. For better performance it is better to spawn it on blocking pool
                #[cfg(feature = "tokio-rt")]
                let document = {
                    let searcher_cloned = searcher.clone();
                    tokio::task::spawn_blocking(move || searcher_cloned.doc(doc_address))
                        .await
                        .expect("Cannot spawn")
                        .expect("Document retrieving failed")
                };
                #[cfg(not(feature = "tokio-rt"))]
                let document = searcher.doc_async(doc_address).await.expect("Document retrieving failed");
                proto::ScoredDocument {
                    document: NamedFieldDocument::from_document(searcher.schema(), fields, multi_fields, &document).to_json(),
                    score: Some(score.into()),
                    position: position as u32,
                    snippets: snippet_generators
                        .iter()
                        .map(|(field_name, snippet_generator)| {
                            (
                                field_name.to_string(),
                                Wrapper::from(snippet_generator.snippet_from_doc(&document)).into_inner(),
                            )
                        })
                        .collect(),
                    index_alias: external_index_alias.to_string(),
                }
            }
        });
        let scored_documents = join_all(document_futures).await;
        Ok(proto::CollectorOutput {
            collector_output: Some(proto::collector_output::CollectorOutput::TopDocs(proto::TopDocsCollectorOutput {
                scored_documents,
                has_next: length > self.limit as usize,
            })),
        })
    }
}

#[derive(Builder)]
#[builder(pattern = "owned", build_fn(error = "BuilderError"))]
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
                                document: NamedFieldDocument::from_document(schema, &self.fields, multi_fields, &searcher.doc(*doc_address)?).to_json(),
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
