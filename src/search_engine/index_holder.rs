use super::default_tokenizers::default_tokenizers;
use super::index_writer_holder::IndexWriterHolder;
use crate::configs::IndexConfigProxy;
use crate::configs::IndexEngine;
use crate::errors::Error::InvalidSyntaxError;
use crate::errors::{Error, SummaResult, ValidationError};
use crate::proto;
use crate::search_engine::collectors::ReservoirSampling;
use crate::search_engine::fruit_extractors::{self, FruitExtractor};
use crate::search_engine::IndexUpdater;
use crate::utils::sync::{Handler, OwningHandler};
use crate::utils::thread_handler::ThreadHandler;
use parking_lot::RwLock;
use std::collections::HashSet;
use std::ops::Bound::Unbounded;
use std::ops::{Bound, Deref};
use std::str::FromStr;
use std::time::Duration;
use tantivy::collector::{Count, MultiCollector, TopDocs};
use tantivy::query::{AllQuery, BooleanQuery, BoostQuery, Occur, PhraseQuery, Query, QueryParser, RangeQuery, RegexQuery, TermQuery};
use tantivy::schema::{Field, FieldEntry, FieldType, IndexRecordOption, Schema};
use tantivy::{Index, IndexReader, IndexSettings, Opstamp, ReloadPolicy, Term};
use tokio::fs::remove_dir_all;
use tokio::sync::oneshot;
use tokio::time;
use tracing::{info, info_span, instrument, warn, Instrument};

pub struct IndexHolder {
    index_name: String,
    index_config_proxy: IndexConfigProxy,
    index: Index,
    cached_schema: Schema,
    index_reader: IndexReader,
    query_parser: QueryParser,
    multi_fields: HashSet<Field>,
    /// All modifying operations are isolated inside `index_updater`
    index_updater: OwningHandler<RwLock<IndexUpdater>>,
    autocommit_thread: Option<ThreadHandler>,
}

impl IndexHolder {
    /// Sets up standard Summa tokenizers
    ///
    /// The set of tokenizers includes standard Tantivy tokenizers as well as `SummaTokenizer` that supports CJK
    fn register_default_tokenizers(index: &Index) {
        for (tokenizer_name, tokenizer) in &default_tokenizers() {
            index.tokenizers().register(tokenizer_name, tokenizer.clone())
        }
    }

    /// Sets up `IndexHolder`
    ///
    /// Creates the autocommitting thread and consumers
    async fn setup(index_name: &str, index: Index, index_config_proxy: IndexConfigProxy) -> SummaResult<IndexHolder> {
        let index_config = index_config_proxy.read().deref().clone();
        IndexHolder::register_default_tokenizers(&index);
        let index_reader = index.reader_builder().reload_policy(ReloadPolicy::OnCommit).try_into()?;
        let index_updater = OwningHandler::new(RwLock::new(IndexUpdater::new(
            &index.schema(),
            index_config_proxy.clone(),
            IndexWriterHolder::new(
                index.writer_with_num_threads(index_config.writer_threads.try_into().unwrap(), index_config.writer_heap_size_bytes.try_into().unwrap())?,
                match index_config.primary_key {
                    Some(ref primary_key) => index.schema().get_field(primary_key).clone(),
                    None => None,
                },
            )?,
        )?));

        let autocommit_thread = match index_config.autocommit_interval_ms.clone() {
            Some(interval_ms) => {
                let index_updater = index_updater.handler().clone();
                let index_name = index_name.to_owned();
                let (shutdown_trigger, mut shutdown_tripwire) = oneshot::channel();
                let mut tick_task = time::interval(Duration::from_millis(interval_ms));
                Some(ThreadHandler::new(
                    tokio::spawn(
                        async move {
                            info!(action = "spawning_autocommit_thread", interval_ms = interval_ms);
                            // The first tick ticks immediately so we skip it
                            tick_task.tick().await;
                            loop {
                                tokio::select! {
                                    _ = tick_task.tick() => {
                                        info!(action = "autocommit_thread_tick");
                                        if let Some(mut index_updater) = index_updater.try_write() {
                                            if let Err(error) = index_updater.commit().await {
                                                warn!(error = ?error);
                                            }
                                        }
                                    }
                                    _ = &mut shutdown_tripwire => {
                                        info!(action = "shutdown_autocommit_thread");
                                        break;
                                    }
                                }
                            }
                            Ok(())
                        }
                        .instrument(info_span!(parent: None, "autocommit_thread", index_name = ?index_name)),
                    ),
                    shutdown_trigger,
                ))
            }
            None => None,
        };

        let cached_schema = index.schema();
        Ok(IndexHolder {
            index_name: String::from(index_name),
            autocommit_thread,
            query_parser: QueryParser::for_index(&index, index_config.default_fields.iter().map(|x| cached_schema.get_field(x).unwrap()).collect()),
            multi_fields: index_config.multi_fields.iter().map(|x| cached_schema.get_field(x).unwrap()).collect(),
            cached_schema,
            index,
            index_reader,
            index_updater,
            index_config_proxy,
        })
    }

    /// Creates index and sets it up via `setup`
    #[instrument(skip(schema, index_config_proxy))]
    pub(crate) async fn create(index_name: &str, schema: &Schema, index_config_proxy: IndexConfigProxy) -> SummaResult<IndexHolder> {
        let index = {
            let index_config = index_config_proxy.read();
            info!(action = "create", engine = ?index_config.index_engine);
            match &index_config.index_engine {
                IndexEngine::Memory(schema) => Index::create_in_ram(schema.clone()),
                IndexEngine::File(index_path) => {
                    if index_path.exists() {
                        Err(ValidationError::ExistingPathError(index_path.to_owned()))?;
                    }
                    std::fs::create_dir_all(index_path)?;
                    let settings = IndexSettings {
                        docstore_compression: index_config.compression.clone(),
                        sort_by_field: index_config.sort_by_field.clone(),
                    };
                    Index::builder().schema(schema.clone()).settings(settings).create_in_dir(index_path)?
                }
            }
        };
        IndexHolder::setup(index_name, index, index_config_proxy).await
    }

    /// Opens index within `index_path` directory and sets it up via `setup`
    #[instrument(skip(index_config_proxy))]
    pub(crate) async fn open(index_name: &str, index_config_proxy: IndexConfigProxy) -> SummaResult<IndexHolder> {
        let index = match &index_config_proxy.read().index_engine {
            IndexEngine::Memory(schema) => Index::create_in_ram(schema.clone()),
            IndexEngine::File(index_path) => Index::open_in_dir(index_path)?,
        };
        IndexHolder::setup(index_name, index, index_config_proxy).await
    }

    /// Index name
    pub(crate) fn index_name(&self) -> &str {
        &self.index_name
    }

    /// `IndexReader` singleton
    pub(crate) fn index_reader(&self) -> &IndexReader {
        &self.index_reader
    }

    /// `IndexUpdater` handler
    pub(crate) fn index_updater(&self) -> Handler<RwLock<IndexUpdater>> {
        self.index_updater.handler()
    }

    /// Index schema
    pub(crate) fn schema(&self) -> &Schema {
        &self.cached_schema
    }

    /// Index schema
    #[inline]
    pub(crate) fn field_and_field_entry(&self, field_name: &str) -> SummaResult<(Field, &FieldEntry)> {
        let field = self.cached_schema.get_field(field_name).ok_or(Error::FieldDoesNotExistError(field_name.to_owned()))?;
        let field_entry = self.cached_schema.get_field_entry(field);
        Ok((field, field_entry))
    }

    /// Stops `IndexHolder` instance
    ///
    /// Both autocommitting thread and consumers are stopped
    #[instrument(skip(self), fields(index_name = %self.index_name))]
    pub(crate) async fn stop(self) -> SummaResult<Opstamp> {
        if let Some(autocommit_thread) = self.autocommit_thread {
            autocommit_thread.stop().await?;
        }
        self.index_updater.into_inner().into_inner().stop_consumers_and_commit().await
    }

    /// Delete `IndexHolder` instance
    ///
    /// Both autocommitting thread and consumers are stopped, then `IndexConfig` is removed from `ApplicationConfig`
    /// and then directory with the index is deleted.
    pub(crate) async fn delete(self) -> SummaResult<()> {
        if let Some(autocommit_thread) = self.autocommit_thread {
            autocommit_thread.stop().await?;
        };
        self.index_updater.into_inner().into_inner().stop().await?;
        match self.index_config_proxy.delete().index_engine {
            IndexEngine::Memory(_) => (),
            IndexEngine::File(ref index_path) => {
                info!(action = "delete_directory");
                remove_dir_all(index_path).await.map_err(|e| Error::IOError((e, Some(index_path.to_path_buf()))))?;
            }
        };
        Ok(())
    }

    fn cast_value_to_term(&self, field: Field, field_type: &FieldType, value: &str) -> SummaResult<Term> {
        // ToDo: more errors and types
        Ok(match field_type {
            FieldType::Str(_) => Term::from_field_text(field, value),
            FieldType::I64(_) => Term::from_field_i64(field, i64::from_str(value).map_err(|_e| InvalidSyntaxError("cannot parse i64".to_owned()))?),
            FieldType::U64(_) => Term::from_field_u64(field, u64::from_str(value).map_err(|_e| InvalidSyntaxError("cannot parse u64".to_owned()))?),
            FieldType::F64(_) => Term::from_field_f64(field, f64::from_str(value).map_err(|_e| InvalidSyntaxError("cannot parse f64".to_owned()))?),
            _ => return Err(InvalidSyntaxError("invalid range type".to_owned()))?,
        })
    }

    fn cast_value_to_bound_term(&self, field: Field, field_type: &FieldType, value: &str, including: bool) -> SummaResult<Bound<Term>> {
        Ok(match value {
            "*" => Unbounded,
            value => {
                let casted_value = self.cast_value_to_term(field, field_type, value)?;
                if including {
                    Bound::Included(casted_value)
                } else {
                    Bound::Excluded(casted_value)
                }
            }
        })
    }

    fn parse_query(&self, query: &proto::Query) -> SummaResult<Box<dyn Query>> {
        Ok(match &query.query {
            None | Some(proto::query::Query::All(_)) => Box::new(AllQuery),
            Some(proto::query::Query::Bool(boolean_query)) => {
                let mut subqueries = vec![];
                for subquery in &boolean_query.subqueries {
                    subqueries.push((
                        match proto::Occur::from_i32(subquery.occur) {
                            None | Some(proto::Occur::Should) => Occur::Should,
                            Some(proto::Occur::Must) => Occur::Must,
                            Some(proto::Occur::MustNot) => Occur::MustNot,
                        },
                        self.parse_query(subquery.query.as_ref().ok_or(Error::EmptyQueryError)?)?,
                    ))
                }
                Box::new(BooleanQuery::new(subqueries))
            }
            Some(proto::query::Query::Match(match_query_proto)) => match self.query_parser.parse_query(&match_query_proto.value) {
                Ok(parsed_query) => Ok(parsed_query),
                Err(tantivy::query::QueryParserError::FieldDoesNotExist(field)) => Err(Error::FieldDoesNotExistError(field)),
                Err(e) => Err(Error::InvalidTantivySyntaxError((e, match_query_proto.value.to_owned()))),
            }?,
            Some(proto::query::Query::Range(range_query_proto)) => {
                let (field, field_entry) = self.field_and_field_entry(&range_query_proto.field)?;
                let value = range_query_proto.value.as_ref().unwrap();
                let left = self.cast_value_to_bound_term(field, field_entry.field_type(), &value.left, value.including_left)?;
                let right = self.cast_value_to_bound_term(field, field_entry.field_type(), &value.right, value.including_right)?;
                Box::new(RangeQuery::new_term_bounds(field, field_entry.field_type().value_type(), &left, &right))
            }
            Some(proto::query::Query::Boost(boost_query_proto)) => Box::new(BoostQuery::new(
                self.parse_query(boost_query_proto.query.as_ref().ok_or(Error::EmptyQueryError)?)?,
                f32::from_str(&boost_query_proto.score).map_err(|_e| InvalidSyntaxError("cannot parse f32".to_owned()))?,
            )),
            Some(proto::query::Query::Regex(regex_query_proto)) => {
                let (field, _) = self.field_and_field_entry(&regex_query_proto.field)?;
                Box::new(RegexQuery::from_pattern(&regex_query_proto.value, field)?)
            }
            Some(proto::query::Query::Phrase(phrase_query_proto)) => {
                let (field, field_entry) = self.field_and_field_entry(&phrase_query_proto.field)?;
                let tokenizer = self.index.tokenizer_for_field(field)?;
                let mut token_stream = tokenizer.token_stream(&phrase_query_proto.value);
                let mut terms = vec![];
                while let Some(token) = token_stream.next() {
                    terms.push(self.cast_value_to_term(field, field_entry.field_type(), &token.text)?)
                }
                if terms.len() == 1 {
                    Box::new(TermQuery::new(
                        terms[0].clone(),
                        field_entry.field_type().index_record_option().unwrap_or(IndexRecordOption::Basic),
                    ))
                } else {
                    let mut phrase_query = PhraseQuery::new(terms);
                    phrase_query.set_slop(phrase_query_proto.slop);
                    Box::new(phrase_query)
                }
            }
            Some(proto::query::Query::Term(term_query_proto)) => {
                let (field, field_entry) = self.field_and_field_entry(&term_query_proto.field)?;
                Box::new(TermQuery::new(
                    self.cast_value_to_term(field, field_entry.field_type(), &term_query_proto.value)?,
                    field_entry.field_type().index_record_option().unwrap_or(IndexRecordOption::Basic),
                ))
            }
            Some(proto::query::Query::MoreLikeThis(_)) => Box::new(AllQuery),
        })
    }

    /// Search `query` in the `IndexHolder` and collecting `Fruit` with a list of `collectors`
    pub(crate) async fn search(&self, query: &proto::Query, collectors: &Vec<proto::Collector>) -> SummaResult<Vec<proto::CollectorResult>> {
        let searcher = self.index_reader.searcher();
        let parsed_query = self.parse_query(query)?;
        let mut multi_collector = MultiCollector::new();
        let mut extractors: Vec<Box<dyn FruitExtractor>> = collectors
            .iter()
            .map(|proto::Collector { collector, .. }| match &collector {
                Some(proto::collector::Collector::TopDocs(top_docs_collector_proto)) => {
                    let top_docs_collector: TopDocs = top_docs_collector_proto.into();
                    Box::new(fruit_extractors::TopDocs::new(
                        multi_collector.add_collector(top_docs_collector),
                        top_docs_collector_proto.limit.try_into().unwrap(),
                    )) as Box<dyn FruitExtractor>
                }
                Some(proto::collector::Collector::ReservoirSampling(reservoir_sampling_collector_proto)) => {
                    let reservoir_sampling_collector: ReservoirSampling = reservoir_sampling_collector_proto.into();
                    Box::new(fruit_extractors::ReservoirSampling(multi_collector.add_collector(reservoir_sampling_collector))) as Box<dyn FruitExtractor>
                }
                Some(proto::collector::Collector::Count(count_collector_proto)) => {
                    let count_collector: Count = count_collector_proto.into();
                    Box::new(fruit_extractors::Count(multi_collector.add_collector(count_collector))) as Box<dyn FruitExtractor>
                }
                None => Box::new(fruit_extractors::Count(multi_collector.add_collector(Count))) as Box<dyn FruitExtractor>,
            })
            .collect();
        info!(target: "query", index_name = ?self.index_name, query = ?parsed_query);
        let multi_fields = self.multi_fields.clone();
        Ok(tokio::task::spawn_blocking(move || -> SummaResult<Vec<proto::CollectorResult>> {
            let mut multifruit = searcher.search(&parsed_query, &multi_collector)?;
            Ok(extractors.drain(..).map(|e| e.extract(&mut multifruit, &searcher, &multi_fields)).collect())
        })
        .await??)
    }
}

impl std::fmt::Debug for IndexHolder {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("IndexHolder").field("index_name", &self.index_name).finish()
    }
}

impl Into<proto::Index> for &IndexHolder {
    fn into(self) -> proto::Index {
        let index_config_proxy = self.index_config_proxy.read();
        proto::Index {
            index_aliases: index_config_proxy.application_config().get_index_aliases_for_index(&self.index_name),
            index_name: self.index_name.to_owned(),
            index_engine: format!("{:?}", index_config_proxy.index_engine),
            num_docs: self.index_reader().searcher().num_docs(),
        }
    }
}
