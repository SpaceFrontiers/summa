use std::collections::hash_map::RandomState;
use std::collections::HashSet;
use std::path::Path;
use std::sync::{Arc, RwLock};

use chrono::{DateTime, Datelike, NaiveDateTime, Utc};
use summa_proto::proto;
use tantivy::json_utils::{convert_to_fast_value_and_get_term, JsonTermWriter};
use tantivy::merge_policy::MergePolicy;
use tantivy::query::Query;
use tantivy::schema::{Field, Value};
use tantivy::{Directory, Document, Index, IndexWriter, Opstamp, SegmentId, SegmentMeta, SingleSegmentIndexWriter, Term};
use tracing::info;

use super::SummaSegmentAttributes;
use crate::configs::core::WriterThreads;
use crate::errors::{SummaResult, ValidationError};
use crate::Error;

fn extract_flatten<T: AsRef<str>>(v: &serde_json::Value, parts: &[T], buffer: &mut Vec<String>) {
    let mut current = v;
    for (i, part) in parts.iter().enumerate() {
        match current {
            serde_json::value::Value::Object(m) => {
                if let Some(next) = m.get(part.as_ref()) {
                    current = next
                }
            }
            serde_json::value::Value::Array(a) => {
                for child in a {
                    extract_flatten(child, &parts[i..], buffer)
                }
            }
            _ => break,
        }
    }
    if let serde_json::Value::String(last_value) = &current {
        buffer.push(last_value.to_string())
    }
}

fn extract_flatten_from_map<T: AsRef<str>>(m: &serde_json::value::Map<String, serde_json::Value>, parts: &[T], buffer: &mut Vec<String>) {
    for (i, part) in parts.iter().enumerate() {
        match m.get(part.as_ref()) {
            Some(v) => match v {
                serde_json::value::Value::Array(a) => {
                    for child in a {
                        extract_flatten(child, &parts[i..], buffer)
                    }
                }
                _ => extract_flatten(v, &parts[i..], buffer),
            },
            None => break,
        }
    }
}

fn cast_to_term(unique_field: &Field, full_path: &str, value: &serde_json::Value) -> Vec<Term> {
    let mut term = Term::with_capacity(128);
    let mut json_term_writer = JsonTermWriter::from_field_and_json_path(*unique_field, full_path, true, &mut term);
    match value {
        serde_json::Value::Number(n) => {
            vec![convert_to_fast_value_and_get_term(&mut json_term_writer, &n.to_string()).expect("incorrect json type")]
        }
        serde_json::Value::String(s) => {
            let mut term = Term::with_capacity(128);
            let mut json_term_writer = JsonTermWriter::from_field_and_json_path(*unique_field, full_path, true, &mut term);
            json_term_writer.set_str(s);
            vec![json_term_writer.term().clone()]
        }
        serde_json::Value::Array(v) => v.iter().flat_map(|e| cast_to_term(unique_field, full_path, e)).collect(),
        _ => unreachable!(),
    }
}

/// Wrap `tantivy::SingleSegmentIndexWriter` and allows to recreate it
pub struct SingleIndexWriter {
    pub index_writer: RwLock<SingleSegmentIndexWriter>,
    pub index: Index,
    pub writer_heap_size_bytes: usize,
}

/// Hold same-thread or pooled implementation of `IndexWriter`
pub enum IndexWriterImpl {
    SameThread(SingleIndexWriter),
    Threaded(IndexWriter),
}

impl IndexWriterImpl {
    pub fn new(index: &Index, writer_threads: WriterThreads, writer_heap_size_bytes: usize, merge_policy: Arc<dyn MergePolicy>) -> SummaResult<Self> {
        Ok(match writer_threads {
            WriterThreads::SameThread => IndexWriterImpl::SameThread(SingleIndexWriter {
                index: index.clone(),
                index_writer: RwLock::new(SingleSegmentIndexWriter::new(index.clone(), writer_heap_size_bytes)?),
                writer_heap_size_bytes,
            }),
            WriterThreads::N(writer_threads) => {
                let index_writer = index.writer_with_num_threads(writer_threads as usize, writer_heap_size_bytes)?;
                index_writer.set_merge_policy(merge_policy);
                IndexWriterImpl::Threaded(index_writer)
            }
        })
    }

    pub fn delete_by_query(&self, query: Box<dyn Query>) -> SummaResult<u64> {
        match self {
            IndexWriterImpl::SameThread(_) => unimplemented!(),
            IndexWriterImpl::Threaded(writer) => Ok(writer.delete_query(query)?),
        }
    }

    pub fn delete_by_term(&self, term: Term) -> u64 {
        match self {
            IndexWriterImpl::SameThread(_) => unimplemented!(),
            IndexWriterImpl::Threaded(writer) => writer.delete_term(term),
        }
    }

    pub fn add_document(&self, document: Document) -> SummaResult<()> {
        match self {
            IndexWriterImpl::SameThread(writer) => {
                writer.index_writer.write().expect("poisoned").add_document(document)?;
            }
            IndexWriterImpl::Threaded(writer) => {
                writer.add_document(document)?;
            }
        };
        Ok(())
    }
    pub fn index(&self) -> &Index {
        match self {
            IndexWriterImpl::SameThread(writer) => &writer.index,
            IndexWriterImpl::Threaded(writer) => writer.index(),
        }
    }
    pub fn merge_with_attributes(&self, segment_ids: &[SegmentId], segment_attributes: Option<serde_json::Value>) -> SummaResult<Option<SegmentMeta>> {
        match self {
            IndexWriterImpl::SameThread(_) => {
                unimplemented!()
            }
            IndexWriterImpl::Threaded(writer) => {
                let target_segment = writer.merge_with_attributes(segment_ids, segment_attributes).wait()?;
                writer.garbage_collect_files().wait()?;
                Ok(target_segment)
            }
        }
    }
    pub fn commit(&mut self) -> SummaResult<Opstamp> {
        match self {
            IndexWriterImpl::SameThread(writer) => {
                let index = writer.index.clone();
                let writer_heap_size_bytes = writer.writer_heap_size_bytes;
                let writer = writer.index_writer.get_mut().expect("poisoned");
                take_mut::take(writer, |writer| {
                    writer.finalize().expect("cannot finalize");
                    SingleSegmentIndexWriter::new(index.clone(), writer_heap_size_bytes).expect("cannot recreate writer")
                });
                Ok(0)
            }
            IndexWriterImpl::Threaded(writer) => {
                info!(action = "commit_files");
                let opstamp = writer.prepare_commit()?.commit()?;
                info!(action = "committed", opstamp = ?opstamp);
                Ok(opstamp)
            }
        }
    }
    pub fn rollback(&mut self) -> SummaResult<()> {
        match self {
            IndexWriterImpl::SameThread(_) => unimplemented!(),
            IndexWriterImpl::Threaded(writer) => {
                info!(action = "rollback_files");
                let opstamp = writer.rollback()?;
                info!(action = "rollbacked", opstamp = ?opstamp);
                Ok(())
            }
        }
    }
}

/// Managing write operations to index
pub struct IndexWriterHolder {
    index_writer: IndexWriterImpl,
    merge_policy: Arc<dyn MergePolicy>,
    unique_fields: Vec<(Field, String)>,
    writer_threads: WriterThreads,
    writer_heap_size_bytes: usize,

    extra_year_field: Option<(Field, Field)>,
    mapped_fields: Vec<((Field, Vec<String>), Field)>,
}

impl IndexWriterHolder {
    /// Creates new `IndexWriterHolder` containing `tantivy::IndexWriter` and primary key
    ///
    /// `IndexWriterHolder` maintains invariant that the only document with the particular primary key exists in the index.
    /// It is reached by deletion of every document with the same primary key as indexing one.
    /// The type of primary key is restricted to I64 but it is subjected to be changed in the future.
    pub(super) fn new(
        index_writer: IndexWriterImpl,
        merge_policy: Arc<dyn MergePolicy>,
        unique_fields: Vec<(Field, String)>,
        mapped_fields: Vec<((Field, Vec<String>), Field)>,
        writer_threads: WriterThreads,
        writer_heap_size_bytes: usize,
    ) -> SummaResult<IndexWriterHolder> {
        let schema = index_writer.index().schema();
        let extra_year_field = if let (Ok(extra_field), Ok(issued_at_field)) = (schema.get_field("extra"), schema.get_field("issued_at")) {
            Some((extra_field, issued_at_field))
        } else {
            None
        };
        Ok(IndexWriterHolder {
            index_writer,
            merge_policy,
            unique_fields,
            writer_threads,
            writer_heap_size_bytes,
            extra_year_field,
            mapped_fields,
        })
    }

    /// Creates new `IndexWriterHolder` from `Index` and `core::Config`
    pub fn create(
        index: &Index,
        writer_threads: WriterThreads,
        writer_heap_size_bytes: usize,
        merge_policy: Arc<dyn MergePolicy>,
    ) -> SummaResult<IndexWriterHolder> {
        let index_writer = IndexWriterImpl::new(index, writer_threads.clone(), writer_heap_size_bytes, merge_policy.clone())?;
        let schema = index_writer.index().schema();
        let metas = index.load_metas()?;
        let mapped_fields = metas
            .index_attributes()?
            .map(|attributes: proto::IndexAttributes| {
                attributes
                    .mapped_fields
                    .iter()
                    .map(|proto::MappedField { source_field, target_field }| {
                        Ok::<((Field, Vec<String>), Field), ValidationError>((
                            schema
                                .find_field(source_field)
                                .ok_or_else(|| ValidationError::MissingField(source_field.to_string()))
                                .map(|(field, full_path)| (field, full_path.split('.').map(|x| x.to_string()).collect()))?,
                            schema
                                .get_field(target_field)
                                .map_err(|_| ValidationError::MissingField(source_field.to_string()))?,
                        ))
                    })
                    .collect::<Result<Vec<(_, _)>, _>>()
            })
            .transpose()?
            .unwrap_or_default();
        let unique_fields = metas
            .index_attributes()?
            .map(|attributes: proto::IndexAttributes| {
                attributes
                    .unique_fields
                    .iter()
                    .map(|unique_field| {
                        schema
                            .find_field(unique_field)
                            .ok_or_else(|| ValidationError::MissingField(unique_field.to_string()))
                            .map(|x| (x.0, x.1.to_string()))
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()?
            .unwrap_or_default();
        IndexWriterHolder::new(index_writer, merge_policy, unique_fields, mapped_fields, writer_threads, writer_heap_size_bytes)
    }

    /// Delete index by its unique fields
    pub(super) fn resolve_conflicts(&self, document: &mut Document, conflict_strategy: proto::ConflictStrategy) -> SummaResult<Option<u64>> {
        if self.unique_fields.is_empty() || matches!(conflict_strategy, proto::ConflictStrategy::DoNothing) {
            return Ok(None);
        }

        let unique_terms: Vec<Term> = self
            .unique_fields
            .iter()
            .filter_map(|(unique_field, full_path)| {
                document.get_first(*unique_field).and_then(|value| match value {
                    Value::Str(s) => Some(Ok(vec![Term::from_field_text(*unique_field, s)])),
                    Value::JsonObject(i) => i.get(full_path).map(|value| Ok(cast_to_term(unique_field, full_path, value))),
                    Value::I64(i) => Some(Ok(vec![Term::from_field_i64(*unique_field, *i)])),
                    Value::U64(i) => Some(Ok(vec![Term::from_field_u64(*unique_field, *i)])),
                    Value::F64(i) => Some(Ok(vec![Term::from_field_f64(*unique_field, *i)])),
                    _ => {
                        let schema = self.index_writer.index().schema();
                        let field_type = schema.get_field_entry(*unique_field).field_type();
                        Some(Err(Error::Validation(Box::new(ValidationError::InvalidUniqueFieldType(field_type.clone())))))
                    }
                })
            })
            .collect::<SummaResult<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();

        if unique_terms.is_empty() {
            Err(ValidationError::MissingUniqueField(format!(
                "{:?}",
                self.index_writer.index().schema().to_named_doc(document)
            )))?
        }

        let mut last_opstamp = None;
        for term in unique_terms {
            last_opstamp = Some(self.delete_by_term(term))
        }

        Ok(last_opstamp)
    }

    /// Delete documents by query
    pub(super) fn delete_by_query(&self, query: Box<dyn Query>) -> SummaResult<u64> {
        self.index_writer.delete_by_query(query)
    }

    /// Delete documents by `Term`
    pub(super) fn delete_by_term(&self, term: Term) -> u64 {
        self.index_writer.delete_by_term(term)
    }

    /// Tantivy `Index`
    pub(super) fn index(&self) -> &Index {
        self.index_writer.index()
    }

    #[inline]
    fn process_dynamic_fields(&self, document: &mut Document) -> SummaResult<()> {
        if let Some((extra_field, issued_at_field)) = self.extra_year_field {
            if let Some(issued_at_value) = document.get_first(issued_at_field) {
                if let Some(issued_at_value) = issued_at_value.as_i64() {
                    if let Some(correct_timestamp) = NaiveDateTime::from_timestamp_opt(issued_at_value, 0) {
                        let datetime: DateTime<Utc> = DateTime::from_utc(correct_timestamp, Utc);
                        document.add_text(extra_field, datetime.year().to_string())
                    }
                }
            }
        }

        let mut buffer = vec![];
        for ((source_field, source_full_path), target_field) in &self.mapped_fields {
            for value in document.get_all(*source_field) {
                match value {
                    Value::Str(s) => buffer.push(s.to_string()),
                    Value::JsonObject(m) => {
                        extract_flatten_from_map(m, source_full_path, &mut buffer);
                    }
                    _ => unimplemented!(),
                }
            }
            for v in &buffer {
                document.add_text(*target_field, v)
            }
            buffer.clear();
        }
        Ok(())
    }

    /// Put document to the index. Before comes searchable it must be committed
    pub fn index_document(&self, mut document: Document, conflict_strategy: proto::ConflictStrategy) -> SummaResult<()> {
        self.process_dynamic_fields(&mut document)?;
        self.resolve_conflicts(&mut document, conflict_strategy)?;
        self.index_writer.add_document(document)?;
        Ok(())
    }

    /// Merge segments into one.
    ///
    /// Also cleans deleted documents and do recompression. Possible to pass the only segment in `segment_ids` to do recompression or clean up.
    /// It is heavy operation that also blocks on `.await` so should be spawned if non-blocking behaviour is required
    pub fn merge(&self, segment_ids: &[SegmentId], segment_attributes: Option<SummaSegmentAttributes>) -> SummaResult<Option<SegmentMeta>> {
        info!(action = "merge_segments", segment_ids = ?segment_ids);
        let segment_meta = self.index_writer.merge_with_attributes(
            segment_ids,
            segment_attributes.map(|segment_attributes| serde_json::to_value(segment_attributes).expect("cannot serialize")),
        )?;
        info!(action = "merged_segments", segment_ids = ?segment_ids, merged_segment_meta = ?segment_meta);
        Ok(segment_meta)
    }

    /// Commits already indexed documents
    ///
    /// Committing makes indexed documents visible
    /// It is heavy operation that also blocks on `.await` so should be spawned if non-blocking behaviour is required
    pub fn commit(&mut self) -> SummaResult<Opstamp> {
        self.index_writer.commit()
    }

    pub fn rollback(&mut self) -> SummaResult<()> {
        self.index_writer.rollback()
    }

    pub fn vacuum(&self, segment_attributes: Option<SummaSegmentAttributes>, excluded_segments: Vec<String>) -> SummaResult<()> {
        let mut segments = self.index().searchable_segments()?;
        segments.sort_by_key(|segment| segment.meta().num_deleted_docs());

        let excluded_segments: HashSet<SegmentId, RandomState> = excluded_segments
            .into_iter()
            .map(|s| SegmentId::from_uuid_string(&s))
            .collect::<Result<_, _>>()
            .map_err(|e| Error::InvalidSegmentId(e.to_string()))?;

        let segments = segments
            .into_iter()
            .filter(|segment| {
                let is_frozen = segment
                    .meta()
                    .segment_attributes()
                    .as_ref()
                    .map(|segment_attributes| {
                        let parsed_attributes = serde_json::from_value::<SummaSegmentAttributes>(segment_attributes.clone());
                        parsed_attributes.map(|v| v.is_frozen).unwrap_or(false)
                    })
                    .unwrap_or(false);
                let is_excluded = excluded_segments.contains(&segment.id());
                !is_frozen && !is_excluded
            })
            .collect::<Vec<_>>();
        if !segments.is_empty() {
            self.merge(&segments.iter().map(|segment| segment.id()).collect::<Vec<_>>(), segment_attributes)?;
        }
        Ok(())
    }

    pub fn wait_merging_threads(&mut self) {
        match &mut self.index_writer {
            IndexWriterImpl::SameThread(_) => (),
            IndexWriterImpl::Threaded(index_writer) => take_mut::take(index_writer, |index_writer| {
                let index = index_writer.index().clone();
                info!(action = "wait_merging_threads", mode = "threaded");
                index_writer.wait_merging_threads().expect("cannot wait merging threads");
                info!(action = "merging_threads_finished", mode = "threaded");
                let index_writer = index
                    .writer_with_num_threads(self.writer_threads.threads() as usize, self.writer_heap_size_bytes)
                    .expect("cannot create index writer_holder");
                index_writer.set_merge_policy(self.merge_policy.clone());
                index_writer
            }),
        };
    }

    /// Locking index files for executing operation on them
    pub fn commit_and_prepare(&mut self, with_hotcache: bool) -> SummaResult<Opstamp> {
        let opstamp = self.commit()?;
        self.wait_merging_threads();

        if with_hotcache {
            let directory = self.index().directory();
            let hotcache_bytes = crate::directories::create_hotcache(
                directory
                    .underlying_directory()
                    .expect("managed directory should contain nested directory")
                    .box_clone(),
            )?;
            directory.atomic_write(Path::new(&format!("hotcache.{}.bin", opstamp)), &hotcache_bytes)?;
        }
        Ok(opstamp)
    }

    pub fn lock_files(&mut self, with_hotcache: bool) -> SummaResult<Vec<String>> {
        let mut segment_files = vec![".managed.json".to_string(), "meta.json".to_string()];
        let opstamp = self.commit_and_prepare(with_hotcache)?;
        if with_hotcache {
            segment_files.push(format!("hotcache.{}.bin", opstamp))
        }
        segment_files.extend(self.get_index_files()?);
        Ok(segment_files)
    }

    /// Get segments
    fn get_index_files(&self) -> SummaResult<impl Iterator<Item = String> + '_> {
        Ok(self.index().searchable_segments()?.into_iter().flat_map(|segment| {
            tantivy::SegmentComponent::iterator()
                .filter_map(|segment_component| {
                    let filepath = segment.meta().relative_path(*segment_component);
                    let file_name = filepath.to_string_lossy().to_string();
                    self.index().directory().exists(&filepath).expect("cannot parse").then_some(file_name)
                })
                .collect::<Vec<_>>()
        }))
    }
}
