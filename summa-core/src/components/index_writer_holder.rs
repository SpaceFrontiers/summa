use std::collections::hash_map::RandomState;
use std::collections::HashSet;
use std::path::Path;
use std::sync::{Arc, RwLock};

use summa_proto::proto;
use tantivy::merge_policy::MergePolicy;
use tantivy::query::Query;
use tantivy::schema::{Field, Value};
use tantivy::{Directory, Document, Index, IndexWriter, Opstamp, SegmentId, SegmentMeta, SingleSegmentIndexWriter, Term};
use tracing::info;

use super::SummaSegmentAttributes;
use crate::configs::core::WriterThreads;
use crate::errors::{SummaResult, ValidationError};
use crate::Error;

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
    pub fn merge_with_attributes(&mut self, segment_ids: &[SegmentId], segment_attributes: Option<serde_json::Value>) -> SummaResult<Option<SegmentMeta>> {
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
    unique_fields: Vec<Field>,
    writer_threads: WriterThreads,
    writer_heap_size_bytes: usize,
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
        unique_fields: Vec<Field>,
        writer_threads: WriterThreads,
        writer_heap_size_bytes: usize,
    ) -> SummaResult<IndexWriterHolder> {
        Ok(IndexWriterHolder {
            index_writer,
            merge_policy,
            unique_fields,
            writer_threads,
            writer_heap_size_bytes,
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
        let unique_fields = index
            .load_metas()?
            .index_attributes()?
            .map(|attributes: proto::IndexAttributes| {
                attributes
                    .unique_fields
                    .iter()
                    .map(|unique_field| index.schema().get_field(unique_field))
                    .collect::<Result<_, _>>()
            })
            .transpose()?
            .unwrap_or_default();
        IndexWriterHolder::new(index_writer, merge_policy, unique_fields, writer_threads, writer_heap_size_bytes)
    }

    /// Delete index by its unique fields
    pub(super) fn resolve_conflicts(&self, document: &Document, conflict_strategy: proto::ConflictStrategy) -> SummaResult<Option<u64>> {
        if self.unique_fields.is_empty() {
            return Ok(None);
        }

        let unique_terms = self
            .unique_fields
            .iter()
            .filter_map(|unique_field| {
                document.get_first(*unique_field).map(|value| match value {
                    Value::Str(s) => Ok(Term::from_field_text(*unique_field, s)),
                    Value::I64(i) => Ok(Term::from_field_i64(*unique_field, *i)),
                    _ => Err(Error::Validation(Box::new(ValidationError::InvalidUniqueFieldType(
                        self.index_writer.index().schema().get_field_entry(*unique_field).field_type().clone(),
                    )))),
                })
            })
            .collect::<SummaResult<Vec<_>>>()?;

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

    /// Put document to the index. Before comes searchable it must be committed
    pub fn index_document(&self, document: Document, conflict_strategy: proto::ConflictStrategy) -> SummaResult<()> {
        self.resolve_conflicts(&document, conflict_strategy)?;
        self.index_writer.add_document(document)?;
        Ok(())
    }

    /// Merge segments into one.
    ///
    /// Also cleans deleted documents and do recompression. Possible to pass the only segment in `segment_ids` to do recompression or clean up.
    /// It is heavy operation that also blocks on `.await` so should be spawned if non-blocking behaviour is required
    pub fn merge(&mut self, segment_ids: &[SegmentId], segment_attributes: Option<SummaSegmentAttributes>) -> SummaResult<Option<SegmentMeta>> {
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

    pub fn vacuum(&mut self, segment_attributes: Option<SummaSegmentAttributes>, excluded_segments: Vec<String>) -> SummaResult<()> {
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
