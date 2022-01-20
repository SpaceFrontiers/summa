use std::fmt::{Debug, Formatter};
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::RwLock;

use summa_proto::proto;
use tantivy::query::{BooleanQuery, Query};
use tantivy::schema::{Field, Value};
use tantivy::{Directory, Document, Index, IndexWriter, SegmentId, SegmentMeta, SingleSegmentIndexWriter, Term};
use tracing::info;

use super::SummaSegmentAttributes;
use crate::components::frozen_log_merge_policy::FrozenLogMergePolicy;
use crate::errors::{SummaResult, ValidationError};

pub struct ComponentFile {
    file_name: String,
    boxed_reader: Box<dyn Future<Output = Vec<u8>> + Send>,
}

impl Debug for ComponentFile {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComponentFile").field("file_name", &self.file_name).finish()
    }
}

impl ComponentFile {
    pub fn new(file_name: &str, boxed_reader: Box<dyn Future<Output = Vec<u8>> + Send>) -> ComponentFile {
        ComponentFile {
            file_name: file_name.to_string(),
            boxed_reader,
        }
    }
    pub fn from_directory<D: Directory + Clone>(directory: &D, file_name: &str) -> ComponentFile {
        let directory = directory.clone();
        let filepath = PathBuf::from(file_name);
        ComponentFile::new(
            file_name,
            Box::new(async move { directory.atomic_read_async(&filepath).await.expect("cannot open") }),
        )
    }
    pub fn file_name(&self) -> &str {
        &self.file_name
    }
    pub fn into_reader(self) -> Pin<Box<dyn Future<Output = Vec<u8>> + Send>> {
        Box::into_pin(self.boxed_reader)
    }
}

pub struct SingleIndexWriter {
    pub index_writer: RwLock<SingleSegmentIndexWriter>,
    pub index: Index,
    pub writer_heap_size_bytes: usize,
}

pub enum IndexWriterImpl {
    Single(SingleIndexWriter),
    Threaded(IndexWriter),
}

impl IndexWriterImpl {
    pub fn new(index: &Index, writer_threads: usize, writer_heap_size_bytes: usize) -> SummaResult<Self> {
        Ok(if writer_threads == 0 {
            IndexWriterImpl::Single(SingleIndexWriter {
                index: index.clone(),
                index_writer: RwLock::new(SingleSegmentIndexWriter::new(index.clone(), writer_heap_size_bytes)?),
                writer_heap_size_bytes,
            })
        } else {
            let index_writer = index.writer_with_num_threads(writer_threads, writer_heap_size_bytes)?;
            index_writer.set_merge_policy(Box::<FrozenLogMergePolicy>::default());
            IndexWriterImpl::Threaded(index_writer)
        })
    }

    pub fn delete_documents(&self, query: Box<dyn Query>) -> SummaResult<u64> {
        match self {
            IndexWriterImpl::Single(_) => Ok(0),
            IndexWriterImpl::Threaded(writer) => Ok(writer.delete_query(query)?),
        }
    }
    pub fn add_document(&self, document: Document) -> SummaResult<()> {
        match self {
            IndexWriterImpl::Single(writer) => {
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
            IndexWriterImpl::Single(writer) => &writer.index,
            IndexWriterImpl::Threaded(writer) => writer.index(),
        }
    }
    pub async fn merge_with_attributes(
        &mut self,
        segment_ids: &[SegmentId],
        segment_attributes: Option<serde_json::Value>,
    ) -> SummaResult<Option<SegmentMeta>> {
        match self {
            IndexWriterImpl::Single(_) => {
                unimplemented!()
            }
            IndexWriterImpl::Threaded(writer) => Ok(writer.merge_with_attributes(segment_ids, segment_attributes).await?),
        }
    }
    pub async fn commit(&mut self) -> SummaResult<()> {
        match self {
            IndexWriterImpl::Single(writer) => {
                let index = writer.index.clone();
                let writer_heap_size_bytes = writer.writer_heap_size_bytes;
                let writer = writer.index_writer.get_mut().expect("poisoned");
                take_mut::take(writer, |writer| {
                    writer.finalize().expect("cannot finalize");
                    SingleSegmentIndexWriter::new(index.clone(), writer_heap_size_bytes).expect("cannot recreate writer")
                });
                Ok(())
            }
            IndexWriterImpl::Threaded(writer) => {
                info!(action = "commit_index");
                let opstamp = writer.prepare_commit()?.commit_future().await?;
                info!(action = "committed_index", opstamp = ?opstamp);
                Ok(())
            }
        }
    }
    pub async fn rollback(&mut self) -> SummaResult<()> {
        match self {
            IndexWriterImpl::Single(_) => unimplemented!(),
            IndexWriterImpl::Threaded(writer) => {
                info!(action = "commit_index");
                let opstamp = writer.rollback_async().await?;
                info!(action = "committed_index", opstamp = ?opstamp);
                Ok(())
            }
        }
    }
}

/// Managing write operations to index
pub struct IndexWriterHolder {
    index_writer: IndexWriterImpl,
    unique_fields: Vec<Field>,
    writer_threads: usize,
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
        unique_fields: Vec<Field>,
        writer_threads: usize,
        writer_heap_size_bytes: usize,
    ) -> SummaResult<IndexWriterHolder> {
        Ok(IndexWriterHolder {
            index_writer,
            unique_fields,
            writer_threads,
            writer_heap_size_bytes,
        })
    }

    /// Creates new `IndexWriterHolder` from `Index` and `core::Config`
    pub fn from_config(index: &Index, core_config: &crate::configs::core::Config) -> SummaResult<IndexWriterHolder> {
        let index_writer = IndexWriterImpl::new(index, core_config.writer_threads as usize, core_config.writer_heap_size_bytes as usize)?;
        let unique_fields = index
            .load_metas()?
            .index_attributes()?
            .map(|attributes: proto::IndexAttributes| {
                attributes
                    .unique_fields
                    .iter()
                    .map(|unique_field| {
                        index
                            .schema()
                            .get_field(unique_field)
                            .ok_or(ValidationError::MissingUniqueField(unique_field.to_string()).into())
                    })
                    .collect::<SummaResult<_>>()
            })
            .transpose()?
            .unwrap_or_default();
        IndexWriterHolder::new(
            index_writer,
            unique_fields,
            core_config.writer_threads as usize,
            core_config.writer_heap_size_bytes as usize,
        )
    }

    /// Delete index by its unique fields
    pub(super) fn delete_documents_by_unique_fields(&self, document: &Document) -> SummaResult<u64> {
        let unique_terms = self
            .unique_fields
            .iter()
            .map(|unique_field| {
                Ok(
                    match document
                        .get_first(*unique_field)
                        .ok_or_else(|| ValidationError::MissingUniqueField(format!("{:?}", self.index_writer.index().schema().to_named_doc(document))))?
                    {
                        Value::Str(s) => Term::from_field_text(*unique_field, s),
                        Value::I64(i) => Term::from_field_i64(*unique_field, *i),
                        _ => Err(ValidationError::InvalidUniqueFieldType(
                            self.index_writer.index().schema().get_field_entry(*unique_field).field_type().clone(),
                        ))?,
                    },
                )
            })
            .collect::<SummaResult<_>>()?;
        self.delete_documents(Box::new(BooleanQuery::new_multiterms_query(unique_terms)))
    }

    /// Delete index by its primary key
    pub(super) fn delete_documents(&self, query: Box<dyn Query>) -> SummaResult<u64> {
        self.index_writer.delete_documents(query)
    }

    /// Tantivy `Index`
    pub(super) fn index(&self) -> &Index {
        self.index_writer.index()
    }

    /// Put document to the index. Before comes searchable it must be committed
    pub fn index_document(&self, document: Document) -> SummaResult<u64> {
        let deleted_documents = self.delete_documents_by_unique_fields(&document)?;
        self.index_writer.add_document(document)?;
        Ok(deleted_documents)
    }

    /// Merge segments into one.
    ///
    /// Also cleans deleted documents and do recompression. Possible to pass the only segment in `segment_ids` to do recompression or clean up.
    /// It is heavy operation that also blocks on `.await` so should be spawned if non-blocking behaviour is required
    pub async fn merge(&mut self, segment_ids: &[SegmentId], segment_attributes: Option<SummaSegmentAttributes>) -> SummaResult<Option<SegmentMeta>> {
        info!(action = "merge_segments", segment_ids = ?segment_ids);
        let segment_meta = self
            .index_writer
            .merge_with_attributes(
                segment_ids,
                segment_attributes.map(|segment_attributes| serde_json::to_value(segment_attributes).expect("cannot serialize")),
            )
            .await?;
        info!(action = "merged_segments", segment_ids = ?segment_ids, merged_segment_meta = ?segment_meta);
        Ok(segment_meta)
    }

    /// Commits already indexed documents
    ///
    /// Committing makes indexed documents visible
    /// It is heavy operation that also blocks on `.await` so should be spawned if non-blocking behaviour is required
    pub async fn commit(&mut self) -> SummaResult<()> {
        info!(action = "commit");
        let result = self.index_writer.commit().await;
        info!(action = "committed", result = ?result);
        result
    }

    pub async fn rollback(&mut self) -> SummaResult<()> {
        info!(action = "rollback");
        let result = self.index_writer.rollback().await;
        info!(action = "rollbacked", result = ?result);
        result
    }

    pub async fn vacuum(&mut self, segment_attributes: Option<SummaSegmentAttributes>) -> SummaResult<()> {
        let mut segments = self.index().searchable_segments_async().await?;
        segments.sort_by_key(|segment| segment.meta().num_deleted_docs());

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
                !is_frozen
            })
            .collect::<Vec<_>>();
        if !segments.is_empty() {
            self.merge(&segments.iter().map(|segment| segment.id()).collect::<Vec<_>>(), segment_attributes.clone())
                .await?;
        }
        Ok(())
    }

    pub fn wait_merging_threads(&mut self) {
        match &mut self.index_writer {
            IndexWriterImpl::Single(_) => (),
            IndexWriterImpl::Threaded(index_writer) => take_mut::take(index_writer, |index_writer| {
                let index = index_writer.index().clone();
                index_writer.wait_merging_threads().expect("cannot wait merging threads");
                index
                    .writer_with_num_threads(self.writer_threads, self.writer_heap_size_bytes)
                    .expect("cannot create index writer_holder")
            }),
        };
    }

    /// Locking index files for executing operation on them
    #[cfg(feature = "fs")]
    pub async fn lock_files<O, Fut>(&mut self, with_hotcache: bool, f: impl FnOnce(Vec<ComponentFile>) -> Fut) -> SummaResult<O>
    where
        Fut: Future<Output = SummaResult<O>>,
    {
        let segment_attributes = SummaSegmentAttributes { is_frozen: true };

        self.commit().await?;
        self.vacuum(Some(segment_attributes)).await?;
        self.commit().await?;

        self.wait_merging_threads();

        let directory = self.index().directory();

        let segment_files_iter = [".managed.json", "meta.json"]
            .into_iter()
            .map(String::from)
            .map(move |file_name| ComponentFile::from_directory(directory, &file_name))
            .chain(self.get_index_files().await?);
        let segment_files = match with_hotcache {
            true => {
                let hotcache_bytes = crate::directories::write_hotcache(directory.inner_directory().box_clone(), 16384).await?;
                segment_files_iter
                    .chain(std::iter::once(ComponentFile::new("hotcache.bin", Box::new(async move { hotcache_bytes }))))
                    .collect()
            }
            false => segment_files_iter.collect(),
        };
        f(segment_files).await
    }

    /// Get segments
    #[cfg(feature = "fs")]
    async fn get_index_files(&self) -> SummaResult<impl Iterator<Item = ComponentFile>> {
        let directory = self.index().directory().clone();
        Ok(self.index().searchable_segments_async().await?.into_iter().flat_map(move |segment| {
            let directory = directory.clone();
            tantivy::SegmentComponent::iterator()
                .filter_map(move |segment_component| {
                    let filepath = segment.meta().relative_path(*segment_component);
                    let file_name = filepath.to_string_lossy().to_string();
                    directory
                        .exists(&filepath)
                        .expect("cannot parse")
                        .then(|| ComponentFile::from_directory(&directory, &file_name))
                })
                .collect::<Vec<_>>()
        }))
    }
}
