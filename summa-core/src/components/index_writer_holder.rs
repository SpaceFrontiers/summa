use std::fmt::{Debug, Formatter};
use std::path::{Path, PathBuf};
use std::sync::RwLock;

use summa_proto::proto;
use tantivy::schema::{Field, FieldType, Value};
use tantivy::{Document, Index, IndexWriter, SegmentId, SegmentMeta, SingleSegmentIndexWriter, Term};
use tracing::info;

use super::SummaSegmentAttributes;
use crate::components::frozen_log_merge_policy::FrozenLogMergePolicy;
use crate::configs::ApplicationConfig;
use crate::errors::{SummaResult, ValidationError};

#[derive(Clone)]
pub struct SegmentComponent {
    pub path: PathBuf,
    pub segment_component: tantivy::SegmentComponent,
}

impl Debug for SegmentComponent {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.path.fmt(f)
    }
}

#[derive(Clone, Debug)]
pub enum ComponentFile {
    SegmentComponent(SegmentComponent),
    Other(PathBuf),
}

impl ComponentFile {
    pub fn path(&self) -> &Path {
        match self {
            ComponentFile::SegmentComponent(segment_component) => &segment_component.path,
            ComponentFile::Other(path) => path,
        }
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

    pub fn delete_term(&self, term: Term) {
        match self {
            IndexWriterImpl::Single(_) => {}
            IndexWriterImpl::Threaded(writer) => {
                writer.delete_term(term);
            }
        };
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
    pub async fn commit(&mut self, payload: Option<String>) -> SummaResult<()> {
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
                let mut prepared_commit = writer.prepare_commit()?;
                if let Some(payload) = payload {
                    prepared_commit.set_payload(&payload);
                }
                let opstamp = prepared_commit.commit_future().await?;
                info!(action = "committed_index", opstamp = ?opstamp);
                Ok(())
            }
        }
    }
}

/// Managing write operations to index
pub struct IndexWriterHolder {
    index_writer: IndexWriterImpl,
    primary_key: Option<Field>,
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
        primary_key: Option<Field>,
        writer_threads: usize,
        writer_heap_size_bytes: usize,
    ) -> SummaResult<IndexWriterHolder> {
        if let Some(primary_key) = primary_key {
            match index_writer.index().schema().get_field_entry(primary_key).field_type() {
                FieldType::I64(_) => Ok(()),
                FieldType::Str(_) => Ok(()),
                another_type => Err(ValidationError::InvalidPrimaryKeyType(another_type.to_owned())),
            }?
        }
        Ok(IndexWriterHolder {
            index_writer,
            primary_key,
            writer_threads,
            writer_heap_size_bytes,
        })
    }

    /// Creates new `IndexWriterHolder` from `Index` and `ApplicationConfig`
    pub fn from_config(index: &Index, application_config: &ApplicationConfig) -> SummaResult<IndexWriterHolder> {
        let index_writer = IndexWriterImpl::new(
            index,
            application_config.writer_threads as usize,
            application_config.writer_heap_size_bytes as usize,
        )?;
        let primary_key = index
            .load_metas()?
            .attributes()?
            .and_then(|attributes: proto::IndexAttributes| {
                attributes.primary_key.map(|primary_key| {
                    index
                        .schema()
                        .get_field(&primary_key)
                        .ok_or(ValidationError::MissingPrimaryKey(Some(primary_key.to_string())))
                })
            })
            .transpose()?;
        IndexWriterHolder::new(
            index_writer,
            primary_key,
            application_config.writer_heap_size_bytes as usize,
            application_config.writer_threads as usize,
        )
    }

    /// Delete index by its primary key
    pub(super) fn delete_document(&self, document: &Document) -> SummaResult<()> {
        if let Some(primary_key) = self.primary_key {
            self.index_writer.delete_term(
                match document
                    .get_first(primary_key)
                    .ok_or_else(|| ValidationError::MissingPrimaryKey(Some(format!("{:?}", self.index_writer.index().schema().to_named_doc(document)))))?
                {
                    Value::Str(s) => Term::from_field_text(primary_key, s),
                    Value::I64(i) => Term::from_field_i64(primary_key, *i),
                    _ => Err(ValidationError::InvalidPrimaryKeyType(
                        self.index_writer.index().schema().get_field_entry(primary_key).field_type().clone(),
                    ))?,
                },
            )
        }
        Ok(())
    }

    /// Delete index by its primary key
    pub(super) fn delete_document_by_primary_key(&self, primary_key_value: Option<proto::PrimaryKey>) -> SummaResult<()> {
        self.primary_key
            .and_then(|primary_key| {
                primary_key_value.and_then(|primary_key_value| {
                    primary_key_value.value.map(|value| match value {
                        proto::primary_key::Value::Str(s) => self.index_writer.delete_term(Term::from_field_text(primary_key, &s)),
                        proto::primary_key::Value::I64(i) => self.index_writer.delete_term(Term::from_field_i64(primary_key, i)),
                    })
                })
            })
            .ok_or_else(|| ValidationError::MissingPrimaryKey(None).into())
    }

    /// Tantivy `Index`
    pub(super) fn index(&self) -> &Index {
        self.index_writer.index()
    }

    /// Put document to the index. Before comes searchable it must be committed
    pub fn index_document(&self, document: Document) -> SummaResult<()> {
        self.delete_document(&document)?;
        self.index_writer.add_document(document)
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
    pub async fn commit(&mut self, payload: Option<String>) -> SummaResult<()> {
        info!(action = "commit");
        let result = self.index_writer.commit(payload).await;
        info!(action = "committed");
        result
    }

    pub async fn vacuum(&mut self, segment_attributes: Option<SummaSegmentAttributes>) -> SummaResult<()> {
        let mut segments = self.index().searchable_segments()?;
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
    pub async fn lock_files<P, O, Fut>(&mut self, index_path: P, payload: Option<String>, f: impl FnOnce(Vec<ComponentFile>) -> Fut) -> SummaResult<O>
    where
        P: AsRef<Path>,
        Fut: std::future::Future<Output = SummaResult<O>>,
    {
        use tantivy::Directory;

        let segment_attributes = SummaSegmentAttributes { is_frozen: true };

        self.commit(None).await?;
        self.vacuum(Some(segment_attributes)).await?;
        self.commit(payload).await?;

        self.wait_merging_threads();

        let mut hotcache_bytes = vec![];

        let read_directory = tantivy::directory::MmapDirectory::open(&index_path)?;
        crate::directories::write_hotcache(read_directory, 16384, &mut hotcache_bytes)?;
        self.index()
            .directory()
            .atomic_write(&PathBuf::from("hotcache.bin".to_string()), &hotcache_bytes)?;

        let segment_files = [
            ComponentFile::Other(PathBuf::from(".managed.json")),
            ComponentFile::Other(PathBuf::from("meta.json")),
            ComponentFile::Other(PathBuf::from("hotcache.bin")),
        ]
        .into_iter()
        .chain(self.get_index_files(index_path.as_ref().to_path_buf())?)
        .collect();
        f(segment_files).await
    }

    /// Get segments
    #[cfg(feature = "fs")]
    fn get_index_files(&self, index_path: PathBuf) -> SummaResult<impl Iterator<Item = ComponentFile>> {
        Ok(self.index().searchable_segments()?.into_iter().flat_map(move |segment| {
            tantivy::SegmentComponent::iterator()
                .filter_map(|segment_component| {
                    let relative_path = segment.meta().relative_path(*segment_component);
                    index_path.join(relative_path).exists().then(|| {
                        ComponentFile::SegmentComponent(SegmentComponent {
                            path: segment.meta().relative_path(*segment_component),
                            segment_component: *segment_component,
                        })
                    })
                })
                .collect::<Vec<_>>()
        }))
    }
}
