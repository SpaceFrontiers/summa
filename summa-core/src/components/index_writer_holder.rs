use std::sync::RwLock;

use summa_proto::proto;
use tantivy::schema::{Field, FieldType};
use tantivy::{Document, Index, IndexWriter, SegmentId, SegmentMeta, SingleSegmentIndexWriter, Term};
use tracing::info;

use super::SummaSegmentAttributes;
use crate::components::frozen_log_merge_policy::FrozenLogMergePolicy;
use crate::configs::IndexConfig;
use crate::errors::{SummaResult, ValidationError};

pub struct SingleIndexWriter {
    pub index_writer: RwLock<SingleSegmentIndexWriter>,
    pub index_config: IndexConfig,
    pub index: Index,
}

pub enum IndexWriterImpl {
    Single(SingleIndexWriter),
    Threaded(IndexWriter),
}

impl IndexWriterImpl {
    pub fn delete_term(&self, term: Term) {
        match self {
            IndexWriterImpl::Single(_) => {
                unimplemented!()
            }
            IndexWriterImpl::Threaded(writer) => writer.delete_term(term),
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
    pub fn wait_merging_threads(self) -> SummaResult<()> {
        match self {
            IndexWriterImpl::Single(_) => Ok(()),
            IndexWriterImpl::Threaded(writer) => Ok(writer.wait_merging_threads()?),
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
                let writer_heap_size_bytes = writer.index_config.writer_heap_size_bytes as usize;
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
}

impl IndexWriterHolder {
    /// Creates new `IndexWriterHolder` containing `tantivy::IndexWriter` and primary key
    ///
    /// `IndexWriterHolder` maintains invariant that the only document with the particular primary key exists in the index.
    /// It is reached by deletion of every document with the same primary key as indexing one.
    /// The type of primary key is restricted to I64 but it is subjected to be changed in the future.
    pub(super) fn new(index_writer: IndexWriterImpl, primary_key: Option<Field>) -> SummaResult<IndexWriterHolder> {
        if let Some(primary_key) = primary_key {
            match index_writer.index().schema().get_field_entry(primary_key).field_type() {
                FieldType::I64(_) => Ok(()),
                FieldType::Str(_) => Ok(()),
                another_type => Err(ValidationError::InvalidPrimaryKeyType(another_type.to_owned())),
            }?
        }
        Ok(IndexWriterHolder { index_writer, primary_key })
    }

    /// Creates new `IndexWriterHolder` from `Index` and `IndexConfig`
    pub(super) fn from_config(index: &Index, index_config: &IndexConfig) -> SummaResult<IndexWriterHolder> {
        let index_writer = if index_config.writer_threads == 0 {
            IndexWriterImpl::Single(SingleIndexWriter {
                index: index.clone(),
                index_config: index_config.clone(),
                index_writer: RwLock::new(SingleSegmentIndexWriter::new(index.clone(), index_config.writer_heap_size_bytes as usize)?),
            })
        } else {
            let index_writer = index.writer_with_num_threads(index_config.writer_threads as usize, index_config.writer_heap_size_bytes as usize)?;
            index_writer.set_merge_policy(Box::<FrozenLogMergePolicy>::default());
            IndexWriterImpl::Threaded(index_writer)
        };
        IndexWriterHolder::new(
            index_writer,
            index_config.primary_key.as_ref().and_then(|primary_key| index.schema().get_field(primary_key)),
        )
    }

    /// Delete index by its primary key
    pub(super) fn delete_document(&self, document: &Document) -> SummaResult<()> {
        if let Some(primary_key) = self.primary_key {
            self.index_writer.delete_term(Term::from_field_i64(
                primary_key,
                document
                    .get_first(primary_key)
                    .ok_or_else(|| ValidationError::MissingPrimaryKey(Some(format!("{:?}", self.index_writer.index().schema().to_named_doc(document)))))?
                    .as_i64()
                    .ok_or_else(|| {
                        ValidationError::InvalidPrimaryKeyType(self.index_writer.index().schema().get_field_entry(primary_key).field_type().clone())
                    })?,
            ));
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
    pub(super) fn index_document(&self, document: Document) -> SummaResult<()> {
        self.delete_document(&document)?;
        self.index_writer.add_document(document)
    }

    /// Merge segments into one.
    ///
    /// Also cleans deleted documents and do recompression. Possible to pass the only segment in `segment_ids` to do recompression or clean up.
    /// It is heavy operation that also blocks on `.await` so should be spawned if non-blocking behaviour is required
    pub(super) async fn merge(&mut self, segment_ids: &[SegmentId], segment_attributes: Option<SummaSegmentAttributes>) -> SummaResult<Option<SegmentMeta>> {
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
    pub(super) async fn commit(&mut self, payload: Option<String>) -> SummaResult<()> {
        self.index_writer.commit(payload).await
    }

    pub(super) async fn vacuum(&mut self, segment_attributes: Option<SummaSegmentAttributes>) -> SummaResult<()> {
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

    pub(super) fn wait_merging_threads(self) -> SummaResult<()> {
        self.index_writer.wait_merging_threads()
    }
}
