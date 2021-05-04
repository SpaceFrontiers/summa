use crate::errors::SummaResult;
use crate::errors::ValidationError;
use crate::errors::ValidationError::MissingPrimaryKeyError;
use tantivy::directory::GarbageCollectionResult;
use tantivy::schema::{Field, FieldType};
use tantivy::{Document, IndexWriter, Opstamp, SegmentId, SegmentMeta, Term};
use tracing::info;

/// Managing write operations to index
pub(super) struct IndexWriterHolder {
    index_writer: IndexWriter,
    primary_key: Option<Field>,
}

impl IndexWriterHolder {
    /// Creates new `IndexWriterHolder` containing `tantivy::IndexWriter` and primary key
    ///
    /// `IndexWriterHolder` maintains invariant that the only document with the particular primary key exists in the index.
    /// It is reached by deletion of every document with the same primary key as indexing one.
    /// The type of primary key is restricted to I64 but it is subjected to be changed in the future.
    pub(super) fn new(index_writer: IndexWriter, primary_key: Option<Field>) -> SummaResult<IndexWriterHolder> {
        if let Some(primary_key) = primary_key {
            match index_writer.index().schema().get_field_entry(primary_key).field_type() {
                FieldType::I64(_) => Ok(()),
                another_type => Err(ValidationError::InvalidPrimaryKeyType(another_type.to_owned())),
            }?
        }
        Ok(IndexWriterHolder { index_writer, primary_key })
    }

    /// Put document to the index. Before comes searchable it must be committed
    pub(super) fn index_document(&self, document: Document) -> SummaResult<Opstamp> {
        if let Some(primary_key) = self.primary_key {
            self.index_writer.delete_term(Term::from_field_i64(
                primary_key,
                document
                    .get_first(primary_key)
                    .ok_or_else(|| MissingPrimaryKeyError(Some(format!("{:?}", self.index_writer.index().schema().to_named_doc(&document)))))?
                    .as_i64()
                    .unwrap(),
            ));
        }
        Ok(self.index_writer.add_document(document)?)
    }

    /// Merge segments into one.
    ///
    /// Also cleans deleted documents and do recompression. Possible to pass the only segment in `segment_ids` to do recompression or clean up.
    /// It is heavy operation that also blocks on `.await` so should be spawned if non-blocking behaviour is required
    pub(super) async fn merge(&mut self, segment_ids: &[SegmentId]) -> SummaResult<SegmentMeta> {
        info!(action = "merge_segments", segment_ids = ?segment_ids);
        let segment_meta = self.index_writer.merge(segment_ids).await?;
        info!(action = "merged_segments", segment_ids = ?segment_ids, merged_segment_meta = ?segment_meta);
        Ok(segment_meta)
    }

    /// Commits already indexed documents
    ///
    /// Committing makes indexed documents visible
    /// It is heavy operation that also blocks on `.await` so should be spawned if non-blocking behaviour is required
    pub(super) async fn commit(&mut self) -> SummaResult<Opstamp> {
        info!(action = "commit_index");
        let result = self.index_writer.prepare_commit()?.commit_future().await;
        info!(action = "committed_index", result = ?result);
        Ok(result?)
    }

    /// Remove non-used files
    ///
    /// Basically, it dones automatically so there is no need to call it manually
    pub(super) async fn garbage_collect_files(&self) -> SummaResult<GarbageCollectionResult> {
        info!(action = "garbage_collect_files");
        let result = self.index_writer.garbage_collect_files().await?;
        info!(action = "garbage_collected_files", deleted_files = ?result.deleted_files);
        Ok(result)
    }
}
