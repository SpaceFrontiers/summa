use crate::errors::SummaResult;

use tantivy::directory::GarbageCollectionResult;
use tantivy::schema::Field;
use tantivy::{Document, IndexWriter, Opstamp, SegmentId, SegmentMeta, Term};
use tracing::info;

/// Managing write operations to index
pub(crate) struct IndexWriterHolder {
    index_writer: IndexWriter,
    primary_key: Option<Field>,
}

impl IndexWriterHolder {
    pub fn new(index_writer: IndexWriter, primary_key: Option<Field>) -> SummaResult<IndexWriterHolder> {
        Ok(IndexWriterHolder { index_writer, primary_key })
    }

    pub fn index_document(&self, document: Document, reindex: bool) -> SummaResult<Opstamp> {
        if reindex {
            if let Some(primary_key) = self.primary_key {
                self.index_writer
                    .delete_term(Term::from_field_i64(primary_key, document.get_first(primary_key).unwrap().as_i64().unwrap()));
            }
        }
        Ok(self.index_writer.add_document(document)?)
    }

    pub async fn merge(&mut self, segment_ids: &[SegmentId]) -> SummaResult<SegmentMeta> {
        info!(action = "merge_segments", segment_ids = ?segment_ids);
        let segment_meta = self.index_writer.merge(segment_ids).await?;
        info!(action = "merged_segments", segment_ids = ?segment_ids, merged_segment_meta = ?segment_meta);
        Ok(segment_meta)
    }

    pub async fn commit(&mut self) -> SummaResult<Opstamp> {
        info!(action = "commit_index");
        let opstamp = self.index_writer.prepare_commit()?.commit_async().await?;
        info!(action = "committed_index");
        Ok(opstamp)
    }

    pub async fn garbage_collect_files(&self) -> SummaResult<GarbageCollectionResult> {
        info!(action = "garbage_collect_files");
        let result = self.index_writer.garbage_collect_files().await?;
        info!(action = "garbage_collected_files", deleted_files = ?result.deleted_files);
        Ok(result)
    }
}
