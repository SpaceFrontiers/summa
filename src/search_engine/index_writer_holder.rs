use crate::configurator::configs::IndexConfigHolder;
use crate::errors::{Error, SummaResult};
use std::str::from_utf8;
use tantivy::schema::{Field, Schema};
use tantivy::{Index, IndexWriter, Opstamp, Term};
use tracing::info;

/// Managing write operations to index
pub(crate) struct IndexWriterHolder {
    index_writer: IndexWriter,
    index_name: String,
    schema: Schema,
    primary_key: Option<Field>,
}

impl IndexWriterHolder {
    pub fn new(index_name: &str, index: &Index, index_config: &IndexConfigHolder) -> SummaResult<IndexWriterHolder> {
        let schema = index.schema();
        let index_config = index_config;
        let index_writer = index.writer_with_num_threads(index_config.writer_threads.try_into().unwrap(), index_config.writer_heap_size_bytes.try_into().unwrap())?;
        Ok(IndexWriterHolder {
            index_writer,
            index_name: index_name.to_string(),
            schema,
            primary_key: index_config.primary_key.clone(),
        })
    }

    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    pub fn index_document(&self, raw_document: &[u8], reindex: bool) -> SummaResult<Opstamp> {
        let text_document = from_utf8(raw_document).map_err(|e| Error::Utf8Error(e))?;
        let tantivy_document = self.schema().parse_document(text_document).map_err(|e| Error::ParseError(e))?;
        if reindex {
            if let Some(primary_key) = self.primary_key {
                self.index_writer
                    .delete_term(Term::from_field_i64(primary_key, tantivy_document.get_first(primary_key).unwrap().as_i64().unwrap()));
            }
        }
        Ok(self.index_writer.add_document(tantivy_document)?)
    }

    pub fn commit(&mut self) -> SummaResult<()> {
        info!(action = "commit_index", index_name = ?self.index_name);
        self.index_writer.commit()?;
        info!(action = "commited_index", index_name = ?self.index_name);
        Ok(())
    }
}

impl std::fmt::Debug for IndexWriterHolder {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self.index_name)
    }
}
