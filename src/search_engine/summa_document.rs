use crate::errors::{Error, SummaResult};
use crate::search_engine::SummaDocument::BoundJsonBytes;
use std::str::from_utf8;
use tantivy::schema::Schema;
use tantivy::Document;

pub enum SummaDocument<'a> {
    BoundJsonBytes((&'a Schema, &'a [u8])),
    UnboundJsonBytes(&'a [u8]),
    TantivyDocument(Document),
}

impl<'a> SummaDocument<'a> {
    pub fn bound_with(self, schema: &'a Schema) -> SummaDocument {
        match self {
            SummaDocument::UnboundJsonBytes(json_bytes) => BoundJsonBytes((schema, json_bytes)),
            SummaDocument::BoundJsonBytes((_, json_bytes)) => BoundJsonBytes((schema, json_bytes)),
            other => other,
        }
    }
}

impl<'a> TryInto<Document> for SummaDocument<'a> {
    type Error = Error;

    fn try_into(self) -> SummaResult<Document> {
        match self {
            SummaDocument::BoundJsonBytes((schema, json_bytes)) => {
                let text_document = from_utf8(json_bytes).map_err(|e| Error::Utf8Error(e))?;
                Ok(schema.parse_document(text_document).map_err(|e| Error::ParseError(e))?)
            }
            SummaDocument::UnboundJsonBytes(_) => Err(Error::UnboundDocumentError)?,
            SummaDocument::TantivyDocument(document) => Ok(document),
        }
    }
}

impl<'a> From<&'a Vec<u8>> for SummaDocument<'a> {
    fn from(v: &'a Vec<u8>) -> Self {
        SummaDocument::UnboundJsonBytes(&v)
    }
}
