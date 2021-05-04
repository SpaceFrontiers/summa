use crate::proto;
use tantivy::collector::TopDocs;

impl Into<tantivy::collector::TopDocs> for &proto::TopDocsCollector {
    fn into(self) -> TopDocs {
        TopDocs::with_limit(self.limit.try_into().unwrap()).and_offset(self.offset.try_into().unwrap())
    }
}
