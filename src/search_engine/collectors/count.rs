use crate::proto;
use tantivy::collector::Count;

impl Into<tantivy::collector::Count> for &proto::CountCollector {
    fn into(self) -> Count {
        Count
    }
}
