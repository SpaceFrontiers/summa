use super::safe_into_f64::SafeIntoF64;
use tantivy::fastfield::{DynamicFastFieldReader, FastFieldReader, FastValue};
use tantivy::DocId;

pub(crate) trait FastFieldIterator {
    fn advance(&mut self, doc_id: DocId);
    fn value(&self) -> &f64;
}

pub(crate) struct FastFieldIteratorImpl<T: FastValue + SafeIntoF64> {
    value: f64,
    ff: DynamicFastFieldReader<T>,
}

impl<T: FastValue + SafeIntoF64> FastFieldIteratorImpl<T> {
    pub fn from_fast_field_reader(ff: DynamicFastFieldReader<T>) -> Box<dyn FastFieldIterator> {
        Box::new(FastFieldIteratorImpl { value: 0f64, ff })
    }
}

impl<T: FastValue + SafeIntoF64> FastFieldIterator for FastFieldIteratorImpl<T> {
    fn advance(&mut self, doc_id: DocId) {
        self.value = self.ff.get(doc_id).safe_into_f64();
    }
    fn value(&self) -> &f64 {
        &self.value
    }
}
