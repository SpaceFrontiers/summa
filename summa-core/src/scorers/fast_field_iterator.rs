use tantivy::fastfield::{Column, FastValue};
use tantivy::DocId;

use super::safe_into_f64::SafeIntoF64;

pub(crate) trait FastFieldIterator: Send + Sync {
    fn advance(&mut self, doc_id: DocId);
    fn value(&self) -> &f64;
}

pub(crate) struct FastFieldIteratorImpl<T: FastValue + SafeIntoF64> {
    value: f64,
    ff: Column<T>,
}

impl<T: FastValue + SafeIntoF64> FastFieldIteratorImpl<T> {
    pub fn from_fast_field_reader(ff: Column<T>) -> Box<dyn FastFieldIterator> {
        Box::new(FastFieldIteratorImpl { value: 0f64, ff })
    }
}

impl<T: FastValue + SafeIntoF64> FastFieldIterator for FastFieldIteratorImpl<T> {
    fn advance(&mut self, doc_id: DocId) {
        self.value = self.ff.first(doc_id).map(|x| x.safe_into_f64()).unwrap_or(0.0f64);
    }
    fn value(&self) -> &f64 {
        &self.value
    }
}
