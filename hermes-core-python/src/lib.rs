//! Python bindings for Hermes search engine

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use tokio::runtime::Runtime;

use hermes_core::{
    BooleanQuery, FieldValue, FsDirectory, Index, IndexConfig, Schema, TermQuery, search_segment,
};

/// Create a tokio runtime for async operations
fn get_runtime() -> &'static Runtime {
    use std::sync::OnceLock;
    static RUNTIME: OnceLock<Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| Runtime::new().expect("Failed to create Tokio runtime"))
}

/// Python wrapper for Hermes Index (read-only)
#[pyclass]
struct HermesIndex {
    index: Arc<Index<FsDirectory>>,
    schema: Arc<Schema>,
}

#[pymethods]
impl HermesIndex {
    /// Open an existing index
    #[staticmethod]
    fn open(path: &str) -> PyResult<Self> {
        let rt = get_runtime();

        rt.block_on(async {
            let dir = FsDirectory::new(PathBuf::from(path));

            let config = IndexConfig::default();
            let index = Index::open(dir, config)
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to open index: {}", e)))?;

            let schema = Arc::new(index.schema().clone());

            Ok(HermesIndex {
                index: Arc::new(index),
                schema,
            })
        })
    }

    /// Get the number of documents in the index
    fn num_docs(&self) -> u32 {
        self.index.num_docs()
    }

    /// Get the number of segments
    fn num_segments(&self) -> usize {
        self.index.segment_readers().len()
    }

    /// Get field names
    fn field_names(&self) -> Vec<String> {
        self.schema
            .fields()
            .map(|(_, entry)| entry.name.clone())
            .collect()
    }

    /// Get a document by ID
    fn get_document(&self, doc_id: u32) -> PyResult<Option<HashMap<String, Py<PyAny>>>> {
        let rt = get_runtime();

        rt.block_on(async {
            let doc =
                self.index.doc(doc_id).await.map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to get document: {}", e))
                })?;

            match doc {
                Some(doc) => Python::attach(|py| {
                    let mut result = HashMap::new();
                    for (field, value) in doc.field_values() {
                        if let Some(entry) = self.schema.get_field_entry(*field) {
                            let py_value = field_value_to_py(py, value);
                            result.insert(entry.name.clone(), py_value);
                        }
                    }
                    Ok(Some(result))
                }),
                None => Ok(None),
            }
        })
    }

    /// Search the index with a term query
    fn search_term(
        &self,
        field: &str,
        term: &str,
        limit: Option<usize>,
    ) -> PyResult<Vec<(u32, f32)>> {
        let field_id = self
            .schema
            .get_field(field)
            .ok_or_else(|| PyValueError::new_err(format!("Field '{}' not found", field)))?;

        let query = TermQuery::text(field_id, term);
        let limit = limit.unwrap_or(10);

        let rt = get_runtime();

        rt.block_on(async {
            let mut all_results = Vec::new();

            for segment in self.index.segment_readers() {
                let results = search_segment(&segment, &query, limit)
                    .await
                    .map_err(|e| PyRuntimeError::new_err(format!("Search failed: {}", e)))?;

                for result in results {
                    all_results.push((result.doc_id + segment.doc_id_offset(), result.score));
                }
            }

            // Sort by score descending
            all_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            all_results.truncate(limit);

            Ok(all_results)
        })
    }

    /// Search with a boolean query
    fn search_boolean(
        &self,
        must: Option<Vec<(String, String)>>,
        should: Option<Vec<(String, String)>>,
        must_not: Option<Vec<(String, String)>>,
        limit: Option<usize>,
    ) -> PyResult<Vec<(u32, f32)>> {
        let mut query = BooleanQuery::new();

        if let Some(must_terms) = must {
            for (field, term) in must_terms {
                let field_id = self
                    .schema
                    .get_field(&field)
                    .ok_or_else(|| PyValueError::new_err(format!("Field '{}' not found", field)))?;
                query = query.must(TermQuery::text(field_id, &term));
            }
        }

        if let Some(should_terms) = should {
            for (field, term) in should_terms {
                let field_id = self
                    .schema
                    .get_field(&field)
                    .ok_or_else(|| PyValueError::new_err(format!("Field '{}' not found", field)))?;
                query = query.should(TermQuery::text(field_id, &term));
            }
        }

        if let Some(must_not_terms) = must_not {
            for (field, term) in must_not_terms {
                let field_id = self
                    .schema
                    .get_field(&field)
                    .ok_or_else(|| PyValueError::new_err(format!("Field '{}' not found", field)))?;
                query = query.must_not(TermQuery::text(field_id, &term));
            }
        }

        let limit = limit.unwrap_or(10);
        let rt = get_runtime();

        rt.block_on(async {
            let mut all_results = Vec::new();

            for segment in self.index.segment_readers() {
                let results = search_segment(&segment, &query, limit)
                    .await
                    .map_err(|e| PyRuntimeError::new_err(format!("Search failed: {}", e)))?;

                for result in results {
                    all_results.push((result.doc_id + segment.doc_id_offset(), result.score));
                }
            }

            all_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            all_results.truncate(limit);

            Ok(all_results)
        })
    }

    /// Reload the index to see new segments
    fn reload(&self) -> PyResult<()> {
        let rt = get_runtime();
        rt.block_on(async {
            self.index
                .reload()
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("Reload failed: {}", e)))
        })
    }
}

fn field_value_to_py(py: Python<'_>, value: &FieldValue) -> Py<PyAny> {
    match value {
        FieldValue::Text(s) => s.into_pyobject(py).unwrap().into_any().unbind(),
        FieldValue::U64(n) => n.into_pyobject(py).unwrap().into_any().unbind(),
        FieldValue::I64(n) => n.into_pyobject(py).unwrap().into_any().unbind(),
        FieldValue::F64(n) => n.into_pyobject(py).unwrap().into_any().unbind(),
        FieldValue::Bytes(b) => b.into_pyobject(py).unwrap().into_any().unbind(),
    }
}

/// Python module
#[pymodule]
fn hermes_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<HermesIndex>()?;
    Ok(())
}
