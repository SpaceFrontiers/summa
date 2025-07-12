//! Python bindings for the Hermes MAL parser.
//!
//! This is a thin PyO3 wrapper around the [`hermes_mal`] crate, which is the
//! single source of truth for parsing the Model Architecture Language (MAL).
//! It exposes exactly one function, [`parse_mal`], returning the same JSON that
//! `hermes-llm export` emits (serde JSON of `hermes_mal::ModelDef`).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Parse MAL source and return the model definition as a JSON string.
///
/// The returned string is byte-for-byte what `hermes-llm export` emits for the
/// same source. Any syntax error, unknown key, or undefined reference is raised
/// as a Python `ValueError`.
#[pyfunction]
fn parse_mal(source: &str) -> PyResult<String> {
    let model = mal::parse_mal(source).map_err(|e| PyValueError::new_err(e.to_string()))?;
    serde_json::to_string(&model).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// The `hermes_mal` Python module.
#[pymodule]
fn hermes_mal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_mal, m)?)?;
    Ok(())
}
