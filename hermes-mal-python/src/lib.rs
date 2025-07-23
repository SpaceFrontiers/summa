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
    parse_mal_json(source).map_err(PyValueError::new_err)
}

fn parse_mal_json(source: &str) -> Result<String, String> {
    let model = mal::parse_mal(source).map_err(|error| error.to_string())?;
    serde_json::to_string(&model).map_err(|error| error.to_string())
}

/// The `hermes_mal` Python module.
#[pymodule]
fn hermes_mal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_mal, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn binding_serializes_the_core_parser_result_without_a_second_model() {
        let source = r#"
            model tiny {
                vocab_size: 256
                max_seq_len: 32
                hidden_size: 16
                num_layers: 1
                block: {
                    attention: { num_heads: 2 }
                    ffn: { hidden_dim: 32 }
                }
            }
        "#;
        let expected = serde_json::to_string(&mal::parse_mal(source).unwrap()).unwrap();

        assert_eq!(super::parse_mal_json(source).unwrap(), expected);
    }
}
