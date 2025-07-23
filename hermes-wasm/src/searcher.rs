//! Shared adapters from core searcher values to JavaScript values.

use std::sync::Arc;

use hermes_core::{Directory, IndexMetadata, Schema, Searcher};
use serde::Serialize;
use wasm_bindgen::JsValue;

pub(crate) async fn open<D: Directory + 'static>(
    directory: Arc<D>,
) -> Result<Searcher<D>, JsValue> {
    let metadata = IndexMetadata::load(directory.as_ref())
        .await
        .map_err(|error| JsValue::from_str(&format!("Failed to load metadata: {error}")))?;
    let schema = Arc::new(metadata.schema.clone());
    let segment_ids = metadata.segment_ids();

    Searcher::open(directory, schema, &segment_ids, 32)
        .await
        .map_err(|error| JsValue::from_str(&format!("Failed to open searcher: {error}")))
}

pub(crate) fn field_names(schema: Option<&Schema>) -> JsValue {
    let names: Vec<String> = schema
        .map(|schema| {
            schema
                .fields()
                .map(|(_, field)| field.name.clone())
                .collect()
        })
        .unwrap_or_default();
    serde_wasm_bindgen::to_value(&names).unwrap_or(JsValue::NULL)
}

pub(crate) fn default_fields<D: Directory>(searcher: Option<&Searcher<D>>) -> JsValue {
    let names: Vec<String> = searcher
        .map(|searcher| {
            searcher
                .default_fields()
                .iter()
                .filter_map(|field| searcher.schema().get_field_name(*field).map(str::to_owned))
                .collect()
        })
        .unwrap_or_default();
    serde_wasm_bindgen::to_value(&names).unwrap_or(JsValue::NULL)
}

pub(crate) async fn search_offset<D: Directory>(
    searcher: &Searcher<D>,
    query: &str,
    limit: usize,
    offset: usize,
) -> Result<JsValue, JsValue> {
    let response = searcher
        .query_offset(query, limit, offset)
        .await
        .map_err(|error| JsValue::from_str(&format!("Search error: {error}")))?;

    response
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(|error| JsValue::from_str(&format!("Serialization error: {error}")))
}

pub(crate) async fn get_document<D: Directory>(
    searcher: &Searcher<D>,
    segment_id: &str,
    doc_id: u32,
    fields: Option<&rustc_hash::FxHashSet<u32>>,
    invalid_segment_message: &str,
) -> Result<JsValue, JsValue> {
    let segment_id = u128::from_str_radix(segment_id, 16)
        .map_err(|error| JsValue::from_str(&format!("{invalid_segment_message}: {error}")))?;
    let address = hermes_core::query::DocAddress::new(segment_id, doc_id);

    let document = searcher
        .get_document_with_fields(&address, fields)
        .await
        .map_err(|error| JsValue::from_str(&format!("Get document error: {error}")))?;

    match document {
        Some(document) => document
            .to_json(searcher.schema())
            .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
            .map_err(|error| JsValue::from_str(&format!("Serialization error: {error}"))),
        None => Ok(JsValue::NULL),
    }
}
