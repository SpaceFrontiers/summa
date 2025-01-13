//! Schema definitions for documents and fields

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Field identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Field(pub u32);

/// Types of fields supported
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FieldType {
    /// Text field - tokenized and indexed
    #[serde(rename = "text")]
    Text,
    /// Unsigned 64-bit integer
    #[serde(rename = "u64")]
    U64,
    /// Signed 64-bit integer
    #[serde(rename = "i64")]
    I64,
    /// 64-bit floating point
    #[serde(rename = "f64")]
    F64,
    /// Raw bytes (not tokenized)
    #[serde(rename = "bytes")]
    Bytes,
    /// Sparse vector field - indexed as inverted posting lists with quantized weights
    #[serde(rename = "sparse_vector")]
    SparseVector,
}

/// Field options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldEntry {
    pub name: String,
    pub field_type: FieldType,
    pub indexed: bool,
    pub stored: bool,
    /// Name of the tokenizer to use for this field (for text fields)
    pub tokenizer: Option<String>,
    /// Whether this field can have multiple values (serialized as array in JSON)
    #[serde(default)]
    pub multi: bool,
    /// Configuration for sparse vector fields (index size, weight quantization)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sparse_vector_config: Option<crate::structures::SparseVectorConfig>,
}

use super::query_field_router::QueryRouterRule;

/// Schema defining document structure
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Schema {
    fields: Vec<FieldEntry>,
    name_to_field: HashMap<String, Field>,
    /// Default fields for query parsing (when no field is specified)
    #[serde(default)]
    default_fields: Vec<Field>,
    /// Query router rules for routing queries to specific fields based on regex patterns
    #[serde(default)]
    query_routers: Vec<QueryRouterRule>,
}

impl Schema {
    pub fn builder() -> SchemaBuilder {
        SchemaBuilder::default()
    }

    pub fn get_field(&self, name: &str) -> Option<Field> {
        self.name_to_field.get(name).copied()
    }

    pub fn get_field_entry(&self, field: Field) -> Option<&FieldEntry> {
        self.fields.get(field.0 as usize)
    }

    pub fn get_field_name(&self, field: Field) -> Option<&str> {
        self.fields.get(field.0 as usize).map(|e| e.name.as_str())
    }

    pub fn fields(&self) -> impl Iterator<Item = (Field, &FieldEntry)> {
        self.fields
            .iter()
            .enumerate()
            .map(|(i, e)| (Field(i as u32), e))
    }

    pub fn num_fields(&self) -> usize {
        self.fields.len()
    }

    /// Get the default fields for query parsing
    pub fn default_fields(&self) -> &[Field] {
        &self.default_fields
    }

    /// Set default fields (used by builder)
    pub fn set_default_fields(&mut self, fields: Vec<Field>) {
        self.default_fields = fields;
    }

    /// Get the query router rules
    pub fn query_routers(&self) -> &[QueryRouterRule] {
        &self.query_routers
    }

    /// Set query router rules
    pub fn set_query_routers(&mut self, rules: Vec<QueryRouterRule>) {
        self.query_routers = rules;
    }
}

/// Builder for Schema
#[derive(Debug, Default)]
pub struct SchemaBuilder {
    fields: Vec<FieldEntry>,
    default_fields: Vec<String>,
    query_routers: Vec<QueryRouterRule>,
}

impl SchemaBuilder {
    pub fn add_text_field(&mut self, name: &str, indexed: bool, stored: bool) -> Field {
        self.add_field_with_tokenizer(
            name,
            FieldType::Text,
            indexed,
            stored,
            Some("default".to_string()),
        )
    }

    pub fn add_text_field_with_tokenizer(
        &mut self,
        name: &str,
        indexed: bool,
        stored: bool,
        tokenizer: &str,
    ) -> Field {
        self.add_field_with_tokenizer(
            name,
            FieldType::Text,
            indexed,
            stored,
            Some(tokenizer.to_string()),
        )
    }

    pub fn add_u64_field(&mut self, name: &str, indexed: bool, stored: bool) -> Field {
        self.add_field(name, FieldType::U64, indexed, stored)
    }

    pub fn add_i64_field(&mut self, name: &str, indexed: bool, stored: bool) -> Field {
        self.add_field(name, FieldType::I64, indexed, stored)
    }

    pub fn add_f64_field(&mut self, name: &str, indexed: bool, stored: bool) -> Field {
        self.add_field(name, FieldType::F64, indexed, stored)
    }

    pub fn add_bytes_field(&mut self, name: &str, stored: bool) -> Field {
        self.add_field(name, FieldType::Bytes, false, stored)
    }

    /// Add a sparse vector field with default configuration
    ///
    /// Sparse vectors are indexed as inverted posting lists where each dimension
    /// becomes a "term" and documents have quantized weights for each dimension.
    pub fn add_sparse_vector_field(&mut self, name: &str, indexed: bool, stored: bool) -> Field {
        self.add_sparse_vector_field_with_config(
            name,
            indexed,
            stored,
            crate::structures::SparseVectorConfig::default(),
        )
    }

    /// Add a sparse vector field with custom configuration
    ///
    /// Use `SparseVectorConfig::splade()` for SPLADE models (u16 indices, uint8 weights).
    /// Use `SparseVectorConfig::compact()` for maximum compression (u16 indices, uint4 weights).
    pub fn add_sparse_vector_field_with_config(
        &mut self,
        name: &str,
        indexed: bool,
        stored: bool,
        config: crate::structures::SparseVectorConfig,
    ) -> Field {
        let field = Field(self.fields.len() as u32);
        self.fields.push(FieldEntry {
            name: name.to_string(),
            field_type: FieldType::SparseVector,
            indexed,
            stored,
            tokenizer: None,
            multi: false,
            sparse_vector_config: Some(config),
        });
        field
    }

    /// Set sparse vector configuration for an existing field
    pub fn set_sparse_vector_config(
        &mut self,
        field: Field,
        config: crate::structures::SparseVectorConfig,
    ) {
        if let Some(entry) = self.fields.get_mut(field.0 as usize) {
            entry.sparse_vector_config = Some(config);
        }
    }

    fn add_field(
        &mut self,
        name: &str,
        field_type: FieldType,
        indexed: bool,
        stored: bool,
    ) -> Field {
        self.add_field_with_tokenizer(name, field_type, indexed, stored, None)
    }

    fn add_field_with_tokenizer(
        &mut self,
        name: &str,
        field_type: FieldType,
        indexed: bool,
        stored: bool,
        tokenizer: Option<String>,
    ) -> Field {
        self.add_field_full(name, field_type, indexed, stored, tokenizer, false)
    }

    fn add_field_full(
        &mut self,
        name: &str,
        field_type: FieldType,
        indexed: bool,
        stored: bool,
        tokenizer: Option<String>,
        multi: bool,
    ) -> Field {
        let field = Field(self.fields.len() as u32);
        self.fields.push(FieldEntry {
            name: name.to_string(),
            field_type,
            indexed,
            stored,
            tokenizer,
            multi,
            sparse_vector_config: None,
        });
        field
    }

    /// Set the multi attribute on the last added field
    pub fn set_multi(&mut self, field: Field, multi: bool) {
        if let Some(entry) = self.fields.get_mut(field.0 as usize) {
            entry.multi = multi;
        }
    }

    /// Set default fields by name
    pub fn set_default_fields(&mut self, field_names: Vec<String>) {
        self.default_fields = field_names;
    }

    /// Set query router rules
    pub fn set_query_routers(&mut self, rules: Vec<QueryRouterRule>) {
        self.query_routers = rules;
    }

    pub fn build(self) -> Schema {
        let mut name_to_field = HashMap::new();
        for (i, entry) in self.fields.iter().enumerate() {
            name_to_field.insert(entry.name.clone(), Field(i as u32));
        }

        // Resolve default field names to Field IDs
        let default_fields: Vec<Field> = self
            .default_fields
            .iter()
            .filter_map(|name| name_to_field.get(name).copied())
            .collect();

        Schema {
            fields: self.fields,
            name_to_field,
            default_fields,
            query_routers: self.query_routers,
        }
    }
}

/// Value that can be stored in a field
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FieldValue {
    #[serde(rename = "text")]
    Text(String),
    #[serde(rename = "u64")]
    U64(u64),
    #[serde(rename = "i64")]
    I64(i64),
    #[serde(rename = "f64")]
    F64(f64),
    #[serde(rename = "bytes")]
    Bytes(Vec<u8>),
    /// Sparse vector: (dimension_ids, weights)
    #[serde(rename = "sparse_vector")]
    SparseVector { indices: Vec<u32>, values: Vec<f32> },
}

impl FieldValue {
    pub fn as_text(&self) -> Option<&str> {
        match self {
            FieldValue::Text(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            FieldValue::U64(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            FieldValue::I64(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            FieldValue::F64(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            FieldValue::Bytes(b) => Some(b),
            _ => None,
        }
    }

    pub fn as_sparse_vector(&self) -> Option<(&[u32], &[f32])> {
        match self {
            FieldValue::SparseVector { indices, values } => Some((indices, values)),
            _ => None,
        }
    }
}

/// A document to be indexed
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Document {
    field_values: Vec<(Field, FieldValue)>,
}

impl Document {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_text(&mut self, field: Field, value: impl Into<String>) {
        self.field_values
            .push((field, FieldValue::Text(value.into())));
    }

    pub fn add_u64(&mut self, field: Field, value: u64) {
        self.field_values.push((field, FieldValue::U64(value)));
    }

    pub fn add_i64(&mut self, field: Field, value: i64) {
        self.field_values.push((field, FieldValue::I64(value)));
    }

    pub fn add_f64(&mut self, field: Field, value: f64) {
        self.field_values.push((field, FieldValue::F64(value)));
    }

    pub fn add_bytes(&mut self, field: Field, value: Vec<u8>) {
        self.field_values.push((field, FieldValue::Bytes(value)));
    }

    pub fn add_sparse_vector(&mut self, field: Field, indices: Vec<u32>, values: Vec<f32>) {
        debug_assert_eq!(
            indices.len(),
            values.len(),
            "Sparse vector indices and values must have same length"
        );
        self.field_values
            .push((field, FieldValue::SparseVector { indices, values }));
    }

    pub fn get_first(&self, field: Field) -> Option<&FieldValue> {
        self.field_values
            .iter()
            .find(|(f, _)| *f == field)
            .map(|(_, v)| v)
    }

    pub fn get_all(&self, field: Field) -> impl Iterator<Item = &FieldValue> {
        self.field_values
            .iter()
            .filter(move |(f, _)| *f == field)
            .map(|(_, v)| v)
    }

    pub fn field_values(&self) -> &[(Field, FieldValue)] {
        &self.field_values
    }

    /// Convert document to a JSON object using field names from schema
    ///
    /// Fields marked as `multi` in the schema are always returned as JSON arrays.
    /// Other fields with multiple values are also returned as arrays.
    /// Fields with a single value (and not marked multi) are returned as scalar values.
    pub fn to_json(&self, schema: &Schema) -> serde_json::Value {
        use std::collections::HashMap;

        // Group values by field, keeping track of field entry for multi check
        let mut field_values_map: HashMap<Field, (String, bool, Vec<serde_json::Value>)> =
            HashMap::new();

        for (field, value) in &self.field_values {
            if let Some(entry) = schema.get_field_entry(*field) {
                let json_value = match value {
                    FieldValue::Text(s) => serde_json::Value::String(s.clone()),
                    FieldValue::U64(n) => serde_json::Value::Number((*n).into()),
                    FieldValue::I64(n) => serde_json::Value::Number((*n).into()),
                    FieldValue::F64(n) => serde_json::json!(n),
                    FieldValue::Bytes(b) => {
                        use base64::Engine;
                        serde_json::Value::String(
                            base64::engine::general_purpose::STANDARD.encode(b),
                        )
                    }
                    FieldValue::SparseVector { indices, values } => {
                        serde_json::json!({
                            "indices": indices,
                            "values": values
                        })
                    }
                };
                field_values_map
                    .entry(*field)
                    .or_insert_with(|| (entry.name.clone(), entry.multi, Vec::new()))
                    .2
                    .push(json_value);
            }
        }

        // Convert to JSON object, using arrays for multi fields or when multiple values exist
        let mut map = serde_json::Map::new();
        for (_field, (name, is_multi, values)) in field_values_map {
            let json_value = if is_multi || values.len() > 1 {
                serde_json::Value::Array(values)
            } else {
                values.into_iter().next().unwrap()
            };
            map.insert(name, json_value);
        }

        serde_json::Value::Object(map)
    }

    /// Create a Document from a JSON object using field names from schema
    ///
    /// Supports:
    /// - String values -> Text fields
    /// - Number values -> U64/I64/F64 fields (based on schema type)
    /// - Array values -> Multiple values for the same field (multifields)
    ///
    /// Unknown fields (not in schema) are silently ignored.
    pub fn from_json(json: &serde_json::Value, schema: &Schema) -> Option<Self> {
        let obj = json.as_object()?;
        let mut doc = Document::new();

        for (key, value) in obj {
            if let Some(field) = schema.get_field(key) {
                let field_entry = schema.get_field_entry(field)?;
                Self::add_json_value(&mut doc, field, &field_entry.field_type, value);
            }
        }

        Some(doc)
    }

    /// Helper to add a JSON value to a document, handling type conversion
    fn add_json_value(
        doc: &mut Document,
        field: Field,
        field_type: &FieldType,
        value: &serde_json::Value,
    ) {
        match value {
            serde_json::Value::String(s) => {
                if matches!(field_type, FieldType::Text) {
                    doc.add_text(field, s.clone());
                }
            }
            serde_json::Value::Number(n) => {
                match field_type {
                    FieldType::I64 => {
                        if let Some(i) = n.as_i64() {
                            doc.add_i64(field, i);
                        }
                    }
                    FieldType::U64 => {
                        if let Some(u) = n.as_u64() {
                            doc.add_u64(field, u);
                        } else if let Some(i) = n.as_i64() {
                            // Allow positive i64 as u64
                            if i >= 0 {
                                doc.add_u64(field, i as u64);
                            }
                        }
                    }
                    FieldType::F64 => {
                        if let Some(f) = n.as_f64() {
                            doc.add_f64(field, f);
                        }
                    }
                    _ => {}
                }
            }
            // Handle arrays (multifields) - add each element separately
            serde_json::Value::Array(arr) => {
                for item in arr {
                    Self::add_json_value(doc, field, field_type, item);
                }
            }
            // Handle sparse vector objects
            serde_json::Value::Object(obj) if matches!(field_type, FieldType::SparseVector) => {
                if let (Some(indices_val), Some(values_val)) =
                    (obj.get("indices"), obj.get("values"))
                {
                    let indices: Vec<u32> = indices_val
                        .as_array()
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_u64().map(|n| n as u32))
                                .collect()
                        })
                        .unwrap_or_default();
                    let values: Vec<f32> = values_val
                        .as_array()
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_f64().map(|n| n as f32))
                                .collect()
                        })
                        .unwrap_or_default();
                    if indices.len() == values.len() {
                        doc.add_sparse_vector(field, indices, values);
                    }
                }
            }
            serde_json::Value::Object(_) => {}
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_builder() {
        let mut builder = Schema::builder();
        let title = builder.add_text_field("title", true, true);
        let body = builder.add_text_field("body", true, false);
        let count = builder.add_u64_field("count", true, true);
        let schema = builder.build();

        assert_eq!(schema.get_field("title"), Some(title));
        assert_eq!(schema.get_field("body"), Some(body));
        assert_eq!(schema.get_field("count"), Some(count));
        assert_eq!(schema.get_field("nonexistent"), None);
    }

    #[test]
    fn test_document() {
        let mut builder = Schema::builder();
        let title = builder.add_text_field("title", true, true);
        let count = builder.add_u64_field("count", true, true);
        let _schema = builder.build();

        let mut doc = Document::new();
        doc.add_text(title, "Hello World");
        doc.add_u64(count, 42);

        assert_eq!(doc.get_first(title).unwrap().as_text(), Some("Hello World"));
        assert_eq!(doc.get_first(count).unwrap().as_u64(), Some(42));
    }

    #[test]
    fn test_document_serialization() {
        let mut builder = Schema::builder();
        let title = builder.add_text_field("title", true, true);
        let count = builder.add_u64_field("count", true, true);
        let _schema = builder.build();

        let mut doc = Document::new();
        doc.add_text(title, "Hello World");
        doc.add_u64(count, 42);

        // Serialize
        let json = serde_json::to_string(&doc).unwrap();
        println!("Serialized doc: {}", json);

        // Deserialize
        let doc2: Document = serde_json::from_str(&json).unwrap();
        assert_eq!(
            doc2.field_values().len(),
            2,
            "Should have 2 field values after deserialization"
        );
        assert_eq!(
            doc2.get_first(title).unwrap().as_text(),
            Some("Hello World")
        );
        assert_eq!(doc2.get_first(count).unwrap().as_u64(), Some(42));
    }

    #[test]
    fn test_multivalue_field() {
        let mut builder = Schema::builder();
        let uris = builder.add_text_field("uris", true, true);
        let title = builder.add_text_field("title", true, true);
        let schema = builder.build();

        // Create document with multiple values for the same field
        let mut doc = Document::new();
        doc.add_text(uris, "one");
        doc.add_text(uris, "two");
        doc.add_text(title, "Test Document");

        // Verify get_first returns the first value
        assert_eq!(doc.get_first(uris).unwrap().as_text(), Some("one"));

        // Verify get_all returns all values
        let all_uris: Vec<_> = doc.get_all(uris).collect();
        assert_eq!(all_uris.len(), 2);
        assert_eq!(all_uris[0].as_text(), Some("one"));
        assert_eq!(all_uris[1].as_text(), Some("two"));

        // Verify to_json returns array for multi-value field
        let json = doc.to_json(&schema);
        let uris_json = json.get("uris").unwrap();
        assert!(uris_json.is_array(), "Multi-value field should be an array");
        let uris_arr = uris_json.as_array().unwrap();
        assert_eq!(uris_arr.len(), 2);
        assert_eq!(uris_arr[0].as_str(), Some("one"));
        assert_eq!(uris_arr[1].as_str(), Some("two"));

        // Verify single-value field is NOT an array
        let title_json = json.get("title").unwrap();
        assert!(
            title_json.is_string(),
            "Single-value field should be a string"
        );
        assert_eq!(title_json.as_str(), Some("Test Document"));
    }

    #[test]
    fn test_multivalue_from_json() {
        let mut builder = Schema::builder();
        let uris = builder.add_text_field("uris", true, true);
        let title = builder.add_text_field("title", true, true);
        let schema = builder.build();

        // Create JSON with array value
        let json = serde_json::json!({
            "uris": ["one", "two"],
            "title": "Test Document"
        });

        // Parse from JSON
        let doc = Document::from_json(&json, &schema).unwrap();

        // Verify all values are present
        let all_uris: Vec<_> = doc.get_all(uris).collect();
        assert_eq!(all_uris.len(), 2);
        assert_eq!(all_uris[0].as_text(), Some("one"));
        assert_eq!(all_uris[1].as_text(), Some("two"));

        // Verify single value
        assert_eq!(
            doc.get_first(title).unwrap().as_text(),
            Some("Test Document")
        );

        // Verify roundtrip: to_json should produce equivalent JSON
        let json_out = doc.to_json(&schema);
        let uris_out = json_out.get("uris").unwrap().as_array().unwrap();
        assert_eq!(uris_out.len(), 2);
        assert_eq!(uris_out[0].as_str(), Some("one"));
        assert_eq!(uris_out[1].as_str(), Some("two"));
    }

    #[test]
    fn test_multi_attribute_forces_array() {
        // Test that fields marked as 'multi' are always serialized as arrays,
        // even when they have only one value
        let mut builder = Schema::builder();
        let uris = builder.add_text_field("uris", true, true);
        builder.set_multi(uris, true); // Mark as multi
        let title = builder.add_text_field("title", true, true);
        let schema = builder.build();

        // Verify the multi attribute is set
        assert!(schema.get_field_entry(uris).unwrap().multi);
        assert!(!schema.get_field_entry(title).unwrap().multi);

        // Create document with single value for multi field
        let mut doc = Document::new();
        doc.add_text(uris, "only_one");
        doc.add_text(title, "Test Document");

        // Verify to_json returns array for multi field even with single value
        let json = doc.to_json(&schema);

        let uris_json = json.get("uris").unwrap();
        assert!(
            uris_json.is_array(),
            "Multi field should be array even with single value"
        );
        let uris_arr = uris_json.as_array().unwrap();
        assert_eq!(uris_arr.len(), 1);
        assert_eq!(uris_arr[0].as_str(), Some("only_one"));

        // Verify non-multi field with single value is NOT an array
        let title_json = json.get("title").unwrap();
        assert!(
            title_json.is_string(),
            "Non-multi single-value field should be a string"
        );
        assert_eq!(title_json.as_str(), Some("Test Document"));
    }

    #[test]
    fn test_sparse_vector_field() {
        let mut builder = Schema::builder();
        let embedding = builder.add_sparse_vector_field("embedding", true, true);
        let title = builder.add_text_field("title", true, true);
        let schema = builder.build();

        assert_eq!(schema.get_field("embedding"), Some(embedding));
        assert_eq!(
            schema.get_field_entry(embedding).unwrap().field_type,
            FieldType::SparseVector
        );

        // Create document with sparse vector
        let mut doc = Document::new();
        doc.add_sparse_vector(embedding, vec![0, 5, 10], vec![1.0, 2.5, 0.5]);
        doc.add_text(title, "Test Document");

        // Verify accessor
        let (indices, values) = doc
            .get_first(embedding)
            .unwrap()
            .as_sparse_vector()
            .unwrap();
        assert_eq!(indices, &[0, 5, 10]);
        assert_eq!(values, &[1.0, 2.5, 0.5]);

        // Verify JSON roundtrip
        let json = doc.to_json(&schema);
        let embedding_json = json.get("embedding").unwrap();
        assert!(embedding_json.is_object());
        assert_eq!(
            embedding_json
                .get("indices")
                .unwrap()
                .as_array()
                .unwrap()
                .len(),
            3
        );

        // Parse back from JSON
        let doc2 = Document::from_json(&json, &schema).unwrap();
        let (indices2, values2) = doc2
            .get_first(embedding)
            .unwrap()
            .as_sparse_vector()
            .unwrap();
        assert_eq!(indices2, &[0, 5, 10]);
        assert!((values2[0] - 1.0).abs() < 1e-6);
        assert!((values2[1] - 2.5).abs() < 1e-6);
        assert!((values2[2] - 0.5).abs() < 1e-6);
    }
}
