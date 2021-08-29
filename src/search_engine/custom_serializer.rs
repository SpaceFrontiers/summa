use serde::{Serialize, Serializer};
use std::collections::{BTreeMap, HashSet};
use tantivy::schema::{Field, Schema};
use tantivy::Document;

#[derive(Debug)]
pub enum Value {
    SingleValue(Option<tantivy::schema::Value>),
    MultipleValue(Vec<tantivy::schema::Value>),
}

/// Internal representation of a document used for JSON
/// serialization.
///
/// A `NamedFieldDocument` is a simple representation of a document
/// as a `BTreeMap<String, Vec<Value>>`. It is base on `tantivy::schema::NamedFieldDocument`
/// but with a support of multi fields
#[derive(Debug, Serialize)]
pub struct NamedFieldDocument(pub BTreeMap<String, Value>);

impl NamedFieldDocument {
    pub fn from_document(schema: &Schema, fields: &Option<HashSet<Field>>, multi_fields: &HashSet<Field>, document: &Document) -> Self {
        let mut field_map = BTreeMap::new();
        for (field, field_values) in document.get_sorted_field_values() {
            let field_name = schema.get_field_name(field);
            if let Some(fields) = fields {
                if !fields.contains(&field) {
                    continue;
                }
            }
            let values = if multi_fields.contains(&field) {
                Value::MultipleValue(field_values.into_iter().cloned().collect())
            } else {
                let value = field_values.get(0).cloned().cloned();
                Value::SingleValue(value)
            };
            field_map.insert(field_name.to_string(), values);
        }
        NamedFieldDocument(field_map)
    }
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap()
    }
}

impl Serialize for Value {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Value::SingleValue(value) => value.serialize(serializer),
            Value::MultipleValue(value) => value.serialize(serializer),
        }
    }
}
