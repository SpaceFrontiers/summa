//! Proto conversion helpers

use hermes_core::query::{
    DenseVectorQuery, LazyGlobalStats, MultiValueCombiner, SparseVectorQuery,
};
use hermes_core::structures::QueryWeighting;
use hermes_core::tokenizer::tokenizer_cache;
use hermes_core::{
    BooleanQuery, BoostQuery, Document, FieldValue as CoreFieldValue, Query, Schema, TermQuery,
};

use crate::proto;
use crate::proto::field_value::Value;
use crate::proto::query::Query as ProtoQueryType;

fn convert_combiner(combiner: i32) -> MultiValueCombiner {
    match combiner {
        1 => MultiValueCombiner::Max,
        2 => MultiValueCombiner::Avg,
        _ => MultiValueCombiner::Sum, // 0 or default
    }
}

pub fn convert_query(
    query: &proto::Query,
    schema: &Schema,
    global_stats: Option<&LazyGlobalStats>,
) -> Result<Box<dyn Query>, String> {
    match &query.query {
        Some(ProtoQueryType::Term(term_query)) => {
            let field = schema
                .get_field(&term_query.field)
                .ok_or_else(|| format!("Field '{}' not found", term_query.field))?;
            Ok(Box::new(TermQuery::text(field, &term_query.term)))
        }
        Some(ProtoQueryType::Boolean(bool_query)) => {
            let mut bq = BooleanQuery::new();
            for q in &bool_query.must {
                let inner = convert_query(q, schema, global_stats)?;
                bq.must.push(inner.into());
            }
            for q in &bool_query.should {
                let inner = convert_query(q, schema, global_stats)?;
                bq.should.push(inner.into());
            }
            for q in &bool_query.must_not {
                let inner = convert_query(q, schema, global_stats)?;
                bq.must_not.push(inner.into());
            }
            Ok(Box::new(bq))
        }
        Some(ProtoQueryType::Boost(boost_query)) => {
            let inner = boost_query
                .query
                .as_ref()
                .ok_or_else(|| "Boost query requires inner query".to_string())?;
            let inner_query = convert_query(inner, schema, global_stats)?;
            Ok(Box::new(BoostQuery {
                inner: inner_query.into(),
                boost: boost_query.boost,
            }))
        }
        Some(ProtoQueryType::All(_)) => {
            // Match all - use a boolean query with no clauses that matches everything
            // For now, return an error as we don't have AllQuery implemented
            Err("AllQuery not yet implemented".to_string())
        }
        Some(ProtoQueryType::SparseVector(sv_query)) => {
            let field = schema
                .get_field(&sv_query.field)
                .ok_or_else(|| format!("Field '{}' not found", sv_query.field))?;

            let vector: Vec<(u32, f32)> = if !sv_query.text.is_empty() {
                // Text provided - tokenize server-side
                let field_entry = schema
                    .get_field_entry(field)
                    .ok_or_else(|| format!("Field entry for '{}' not found", sv_query.field))?;
                let sparse_config = field_entry.sparse_vector_config.as_ref().ok_or_else(|| {
                    format!("Field '{}' is not a sparse vector field", sv_query.field)
                })?;
                let query_config = sparse_config
                    .query_config
                    .as_ref()
                    .ok_or_else(|| format!("Field '{}' has no query config", sv_query.field))?;
                let tokenizer_name = query_config.tokenizer.as_ref().ok_or_else(|| {
                    format!("Field '{}' has no tokenizer configured", sv_query.field)
                })?;

                let tokenizer = tokenizer_cache()
                    .get_or_load(tokenizer_name)
                    .map_err(|e| format!("Failed to load tokenizer '{}': {}", tokenizer_name, e))?;

                let token_counts = tokenizer
                    .tokenize(&sv_query.text)
                    .map_err(|e| format!("Tokenization failed: {}", e))?;

                // Convert (token_id, count) to (token_id, weight)
                // Apply IDF weighting if configured
                let token_ids: Vec<u32> = token_counts.iter().map(|(id, _)| *id).collect();
                let weights: Vec<f32> = match query_config.weighting {
                    QueryWeighting::One => token_counts
                        .iter()
                        .map(|(_, count)| *count as f32)
                        .collect(),
                    QueryWeighting::Idf => {
                        if let Some(stats) = global_stats {
                            let idf_weights = stats.sparse_idf_weights(field, &token_ids);
                            // Multiply count by IDF weight
                            token_counts
                                .iter()
                                .zip(idf_weights.iter())
                                .map(|((_, count), idf)| *count as f32 * idf)
                                .collect()
                        } else {
                            // No global stats available, fall back to count
                            token_counts
                                .iter()
                                .map(|(_, count)| *count as f32)
                                .collect()
                        }
                    }
                };
                token_ids.into_iter().zip(weights).collect()
            } else {
                // Pre-computed indices/values provided (from embedding model)
                // Filter out entries with negative or zero weights - negative weights
                // from SPLADE indicate "do not match this token" which we handle by
                // simply not including them in the query
                sv_query
                    .indices
                    .iter()
                    .copied()
                    .zip(sv_query.values.iter().copied())
                    .filter(|(_, weight)| *weight > 0.0)
                    .collect()
            };

            let combiner = convert_combiner(sv_query.combiner);
            Ok(Box::new(
                SparseVectorQuery::new(field, vector).with_combiner(combiner),
            ))
        }
        Some(ProtoQueryType::DenseVector(dv_query)) => {
            let field = schema
                .get_field(&dv_query.field)
                .ok_or_else(|| format!("Field '{}' not found", dv_query.field))?;
            let mut query = DenseVectorQuery::new(field, dv_query.vector.clone());
            if dv_query.nprobe > 0 {
                query = query.with_nprobe(dv_query.nprobe as usize);
            }
            if dv_query.rerank_factor > 0 {
                query = query.with_rerank_factor(dv_query.rerank_factor as usize);
            }
            let combiner = convert_combiner(dv_query.combiner);
            query = query.with_combiner(combiner);
            Ok(Box::new(query))
        }
        None => Err("Query type is required".to_string()),
    }
}

pub fn convert_field_value(value: &CoreFieldValue) -> proto::FieldValue {
    let v = match value {
        CoreFieldValue::Text(s) => Value::Text(s.clone()),
        CoreFieldValue::U64(n) => Value::U64(*n),
        CoreFieldValue::I64(n) => Value::I64(*n),
        CoreFieldValue::F64(n) => Value::F64(*n),
        CoreFieldValue::Bytes(b) => Value::BytesValue(b.clone()),
        CoreFieldValue::SparseVector(entries) => {
            let (indices, values): (Vec<u32>, Vec<f32>) = entries.iter().copied().unzip();
            Value::SparseVector(proto::SparseVector { indices, values })
        }
        CoreFieldValue::DenseVector(values) => Value::DenseVector(proto::DenseVector {
            values: values.clone(),
        }),
        CoreFieldValue::Json(json_val) => {
            Value::JsonValue(serde_json::to_string(json_val).unwrap_or_default())
        }
    };
    proto::FieldValue { value: Some(v) }
}

/// Convert Schema to SDL string representation
pub fn schema_to_sdl(schema: &Schema) -> String {
    use hermes_core::FieldType;

    let mut lines = vec!["index _ {".to_string()];
    for (_, entry) in schema.fields() {
        let type_str = match entry.field_type {
            FieldType::Text => "text",
            FieldType::U64 => "u64",
            FieldType::I64 => "i64",
            FieldType::F64 => "f64",
            FieldType::Bytes => "bytes",
            FieldType::SparseVector => "sparse_vector",
            FieldType::DenseVector => "dense_vector",
            FieldType::Json => "json",
        };
        let mut flags = Vec::new();
        if entry.indexed {
            flags.push("indexed");
        }
        if entry.stored {
            flags.push("stored");
        }
        lines.push(format!(
            "    {}: {} {}",
            entry.name,
            type_str,
            flags.join(" ")
        ));
    }
    lines.push("}".to_string());
    lines.join("\n")
}

pub fn convert_proto_to_document(
    fields: &[proto::FieldEntry],
    schema: &Schema,
) -> Result<Document, String> {
    let mut doc = Document::new();

    for entry in fields {
        let name = &entry.name;
        let value = entry
            .value
            .as_ref()
            .ok_or_else(|| format!("Field '{}' has no value", name))?;

        let field = schema
            .get_field(name)
            .ok_or_else(|| format!("Field '{}' not found in schema", name))?;

        match &value.value {
            Some(Value::Text(s)) => doc.add_text(field, s),
            Some(Value::U64(n)) => doc.add_u64(field, *n),
            Some(Value::I64(n)) => doc.add_i64(field, *n),
            Some(Value::F64(n)) => doc.add_f64(field, *n),
            Some(Value::BytesValue(b)) => doc.add_bytes(field, b.clone()),
            Some(Value::SparseVector(sv)) => {
                let entries: Vec<(u32, f32)> = sv
                    .indices
                    .iter()
                    .copied()
                    .zip(sv.values.iter().copied())
                    .collect();
                doc.add_sparse_vector(field, entries);
            }
            Some(Value::DenseVector(dv)) => {
                doc.add_dense_vector(field, dv.values.clone());
            }
            Some(Value::JsonValue(json_str)) => {
                let json_val: serde_json::Value = serde_json::from_str(json_str)
                    .map_err(|e| format!("Invalid JSON in field '{}': {}", name, e))?;
                doc.add_json(field, json_val);
            }
            None => return Err(format!("Field '{}' has no value", name)),
        }
    }

    Ok(doc)
}
