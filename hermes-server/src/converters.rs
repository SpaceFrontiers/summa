//! Proto conversion helpers

use std::sync::LazyLock;

use hermes_core::query::{
    BinaryDenseVectorQuery, DenseVectorQuery, LazyGlobalStats, MAX_DENSE_NPROBE,
    MultiValueCombiner, RerankerConfig, SparseVectorQuery,
};
use hermes_core::structures::QueryWeighting;
use hermes_core::tokenizer::{idf_weights_cache, tokenizer_cache};
use hermes_core::{
    BooleanQuery, BoostQuery, Document, FieldValue as CoreFieldValue, PrefixQuery, Query, Schema,
    TermQuery, TokenizerRegistry,
};
use log::{debug, warn};

static TOKENIZER_REGISTRY: LazyLock<TokenizerRegistry> = LazyLock::new(TokenizerRegistry::new);

use crate::proto;
use crate::proto::field_value::Value;
use crate::proto::query::Query as ProtoQueryType;

const MAX_TEXT_QUERY_TOKENS: usize = 256;
const MAX_SPARSE_TOKEN_DIMENSIONS: usize = 4_096;

fn validate_token_expansion(kind: &str, count: usize, maximum: usize) -> Result<(), String> {
    if count > maximum {
        return Err(format!(
            "{kind} expands to {count} tokens; maximum is {maximum}"
        ));
    }
    Ok(())
}

/// Convert proto combiner enum to core MultiValueCombiner
/// Parameters (temperature, k, decay) are passed separately from the query message
fn convert_combiner(combiner: i32, temperature: f32, top_k: u32, decay: f32) -> MultiValueCombiner {
    match combiner {
        1 => MultiValueCombiner::Max,
        2 => MultiValueCombiner::Avg,
        3 => MultiValueCombiner::Sum,
        4 => MultiValueCombiner::WeightedTopK {
            k: if top_k > 0 { top_k as usize } else { 5 },
            decay: if decay > 0.0 { decay } else { 0.7 },
        },
        _ => MultiValueCombiner::LogSumExp {
            // 0 or default: LogSumExp
            temperature: if temperature > 0.0 { temperature } else { 1.5 },
        },
    }
}

/// Convert a non-zero proto combiner value for fusion chunk combination.
/// (0/unset is handled by the caller: fusion defaults to Max, not LogSumExp.)
pub fn convert_fusion_combiner(combiner: i32) -> MultiValueCombiner {
    convert_combiner(combiner, 0.0, 0, 0.0)
}

pub fn convert_query(
    query: &proto::Query,
    schema: &Schema,
    global_stats: Option<&LazyGlobalStats>,
    idf_cache_dir: Option<&std::path::Path>,
) -> Result<Box<dyn Query>, String> {
    match &query.query {
        Some(ProtoQueryType::Term(term_query)) => {
            let field = schema
                .get_field(&term_query.field)
                .ok_or_else(|| format!("Field '{}' not found", term_query.field))?;
            let entry = schema.get_field_entry(field);
            if let Some(e) = entry
                && e.field_type != hermes_core::FieldType::Text
            {
                return Err(format!(
                    "TermQuery requires a text field, but '{}' is {:?}. Use RangeQuery for numeric fields.",
                    term_query.field, e.field_type
                ));
            }
            // Tokenize the term using the field's configured tokenizer so that
            // stemmers (e.g. ru_stem) are applied, matching the indexing path.
            let tokenizer_name = entry
                .and_then(|e| e.tokenizer.as_deref())
                .unwrap_or("simple");
            let tokenizer = TOKENIZER_REGISTRY
                .get(tokenizer_name)
                .unwrap_or_else(|| Box::new(hermes_core::SimpleTokenizer));
            let tokens: Vec<String> = tokenizer
                .tokenize(&term_query.term)
                .into_iter()
                .map(|t| t.text)
                .collect();
            validate_token_expansion("TermQuery", tokens.len(), MAX_TEXT_QUERY_TOKENS)?;
            if tokens.is_empty() {
                return Err(format!("No tokens in term '{}'", term_query.term));
            }
            if tokens.len() == 1 {
                Ok(Box::new(TermQuery::text(field, &tokens[0])))
            } else {
                let mut query = BooleanQuery::new();
                for token in tokens {
                    query = query.must(TermQuery::text(field, &token));
                }
                Ok(Box::new(query))
            }
        }
        Some(ProtoQueryType::Match(match_query)) => {
            let field = schema
                .get_field(&match_query.field)
                .ok_or_else(|| format!("Field '{}' not found", match_query.field))?;

            // Trailing `*` → PrefixQuery (no tokenization, raw lowercased prefix)
            if let Some(prefix) = match_query.text.strip_suffix('*') {
                if prefix.is_empty() {
                    return Err("Prefix query must not be empty".to_string());
                }
                return Ok(Box::new(PrefixQuery::text(field, prefix)));
            }

            // Get the field's configured tokenizer (or default)
            let tokenizer_name = schema
                .get_field_entry(field)
                .and_then(|entry| entry.tokenizer.as_deref())
                .unwrap_or("simple");

            let tokenizer = TOKENIZER_REGISTRY
                .get(tokenizer_name)
                .unwrap_or_else(|| Box::new(hermes_core::SimpleTokenizer));

            let tokens: Vec<String> = tokenizer
                .tokenize(&match_query.text)
                .into_iter()
                .map(|t| t.text)
                .collect();
            validate_token_expansion("MatchQuery", tokens.len(), MAX_TEXT_QUERY_TOKENS)?;

            if tokens.is_empty() {
                return Err(format!(
                    "No tokens in match query text '{}'",
                    match_query.text
                ));
            }

            if tokens.len() == 1 {
                // Single token - use TermQuery directly
                return Ok(Box::new(TermQuery::text(field, &tokens[0])));
            }

            // Multiple tokens - use BooleanQuery with SHOULD clauses (MaxScore fast path)
            let mut query = BooleanQuery::new();
            for token in tokens {
                query = query.should(TermQuery::text(field, &token));
            }
            Ok(Box::new(query))
        }
        Some(ProtoQueryType::Boolean(bool_query)) => {
            convert_boolean_query(bool_query, schema, global_stats, idf_cache_dir)
        }
        Some(ProtoQueryType::Boost(boost_query)) => {
            if !boost_query.boost.is_finite() {
                return Err("Boost must be a finite number".to_string());
            }
            let inner = boost_query
                .query
                .as_ref()
                .ok_or_else(|| "Boost query requires inner query".to_string())?;
            let inner_query = convert_query(inner, schema, global_stats, idf_cache_dir)?;
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
            let field_entry = schema
                .get_field_entry(field)
                .ok_or_else(|| format!("Field entry for '{}' not found", sv_query.field))?;
            if field_entry.field_type != hermes_core::FieldType::SparseVector {
                return Err(format!(
                    "SparseVectorQuery requires a sparse_vector field, but '{}' is {:?}",
                    sv_query.field, field_entry.field_type
                ));
            }
            if sv_query.text.is_empty() && sv_query.indices.len() != sv_query.values.len() {
                return Err(format!(
                    "Sparse query has {} indices but {} values",
                    sv_query.indices.len(),
                    sv_query.values.len()
                ));
            }
            if let Some((index, value)) = sv_query
                .values
                .iter()
                .enumerate()
                .find(|(_, value)| !value.is_finite())
            {
                return Err(format!(
                    "Sparse query contains non-finite value {value} at index {index}"
                ));
            }
            if !sv_query.heap_factor.is_finite()
                || sv_query.heap_factor < 0.0
                || sv_query.heap_factor > 1.0
            {
                return Err(format!(
                    "Sparse query heap_factor must be finite and in [0, 1], got {}",
                    sv_query.heap_factor
                ));
            }
            if !sv_query.weight_threshold.is_finite() || sv_query.weight_threshold < 0.0 {
                return Err(format!(
                    "Sparse query weight_threshold must be finite and non-negative, got {}",
                    sv_query.weight_threshold
                ));
            }
            if !sv_query.pruning.is_finite() || sv_query.pruning < 0.0 || sv_query.pruning > 1.0 {
                return Err(format!(
                    "Sparse query pruning must be finite and in [0, 1], got {}",
                    sv_query.pruning
                ));
            }

            let vector: Vec<(u32, f32)> = if !sv_query.text.is_empty() {
                // Text provided - tokenize server-side
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
                validate_token_expansion(
                    "SparseVectorQuery.text",
                    token_counts.len(),
                    MAX_SPARSE_TOKEN_DIMENSIONS,
                )?;

                // Convert (token_id, count) to (token_id, weight)
                // Apply IDF weighting if configured
                let token_ids: Vec<u32> = token_counts.iter().map(|(id, _)| *id).collect();
                let weights: Vec<f32> = match query_config.weighting {
                    QueryWeighting::One => token_counts
                        .iter()
                        .map(|(_, count)| *count as f32)
                        .collect(),
                    QueryWeighting::Idf => {
                        // Use real IDF from global index statistics
                        if let Some(stats) = global_stats {
                            let idf_weights = stats.sparse_idf_weights(field, &token_ids);
                            let final_weights: Vec<f32> = token_counts
                                .iter()
                                .zip(idf_weights.iter())
                                .map(|((_, count), idf)| *count as f32 * idf)
                                .collect();
                            if log::log_enabled!(log::Level::Debug) {
                                let paired: Vec<_> = token_ids
                                    .iter()
                                    .zip(final_weights.iter())
                                    .map(|(id, w)| {
                                        let tok = tokenizer.id_to_token(*id).unwrap_or_default();
                                        format!("({:?},{},{:.4})", tok, id, w)
                                    })
                                    .collect();
                                debug!(
                                    "Sparse IDF (global stats): field={}, total_docs={}, tokens=[{}]",
                                    sv_query.field,
                                    stats.total_docs(),
                                    paired.join(", "),
                                );
                            }
                            final_weights
                        } else {
                            warn!(
                                "Sparse IDF: no global_stats available for field={}, falling back to count",
                                sv_query.field,
                            );
                            token_counts
                                .iter()
                                .map(|(_, count)| *count as f32)
                                .collect()
                        }
                    }
                    QueryWeighting::IdfFile => {
                        // Use pre-computed IDF from model's idf.json
                        let precomputed =
                            idf_weights_cache().get_or_load(tokenizer_name, idf_cache_dir);

                        if let Some(idf_weights) = &precomputed {
                            let weights: Vec<f32> = token_counts
                                .iter()
                                .map(|&(id, count)| count as f32 * idf_weights.get(id))
                                .collect();
                            if log::log_enabled!(log::Level::Debug) {
                                let paired: Vec<_> = token_ids
                                    .iter()
                                    .zip(weights.iter())
                                    .map(|(id, w)| {
                                        let tok = tokenizer.id_to_token(*id).unwrap_or_default();
                                        format!("({:?},{},{:.4})", tok, id, w)
                                    })
                                    .collect();
                                debug!(
                                    "Sparse IDF (idf.json): tokenizer={}, tokens=[{}]",
                                    tokenizer_name,
                                    paired.join(", "),
                                );
                            }
                            weights
                        } else if let Some(stats) = global_stats {
                            // Fallback: use index-derived IDF from global stats.
                            // Without IDF weighting, all query dimensions get equal weight,
                            // which disables MaxScore pruning and causes full posting list scans.
                            warn!(
                                "Sparse IdfFile: no idf.json for model '{}', field={}, falling back to index-derived IDF",
                                tokenizer_name, sv_query.field,
                            );
                            let idf_weights = stats.sparse_idf_weights(field, &token_ids);
                            token_counts
                                .iter()
                                .zip(idf_weights.iter())
                                .map(|((_, count), idf)| *count as f32 * idf)
                                .collect()
                        } else {
                            warn!(
                                "Sparse IdfFile: no idf.json and no global stats for field={}, falling back to count",
                                sv_query.field,
                            );
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

            let combiner = convert_combiner(
                sv_query.combiner,
                sv_query.combiner_temperature,
                sv_query.combiner_top_k,
                sv_query.combiner_decay,
            );
            // SearchRequest.candidate_limit is the single candidate budget.
            // Query collectors must not multiply it independently.
            let mut query = SparseVectorQuery::new(field, vector)
                .with_combiner(combiner)
                .with_over_fetch_factor(1.0);

            // Apply SDL query_config defaults, then override with per-request values
            let sparse_config = schema
                .get_field_entry(field)
                .and_then(|entry| entry.sparse_vector_config.as_ref());
            let schema_qc = sparse_config.and_then(|config| config.query_config.as_ref());

            // heap_factor: per-request > schema default > 1.0
            if sv_query.heap_factor > 0.0 {
                query = query.with_heap_factor(sv_query.heap_factor);
            } else if let Some(qc) = schema_qc {
                query = query.with_heap_factor(qc.heap_factor);
            }

            // weight_threshold: per-request > schema default > 0.0
            if sv_query.weight_threshold > 0.0 {
                query = query.with_weight_threshold(sv_query.weight_threshold);
            } else if let Some(qc) = schema_qc {
                query = query.with_weight_threshold(qc.weight_threshold);
            }

            // max_query_dims: per-request > schema default > None
            if sv_query.max_query_dims > 0 {
                query = query.with_max_query_dims(sv_query.max_query_dims as usize);
            } else if let Some(Some(max_dims)) = schema_qc.map(|qc| qc.max_query_dims) {
                query = query.with_max_query_dims(max_dims);
            }

            // pruning: per-request > schema default > None
            if sv_query.pruning > 0.0 {
                query = query.with_pruning(sv_query.pruning);
            } else if let Some(Some(p)) = schema_qc.map(|qc| qc.pruning) {
                query = query.with_pruning(p);
            }

            // min_query_dims: schema default (no per-request override)
            if let Some(qc) = schema_qc {
                query = query.with_min_query_dims(qc.min_query_dims);
            }

            // LSP/0 gamma: per-request > schema default > depth-derived.
            if let Some(gamma) = sv_query.lsp_gamma {
                query = query.with_lsp_gamma(gamma as usize);
            } else if let Some(gamma) = schema_qc.and_then(|config| config.lsp_gamma) {
                query = query.with_lsp_gamma(gamma);
            }

            Ok(Box::new(query))
        }
        Some(ProtoQueryType::DenseVector(dv_query)) => {
            let field = schema
                .get_field(&dv_query.field)
                .ok_or_else(|| format!("Field '{}' not found", dv_query.field))?;
            let entry = schema
                .get_field_entry(field)
                .ok_or_else(|| format!("Field entry for '{}' not found", dv_query.field))?;
            if entry.field_type != hermes_core::FieldType::DenseVector {
                return Err(format!(
                    "DenseVectorQuery requires a dense_vector field, but '{}' is {:?}",
                    dv_query.field, entry.field_type
                ));
            }
            let config = entry.dense_vector_config.as_ref().ok_or_else(|| {
                format!(
                    "Dense vector field '{}' has no dense vector configuration",
                    dv_query.field
                )
            })?;
            if dv_query.vector.is_empty() {
                return Err("Dense query vector must not be empty".to_string());
            }
            if dv_query.vector.len() != config.dim {
                return Err(format!(
                    "Dense query vector dimension {} does not match field '{}' dimension {}",
                    dv_query.vector.len(),
                    dv_query.field,
                    config.dim
                ));
            }
            if let Some((index, value)) = dv_query
                .vector
                .iter()
                .enumerate()
                .find(|(_, value)| !value.is_finite())
            {
                return Err(format!(
                    "Dense query vector contains non-finite value {value} at index {index}"
                ));
            }

            let nprobe = if dv_query.nprobe == 0 {
                config.nprobe
            } else {
                dv_query.nprobe as usize
            };
            if nprobe > MAX_DENSE_NPROBE {
                return Err(format!(
                    "Dense query nprobe must be at most {MAX_DENSE_NPROBE}, got {nprobe}"
                ));
            }

            let mut query = DenseVectorQuery::new(field, dv_query.vector.clone())
                .with_nprobe(nprobe)
                .with_rerank_factor(1.0);
            let combiner = convert_combiner(
                dv_query.combiner,
                dv_query.combiner_temperature,
                dv_query.combiner_top_k,
                dv_query.combiner_decay,
            );
            query = query.with_combiner(combiner);
            Ok(Box::new(query))
        }
        Some(ProtoQueryType::BinaryDenseVector(bv_query)) => {
            let field = schema
                .get_field(&bv_query.field)
                .ok_or_else(|| format!("Field '{}' not found", bv_query.field))?;
            let entry = schema
                .get_field_entry(field)
                .ok_or_else(|| format!("Field entry for '{}' not found", bv_query.field))?;
            if entry.field_type != hermes_core::FieldType::BinaryDenseVector {
                return Err(format!(
                    "BinaryDenseVectorQuery requires a binary_dense_vector field, but '{}' is {:?}",
                    bv_query.field, entry.field_type
                ));
            }
            let config = entry.binary_dense_vector_config.as_ref().ok_or_else(|| {
                format!(
                    "Binary dense vector field '{}' has no configuration",
                    bv_query.field
                )
            })?;
            if bv_query.vector.len() != config.byte_len() {
                return Err(format!(
                    "Binary query byte length {} does not match field '{}' byte length {}",
                    bv_query.vector.len(),
                    bv_query.field,
                    config.byte_len()
                ));
            }
            let mut query = BinaryDenseVectorQuery::new(field, bv_query.vector.clone());
            let combiner = convert_combiner(
                bv_query.combiner,
                bv_query.combiner_temperature,
                bv_query.combiner_top_k,
                bv_query.combiner_decay,
            );
            query = query.with_combiner(combiner);
            Ok(Box::new(query))
        }
        Some(ProtoQueryType::Range(range_query)) => convert_range_query(range_query, schema),
        Some(ProtoQueryType::Prefix(prefix_query)) => {
            let field = schema
                .get_field(&prefix_query.field)
                .ok_or_else(|| format!("Field '{}' not found", prefix_query.field))?;
            if prefix_query.prefix.is_empty() {
                return Err("Prefix query must not be empty".to_string());
            }
            Ok(Box::new(PrefixQuery::text(field, &prefix_query.prefix)))
        }
        Some(ProtoQueryType::Fusion(_)) => {
            Err("FusionQuery is only supported at the top level of SearchRequest.query".to_string())
        }
        None => Err("Query type is required".to_string()),
    }
}

/// Convert a BooleanQuery. MUST/SHOULD/MUST_NOT clauses are mapped to
/// the corresponding BooleanQuery fields. Intersection between MUST clauses
/// (including term filters and vector queries) is handled by BooleanScorer's
/// DocSet-based seek optimization.
fn convert_boolean_query(
    bool_query: &proto::BooleanQuery,
    schema: &Schema,
    global_stats: Option<&LazyGlobalStats>,
    idf_cache_dir: Option<&std::path::Path>,
) -> Result<Box<dyn Query>, String> {
    let mut bq = BooleanQuery::new();
    for q in &bool_query.must {
        let inner = convert_query(q, schema, global_stats, idf_cache_dir)?;
        bq.must.push(inner.into());
    }
    for q in &bool_query.should {
        let inner = convert_query(q, schema, global_stats, idf_cache_dir)?;
        bq.should.push(inner.into());
    }
    for q in &bool_query.must_not {
        let inner = convert_query(q, schema, global_stats, idf_cache_dir)?;
        bq.must_not.push(inner.into());
    }
    Ok(Box::new(bq))
}

/// Convert a RangeQuery from proto to core.
///
/// Detects the type from which bounds are set:
/// - min_u64/max_u64 → U64 range
/// - min_i64/max_i64 → I64 range
/// - min_f64/max_f64 → F64 range
///   Field must have fast=true in the schema.
fn convert_range_query(rq: &proto::RangeQuery, schema: &Schema) -> Result<Box<dyn Query>, String> {
    use hermes_core::query::{RangeBound, RangeQuery};

    let field = schema
        .get_field(&rq.field)
        .ok_or_else(|| format!("Range query field '{}' not found", rq.field))?;

    let entry = schema
        .get_field_entry(field)
        .ok_or_else(|| format!("Field entry for '{}' not found", rq.field))?;

    if !entry.fast {
        return Err(format!(
            "Range query field '{}' must have fast=true in schema",
            rq.field
        ));
    }

    // Detect which type of bounds are provided
    let bound = if rq.min_u64.is_some() || rq.max_u64.is_some() {
        RangeBound::U64 {
            min: rq.min_u64,
            max: rq.max_u64,
        }
    } else if rq.min_i64.is_some() || rq.max_i64.is_some() {
        RangeBound::I64 {
            min: rq.min_i64,
            max: rq.max_i64,
        }
    } else if rq.min_f64.is_some() || rq.max_f64.is_some() {
        RangeBound::F64 {
            min: rq.min_f64,
            max: rq.max_f64,
        }
    } else {
        // No bounds specified — match all docs that have a value (exists check)
        // Use full u64 range which excludes FAST_FIELD_MISSING
        RangeBound::U64 {
            min: None,
            max: None,
        }
    };

    Ok(Box::new(RangeQuery::new(field, bound)))
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
        CoreFieldValue::BinaryDenseVector(b) => Value::BinaryDenseVector(b.clone()),
    };
    proto::FieldValue { value: Some(v) }
}

/// Convert Schema to SDL string representation
///
/// Produces a faithful round-trippable SDL including tokenizer, multi, fast,
/// positions, and full vector configuration (dense/sparse).
pub fn schema_to_sdl(schema: &Schema) -> String {
    use hermes_core::dsl::{
        BinaryIndexType, DenseVectorQuantization, FieldType, IvfRoutingMode, PositionMode,
        VectorIndexType,
    };
    use hermes_core::structures::{IndexSize, WeightQuantization};

    let mut lines = vec!["index _ {".to_string()];
    for (_, entry) in schema.fields() {
        // --- type name + optional type-level config ---
        let mut type_part = match entry.field_type {
            FieldType::Text => "text".to_string(),
            FieldType::U64 => "u64".to_string(),
            FieldType::I64 => "i64".to_string(),
            FieldType::F64 => "f64".to_string(),
            FieldType::Bytes => "bytes".to_string(),
            FieldType::Json => "json".to_string(),
            FieldType::SparseVector => "sparse_vector".to_string(),
            FieldType::DenseVector => "dense_vector".to_string(),
            FieldType::BinaryDenseVector => "binary_dense_vector".to_string(),
        };

        // Text tokenizer: text<en_stem>
        if entry.field_type == FieldType::Text
            && let Some(ref tok) = entry.tokenizer
        {
            type_part.push_str(&format!("<{}>", tok));
        }

        // Sparse vector type config: sparse_vector<u16>
        if let Some(ref cfg) = entry.sparse_vector_config {
            let idx = match cfg.index_size {
                IndexSize::U16 => "u16",
                IndexSize::U32 => "u32",
            };
            type_part.push_str(&format!("<{}>", idx));
        }

        // Dense vector type config: dense_vector<768> or dense_vector<768, f16>
        if let Some(ref cfg) = entry.dense_vector_config {
            let quant_suffix = match cfg.quantization {
                DenseVectorQuantization::F32 => String::new(),
                DenseVectorQuantization::F16 => ", f16".to_string(),
                DenseVectorQuantization::UInt8 => ", uint8".to_string(),
                DenseVectorQuantization::Binary => String::new(), // binary uses BinaryDenseVector field type
            };
            type_part.push_str(&format!("<{}{}>", cfg.dim, quant_suffix));
        }

        // Binary dense vector type config: binary_dense_vector<128>
        if let Some(ref cfg) = entry.binary_dense_vector_config {
            type_part.push_str(&format!("<{}>", cfg.dim));
        }

        // --- attributes: [indexed<...>, stored<multi>, fast] ---
        let mut attrs = Vec::new();

        if entry.indexed {
            let mut idx_params = Vec::new();

            // Positions (for text/sparse)
            if let Some(pos) = entry.positions {
                idx_params.push(match pos {
                    PositionMode::Ordinal => "ordinal".to_string(),
                    PositionMode::TokenPosition => "token_position".to_string(),
                    PositionMode::Full => "positions".to_string(),
                });
            }

            // Dense vector index params
            if let Some(ref cfg) = entry.dense_vector_config {
                let idx_name = match cfg.index_type {
                    VectorIndexType::Flat => "flat",
                    // Unreachable in practice: schemas with the retired
                    // ivf_pq type are rejected at index create/open.
                    VectorIndexType::IvfPq => "ivf_pq",
                    VectorIndexType::Tq => "tq",
                    VectorIndexType::IvfTq => "ivf_tq",
                };
                idx_params.push(idx_name.to_string());
                // TQ scans every code: IVF knobs do not apply and re-parsing
                // them would warn.
                if cfg.index_type != VectorIndexType::Tq {
                    if let Some(nc) = cfg.num_clusters {
                        idx_params.push(format!("num_clusters: {}", nc));
                    }
                    if cfg.nprobe != 64 {
                        idx_params.push(format!("nprobe: {}", cfg.nprobe));
                    }
                    if cfg.ivf_routing != IvfRoutingMode::Auto {
                        let routing = match cfg.ivf_routing {
                            IvfRoutingMode::Auto => unreachable!(),
                            IvfRoutingMode::Flat => "flat",
                            IvfRoutingMode::TwoLevel => "two_level",
                            IvfRoutingMode::Hnsw => "hnsw",
                        };
                        idx_params.push(format!("routing: {routing}"));
                    }
                    if let Some(soar) = &cfg.soar {
                        let mode = if soar.selective {
                            "selective"
                        } else if soar.num_secondary > 1 {
                            "aggressive"
                        } else {
                            "full"
                        };
                        idx_params.push(format!("soar: {mode}"));
                    }
                }
            }

            if let Some(ref cfg) = entry.binary_dense_vector_config {
                idx_params.push(
                    match cfg.index_type {
                        BinaryIndexType::Flat => "flat",
                        BinaryIndexType::Ivf => "ivf",
                    }
                    .to_string(),
                );
                if let Some(num_clusters) = cfg.num_clusters {
                    idx_params.push(format!("num_clusters: {num_clusters}"));
                }
                if cfg.nprobe != 64 {
                    idx_params.push(format!("nprobe: {}", cfg.nprobe));
                }
                if cfg.ivf_routing != IvfRoutingMode::Auto {
                    let routing = match cfg.ivf_routing {
                        IvfRoutingMode::Auto => unreachable!(),
                        IvfRoutingMode::Flat => "flat",
                        IvfRoutingMode::TwoLevel => "two_level",
                        IvfRoutingMode::Hnsw => "hnsw",
                    };
                    idx_params.push(format!("routing: {routing}"));
                }
            }

            // Sparse vector index params
            if let Some(ref cfg) = entry.sparse_vector_config {
                let quant = match cfg.weight_quantization {
                    WeightQuantization::Float32 => None,
                    WeightQuantization::Float16 => Some("float16"),
                    WeightQuantization::UInt8 => Some("uint8"),
                    WeightQuantization::UInt4 => Some("uint4"),
                };
                if let Some(q) = quant {
                    idx_params.push(format!("quantization: {}", q));
                }
                if cfg.weight_threshold > 0.0 {
                    idx_params.push(format!("weight_threshold: {}", cfg.weight_threshold));
                }
                if cfg.block_size != 128 {
                    idx_params.push(format!("block_size: {}", cfg.block_size));
                }
                if let Some(p) = cfg.pruning {
                    idx_params.push(format!("pruning: {}", p));
                }
                if cfg.min_terms != 4 {
                    idx_params.push(format!("min_terms: {}", cfg.min_terms));
                }
                // Query config sub-block
                if let Some(ref qc) = cfg.query_config {
                    let mut qparams = Vec::new();
                    if let Some(ref t) = qc.tokenizer {
                        qparams.push(format!("tokenizer: \"{}\"", t));
                    }
                    if qc.weighting != hermes_core::structures::QueryWeighting::One {
                        let w = match qc.weighting {
                            hermes_core::structures::QueryWeighting::Idf => "idf",
                            hermes_core::structures::QueryWeighting::IdfFile => "idf_file",
                            _ => "one",
                        };
                        qparams.push(format!("weighting: {}", w));
                    }
                    if qc.weight_threshold > 0.0 {
                        qparams.push(format!("weight_threshold: {}", qc.weight_threshold));
                    }
                    if let Some(md) = qc.max_query_dims {
                        qparams.push(format!("max_dims: {}", md));
                    }
                    if let Some(p) = qc.pruning {
                        qparams.push(format!("pruning: {}", p));
                    }
                    if qc.min_query_dims != 4 {
                        qparams.push(format!("min_query_dims: {}", qc.min_query_dims));
                    }
                    if let Some(gamma) = qc.lsp_gamma {
                        qparams.push(format!("lsp_gamma: {gamma}"));
                    }
                    if !qparams.is_empty() {
                        idx_params.push(format!("query<{}>", qparams.join(", ")));
                    }
                }
            }

            if idx_params.is_empty() {
                attrs.push("indexed".to_string());
            } else {
                attrs.push(format!("indexed<{}>", idx_params.join(", ")));
            }
        }

        if entry.stored {
            if entry.multi {
                attrs.push("stored<multi>".to_string());
            } else {
                attrs.push("stored".to_string());
            }
        }

        if entry.fast {
            attrs.push("fast".to_string());
        }

        if entry.primary_key {
            attrs.push("primary".to_string());
        }

        if attrs.is_empty() {
            lines.push(format!("    field {}: {}", entry.name, type_part));
        } else {
            lines.push(format!(
                "    field {}: {} [{}]",
                entry.name,
                type_part,
                attrs.join(", ")
            ));
        }
    }
    lines.push("}".to_string());
    lines.join("\n")
}

pub fn convert_reranker(
    reranker: &proto::Reranker,
    schema: &Schema,
) -> Result<RerankerConfig, String> {
    let field = schema
        .get_field(&reranker.field)
        .ok_or_else(|| format!("Reranker field '{}' not found", reranker.field))?;

    let entry = schema
        .get_field_entry(field)
        .ok_or_else(|| format!("Field entry for '{}' not found", reranker.field))?;

    let is_binary = entry.field_type == hermes_core::FieldType::BinaryDenseVector;

    if !reranker.rrf_k.is_finite() || reranker.rrf_k < 0.0 {
        return Err(format!(
            "Reranker rrf_k must be finite and non-negative, got {}",
            reranker.rrf_k
        ));
    }

    if entry.field_type != hermes_core::FieldType::DenseVector && !is_binary {
        return Err(format!(
            "Reranker field '{}' must be dense_vector or binary_dense_vector, got {:?}",
            reranker.field, entry.field_type
        ));
    }

    // Validate query vector
    if is_binary {
        if reranker.binary_vector.is_empty() {
            return Err(
                "Reranker binary_vector must not be empty for binary_dense_vector field"
                    .to_string(),
            );
        }
        if let Some(ref bv_config) = entry.binary_dense_vector_config {
            let expected_bytes = bv_config.byte_len();
            if reranker.binary_vector.len() != expected_bytes {
                return Err(format!(
                    "Reranker binary_vector byte length {} does not match field '{}' expected {} (dim={})",
                    reranker.binary_vector.len(),
                    reranker.field,
                    expected_bytes,
                    bv_config.dim
                ));
            }
        }
    } else {
        if reranker.vector.is_empty() {
            return Err("Reranker query vector must not be empty".to_string());
        }
        if let Some(ref dv_config) = entry.dense_vector_config
            && reranker.vector.len() != dv_config.dim
        {
            return Err(format!(
                "Reranker query vector dimension {} does not match field '{}' dimension {}",
                reranker.vector.len(),
                reranker.field,
                dv_config.dim
            ));
        }
        if let Some((index, value)) = reranker
            .vector
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(format!(
                "Reranker query vector contains non-finite value {value} at index {index}"
            ));
        }
    }

    // Default reranker combiner to WeightedTopK(k=3, decay=0.7) — decaying
    // combination of top-3 best-matching chunks. LogSumExp heavily biases
    // toward documents with many vectors regardless of relevance.
    // Proto enum default 0 = LOG_SUM_EXP — if nothing was explicitly set
    // (combiner=0, temperature=0), override for reranking.
    let combiner = if reranker.combiner == 0 && reranker.combiner_temperature == 0.0 {
        MultiValueCombiner::WeightedTopK { k: 3, decay: 0.7 }
    } else {
        convert_combiner(
            reranker.combiner,
            reranker.combiner_temperature,
            reranker.combiner_top_k,
            reranker.combiner_decay,
        )
    };

    let unit_norm = entry
        .dense_vector_config
        .as_ref()
        .is_some_and(|c| c.unit_norm);

    let matryoshka_dims = if reranker.matryoshka_dims > 0 {
        Some(reranker.matryoshka_dims as usize)
    } else {
        None
    };
    if is_binary && matryoshka_dims.is_some() {
        return Err("Reranker matryoshka_dims is not supported for binary vectors".to_string());
    }
    if let (Some(dims), Some(config)) = (matryoshka_dims, entry.dense_vector_config.as_ref())
        && dims > config.dim
    {
        return Err(format!(
            "Reranker matryoshka_dims {dims} exceeds field '{}' dimension {}",
            reranker.field, config.dim
        ));
    }

    Ok(RerankerConfig {
        field,
        vector: reranker.vector.clone(),
        binary_vector: reranker.binary_vector.clone(),
        combiner,
        unit_norm,
        matryoshka_dims,
        rrf_k: reranker.rrf_k,
    })
}

fn validate_binary_document_value(
    name: &str,
    field: hermes_core::Field,
    bytes: &[u8],
    schema: &Schema,
) -> Result<(), String> {
    let config = schema
        .get_field_entry(field)
        .and_then(|entry| entry.binary_dense_vector_config.as_ref())
        .ok_or_else(|| format!("Field '{}' has no binary dense vector config", name))?;
    if bytes.len() != config.byte_len() {
        return Err(format!(
            "Field '{}': binary vector byte length {} does not match schema byte length {}",
            name,
            bytes.len(),
            config.byte_len()
        ));
    }
    Ok(())
}

pub fn convert_proto_to_document(
    fields: &[proto::FieldEntry],
    schema: &Schema,
) -> Result<Document, String> {
    use hermes_core::FieldType;

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

        let field_type = schema
            .get_field_entry(field)
            .map(|e| &e.field_type)
            .ok_or_else(|| format!("Field '{}' has no entry", name))?;

        // Extract a numeric value from any proto numeric variant for coercion.
        // Clients infer the proto type from the native value (e.g. Python sends
        // positive ints as u64 even for i64/f64 schema fields), so we coerce
        // to match the schema field type.
        match (&value.value, field_type) {
            // ── Text ──
            (Some(Value::Text(s)), _) => doc.add_text(field, s),

            // ── Numeric: coerce any numeric proto variant to the schema type ──
            (Some(Value::U64(n)), FieldType::U64) => doc.add_u64(field, *n),
            (Some(Value::U64(n)), FieldType::I64) => doc.add_i64(field, *n as i64),
            (Some(Value::U64(n)), FieldType::F64) => doc.add_f64(field, *n as f64),

            (Some(Value::I64(n)), FieldType::I64) => doc.add_i64(field, *n),
            (Some(Value::I64(n)), FieldType::U64) => doc.add_u64(field, *n as u64),
            (Some(Value::I64(n)), FieldType::F64) => doc.add_f64(field, *n as f64),

            (Some(Value::F64(n)), FieldType::F64) => doc.add_f64(field, *n),
            (Some(Value::F64(n)), FieldType::U64) => doc.add_u64(field, *n as u64),
            (Some(Value::F64(n)), FieldType::I64) => doc.add_i64(field, *n as i64),

            // ── Non-numeric types: no coercion needed ──
            // bytes_value coerced to binary_dense_vector when schema says so
            (Some(Value::BytesValue(b)), FieldType::BinaryDenseVector) => {
                validate_binary_document_value(name, field, b, schema)?;
                doc.add_binary_dense_vector(field, b.clone());
            }
            (Some(Value::BytesValue(b)), _) => doc.add_bytes(field, b.clone()),
            (Some(Value::BinaryDenseVector(b)), FieldType::BinaryDenseVector) => {
                validate_binary_document_value(name, field, b, schema)?;
                doc.add_binary_dense_vector(field, b.clone());
            }
            (Some(Value::SparseVector(sv)), FieldType::SparseVector) => {
                if sv.indices.len() != sv.values.len() {
                    return Err(format!(
                        "Field '{}': sparse vector has {} indices but {} values",
                        name,
                        sv.indices.len(),
                        sv.values.len()
                    ));
                }
                if let Some((index, value)) =
                    sv.values.iter().enumerate().find(|(_, v)| !v.is_finite())
                {
                    return Err(format!(
                        "Field '{}': sparse vector contains non-finite value {value} at index {index}",
                        name
                    ));
                }
                let entries: Vec<(u32, f32)> = sv
                    .indices
                    .iter()
                    .copied()
                    .zip(sv.values.iter().copied())
                    .collect();
                doc.add_sparse_vector(field, entries);
            }
            (Some(Value::DenseVector(dv)), FieldType::DenseVector) => {
                let expected_dim = schema
                    .get_field_entry(field)
                    .and_then(|field| field.dense_vector_config.as_ref())
                    .map(|config| config.dim)
                    .ok_or_else(|| format!("Field '{}' has no dense vector config", name))?;
                if dv.values.len() != expected_dim {
                    return Err(format!(
                        "Field '{}': dense vector dimension {} does not match schema dimension {}",
                        name,
                        dv.values.len(),
                        expected_dim
                    ));
                }
                if let Some((index, value)) = dv
                    .values
                    .iter()
                    .enumerate()
                    .find(|(_, value)| !value.is_finite())
                {
                    return Err(format!(
                        "Field '{}': dense vector contains non-finite value {value} at index {index}",
                        name
                    ));
                }
                doc.add_dense_vector(field, dv.values.clone());
            }
            (Some(Value::DenseVector(_)), got)
            | (Some(Value::SparseVector(_)), got)
            | (Some(Value::BinaryDenseVector(_)), got) => {
                return Err(format!(
                    "Field '{}': vector value does not match schema type {:?}",
                    name, got
                ));
            }
            // ── JSON: expand string arrays into multi-valued text fields ──
            // Python client serializes list[str] as json_value '["en","fr"]'
            // because it can't distinguish from a generic list. When the schema
            // field is Text, expand the array into multiple add_text calls.
            (Some(Value::JsonValue(json_str)), FieldType::Text) => {
                let json_val: serde_json::Value = serde_json::from_str(json_str)
                    .map_err(|e| format!("Invalid JSON in field '{}': {}", name, e))?;
                if let serde_json::Value::Array(arr) = &json_val {
                    for item in arr {
                        if let serde_json::Value::String(s) = item {
                            doc.add_text(field, s);
                        } else {
                            return Err(format!(
                                "Field '{}': expected string in JSON array, got {}",
                                name, item
                            ));
                        }
                    }
                } else if let serde_json::Value::String(s) = &json_val {
                    doc.add_text(field, s);
                } else {
                    return Err(format!(
                        "Field '{}': expected JSON string array for text field, got {}",
                        name, json_val
                    ));
                }
            }
            (Some(Value::JsonValue(json_str)), _) => {
                let json_val: serde_json::Value = serde_json::from_str(json_str)
                    .map_err(|e| format!("Invalid JSON in field '{}': {}", name, e))?;
                doc.add_json(field, json_val);
            }
            (None, _) => return Err(format!("Field '{}' has no value", name)),
            // Numeric value sent to a non-numeric field (e.g. u64 to text) — skip with warning
            (Some(_), _) => {
                warn!(
                    "Field '{}': proto value type does not match schema type {:?}, skipping",
                    name, field_type
                );
            }
        }
    }

    Ok(doc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_expansion_is_bounded_before_query_construction() {
        assert!(validate_token_expansion("test", 256, 256).is_ok());
        assert!(validate_token_expansion("test", 257, 256).is_err());
    }

    fn dense_test_schema(nprobe: usize) -> Schema {
        let mut builder = hermes_core::SchemaBuilder::default();
        let mut config = hermes_core::dsl::DenseVectorConfig::new(3);
        config.nprobe = nprobe;
        builder.add_dense_vector_field_with_config("embedding", true, false, config);
        builder.add_text_field("title", true, false);
        builder.build()
    }

    fn dense_proto_query(vector: Vec<f32>) -> proto::Query {
        proto::Query {
            query: Some(ProtoQueryType::DenseVector(proto::DenseVectorQuery {
                field: "embedding".to_string(),
                vector,
                ..Default::default()
            })),
        }
    }

    fn vector_test_schema() -> Schema {
        let mut builder = hermes_core::SchemaBuilder::default();
        builder.add_sparse_vector_field("sparse", true, false);
        builder.add_binary_dense_vector_field("binary", 16, true, false);
        builder.add_text_field("title", true, false);
        builder.build()
    }

    fn bmp_sparse_test_schema() -> Schema {
        let mut builder = hermes_core::SchemaBuilder::default();
        let mut config = hermes_core::structures::SparseVectorConfig::splade_bmp();
        config.dims = Some(16);
        builder.add_sparse_vector_field_with_config("sparse", true, false, config);
        builder.build()
    }

    fn sparse_proto_query(pruning: f32) -> proto::Query {
        proto::Query {
            query: Some(ProtoQueryType::SparseVector(proto::SparseVectorQuery {
                field: "sparse".to_string(),
                indices: vec![1, 2, 3, 4, 5, 6],
                values: vec![1.0; 6],
                pruning,
                ..Default::default()
            })),
        }
    }

    #[test]
    fn bmp_query_dimension_pruning_is_explicit_not_a_server_fallback() {
        let schema = bmp_sparse_test_schema();
        let default_query = convert_query(&sparse_proto_query(0.0), &schema, None, None).unwrap();
        assert!(!default_query.to_string().contains("orig="));

        let pruned_query = convert_query(&sparse_proto_query(0.33), &schema, None, None).unwrap();
        assert!(pruned_query.to_string().contains("orig=6"));
    }

    #[test]
    fn dense_query_uses_schema_nprobe_when_request_omits_it() {
        let schema = dense_test_schema(17);
        let query =
            convert_query(&dense_proto_query(vec![1.0, 2.0, 3.0]), &schema, None, None).unwrap();

        assert!(query.to_string().contains("nprobe=17"));
    }

    #[test]
    fn server_conversion_uses_the_shared_candidate_budget() {
        let schema = dense_test_schema(17);
        let query =
            convert_query(&dense_proto_query(vec![1.0, 2.0, 3.0]), &schema, None, None).unwrap();

        assert!(query.to_string().contains("rerank=1"));
    }

    #[test]
    fn dense_query_rejects_invalid_dimensions_and_values() {
        let schema = dense_test_schema(17);
        for vector in [
            vec![],
            vec![1.0, 2.0],
            vec![1.0, 2.0, 3.0, 4.0],
            vec![1.0, f32::NAN, 3.0],
            vec![1.0, f32::INFINITY, 3.0],
            vec![1.0, f32::NEG_INFINITY, 3.0],
        ] {
            assert!(
                convert_query(&dense_proto_query(vector), &schema, None, None).is_err(),
                "invalid vector should be rejected"
            );
        }
    }

    #[test]
    fn dense_query_rejects_wrong_field_and_unbounded_search_parameters() {
        let schema = dense_test_schema(17);
        let mut wrong_field = dense_proto_query(vec![1.0, 2.0, 3.0]);
        let Some(ProtoQueryType::DenseVector(query)) = wrong_field.query.as_mut() else {
            unreachable!()
        };
        query.field = "title".to_string();
        assert!(convert_query(&wrong_field, &schema, None, None).is_err());

        let mut proto = dense_proto_query(vec![1.0, 2.0, 3.0]);
        let Some(ProtoQueryType::DenseVector(query)) = proto.query.as_mut() else {
            unreachable!()
        };
        query.nprobe = MAX_DENSE_NPROBE as u32 + 1;
        assert!(convert_query(&proto, &schema, None, None).is_err());
    }

    #[test]
    fn prefix_query_rejects_empty_prefixes() {
        let schema = dense_test_schema(17);
        let match_all_prefix = proto::Query {
            query: Some(ProtoQueryType::Match(proto::MatchQuery {
                field: "title".to_string(),
                text: "*".to_string(),
            })),
        };
        let explicit_empty_prefix = proto::Query {
            query: Some(ProtoQueryType::Prefix(proto::PrefixQuery {
                field: "title".to_string(),
                prefix: String::new(),
            })),
        };

        assert!(convert_query(&match_all_prefix, &schema, None, None).is_err());
        assert!(convert_query(&explicit_empty_prefix, &schema, None, None).is_err());
    }

    #[test]
    fn binary_query_and_document_require_exact_schema_width() {
        let schema = vector_test_schema();
        let query = |field: &str, vector: Vec<u8>| proto::Query {
            query: Some(ProtoQueryType::BinaryDenseVector(
                proto::BinaryDenseVectorQuery {
                    field: field.to_string(),
                    vector,
                    ..Default::default()
                },
            )),
        };

        assert!(convert_query(&query("binary", vec![0, 1]), &schema, None, None).is_ok());
        assert!(convert_query(&query("binary", vec![0]), &schema, None, None).is_err());
        assert!(convert_query(&query("title", vec![0, 1]), &schema, None, None).is_err());

        let bad_document = [proto::FieldEntry {
            name: "binary".to_string(),
            value: Some(proto::FieldValue {
                value: Some(Value::BinaryDenseVector(vec![0])),
            }),
        }];
        assert!(convert_proto_to_document(&bad_document, &schema).is_err());
    }

    #[test]
    fn sparse_query_rejects_malformed_arrays_and_parameters() {
        let schema = vector_test_schema();
        let query = |field: &str, indices: Vec<u32>, values: Vec<f32>| proto::Query {
            query: Some(ProtoQueryType::SparseVector(proto::SparseVectorQuery {
                field: field.to_string(),
                indices,
                values,
                ..Default::default()
            })),
        };

        assert!(convert_query(&query("sparse", vec![1], vec![1.0]), &schema, None, None).is_ok());
        assert!(
            convert_query(&query("sparse", vec![1, 2], vec![1.0]), &schema, None, None).is_err()
        );
        assert!(
            convert_query(
                &query("sparse", vec![1], vec![f32::NAN]),
                &schema,
                None,
                None
            )
            .is_err()
        );
        assert!(convert_query(&query("title", vec![1], vec![1.0]), &schema, None, None).is_err());

        let mut invalid_factor = query("sparse", vec![1], vec![1.0]);
        let Some(ProtoQueryType::SparseVector(query)) = invalid_factor.query.as_mut() else {
            unreachable!()
        };
        query.heap_factor = f32::INFINITY;
        assert!(convert_query(&invalid_factor, &schema, None, None).is_err());
    }

    #[test]
    fn reranker_rejects_invalid_rrf_and_matryoshka_dimensions() {
        let schema = dense_test_schema(17);
        let base = proto::Reranker {
            field: "embedding".to_string(),
            vector: vec![1.0, 2.0, 3.0],
            ..Default::default()
        };

        for rrf_k in [f32::NAN, f32::INFINITY, -1.0] {
            let mut reranker = base.clone();
            reranker.rrf_k = rrf_k;
            assert!(convert_reranker(&reranker, &schema).is_err());
        }

        let mut too_wide = base;
        too_wide.matryoshka_dims = 4;
        assert!(convert_reranker(&too_wide, &schema).is_err());
    }

    #[test]
    fn test_schema_to_sdl_roundtrip() {
        let input_sdl = r#"
            index documents {
                field id: text<raw> [indexed, stored]
                field title: text<en_stem> [indexed, stored]
                field uris: text<default> [indexed, stored<multi>]
                field price: f64 [indexed, fast]
                field count: u64 [indexed, stored, fast]
                field tags: text<raw_ci> [indexed, stored<multi>, fast]
                field sparse_emb: sparse_vector<u32> [indexed<quantization: uint8, weight_threshold: 0.01>, stored<multi>]
                field dense_emb: dense_vector<1024, f16> [indexed<ivf_pq, routing: hnsw, num_clusters: 256>, stored<multi>]
                field meta: json [stored<multi>]
            }
        "#;

        let indexes = hermes_core::dsl::sdl::parse_sdl(input_sdl).unwrap();
        let schema = indexes[0].to_schema();
        let sdl_output = schema_to_sdl(&schema);

        // Verify the output is valid SDL that parses back
        let reparsed = hermes_core::dsl::sdl::parse_sdl(&sdl_output)
            .unwrap_or_else(|e| panic!("Failed to reparse SDL:\n{}\nError: {}", sdl_output, e));
        assert_eq!(reparsed.len(), 1);
        let reparsed_schema = reparsed[0].to_schema();

        // Verify field count matches
        assert_eq!(
            schema.fields().count(),
            reparsed_schema.fields().count(),
            "SDL:\n{}",
            sdl_output
        );

        // Verify each field entry round-trips
        for ((_, orig), (_, reparsed)) in schema.fields().zip(reparsed_schema.fields()) {
            assert_eq!(orig.name, reparsed.name, "field name mismatch");
            assert_eq!(
                orig.field_type, reparsed.field_type,
                "field type mismatch for {}",
                orig.name
            );
            assert_eq!(
                orig.indexed, reparsed.indexed,
                "indexed mismatch for {}",
                orig.name
            );
            assert_eq!(
                orig.stored, reparsed.stored,
                "stored mismatch for {}",
                orig.name
            );
            assert_eq!(
                orig.multi, reparsed.multi,
                "multi mismatch for {}",
                orig.name
            );
            assert_eq!(orig.fast, reparsed.fast, "fast mismatch for {}", orig.name);
            assert_eq!(
                orig.primary_key, reparsed.primary_key,
                "primary_key mismatch for {}",
                orig.name
            );
            assert_eq!(
                orig.tokenizer, reparsed.tokenizer,
                "tokenizer mismatch for {}",
                orig.name
            );
            assert_eq!(
                orig.positions, reparsed.positions,
                "positions mismatch for {}",
                orig.name
            );
            assert_eq!(
                orig.sparse_vector_config, reparsed.sparse_vector_config,
                "sparse config mismatch for {}",
                orig.name
            );
            if let (Some(a), Some(b)) = (&orig.dense_vector_config, &reparsed.dense_vector_config) {
                assert_eq!(a.dim, b.dim, "dense dim mismatch for {}", orig.name);
                assert_eq!(
                    a.quantization, b.quantization,
                    "dense quant mismatch for {}",
                    orig.name
                );
                assert_eq!(
                    a.index_type, b.index_type,
                    "dense index_type mismatch for {}",
                    orig.name
                );
                assert_eq!(
                    a.num_clusters, b.num_clusters,
                    "dense num_clusters mismatch for {}",
                    orig.name
                );
            }
        }
    }
}
