//! Proto conversion helpers

use std::sync::LazyLock;

use hermes_core::query::{
    DenseVectorQuery, LazyGlobalStats, MultiValueCombiner, RerankerConfig, SparseVectorQuery,
};
use hermes_core::structures::QueryWeighting;
use hermes_core::tokenizer::{idf_weights_cache, tokenizer_cache};
use hermes_core::{
    BooleanQuery, BoostQuery, Document, FieldValue as CoreFieldValue, Query, Schema, TermQuery,
    TokenizerRegistry,
};
use log::{debug, warn};

static TOKENIZER_REGISTRY: LazyLock<TokenizerRegistry> = LazyLock::new(TokenizerRegistry::new);

use crate::proto;
use crate::proto::field_value::Value;
use crate::proto::query::Query as ProtoQueryType;

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
            Ok(Box::new(TermQuery::text(field, &term_query.term)))
        }
        Some(ProtoQueryType::Match(match_query)) => {
            let field = schema
                .get_field(&match_query.field)
                .ok_or_else(|| format!("Field '{}' not found", match_query.field))?;

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
            let mut query = SparseVectorQuery::new(field, vector).with_combiner(combiner);

            // Apply SDL query_config defaults, then override with per-request values
            let schema_qc = schema
                .get_field_entry(field)
                .and_then(|e| e.sparse_vector_config.as_ref())
                .and_then(|c| c.query_config.as_ref());

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

            Ok(Box::new(query))
        }
        Some(ProtoQueryType::DenseVector(dv_query)) => {
            let field = schema
                .get_field(&dv_query.field)
                .ok_or_else(|| format!("Field '{}' not found", dv_query.field))?;
            let mut query = DenseVectorQuery::new(field, dv_query.vector.clone());
            if dv_query.nprobe > 0 {
                query = query.with_nprobe(dv_query.nprobe as usize);
            }
            if dv_query.rerank_factor > 0.0 {
                query = query.with_rerank_factor(dv_query.rerank_factor);
            }
            let combiner = convert_combiner(
                dv_query.combiner,
                dv_query.combiner_temperature,
                dv_query.combiner_top_k,
                dv_query.combiner_decay,
            );
            query = query.with_combiner(combiner);
            Ok(Box::new(query))
        }
        Some(ProtoQueryType::Range(range_query)) => convert_range_query(range_query, schema),
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
    };
    proto::FieldValue { value: Some(v) }
}

/// Convert Schema to SDL string representation
///
/// Produces a faithful round-trippable SDL including tokenizer, multi, fast,
/// positions, and full vector configuration (dense/sparse).
pub fn schema_to_sdl(schema: &Schema) -> String {
    use hermes_core::dsl::{DenseVectorQuantization, FieldType, PositionMode, VectorIndexType};
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
            };
            type_part.push_str(&format!("<{}{}>", cfg.dim, quant_suffix));
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
                    VectorIndexType::RaBitQ => "rabitq",
                    VectorIndexType::IvfRaBitQ => "ivf_rabitq",
                    VectorIndexType::ScaNN => "scann",
                };
                idx_params.push(idx_name.to_string());
                if let Some(nc) = cfg.num_clusters {
                    idx_params.push(format!("num_clusters: {}", nc));
                }
                if cfg.nprobe != 32 {
                    idx_params.push(format!("nprobe: {}", cfg.nprobe));
                }
                if let Some(bt) = cfg.build_threshold {
                    idx_params.push(format!("build_threshold: {}", bt));
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
                if let Some(qf) = cfg.quantization_factor {
                    idx_params.push(format!("quantization_factor: {}", qf));
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

    if entry.field_type != hermes_core::FieldType::DenseVector {
        return Err(format!(
            "Reranker field '{}' must be a dense_vector, got {:?}",
            reranker.field, entry.field_type
        ));
    }

    // Dense vectors are always available via lazy flat files (.vectors),
    // no need to check entry.stored — reranking reads from flat data, not store.

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

    Ok(RerankerConfig {
        field,
        vector: reranker.vector.clone(),
        combiner,
        unit_norm,
        matryoshka_dims,
    })
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
            (Some(Value::BytesValue(b)), _) => doc.add_bytes(field, b.clone()),
            (Some(Value::SparseVector(sv)), _) => {
                let entries: Vec<(u32, f32)> = sv
                    .indices
                    .iter()
                    .copied()
                    .zip(sv.values.iter().copied())
                    .collect();
                doc.add_sparse_vector(field, entries);
            }
            (Some(Value::DenseVector(dv)), _) => {
                doc.add_dense_vector(field, dv.values.clone());
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
                field dense_emb: dense_vector<1024, f16> [indexed<ivf_rabitq, num_clusters: 256>, stored<multi>]
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
