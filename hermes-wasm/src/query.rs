//! Structured query builder: converts JS query objects → core `Box<dyn Query>`.
//!
//! ```js
//! // Term query (exact token match after tokenization)
//! { term: { field: "title", value: "rust" } }
//!
//! // Match query (tokenized, multi-token OR)
//! { match: { field: "body", text: "search engine" } }
//!
//! // Boolean query
//! { boolean: { must: [...], should: [...], mustNot: [...] } }
//!
//! // Prefix query
//! { prefix: { field: "title", value: "rus" } }
//!
//! // Sparse vector query
//! { sparseVector: { field: "emb", indices: [1, 5], values: [0.5, 0.3] } }
//!
//! // Dense vector query
//! { denseVector: { field: "emb", vector: [0.1, 0.2, 0.3] } }
//!
//! // Hybrid fusion (top-level only): union of sub-query results,
//! // combined with Reciprocal Rank Fusion or normalized weighted sum
//! { fusion: {
//!     queries: [
//!       { query: { match: { field: "body", text: "rust engine" } }, weight: 1.0 },
//!       { query: { denseVector: { field: "emb", vector: [...] } }, weight: 1.0 },
//!     ],
//!     method: "rrf",        // or "normalizedWeightedSum" (default: "rrf")
//!     rrfK: 60,             // optional RRF rank constant
//!     fetchLimit: 100,      // optional per-sub-query candidate depth
//! } }
//! ```

use hermes_core::query::{
    BooleanQuery, DenseVectorQuery, PrefixQuery, Query, SparseVectorQuery, TermQuery,
};
use hermes_core::tokenizer::TokenizerRegistry;
use hermes_core::{Directory, Schema, Searcher};
use serde::{Deserialize, Serialize};
use wasm_bindgen::JsValue;

/// Top-level query object deserialized from JS.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct JsQuery {
    #[serde(default)]
    term: Option<JsTermQuery>,
    #[serde(default, rename = "match")]
    match_: Option<JsMatchQuery>,
    #[serde(default)]
    boolean: Option<JsBooleanQuery>,
    #[serde(default)]
    prefix: Option<JsPrefixQuery>,
    #[serde(default)]
    sparse_vector: Option<JsSparseVectorQuery>,
    #[serde(default)]
    dense_vector: Option<JsDenseVectorQuery>,
    #[serde(default)]
    fusion: Option<JsFusionQuery>,
}

#[derive(Deserialize)]
pub(crate) struct JsTermQuery {
    field: String,
    value: String,
}

#[derive(Deserialize)]
pub(crate) struct JsMatchQuery {
    field: String,
    text: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct JsBooleanQuery {
    #[serde(default)]
    must: Vec<JsQuery>,
    #[serde(default)]
    should: Vec<JsQuery>,
    #[serde(default)]
    must_not: Vec<JsQuery>,
}

#[derive(Deserialize)]
pub(crate) struct JsPrefixQuery {
    field: String,
    value: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct JsSparseVectorQuery {
    field: String,
    indices: Vec<u32>,
    values: Vec<f32>,
    #[serde(default)]
    heap_factor: Option<f32>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct JsDenseVectorQuery {
    field: String,
    vector: Vec<f32>,
    #[serde(default)]
    nprobe: Option<usize>,
    #[serde(default)]
    rerank_factor: Option<f32>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct JsFusionQuery {
    queries: Vec<JsWeightedQuery>,
    /// "rrf" (default) or "normalizedWeightedSum"
    #[serde(default)]
    method: Option<String>,
    #[serde(default)]
    rrf_k: Option<f32>,
    #[serde(default)]
    fetch_limit: Option<usize>,
}

#[derive(Deserialize)]
pub(crate) struct JsWeightedQuery {
    query: JsQuery,
    #[serde(default)]
    weight: Option<f32>,
}

/// Search request with optional parameters.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct JsSearchRequest {
    pub query: JsQuery,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default)]
    pub offset: usize,
    #[serde(default)]
    pub fields_to_load: Option<Vec<String>>,
}

fn default_limit() -> usize {
    10
}

/// Typed response structs (avoid serde_json::json! intermediate allocations).
#[derive(Serialize)]
pub(crate) struct StructuredSearchResponse {
    hits: Vec<StructuredHit>,
    total_hits: usize,
}

#[derive(Serialize)]
pub(crate) struct StructuredHit {
    address: HitAddress,
    score: f32,
    doc: Option<serde_json::Value>,
}

#[derive(Serialize)]
pub(crate) struct HitAddress {
    segment_id: String,
    doc_id: u32,
}

/// Execute a structured search on any Searcher<D>.
pub(crate) async fn execute_structured_search<D: Directory>(
    searcher: &Searcher<D>,
    request: JsValue,
) -> Result<JsValue, JsValue> {
    let req: JsSearchRequest = serde_wasm_bindgen::from_value(request)
        .map_err(|e| JsValue::from_str(&format!("Invalid search request: {}", e)))?;

    let field_ids = req
        .fields_to_load
        .as_ref()
        .map(|names| crate::resolve_field_ids(searcher.schema(), names))
        .transpose()?;

    // Fusion: run each sub-query independently and fuse the ranked lists
    // (union). Handled here because fusion is a searcher-level operation.
    let results = if let Some(ref fusion) = req.query.fusion {
        let mut sub_queries = Vec::with_capacity(fusion.queries.len());
        for weighted in &fusion.queries {
            if weighted.query.fusion.is_some() {
                return Err(JsValue::from_str("fusion queries cannot be nested"));
            }
            let sub = convert_query(&weighted.query, searcher.schema(), searcher.tokenizers())?;
            sub_queries.push((sub, weighted.weight.unwrap_or(1.0)));
        }
        if sub_queries.is_empty() {
            return Err(JsValue::from_str("fusion requires at least one sub-query"));
        }

        let method = match fusion.method.as_deref() {
            None | Some("rrf") => hermes_core::query::FusionMethod::Rrf {
                k: fusion.rrf_k.unwrap_or(hermes_core::query::DEFAULT_RRF_K),
            },
            Some("normalizedWeightedSum") => {
                hermes_core::query::FusionMethod::NormalizedWeightedSum
            }
            Some(other) => {
                return Err(JsValue::from_str(&format!(
                    "Unknown fusion method '{}': expected 'rrf' or 'normalizedWeightedSum'",
                    other
                )));
            }
        };

        let fused_limit = req.limit + req.offset;
        let fetch_limit = fusion.fetch_limit.unwrap_or((fused_limit * 4).max(50));
        let refs: Vec<(&dyn Query, f32)> =
            sub_queries.iter().map(|(q, w)| (q.as_ref(), *w)).collect();
        let mut fused = searcher
            .search_fused(
                &refs,
                fetch_limit,
                fused_limit,
                method,
                hermes_core::query::MultiValueCombiner::Max,
            )
            .await
            .map_err(|e| JsValue::from_str(&format!("Search error: {}", e)))?;
        if req.offset > 0 {
            fused.drain(..req.offset.min(fused.len()));
        }
        fused
    } else {
        let query = convert_query(&req.query, searcher.schema(), searcher.tokenizers())?;
        let (results, _) = searcher
            .search_with_offset_and_count(query.as_ref(), req.limit, req.offset)
            .await
            .map_err(|e| JsValue::from_str(&format!("Search error: {}", e)))?;
        results
    };

    let mut hits = Vec::with_capacity(results.len());
    for result in &results {
        let address = hermes_core::query::DocAddress::new(result.segment_id, result.doc_id);
        let doc = searcher
            .get_document_with_fields(&address, field_ids.as_ref())
            .await
            .map_err(|e| JsValue::from_str(&format!("Get document error: {}", e)))?;
        hits.push(StructuredHit {
            address: HitAddress {
                segment_id: format!("{:032x}", result.segment_id),
                doc_id: result.doc_id,
            },
            score: result.score,
            doc: doc.map(|d| d.to_json(searcher.schema())),
        });
    }

    let response = StructuredSearchResponse {
        hits,
        total_hits: results.len(),
    };

    response
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

/// Tokenize text using the field's configured tokenizer and build a query.
/// Term queries use MUST (AND), match queries use SHOULD (OR).
fn tokenize_and_build(
    field_name: &str,
    text: &str,
    must: bool,
    schema: &Schema,
    tokenizers: &TokenizerRegistry,
) -> Result<Box<dyn Query>, JsValue> {
    let field = schema
        .get_field(field_name)
        .ok_or_else(|| JsValue::from_str(&format!("Unknown field: '{}'", field_name)))?;
    let tokenizer_name = schema
        .get_field_entry(field)
        .and_then(|e| e.tokenizer.as_deref())
        .unwrap_or("simple");
    let tok = tokenizers
        .get(tokenizer_name)
        .unwrap_or_else(|| Box::new(hermes_core::SimpleTokenizer));
    let tokens: Vec<String> = tok.tokenize(text).into_iter().map(|t| t.text).collect();
    if tokens.is_empty() {
        return Err(JsValue::from_str("No tokens in query"));
    }
    if tokens.len() == 1 {
        return Ok(Box::new(TermQuery::text(field, &tokens[0])));
    }
    let mut bq = BooleanQuery::new();
    for token in tokens {
        if must {
            bq = bq.must(TermQuery::text(field, &token));
        } else {
            bq = bq.should(TermQuery::text(field, &token));
        }
    }
    Ok(Box::new(bq))
}

/// Convert a JS query object into a core `Box<dyn Query>`.
pub(crate) fn convert_query(
    js: &JsQuery,
    schema: &Schema,
    tokenizers: &TokenizerRegistry,
) -> Result<Box<dyn Query>, JsValue> {
    if let Some(ref tq) = js.term {
        return tokenize_and_build(&tq.field, &tq.value, true, schema, tokenizers);
    }

    if let Some(ref mq) = js.match_ {
        return tokenize_and_build(&mq.field, &mq.text, false, schema, tokenizers);
    }

    if let Some(ref bq) = js.boolean {
        let mut query = BooleanQuery::new();
        for q in &bq.must {
            let inner = convert_query(q, schema, tokenizers)?;
            query.must.push(inner.into());
        }
        for q in &bq.should {
            let inner = convert_query(q, schema, tokenizers)?;
            query.should.push(inner.into());
        }
        for q in &bq.must_not {
            let inner = convert_query(q, schema, tokenizers)?;
            query.must_not.push(inner.into());
        }
        return Ok(Box::new(query));
    }

    if let Some(ref pq) = js.prefix {
        let field = schema
            .get_field(&pq.field)
            .ok_or_else(|| JsValue::from_str(&format!("Unknown field: '{}'", pq.field)))?;
        return Ok(Box::new(PrefixQuery::text(field, &pq.value)));
    }

    if let Some(ref sq) = js.sparse_vector {
        let field = schema
            .get_field(&sq.field)
            .ok_or_else(|| JsValue::from_str(&format!("Unknown field: '{}'", sq.field)))?;
        if sq.indices.len() != sq.values.len() {
            return Err(JsValue::from_str(
                "sparseVector: indices and values must have the same length",
            ));
        }
        let vector: Vec<(u32, f32)> = sq
            .indices
            .iter()
            .zip(sq.values.iter())
            .map(|(&i, &v)| (i, v))
            .collect();
        let mut query = SparseVectorQuery::new(field, vector);
        if let Some(hf) = sq.heap_factor {
            query = query.with_heap_factor(hf);
        }
        return Ok(Box::new(query));
    }

    if let Some(ref dq) = js.dense_vector {
        let field = schema
            .get_field(&dq.field)
            .ok_or_else(|| JsValue::from_str(&format!("Unknown field: '{}'", dq.field)))?;
        let mut query = DenseVectorQuery::new(field, dq.vector.clone());
        if let Some(np) = dq.nprobe {
            query = query.with_nprobe(np);
        }
        if let Some(rf) = dq.rerank_factor {
            query = query.with_rerank_factor(rf);
        }
        return Ok(Box::new(query));
    }

    if js.fusion.is_some() {
        return Err(JsValue::from_str(
            "fusion is only supported at the top level of a search request",
        ));
    }

    Err(JsValue::from_str(
        "Invalid query: must contain exactly one of: term, match, boolean, prefix, sparseVector, denseVector, fusion",
    ))
}
