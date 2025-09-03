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

mod diagnostics;

use hermes_core::query::{
    BooleanQuery, DenseVectorQuery, PhraseQuery, PrefixQuery, Query, SparseVectorQuery, TermQuery,
};
use hermes_core::tokenizer::{Purpose, TokenizerRegistry};
use hermes_core::{Directory, Schema, Searcher};
use serde::{Deserialize, Serialize};
use wasm_bindgen::JsValue;

/// Top-level query object deserialized from JS.
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct JsQuery {
    #[serde(default)]
    term: Option<JsTermQuery>,
    #[serde(default, rename = "match")]
    match_: Option<JsMatchQuery>,
    #[serde(default)]
    phrase: Option<JsPhraseQuery>,
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

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct JsTermQuery {
    field: String,
    value: String,
    #[serde(default)]
    tokenizer_hint: Option<String>,
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct JsMatchQuery {
    field: String,
    text: String,
    #[serde(default)]
    tokenizer_hint: Option<String>,
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct JsPhraseQuery {
    field: String,
    text: String,
    #[serde(default)]
    slop: u32,
    #[serde(default)]
    tokenizer_hint: Option<String>,
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct JsBooleanQuery {
    #[serde(default)]
    must: Vec<JsQuery>,
    #[serde(default)]
    should: Vec<JsQuery>,
    #[serde(default)]
    must_not: Vec<JsQuery>,
}

#[derive(Deserialize, Serialize)]
pub(crate) struct JsPrefixQuery {
    field: String,
    value: String,
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct JsSparseVectorQuery {
    field: String,
    indices: Vec<u32>,
    values: Vec<f32>,
    #[serde(default)]
    heap_factor: Option<f32>,
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct JsDenseVectorQuery {
    field: String,
    vector: Vec<f32>,
    #[serde(default)]
    nprobe: Option<usize>,
    #[serde(default)]
    rerank_factor: Option<f32>,
}

#[derive(Deserialize, Serialize)]
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

#[derive(Deserialize, Serialize)]
pub(crate) struct JsWeightedQuery {
    #[serde(default)]
    name: String,
    query: JsQuery,
    #[serde(default)]
    weight: Option<f32>,
}

/// Search request with optional parameters.
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct JsSearchRequest {
    pub query: JsQuery,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default)]
    pub offset: usize,
    #[serde(default)]
    pub fields_to_load: Option<Vec<String>>,
    #[serde(default)]
    pub include_rrf_scores: bool,
    #[serde(default)]
    pub tracing: bool,
}

fn default_limit() -> usize {
    10
}

/// Typed response structs (avoid serde_json::json! intermediate allocations).
#[derive(Serialize)]
pub(crate) struct StructuredSearchResponse<'a> {
    hits: Vec<StructuredHit<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    trace: Option<diagnostics::SearchTrace<'a>>,
    total_hits: usize,
}

#[derive(Serialize)]
pub(crate) struct StructuredHit<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    rrf_score: Option<f32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    rrf_contributions: Vec<diagnostics::RrfContribution<'a>>,
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

    let diagnostics = req.tracing || req.include_rrf_scores;
    let window = req.limit.saturating_add(req.offset);
    if diagnostics && window > 10_000 {
        return Err(JsValue::from_str("diagnostic result window exceeds 10000"));
    }
    if req.include_rrf_scores && req.query.fusion.is_none() {
        return Err(JsValue::from_str("includeRrfScores requires fusion"));
    }
    let mut nomination_lists = Vec::new();
    let mut trace = None;
    let mut budget = diagnostics::Budget::default();
    let field_ids = req
        .fields_to_load
        .as_ref()
        .map(|names| crate::resolve_field_ids(searcher.schema(), names))
        .transpose()?;

    // Fusion: run each sub-query independently and fuse the ranked lists
    // (union). Handled here because fusion is a searcher-level operation.
    let results = if let Some(ref fusion) = req.query.fusion {
        if fusion.queries.is_empty()
            || fusion.queries.len() > hermes_core::query::MAX_FUSION_SUB_QUERIES
        {
            return Err(JsValue::from_str("fusion requires 1..16 branches"));
        }
        if fusion.queries.iter().any(|branch| {
            !branch.weight.unwrap_or(1.0).is_finite()
                || branch.weight.unwrap_or(1.0) < 0.0
                || (diagnostics && branch.name.len() > 128)
        }) || fusion.rrf_k.is_some_and(|k| !k.is_finite() || k < 0.0)
        {
            return Err(JsValue::from_str(
                "invalid fusion weights, rank constant or branch name",
            ));
        }
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

        let fused_limit = req.limit.saturating_add(req.offset);
        let max_fetch_limit = hermes_core::query::max_candidate_limit(fused_limit);
        let fetch_limit = fusion.fetch_limit.unwrap_or(max_fetch_limit);
        if !(fused_limit..=max_fetch_limit).contains(&fetch_limit) {
            return Err(JsValue::from_str(&format!(
                "fusion fetchLimit must be between {fused_limit} and {max_fetch_limit}"
            )));
        }
        let refs: Vec<(&dyn Query, f32)> =
            sub_queries.iter().map(|(q, w)| (q.as_ref(), *w)).collect();
        let mut fused = if diagnostics {
            let queries: Vec<std::sync::Arc<dyn Query>> =
                sub_queries.into_iter().map(|(q, _)| q.into()).collect();
            nomination_lists = searcher
                .search_candidate_lists(&queries, fetch_limit, None)
                .await
                .map_err(|e| JsValue::from_str(&format!("Search error: {e}")))?;
            let lists: Vec<_> = nomination_lists
                .iter()
                .zip(&fusion.queries)
                .map(|((hits, _), branch)| (hits.as_slice(), branch.weight.unwrap_or(1.0)))
                .collect();
            if req.tracing {
                trace = Some(diagnostics::fusion_trace(
                    searcher.schema().index_label(),
                    fusion,
                    &nomination_lists,
                    fetch_limit,
                    &mut budget,
                )?);
            }
            searcher
                .fuse_candidate_lists(
                    &lists,
                    method,
                    hermes_core::query::MultiValueCombiner::Max,
                    fused_limit,
                )
                .map_err(|e| JsValue::from_str(&format!("Search error: {e}")))?
        } else {
            searcher
                .search_fused(
                    &refs,
                    fetch_limit,
                    fused_limit,
                    method,
                    hermes_core::query::MultiValueCombiner::Max,
                )
                .await
                .map_err(|e| JsValue::from_str(&format!("Search error: {e}")))?
        };
        if req.offset > 0 {
            fused.drain(..req.offset.min(fused.len()));
        }
        fused
    } else {
        let query = convert_query(&req.query, searcher.schema(), searcher.tokenizers())?;
        if req.tracing {
            let (mut results, seen) = searcher
                .search_with_count(query.as_ref(), window)
                .await
                .map_err(|e| JsValue::from_str(&format!("Search error: {e}")))?;
            trace = Some(diagnostics::root_trace(
                searcher.schema().index_label(),
                &req.query,
                &results,
                seen,
                window,
                &mut budget,
            )?);
            results.drain(..req.offset.min(results.len()));
            results
        } else {
            searcher
                .search_with_offset_and_count(query.as_ref(), req.limit, req.offset)
                .await
                .map_err(|e| JsValue::from_str(&format!("Search error: {e}")))?
                .0
        }
    };

    if let Some(trace) = &mut trace {
        trace.shards[0].selected = budget.candidates(&results)?;
    }
    let rrf = if req.include_rrf_scores {
        let fusion = req.query.fusion.as_ref().expect("validated fusion");
        let lists: Vec<_> = nomination_lists
            .iter()
            .zip(&fusion.queries)
            .enumerate()
            .map(
                |(query_index, ((hits, _), branch))| hermes_core::query::RrfRankedList {
                    query_index,
                    scope: None,
                    weight: branch.weight.unwrap_or(1.0),
                    hits,
                },
            )
            .collect();
        let selected: Vec<_> = results
            .iter()
            .map(|hit| (hit.segment_id, hit.doc_id))
            .collect();
        Some(
            searcher
                .rrf_scores_for_hits(
                    &lists,
                    &selected,
                    fusion.rrf_k.unwrap_or(hermes_core::query::DEFAULT_RRF_K),
                    hermes_core::query::MultiValueCombiner::Max,
                )
                .map_err(|e| JsValue::from_str(&e.to_string()))?,
        )
    } else {
        None
    };
    let mut scores = rrf.unwrap_or_default().into_iter();
    let mut hits = Vec::with_capacity(results.len());
    for result in &results {
        let address = hermes_core::query::DocAddress::new(result.segment_id, result.doc_id);
        let doc = searcher
            .get_document_with_fields(&address, field_ids.as_ref())
            .await
            .map_err(|e| JsValue::from_str(&format!("Get document error: {}", e)))?;
        let score = scores.next();
        let rrf_score = score.as_ref().map(|score| score.score);
        let rrf_contributions = score.map_or_else(Vec::new, |score| {
            score
                .contributions
                .into_iter()
                .map(|vote| diagnostics::RrfContribution {
                    query_name: &req.query.fusion.as_ref().unwrap().queries[vote.query_index].name,
                    vote,
                })
                .collect()
        });
        hits.push(StructuredHit {
            rrf_score,
            rrf_contributions,
            address: HitAddress {
                segment_id: format!("{:032x}", result.segment_id),
                doc_id: result.doc_id,
            },
            score: result.score,
            doc: doc.map(|d| d.to_json(searcher.schema())),
        });
    }

    let response = StructuredSearchResponse {
        trace,
        hits,
        total_hits: results.len(),
    };

    if diagnostics {
        diagnostics::check_encoded_size(&response)?;
    }
    response
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

/// Resolve a text field and tokenize `text` with its configured tokenizer,
/// passing the optional hint through (dynamic stemmers read it as a language
/// list; static tokenizers ignore it).
fn field_tokens(
    field_name: &str,
    text: &str,
    tokenizer_hint: Option<&str>,
    purpose: Purpose,
    schema: &Schema,
    tokenizers: &TokenizerRegistry,
) -> Result<(hermes_core::Field, Vec<(u32, String)>), JsValue> {
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
    let hint = tokenizer_hint.map(str::trim).filter(|h| !h.is_empty());
    let tokens: Vec<(u32, String)> = tok
        .tokenize_with(text, hint, purpose)
        .into_iter()
        .map(|t| (t.position, t.text))
        .collect();
    if tokens.is_empty() {
        return Err(JsValue::from_str("No tokens in query"));
    }
    Ok((field, tokens))
}

/// Tokenize text using the field's configured tokenizer and build a query.
/// Term queries use MUST (AND), match queries use SHOULD (OR).
fn tokenize_and_build(
    field_name: &str,
    text: &str,
    tokenizer_hint: Option<&str>,
    must: bool,
    schema: &Schema,
    tokenizers: &TokenizerRegistry,
) -> Result<Box<dyn Query>, JsValue> {
    // A term query wants the written form; a match query the stem.
    let purpose = if must { Purpose::Exact } else { Purpose::Match };
    let (field, tokens) = field_tokens(
        field_name,
        text,
        tokenizer_hint,
        purpose,
        schema,
        tokenizers,
    )?;
    if tokens.len() == 1 {
        return Ok(Box::new(TermQuery::text(field, &tokens[0].1)));
    }
    let mut bq = BooleanQuery::new();
    for (_, token) in tokens {
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
        return tokenize_and_build(
            &tq.field,
            &tq.value,
            tq.tokenizer_hint.as_deref(),
            true,
            schema,
            tokenizers,
        );
    }

    if let Some(ref mq) = js.match_ {
        return tokenize_and_build(
            &mq.field,
            &mq.text,
            mq.tokenizer_hint.as_deref(),
            false,
            schema,
            tokenizers,
        );
    }

    if let Some(ref pq) = js.phrase {
        let (field, tokens) = field_tokens(
            &pq.field,
            &pq.text,
            pq.tokenizer_hint.as_deref(),
            Purpose::Exact,
            schema,
            tokenizers,
        )?;
        let terms = tokens
            .into_iter()
            .map(|(offset, token)| (offset, token.into_bytes()))
            .collect();
        return Ok(Box::new(
            PhraseQuery::with_offsets(field, terms).with_slop(pq.slop),
        ));
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
