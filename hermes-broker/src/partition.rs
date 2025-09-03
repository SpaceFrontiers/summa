//! Partitioned indexes: one logical index spread over several shards
//! (`--placement "documents_2026*=2,3,4"`, see `docs/broker.md`, phase 2).
//!
//! Pure helpers the services use: the write router (documents go to the
//! partition their primary key hashes to), the read merge (per-shard
//! responses folded into one), and the primary-key field lookup from a
//! schema. Nothing here talks to the network.

use std::collections::BTreeMap;

use tonic::Status;

use crate::proto::hermes::{
    BatchIndexDocumentsResponse, BooleanQuery, DocumentError, FieldValue, GetIndexInfoResponse,
    IndexingBufferStats, MemoryStats, NamedDocument, Query, SearchHit, SearchResponse,
    SearchTimings, SegmentReaderStats, TermDocFreq, TextFieldStats, TextStats, VectorFieldStats,
    field_value, query,
};

/// FNV-1a 64 of the primary key bytes: the pinned partition hash. Changing
/// it moves every document, so it is part of the on-disk contract.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

/// Partition (0-based) of a primary key among `partitions`.
pub fn partition_of(primary_key: &[u8], partitions: usize) -> usize {
    debug_assert!(partitions > 0);
    (fnv1a64(primary_key) % partitions as u64) as usize
}

/// Name of the field declared `primary` in a schema SDL, e.g.
/// `field id: text<raw_ci> [indexed, stored, fast, primary]`.
pub fn primary_key_field(schema_sdl: &str) -> Option<String> {
    for line in schema_sdl.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("field ") else {
            continue;
        };
        let Some((name, decl)) = rest.split_once(':') else {
            continue;
        };
        let Some(open) = decl.find('[') else {
            continue;
        };
        let attrs = decl[open + 1..].trim_end_matches(']');
        if attrs.split(',').any(|attr| {
            attr.trim()
                .split('<')
                .next()
                .is_some_and(|a| a.trim() == "primary")
        }) {
            return Some(name.trim().to_string());
        }
    }
    None
}

/// Bytes hashed for a primary key value (numbers as their decimal text, so
/// the same key written as text or number lands on the same partition).
fn key_bytes(value: &FieldValue) -> Option<Vec<u8>> {
    match value.value.as_ref()? {
        field_value::Value::Text(text) => Some(text.as_bytes().to_vec()),
        field_value::Value::U64(n) => Some(n.to_string().into_bytes()),
        field_value::Value::I64(n) => Some(n.to_string().into_bytes()),
        field_value::Value::F64(n) => Some(n.to_string().into_bytes()),
        field_value::Value::BytesValue(bytes) => Some(bytes.clone()),
        field_value::Value::JsonValue(text) => Some(text.as_bytes().to_vec()),
        _ => None,
    }
}

/// Documents grouped by partition, remembering each document's position in
/// the request so per-partition error indexes map back to it.
pub struct RoutedBatch {
    /// Per partition: (original positions, documents).
    pub groups: Vec<(Vec<usize>, Vec<NamedDocument>)>,
    /// Documents that could not be routed, as errors at their positions.
    pub unroutable: Vec<DocumentError>,
}

/// Split a batch by the primary key hash.
pub fn route_documents(
    documents: Vec<NamedDocument>,
    primary_key: &str,
    partitions: usize,
) -> RoutedBatch {
    let mut groups: Vec<(Vec<usize>, Vec<NamedDocument>)> =
        (0..partitions).map(|_| (Vec::new(), Vec::new())).collect();
    let mut unroutable = Vec::new();
    for (position, document) in documents.into_iter().enumerate() {
        let key = document
            .fields
            .iter()
            .find(|entry| entry.name == primary_key)
            .and_then(|entry| entry.value.as_ref())
            .and_then(key_bytes);
        match key {
            Some(key) if !key.is_empty() => {
                let partition = partition_of(&key, partitions);
                groups[partition].0.push(position);
                groups[partition].1.push(document);
            }
            _ => unroutable.push(DocumentError {
                index: position as u32,
                error: format!(
                    "document has no '{primary_key}' primary key value; a partitioned index cannot route it"
                ),
            }),
        }
    }
    RoutedBatch { groups, unroutable }
}

/// Collect text-bearing leaves into a query `GetTextStats` can convert.
///
/// Fusion is a searcher-level operation and `hermes-server` deliberately
/// rejects it in the ordinary query converter used by `GetTextStats`. A
/// partitioned hybrid search still needs corpus-wide BM25 statistics, so the
/// broker sends only its term/match/phrase leaves for statistics while the
/// original fusion remains unchanged for the actual search.
pub fn text_stats_query(query: &Query) -> Option<Query> {
    fn collect(query: &Query, leaves: &mut Vec<Query>) {
        match &query.query {
            Some(query::Query::Term(_))
            | Some(query::Query::Match(_))
            | Some(query::Query::Phrase(_)) => leaves.push(query.clone()),
            Some(query::Query::Boolean(boolean)) => {
                for clause in boolean
                    .must
                    .iter()
                    .chain(&boolean.should)
                    .chain(&boolean.must_not)
                {
                    collect(clause, leaves);
                }
            }
            Some(query::Query::Boost(boost)) => {
                if let Some(inner) = boost.query.as_deref() {
                    collect(inner, leaves);
                }
            }
            Some(query::Query::Fusion(fusion)) => {
                for filter in &fusion.filters {
                    collect(filter, leaves);
                }
                for weighted in &fusion.queries {
                    if let Some(inner) = weighted.query.as_ref() {
                        collect(inner, leaves);
                    }
                }
            }
            _ => {}
        }
    }

    let mut leaves = Vec::new();
    collect(query, &mut leaves);
    match leaves.len() {
        0 => None,
        1 => leaves.pop(),
        _ => Some(Query {
            query: Some(query::Query::Boolean(BooleanQuery {
                must: vec![],
                should: leaves,
                must_not: vec![],
            })),
        }),
    }
}

/// Fold per-partition batch responses (with their original positions) into
/// one, error indexes mapped back to request positions.
pub fn merge_batch_responses(
    responses: Vec<(Vec<usize>, BatchIndexDocumentsResponse)>,
    unroutable: Vec<DocumentError>,
) -> BatchIndexDocumentsResponse {
    let mut merged = BatchIndexDocumentsResponse {
        indexed_count: 0,
        error_count: unroutable.len() as u32,
        errors: unroutable,
    };
    for (positions, response) in responses {
        merged.indexed_count = merged.indexed_count.saturating_add(response.indexed_count);
        merged.error_count = merged.error_count.saturating_add(response.error_count);
        for error in response.errors {
            let index = positions
                .get(error.index as usize)
                .map(|p| *p as u32)
                .unwrap_or(error.index);
            merged.errors.push(DocumentError {
                index,
                error: error.error,
            });
        }
    }
    merged.errors.sort_by_key(|e| e.index);
    merged
}

/// Merge per-partition search responses for `offset`/`limit`: hits by score
/// descending (ties by address), `total_hits` summed, timings at their
/// maximum, `truncated` if any partition truncated. Every partition scored
/// with the same global statistics, so scores are comparable; rank-fused
/// is coordinated separately in `ranking`; this merger only orders supplied scores.
pub fn merge_search_responses(
    responses: Vec<SearchResponse>,
    offset: usize,
    limit: usize,
) -> Result<SearchResponse, Status> {
    let mut hits: Vec<SearchHit> = Vec::new();
    let mut total_hits: u64 = 0;
    let mut took_ms: u64 = 0;
    let mut timings: Option<SearchTimings> = None;
    let mut truncated = false;
    let mut ranking_method: Option<String> = None;
    let mut fusion_candidates = BTreeMap::<u32, Vec<crate::proto::hermes::FusionCandidate>>::new();
    for response in responses {
        if ranking_method
            .as_ref()
            .is_some_and(|method| method != &response.ranking_method)
        {
            return Err(Status::failed_precondition(
                "shards returned incompatible ranking methods; complete the Hermes rollout",
            ));
        }
        ranking_method = Some(response.ranking_method);
        total_hits = total_hits.saturating_add(response.total_hits);
        took_ms = took_ms.max(response.took_ms);
        truncated |= response.truncated;
        if let Some(t) = response.timings {
            let merged = timings.get_or_insert_with(SearchTimings::default);
            merged.search_us = merged.search_us.max(t.search_us);
            merged.rerank_us = merged.rerank_us.max(t.rerank_us);
            merged.load_us = merged.load_us.max(t.load_us);
            merged.total_us = merged.total_us.max(t.total_us);
            merged.candidate_scoring_us = merged.candidate_scoring_us.max(t.candidate_scoring_us);
        }
        hits.extend(response.hits);
        for branch in response.fusion_candidates {
            fusion_candidates
                .entry(branch.query_index)
                .or_default()
                .extend(branch.candidates);
        }
    }
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| address_key(a).cmp(&address_key(b)))
    });
    if matches!(
        ranking_method.as_deref(),
        Some("feature_export_v1" | "fusion_candidates_v1")
    ) && (offset != 0 || hits.len() > limit)
    {
        return Err(Status::invalid_argument(format!(
            "feature export needs offset=0 and limit covering the complete cross-shard union ({} candidates)",
            hits.len()
        )));
    }
    let hits: Vec<SearchHit> = hits.into_iter().skip(offset).take(limit).collect();
    Ok(SearchResponse {
        hits,
        total_hits,
        took_ms,
        timings,
        truncated,
        ranking_method: ranking_method.unwrap_or_default(),
        fusion_candidates: fusion_candidates
            .into_iter()
            .map(
                |(query_index, candidates)| crate::proto::hermes::FusionCandidateList {
                    query_index,
                    candidates,
                },
            )
            .collect(),
    })
}

fn address_key(hit: &SearchHit) -> (String, u32) {
    hit.address
        .as_ref()
        .map(|a| (a.segment_id.clone(), a.doc_id))
        .unwrap_or_default()
}

/// Sum per-partition text statistics: document and corpus counts add up,
/// average lengths are weighted by corpus size, term frequencies add up.
pub fn merge_text_stats(parts: Vec<TextStats>) -> TextStats {
    let mut total_docs: u64 = 0;
    // field → (num_docs, sum of field lengths, term → doc freq)
    type FieldAcc = (u64, f64, BTreeMap<Vec<u8>, u64>);
    let mut fields: BTreeMap<String, FieldAcc> = BTreeMap::new();
    for part in parts {
        total_docs = total_docs.saturating_add(part.total_docs);
        for field in part.fields {
            let entry = fields.entry(field.field).or_default();
            entry.0 = entry.0.saturating_add(field.corpus_size);
            entry.1 += f64::from(field.avg_len) * field.corpus_size as f64;
            for term in field.terms {
                let df = entry.2.entry(term.term).or_insert(0);
                *df = df.saturating_add(term.doc_freq);
            }
        }
    }
    TextStats {
        total_docs,
        fields: fields
            .into_iter()
            .map(
                |(field, (corpus_size, weighted_len, terms))| TextFieldStats {
                    field,
                    corpus_size,
                    avg_len: if corpus_size > 0 {
                        (weighted_len / corpus_size as f64) as f32
                    } else {
                        0.0
                    },
                    terms: terms
                        .into_iter()
                        .map(|(term, doc_freq)| TermDocFreq { term, doc_freq })
                        .collect(),
                },
            )
            .collect(),
    }
}

/// Aggregate per-partition index info: counts and byte sizes add up, the
/// schema and text-field facts come from the first partition (every
/// partition was created from the same schema).
pub fn merge_index_info(parts: Vec<GetIndexInfoResponse>) -> GetIndexInfoResponse {
    let mut iter = parts.into_iter();
    let Some(mut merged) = iter.next() else {
        return GetIndexInfoResponse::default();
    };
    let mut vectors: BTreeMap<String, VectorFieldStats> = merged
        .vector_stats
        .drain(..)
        .map(|v| (v.field_name.clone(), v))
        .collect();
    for part in iter {
        merged.candidate_scoring_version = merged
            .candidate_scoring_version
            .min(part.candidate_scoring_version);
        merged
            .unprepared_candidate_fields
            .extend(part.unprepared_candidate_fields);
        merged.num_docs = merged.num_docs.saturating_add(part.num_docs);
        merged.num_segments = merged.num_segments.saturating_add(part.num_segments);
        merged.memory_stats = match (merged.memory_stats.take(), part.memory_stats) {
            (Some(a), Some(b)) => Some(add_memory_stats(a, b)),
            (a, b) => a.or(b),
        };
        for v in part.vector_stats {
            match vectors.get_mut(&v.field_name) {
                Some(existing) => {
                    let total = existing.total_vectors.saturating_add(v.total_vectors);
                    if total > 0 {
                        existing.avg_terms_per_vector = ((f64::from(existing.avg_terms_per_vector)
                            * existing.total_vectors as f64
                            + f64::from(v.avg_terms_per_vector) * v.total_vectors as f64)
                            / total as f64)
                            as f32;
                    }
                    existing.total_vectors = total;
                }
                None => {
                    vectors.insert(v.field_name.clone(), v);
                }
            }
        }
    }
    merged.unprepared_candidate_fields.sort();
    merged.unprepared_candidate_fields.dedup();
    merged.vector_stats = vectors.into_values().collect();
    merged
}

fn add_memory_stats(a: MemoryStats, b: MemoryStats) -> MemoryStats {
    MemoryStats {
        total_bytes: a.total_bytes.saturating_add(b.total_bytes),
        indexing_buffer: match (a.indexing_buffer, b.indexing_buffer) {
            (Some(x), Some(y)) => Some(IndexingBufferStats {
                total_bytes: x.total_bytes.saturating_add(y.total_bytes),
                postings_bytes: x.postings_bytes.saturating_add(y.postings_bytes),
                sparse_vectors_bytes: x
                    .sparse_vectors_bytes
                    .saturating_add(y.sparse_vectors_bytes),
                dense_vectors_bytes: x.dense_vectors_bytes.saturating_add(y.dense_vectors_bytes),
                interner_bytes: x.interner_bytes.saturating_add(y.interner_bytes),
                position_index_bytes: x
                    .position_index_bytes
                    .saturating_add(y.position_index_bytes),
                pending_docs: x.pending_docs.saturating_add(y.pending_docs),
                unique_terms: x.unique_terms.saturating_add(y.unique_terms),
            }),
            (x, y) => x.or(y),
        },
        segment_reader: match (a.segment_reader, b.segment_reader) {
            (Some(x), Some(y)) => Some(SegmentReaderStats {
                total_bytes: x.total_bytes.saturating_add(y.total_bytes),
                term_dict_cache_bytes: x
                    .term_dict_cache_bytes
                    .saturating_add(y.term_dict_cache_bytes),
                store_cache_bytes: x.store_cache_bytes.saturating_add(y.store_cache_bytes),
                sparse_index_bytes: x.sparse_index_bytes.saturating_add(y.sparse_index_bytes),
                dense_index_bytes: x.dense_index_bytes.saturating_add(y.dense_index_bytes),
                num_segments_loaded: x.num_segments_loaded.saturating_add(y.num_segments_loaded),
            }),
            (x, y) => x.or(y),
        },
    }
}

/// A partition's failure fails the whole request: a silently partial
/// answer over a partitioned index is a wrong answer.
pub fn partition_failure(index_name: &str, shard: &str, status: Status) -> Status {
    Status::new(
        status.code(),
        format!(
            "index '{index_name}' partition on shard '{shard}': {}",
            status.message()
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::hermes::{
        DocAddress, FieldEntry, FusionQuery, MatchQuery, PhraseQuery, SparseVectorQuery,
        WeightedQuery,
    };

    fn doc(id: &str) -> NamedDocument {
        NamedDocument {
            fields: vec![FieldEntry {
                name: "id".to_string(),
                value: Some(FieldValue {
                    value: Some(field_value::Value::Text(id.to_string())),
                }),
            }],
        }
    }

    #[test]
    fn hash_is_pinned_and_keys_route_deterministically() {
        // FNV-1a 64 reference values.
        assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a64(b"foobar"), 0x85944171f73967e8);
        let a = partition_of(b"doc-1", 3);
        assert_eq!(partition_of(b"doc-1", 3), a);
        // Numbers route like their decimal text.
        let text = key_bytes(&FieldValue {
            value: Some(field_value::Value::Text("42".to_string())),
        });
        let number = key_bytes(&FieldValue {
            value: Some(field_value::Value::U64(42)),
        });
        assert_eq!(text, number);
    }

    #[test]
    fn primary_key_comes_from_the_schema() {
        let sdl = "index documents {\n    field id: text<raw_ci> [indexed, stored, fast, primary]\n    field uris: text<raw_ci> [indexed, stored<multi>]\n}";
        assert_eq!(primary_key_field(sdl).as_deref(), Some("id"));
        assert_eq!(
            primary_key_field("index x {\n field a: text<raw> [indexed]\n}"),
            None
        );
    }

    #[test]
    fn fusion_stats_query_keeps_only_text_leaves() {
        let text_query = |kind| Query { query: Some(kind) };
        let query = Query {
            query: Some(query::Query::Fusion(FusionQuery {
                queries: vec![
                    WeightedQuery {
                        query: Some(text_query(query::Query::SparseVector(SparseVectorQuery {
                            field: "sparse_vectors".to_string(),
                            ..Default::default()
                        }))),
                        weight: 1.0,

                        ..Default::default()
                    },
                    WeightedQuery {
                        query: Some(Query {
                            query: Some(query::Query::Boolean(BooleanQuery {
                                must: vec![text_query(query::Query::Match(MatchQuery {
                                    field: "content".to_string(),
                                    text: "quantum".to_string(),
                                    ..Default::default()
                                }))],
                                should: vec![text_query(query::Query::Phrase(PhraseQuery {
                                    field: "content".to_string(),
                                    text: "quantum field".to_string(),
                                    ..Default::default()
                                }))],
                                must_not: vec![],
                            })),
                        }),
                        weight: 0.8,

                        ..Default::default()
                    },
                ],
                ..Default::default()
            })),
        };

        let stats = text_stats_query(&query).expect("fusion has text leaves");
        let Some(query::Query::Boolean(boolean)) = stats.query else {
            panic!("multiple text leaves should become a boolean query");
        };
        assert_eq!(boolean.should.len(), 2);
        assert!(matches!(
            boolean.should[0].query,
            Some(query::Query::Match(_))
        ));
        assert!(matches!(
            boolean.should[1].query,
            Some(query::Query::Phrase(_))
        ));

        let vector_only = text_query(query::Query::SparseVector(SparseVectorQuery {
            field: "sparse_vectors".to_string(),
            ..Default::default()
        }));
        assert!(text_stats_query(&vector_only).is_none());
    }

    #[test]
    fn documents_are_routed_by_key_and_errors_map_back() {
        let docs: Vec<NamedDocument> = (0..30).map(|i| doc(&format!("doc-{i}"))).collect();
        let mut unkeyed = NamedDocument::default();
        unkeyed.fields.push(FieldEntry {
            name: "title".to_string(),
            value: None,
        });
        let mut all = docs;
        all.push(unkeyed);
        let routed = route_documents(all, "id", 3);
        assert_eq!(routed.groups.len(), 3);
        let routed_count: usize = routed.groups.iter().map(|(_, d)| d.len()).sum();
        assert_eq!(routed_count, 30);
        assert!(
            routed.groups.iter().all(|(_, d)| !d.is_empty()),
            "every partition gets some"
        );
        assert_eq!(routed.unroutable.len(), 1);
        assert_eq!(routed.unroutable[0].index, 30);
        // The same key always lands on the same partition.
        let again = route_documents(vec![doc("doc-7")], "id", 3);
        let first = routed
            .groups
            .iter()
            .position(|(positions, _)| positions.contains(&7))
            .unwrap();
        assert!(!again.groups[first].1.is_empty());

        // A partition error at its local position maps to the request position.
        let (positions, _) = &routed.groups[first];
        let local = positions.iter().position(|p| *p == 7).unwrap();
        let merged = merge_batch_responses(
            vec![(
                positions.clone(),
                BatchIndexDocumentsResponse {
                    indexed_count: positions.len() as u32 - 1,
                    error_count: 1,
                    errors: vec![DocumentError {
                        index: local as u32,
                        error: "boom".to_string(),
                    }],
                },
            )],
            routed.unroutable,
        );
        assert_eq!(merged.error_count, 2);
        assert_eq!(merged.errors[0].index, 7);
        assert_eq!(merged.errors[1].index, 30);
    }

    fn hit(segment: &str, doc_id: u32, score: f32) -> SearchHit {
        SearchHit {
            address: Some(DocAddress {
                segment_id: segment.to_string(),
                doc_id,
            }),
            score,
            ..Default::default()
        }
    }

    #[test]
    fn search_responses_merge_by_score_with_offset_and_limit() {
        let a = SearchResponse {
            hits: vec![hit("a", 1, 0.9), hit("a", 2, 0.5), hit("a", 3, 0.1)],
            total_hits: 3,
            took_ms: 5,
            timings: Some(SearchTimings {
                search_us: 100,
                ..Default::default()
            }),
            truncated: false,

            ..Default::default()
        };
        let b = SearchResponse {
            hits: vec![hit("b", 1, 0.7), hit("b", 2, 0.5)],
            total_hits: 2,
            took_ms: 9,
            timings: Some(SearchTimings {
                search_us: 300,
                ..Default::default()
            }),
            truncated: true,

            ..Default::default()
        };
        let merged = merge_search_responses(vec![a, b], 1, 3).unwrap();
        let order: Vec<(String, u32)> = merged.hits.iter().map(address_key).collect();
        // 0.9(a1) skipped by offset; then 0.7(b1), 0.5(a2 before b2 by address), 0.5(b2)
        assert_eq!(
            order,
            vec![
                ("b".to_string(), 1),
                ("a".to_string(), 2),
                ("b".to_string(), 2)
            ]
        );
        assert_eq!(merged.total_hits, 5);
        assert_eq!(merged.took_ms, 9);
        assert_eq!(merged.timings.unwrap().search_us, 300);
        assert!(merged.truncated);
    }

    #[test]
    fn linear_shard_scores_and_features_merge_directly_and_reject_mixed_versions() {
        let mut a = hit("a", 1, -2.0);
        a.candidate_scores = Some(crate::proto::hermes::CandidateScores {
            document: std::collections::HashMap::from([("dense".into(), -2.0)]),
            ..Default::default()
        });
        let first = SearchResponse {
            ranking_method: "linear_v1".into(),
            hits: vec![a],
            ..Default::default()
        };
        let second = SearchResponse {
            ranking_method: "linear_v1".into(),
            hits: vec![hit("b", 1, -1.0)],
            ..Default::default()
        };
        let merged = merge_search_responses(vec![first.clone(), second], 0, 2).unwrap();
        assert_eq!(merged.ranking_method, "linear_v1");
        assert_eq!(merged.hits[0].score, -1.0);
        assert_eq!(
            merged.hits[1].candidate_scores.as_ref().unwrap().document["dense"],
            -2.0
        );
        assert_eq!(
            merge_search_responses(vec![first, SearchResponse::default()], 0, 10)
                .unwrap_err()
                .code(),
            tonic::Code::FailedPrecondition
        );
    }

    #[test]
    fn feature_export_never_discards_the_cross_shard_union() {
        let response = |segment| SearchResponse {
            ranking_method: "feature_export_v1".into(),
            hits: vec![hit(segment, 1, 0.0)],
            ..Default::default()
        };
        assert!(merge_search_responses(vec![response("a"), response("b")], 0, 1).is_err());
        assert!(merge_search_responses(vec![response("a")], 1, 10).is_err());
        assert_eq!(
            merge_search_responses(vec![response("a"), response("b")], 0, 2)
                .unwrap()
                .hits
                .len(),
            2
        );
    }

    #[test]
    fn text_stats_and_index_info_add_up() {
        let stats = |docs: u64, corpus: u64, avg: f32, df: u64| TextStats {
            total_docs: docs,
            fields: vec![TextFieldStats {
                field: "content".to_string(),
                corpus_size: corpus,
                avg_len: avg,
                terms: vec![TermDocFreq {
                    term: b"cell".to_vec(),
                    doc_freq: df,
                }],
            }],
        };
        let merged = merge_text_stats(vec![stats(10, 100, 10.0, 3), stats(30, 300, 20.0, 5)]);
        assert_eq!(merged.total_docs, 40);
        assert_eq!(merged.fields[0].corpus_size, 400);
        assert!((merged.fields[0].avg_len - 17.5).abs() < 1e-5);
        assert_eq!(merged.fields[0].terms[0].doc_freq, 8);

        let info = |docs: u32, segments: u32| GetIndexInfoResponse {
            index_name: "documents".to_string(),
            num_docs: docs,
            num_segments: segments,
            schema: "index documents {}".to_string(),
            memory_stats: Some(MemoryStats {
                total_bytes: 10,
                indexing_buffer: None,
                segment_reader: None,
            }),
            vector_stats: vec![VectorFieldStats {
                field_name: "v".to_string(),
                vector_type: "sparse".to_string(),
                total_vectors: docs as u64,
                dimension: 0,
                avg_terms_per_vector: 2.0,
            }],
            text_fields: vec![],

            ..Default::default()
        };
        let merged = merge_index_info(vec![info(5, 2), info(7, 3)]);
        assert_eq!(merged.num_docs, 12);
        assert_eq!(merged.num_segments, 5);
        assert_eq!(merged.memory_stats.unwrap().total_bytes, 20);
        assert_eq!(merged.vector_stats[0].total_vectors, 12);
    }
}
