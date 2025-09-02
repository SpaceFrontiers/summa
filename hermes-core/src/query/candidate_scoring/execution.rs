use super::*;
use crate::directories::Directory;
use crate::index::Searcher;
use crate::query::{GlobalStats, GlobalStatsBuilder, ScoredPosition, SearchResult};
use crate::segment::SegmentReader;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

const MAX_FEATURES: usize = crate::query::MAX_FUSION_SUB_QUERIES;
const MAX_FEATURE_VALUES: usize = 2_000_000;
const MAX_VECTOR_BYTES: usize = 1024 * 1024 * 1024;

impl CandidateScoringPlan {
    pub fn validate(&self, schema: &crate::Schema) -> Result<()> {
        if self.features.is_empty()
            || self.features.len() > MAX_FEATURES
            || self.export_passages == 0
            || self.export_passages > u16::MAX as usize + 1
        {
            return Err(Error::Query("candidate scoring needs 1..16 branches and 1..65536 exported passages per document".into()));
        }
        let mut names = BTreeSet::new();
        for feature in &self.features {
            if feature.name.is_empty()
                || feature.name.len() > 128
                || !feature
                    .name
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || b"._-".contains(&c))
                || !names.insert(feature.name.as_str())
            {
                return Err(Error::Query("candidate scoring requires unique query names (1..128 ASCII letters, digits, '.', '_' or '-')".into()));
            }
            let entry = schema
                .get_field_entry(feature.query.field)
                .ok_or_else(|| Error::FieldNotFound(feature.query.field.0.to_string()))?;
            if !entry.indexed {
                return Err(Error::Query(format!(
                    "L1 branch '{}' needs an indexed field",
                    feature.name
                )));
            }
            if feature.scope == ScoreScope::Chunk
                && entry.field_type == crate::FieldType::Text
                && !entry.chunked
            {
                return Err(Error::Query(format!(
                    "L1 chunk branch '{}' needs chunked text; plain text is document scope",
                    feature.name
                )));
            }
            if feature.query.components.is_empty()
                || feature.query.components.len() > crate::query::MAX_QUERY_TERMS
            {
                return Err(Error::Query(
                    "L1 branch has an invalid component count".into(),
                ));
            }
            for (component, boost) in &feature.query.components {
                if !boost.is_finite() {
                    return Err(Error::Query("non-finite L1 query boost".into()));
                }
                match component {
                    ScoreComponent::Text(terms) => {
                        if entry.field_type != crate::FieldType::Text
                            || terms.len() > crate::query::MAX_QUERY_TERMS
                            || terms.iter().any(|(_, w)| !w.is_finite())
                        {
                            return Err(Error::Query("invalid L1 text feature".into()));
                        }
                    }
                    ScoreComponent::Phrase(query) => {
                        if entry.field_type != crate::FieldType::Text
                            || query.field != feature.query.field
                            || query.terms.len() > crate::query::MAX_QUERY_TERMS
                            || query.offsets.len() != query.terms.len()
                            || !query.offsets.windows(2).all(|p| p[0] < p[1])
                            || (query.terms.len() > 1 && entry.positions.is_none())
                        {
                            return Err(Error::Query(
                                "invalid L1 phrase feature or missing positions".into(),
                            ));
                        }
                    }
                    ScoreComponent::Sparse(terms) => {
                        if entry.field_type != crate::FieldType::SparseVector
                            || terms.len() > crate::query::MAX_QUERY_TERMS
                            || terms.iter().any(|(_, w)| !w.is_finite())
                        {
                            return Err(Error::Query("invalid L1 sparse feature".into()));
                        }
                    }
                    ScoreComponent::Dense(vector) => {
                        let Some(config) = &entry.dense_vector_config else {
                            return Err(Error::Query(
                                "L1 dense feature requires a dense field".into(),
                            ));
                        };
                        if vector.len() != config.dim
                            || vector.is_empty()
                            || vector.iter().any(|v| !v.is_finite())
                        {
                            return Err(Error::Query(
                                "invalid L1 dense dimensions or values".into(),
                            ));
                        }
                    }
                    ScoreComponent::Binary(vector) => {
                        let Some(config) = &entry.binary_dense_vector_config else {
                            return Err(Error::Query(
                                "L1 binary feature requires a binary field".into(),
                            ));
                        };
                        if vector.is_empty() || vector.len() != config.byte_len() {
                            return Err(Error::Query("invalid L1 binary dimension".into()));
                        }
                    }
                }
            }
        }
        if let Some(model) = &self.model {
            model.validate(
                &self
                    .features
                    .iter()
                    .map(|f| f.name.as_str())
                    .collect::<Vec<_>>(),
            )?;
        }
        Ok(())
    }
}

async fn score_field<D: Directory + 'static>(
    searcher: &Searcher<D>,
    reader: &SegmentReader,
    query: &CandidateQuery,
    targets: &[u32],
    stats: &Arc<GlobalStats>,
) -> Result<Vec<f32>> {
    let mut result = vec![0.0; targets.len()];
    for (component, boost) in &query.components {
        let values = match component {
            ScoreComponent::Text(terms) => {
                crate::query::term::score_term_candidates(
                    reader,
                    query.field,
                    terms,
                    targets,
                    Some(stats),
                )
                .await?
            }
            ScoreComponent::Phrase(phrase) => {
                crate::query::phrase::score_phrase_candidates(reader, phrase, targets, Some(stats))
                    .await?
            }
            ScoreComponent::Sparse(terms) => {
                let index = reader
                    .bmp_index(query.field)
                    .ok_or_else(|| Error::Query("L1 sparse feature requires BMP storage".into()))?;
                searcher.install_search_cpu(|| {
                    crate::query::bmp::score_bmp_candidates(index, terms, targets)
                })?
            }
            ScoreComponent::Dense(vector) => {
                let flat = reader.flat_vectors().get(&query.field.0).ok_or_else(|| {
                    Error::Corruption("L1 dense locations lack stored vectors".into())
                })?;
                let unit_norm = reader
                    .schema()
                    .get_field_entry(query.field)
                    .and_then(|e| e.dense_vector_config.as_ref())
                    .is_some_and(|c| c.unit_norm);
                crate::query::reranker::score_vector_candidates(
                    searcher,
                    flat,
                    vector,
                    &[],
                    unit_norm,
                    targets,
                )
                .await?
            }
            ScoreComponent::Binary(vector) => {
                let flat = reader.flat_vectors().get(&query.field.0).ok_or_else(|| {
                    Error::Corruption("L1 binary locations lack stored vectors".into())
                })?;
                crate::query::reranker::score_vector_candidates(
                    searcher,
                    flat,
                    &[],
                    vector,
                    false,
                    targets,
                )
                .await?
            }
        };
        for (total, value) in result.iter_mut().zip(values) {
            *total += value * boost;
            if !total.is_finite() {
                return Err(Error::Query("L1 feature score overflow".into()));
            }
        }
    }
    Ok(result)
}

impl<D: Directory + 'static> Searcher<D> {
    /// Complete BM25 statistics over this immutable snapshot, including on
    /// native-async and WASM where the synchronous lazy cache is unavailable.
    pub async fn candidate_text_stats(
        &self,
        plan: &CandidateScoringPlan,
    ) -> Result<Arc<GlobalStats>> {
        let mut terms = Vec::new();
        for feature in &plan.features {
            feature.query.text_terms(&mut terms);
        }
        terms.sort_unstable_by(|a, b| (a.0.0, &a.1).cmp(&(b.0.0, &b.1)));
        terms.dedup();
        let mut builder = GlobalStatsBuilder::new();
        let mut fields = BTreeSet::new();
        for reader in self.segment_readers() {
            builder.add_segment(reader);
        }
        for (field, term) in terms {
            if fields.insert(field.0) {
                let mut corpus_size = 0u64;
                let mut total_length = 0.0f64;
                for reader in self.segment_readers() {
                    let size = reader.text_corpus_size(field) as u64;
                    corpus_size += size;
                    total_length += f64::from(reader.avg_field_len(field)) * size as f64;
                }
                builder.set_text_corpus_size(field, corpus_size);
                builder.set_avg_field_len(field, (total_length / corpus_size.max(1) as f64) as f32);
            }
            for reader in self.segment_readers() {
                let count = reader
                    .get_postings(field, &term)
                    .await?
                    .map_or(0, |p| p.doc_count());
                builder.add_text_df(
                    field,
                    String::from_utf8_lossy(&term).into_owned(),
                    u64::from(count),
                );
            }
        }
        Ok(Arc::new(builder.build(0)))
    }

    /// Score all requested documents against every named branch, including
    /// fields which did not nominate them. Addresses must belong to this
    /// snapshot; stale addresses and budget overflows fail explicitly.
    pub async fn score_candidates(
        &self,
        candidates: &[SearchResult],
        plan: &CandidateScoringPlan,
        stats: Option<Arc<GlobalStats>>,
    ) -> Result<Vec<ScoredCandidate>> {
        plan.validate(self.schema())?;
        if candidates.len() > crate::query::MAX_FUSION_CANDIDATE_SLOTS {
            return Err(Error::Query(
                "candidate scoring document budget exceeded".into(),
            ));
        }
        let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        let mut addresses = BTreeSet::new();
        for (i, candidate) in candidates.iter().enumerate() {
            let &segment = self
                .segment_map()
                .get(&candidate.segment_id)
                .ok_or_else(|| {
                    Error::Query("candidate address is stale or belongs to another snapshot".into())
                })?;
            if candidate.doc_id >= self.segment_readers()[segment].num_docs()
                || !addresses.insert((candidate.segment_id, candidate.doc_id))
            {
                return Err(Error::Query(
                    "candidate addresses must be valid and unique".into(),
                ));
            }
            groups.entry(segment).or_default().push(i);
        }
        let stats = match stats {
            Some(stats) => stats,
            None => self.candidate_text_stats(plan).await?,
        };
        let names: Vec<&str> = plan.features.iter().map(|f| f.name.as_str()).collect();
        let count = names.len();
        let mut output = Vec::with_capacity(candidates.len());
        let mut scored_values = 0usize;
        let mut vector_bytes = 0usize;
        let mut matrix_values = 0usize;
        for (segment, mut candidate_indices) in groups {
            let reader = &self.segment_readers()[segment];
            candidate_indices.sort_unstable_by_key(|&i| candidates[i].doc_id);
            let documents: Vec<u32> = candidate_indices
                .iter()
                .map(|&i| candidates[i].doc_id)
                .collect();
            let mut doc_values = vec![vec![None; count]; documents.len()];
            let mut passages: BTreeMap<(u32, u16), Vec<Option<f32>>> = BTreeMap::new();
            let chunk_fields: BTreeSet<u32> = plan
                .features
                .iter()
                .filter(|feature| feature.scope == ScoreScope::Chunk)
                .map(|feature| feature.query.field.0)
                .collect();
            let mut nominated = Vec::new();
            if !plan.all_passages {
                for &i in &candidate_indices {
                    for (field, positions) in &candidates[i].positions {
                        if !chunk_fields.contains(field) {
                            continue;
                        }
                        for position in positions {
                            nominated.push(crate::segment::ordinal_lookup::LogicalUnit {
                                doc: candidates[i].doc_id,
                                ordinal: u16::try_from(position.position).map_err(|_| {
                                    Error::Query("invalid nominated passage ordinal".into())
                                })?,
                            });
                            if nominated.len() > crate::query::MAX_FUSION_CHUNK_SLOTS {
                                return Err(Error::Query("too many nominated passages".into()));
                            }
                        }
                    }
                }
                nominated.sort_unstable();
                nominated.dedup();
            }
            for (feature_index, feature) in plan.features.iter().enumerate() {
                let mut locations = if feature.scope == ScoreScope::Chunk && !plan.all_passages {
                    reader.candidate_passage_locations(feature.query.field, &nominated)?
                } else {
                    reader.candidate_locations(
                        feature.query.field,
                        &documents,
                        MAX_FEATURE_VALUES.saturating_sub(scored_values),
                    )?
                };
                if locations.len() > MAX_FEATURE_VALUES.saturating_sub(scored_values) {
                    return Err(Error::Query("L1 scored-value budget exceeded".into()));
                }
                let work = locations
                    .len()
                    .checked_mul(feature.query.components.len())
                    .ok_or_else(|| Error::Query("L1 feature work overflow".into()))?;
                scored_values = scored_values
                    .checked_add(work)
                    .filter(|&n| n <= MAX_FEATURE_VALUES)
                    .ok_or_else(|| Error::Query("L1 scored-component budget exceeded".into()))?;
                if locations.is_empty() {
                    continue;
                }
                if let Some(flat) = reader.flat_vectors().get(&feature.query.field.0) {
                    vector_bytes = vector_bytes
                        .checked_add(
                            locations
                                .len()
                                .checked_mul(feature.query.components.len())
                                .and_then(|n| n.checked_mul(flat.vector_byte_size()))
                                .ok_or_else(|| {
                                    Error::Query("L1 vector byte count overflow".into())
                                })?,
                        )
                        .ok_or_else(|| Error::Query("L1 vector byte count overflow".into()))?;
                    if vector_bytes > MAX_VECTOR_BYTES {
                        return Err(Error::Query("L1 exceeds stored vector read budget".into()));
                    }
                }
                locations.sort_unstable_by_key(|location| location.physical);
                let targets: Vec<u32> =
                    locations.iter().map(|location| location.physical).collect();
                let scores = score_field(self, reader, &feature.query, &targets, &stats).await?;
                for (location, score) in locations.into_iter().zip(scores) {
                    if feature.scope == ScoreScope::Document {
                        let doc_index = documents
                            .binary_search(&location.doc)
                            .expect("resolved selected doc");
                        let value: &mut Option<f32> = &mut doc_values[doc_index][feature_index];
                        *value = Some(value.map_or(score, |v| v.max(score)));
                    } else {
                        let values = match passages.entry((location.doc, location.ordinal)) {
                            std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
                            std::collections::btree_map::Entry::Vacant(entry) => {
                                matrix_values += count;
                                if matrix_values > MAX_FEATURE_VALUES {
                                    return Err(Error::Query(
                                        "L1 feature matrix budget exceeded".into(),
                                    ));
                                }
                                entry.insert(vec![None; count])
                            }
                        };
                        values[feature_index] = Some(score);
                    }
                }
            }
            let mut passages = passages.into_iter().peekable();
            let mut combined = vec![None; count];
            for (doc_index, &candidate_index) in candidate_indices.iter().enumerate() {
                let candidate = &candidates[candidate_index];
                let document = std::mem::take(&mut doc_values[doc_index]);
                let mut rows = Vec::new();
                while passages
                    .peek()
                    .is_some_and(|((doc, _), _)| *doc == candidate.doc_id)
                {
                    let ((_, ordinal), values) = passages.next().expect("peeked row");
                    for j in 0..count {
                        combined[j] = values[j].or(document[j]);
                    }
                    let score = match &plan.model {
                        Some(model) => model.score(&names, &combined)?,
                        None => candidate.score,
                    };
                    rows.push(PassageFeatures {
                        ordinal,
                        score,
                        values,
                    });
                }
                let scored_passages = rows.len();
                if plan.model.is_none() && rows.len() > plan.export_passages {
                    return Err(Error::Query(format!(
                        "feature export would omit {} passages of document {}; increase export_passages or supply l1",
                        rows.len() - plan.export_passages,
                        candidate.doc_id
                    )));
                }
                rows.sort_unstable_by(|a, b| {
                    b.score
                        .total_cmp(&a.score)
                        .then_with(|| a.ordinal.cmp(&b.ordinal))
                });
                let mut result = candidate.clone();
                if let Some(model) = &plan.model {
                    result.score = match rows.first() {
                        Some(row) => row.score,
                        None => model.score(&names, &document)?,
                    };
                }
                rows.truncate(plan.export_passages);
                // A document-only feature never creates an ordinal-zero row.
                if plan.model.is_some() {
                    let fields: BTreeSet<u32> = plan
                        .features
                        .iter()
                        .filter(|f| f.scope == ScoreScope::Chunk)
                        .map(|f| f.query.field.0)
                        .collect();
                    result.positions = fields
                        .into_iter()
                        .map(|field| {
                            (
                                field,
                                rows.iter()
                                    .map(|row| {
                                        ScoredPosition::new(u32::from(row.ordinal), row.score)
                                    })
                                    .collect(),
                            )
                        })
                        .collect();
                }
                output.push(ScoredCandidate {
                    result,
                    features: CandidateScores {
                        document,
                        passages: rows,
                        scored_passages,
                    },
                });
            }
        }
        output.sort_unstable_by(|a, b| {
            crate::query::compare_search_results_desc(&a.result, &b.result)
        });
        Ok(output)
    }
}
