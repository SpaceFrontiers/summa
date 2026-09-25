//! Field-level BP reordering of mapped text fields.
//!
//! A chunked text field keys its postings and positions by a segment-local
//! virtual chunk id and resolves hits through `.chunks`, so its order is its
//! own: reordering it never moves a document id, the store, the fast fields
//! or any vector field (`docs/lexical-vertical.md`, "Field-level
//! reordering"). This module computes a Recursive Graph Bisection order over
//! the field's own postings and rewrites the text files of a segment with the
//! field's virtual ids permuted:
//!
//! - the term dictionary, postings and positions are rewritten term by term
//!   (terms of other fields are copied through the merger's single-source
//!   path, which keeps their bytes and rebuilds offsets),
//! - `.chunks` is written with the reordered field's rows in the new order
//!   and every other section unchanged.
//!
//! Gated by the `reorder` schema attribute on a mapped text field.

use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::Result;
use crate::directories::{Directory, DirectoryWriter};
use crate::dsl::{Field, FieldType, Schema};
use crate::segment::builder::graph_bisection::{
    BpProgressLabel, ForwardIndex, fit_candidates_to_budget, graph_bisection_with_progress,
    select_frequency_candidates,
};
use crate::segment::chunk_map::{ChunkMapBuilder, write_chunk_maps_with_copied_norms};
use crate::segment::reader::SegmentReader;
use crate::segment::types::SegmentFiles;
use crate::segment::{OffsetWriter, SegmentMerger};
use crate::structures::{
    BlockPostingList, PositionStreamEncoder, PostingCodec, SSTableWriter, TERMINATED, TermInfo,
    postings::{PositionRangeSource, PostingBlockSource, PostingStreamWriter},
};

/// Minimum partition of the bisection: one posting block, the pruning unit.
const MIN_PARTITION: usize = crate::structures::POSTING_BLOCK_SIZE;
const BP_ITERATIONS: usize = 20;

/// A computed permutation of one text field's physical scoring-unit IDs.
pub(crate) struct TextReorderPlan {
    pub field: Field,
    /// `order[new] = old` virtual id.
    pub order: Vec<u32>,
    /// `inverse[old] = new` virtual id.
    pub inverse: Vec<u32>,
    pub converged: bool,
}

fn field_of(key: &[u8]) -> u32 {
    u32::from_le_bytes([key[0], key[1], key[2], key[3]])
}

fn check_cancelled(cancellation: Option<&std::sync::atomic::AtomicBool>) -> Result<()> {
    if cancellation.is_some_and(|c| c.load(std::sync::atomic::Ordering::Acquire)) {
        return Err(crate::Error::IndexClosed);
    }
    Ok(())
}

fn admit_scratch(required: usize, budget: usize) -> Result<()> {
    if required > budget {
        return Err(crate::Error::Schema(format!(
            "text reorder needs {required} bytes of scratch; increase bp-memory-budget-mb (budget {budget})"
        )));
    }
    Ok(())
}

pub(crate) fn plan_bytes(plans: &[TextReorderPlan]) -> usize {
    plans.iter().fold(0usize, |bytes, plan| {
        bytes.saturating_add((plan.order.capacity() + plan.inverse.capacity()).saturating_mul(4))
    })
}

/// Plan the reorder of every mapped text field carrying the `reorder`
/// attribute that has data in this segment.
pub(crate) async fn plan_text_reorders(
    reader: &SegmentReader,
    schema: &Schema,
    memory_budget: usize,
    bp_budget: crate::segment::BpBudget,
    cancellation: Option<&std::sync::atomic::AtomicBool>,
    rayon_pool: Option<Arc<rayon::ThreadPool>>,
) -> Result<Vec<TextReorderPlan>> {
    plan_text_reorders_from_sources(
        std::slice::from_ref(reader),
        schema,
        memory_budget,
        bp_budget,
        cancellation,
        rayon_pool,
        false,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn plan_text_reorders_from_sources(
    readers: &[SegmentReader],
    schema: &Schema,
    memory_budget: usize,
    bp_budget: crate::segment::BpBudget,
    cancellation: Option<&std::sync::atomic::AtomicBool>,
    rayon_pool: Option<Arc<rayon::ThreadPool>>,
    allow_unmapped: bool,
) -> Result<Vec<TextReorderPlan>> {
    check_cancelled(cancellation)?;
    let mut plans = Vec::new();
    for (field, entry) in schema.fields() {
        if !(entry.indexed && entry.reorder && entry.field_type == FieldType::Text) {
            continue;
        }
        for reader in readers {
            if reader.chunk_map(field).is_none()
                && !entry.chunked
                && reader
                    .meta()
                    .field_stats
                    .get(&field.0)
                    .is_some_and(|stats| stats.total_tokens > 0)
                && (!allow_unmapped || reader.doc_lengths(field).is_none())
            {
                return Err(crate::Error::Schema(format!(
                    "text reorder of '{}' requires a document map or legacy document lengths; rebuild or migrate legacy segments through a compatible merge first",
                    entry.name
                )));
            }
        }
        if let Some(plan) = plan_field(
            readers,
            schema,
            field,
            memory_budget.saturating_sub(plan_bytes(&plans)),
            bp_budget,
            cancellation,
            rayon_pool.clone(),
        )
        .await?
        {
            plans.push(plan);
        }
    }
    Ok(plans)
}

async fn plan_field(
    readers: &[SegmentReader],
    schema: &Schema,
    field: Field,
    memory_budget: usize,
    bp_budget: crate::segment::BpBudget,
    cancellation: Option<&std::sync::atomic::AtomicBool>,
    rayon_pool: Option<Arc<rayon::ThreadPool>>,
) -> Result<Option<TextReorderPlan>> {
    let started = std::time::Instant::now();
    let field_name = schema.get_field_name(field).unwrap_or("?");
    let chunked = schema.get_field_entry(field).unwrap().chunked;
    let mut unit_offsets = Vec::with_capacity(readers.len());
    let mut total = 0u32;
    for reader in readers {
        unit_offsets.push(total);
        let count = if chunked {
            reader.num_chunks(field)
        } else {
            reader.num_docs()
        };
        total = total
            .checked_add(count)
            .ok_or_else(|| crate::Error::Schema("text reorder exceeds u32 scoring units".into()))?;
    }
    let num_chunks = total as usize;
    if num_chunks < 2 * MIN_PARTITION {
        return Ok(None);
    }
    let prefix = field.0.to_le_bytes();

    // Use the shared graph-bisection low-frequency selection and fitting policy. Text builds
    // a (unit, term) array plus CSR, so its construction costs 12 B/posting.
    // Per-unit scratch includes counts, offsets, fill cursors and BP scratch.
    check_cancelled(cancellation)?;
    let entity_bytes = num_chunks.saturating_mul(32).saturating_add(8);
    admit_scratch(entity_bytes, memory_budget)?;
    let mut dfs: Vec<u32> = Vec::new();
    let mut input_bytes = 0usize;
    let mut sources = Vec::with_capacity(readers.len());
    let mut iter = crate::segment::merger::MergedTerms::new(readers, cancellation).await?;
    while let Some(key) = iter.next(&mut sources).await? {
        check_cancelled(cancellation)?;
        if key[..4] != prefix {
            continue;
        }
        if dfs.len() == dfs.capacity() {
            let capacity = dfs.capacity().saturating_add(1024);
            admit_scratch(
                entity_bytes.saturating_add(capacity.saturating_mul(4)),
                memory_budget,
            )?;
            dfs.reserve_exact(capacity - dfs.len());
        }
        let mut frequency = 0u32;
        for (_, info, _) in &sources {
            frequency = frequency
                .checked_add(info.doc_freq())
                .ok_or_else(|| crate::Error::Corruption("text BP frequency overflow".into()))?;
            input_bytes = input_bytes.max(
                info.external_info()
                    .map_or(0, |(_, len)| usize::try_from(len).unwrap_or(usize::MAX)),
            );
        }
        dfs.push(frequency);
    }
    drop(iter);
    if dfs.is_empty() {
        return Ok(None);
    }
    let vocabulary_bytes = dfs.capacity().saturating_mul(4);
    let fixed_bytes = entity_bytes
        .saturating_add(vocabulary_bytes)
        .saturating_add(input_bytes);
    admit_scratch(fixed_bytes, memory_budget)?;
    let (mut eligible, mut budget_limited) = select_frequency_candidates(
        &dfs,
        2,
        usize::MAX,
        memory_budget.saturating_sub(fixed_bytes),
    );
    let fit = fit_candidates_to_budget(&mut eligible, fixed_bytes, memory_budget, 12);
    budget_limited |= fit.dropped > 0;
    if budget_limited {
        log::warn!(
            "[reorder_text] field {field_name}: budget {memory_budget} limits BP to {} terms ({} postings)",
            eligible.len(),
            fit.retained_postings
        );
    }
    let terms_count = dfs.len();
    drop(dfs);
    if eligible.is_empty() {
        if budget_limited {
            return Err(crate::Error::Schema(
                "text reorder budget cannot fit any active graph terms".into(),
            ));
        }
        return Ok(None);
    }
    eligible.sort_unstable_by_key(|&(term, frequency)| (frequency, term));
    let kept = fit.retained_postings;
    let compact = eligible.len();
    let mut active = vec![u32::MAX; terms_count];
    for (id, &(ordinal, _)) in eligible.iter().enumerate() {
        active[ordinal as usize] = id as u32;
    }
    drop(eligible);

    let mut pairs = Vec::with_capacity(kept);
    let mut iter = crate::segment::merger::MergedTerms::new(readers, cancellation).await?;
    let mut ordinal = 0usize;
    while let Some(key) = iter.next(&mut sources).await? {
        check_cancelled(cancellation)?;
        if key[..4] != prefix {
            continue;
        }
        let compact_id = active[ordinal];
        ordinal += 1;
        if compact_id == u32::MAX {
            continue;
        }
        for &(source, ref info, _) in &sources {
            let reader = &readers[source];
            let base = unit_offsets[source];
            let units = if chunked {
                reader.num_chunks(field)
            } else {
                reader.num_docs()
            };
            let first_pair = pairs.len();
            let mut append = |vid: u32| -> Result<()> {
                if pairs.len().is_multiple_of(4096) {
                    check_cancelled(cancellation)?;
                }
                if vid >= units || pairs.len() >= kept {
                    return Err(crate::Error::Corruption(
                        "text reorder posting exceeds its map or declared frequency".into(),
                    ));
                }
                pairs.push((base + vid, compact_id));
                Ok(())
            };
            if let Some((ids, _)) = info.decode_inline() {
                for vid in ids {
                    append(vid)?;
                }
            } else {
                let (offset, len) = info
                    .external_info()
                    .ok_or_else(|| crate::Error::Corruption("missing text postings".into()))?;
                let bytes = reader.read_postings(offset, len).await?;
                let list = BlockPostingList::deserialize_zero_copy(bytes)?;
                let mut postings = list.iterator();
                while postings.doc() != TERMINATED {
                    append(postings.doc())?;
                    postings.advance();
                }
            }
            if pairs.len() - first_pair != info.doc_freq() as usize {
                return Err(crate::Error::Corruption(
                    "text BP posting count disagrees with dictionary".into(),
                ));
            }
        }
    }
    drop(iter);
    drop(sources);
    drop(active);
    let mut counts = vec![0u32; num_chunks];
    for &(vid, _) in &pairs {
        counts[vid as usize] += 1;
    }
    let mut offsets = Vec::with_capacity(num_chunks + 1);
    offsets.push(0u64);
    for &c in &counts {
        offsets.push(offsets.last().unwrap() + u64::from(c));
    }
    let mut terms = vec![0u32; pairs.len()];
    let mut fill: Vec<u64> = offsets[..num_chunks].to_vec();
    for &(vid, term) in &pairs {
        let at = &mut fill[vid as usize];
        terms[*at as usize] = term;
        *at += 1;
    }
    drop(pairs);
    drop(counts);
    drop(fill);
    let fwd = ForwardIndex::from_csr(terms, offsets, compact, memory_budget, budget_limited);

    let bisect = || {
        graph_bisection_with_progress(
            &fwd,
            MIN_PARTITION,
            BP_ITERATIONS,
            bp_budget,
            cancellation,
            BpProgressLabel {
                index: schema.index_label(),
                field: field_name,
                entity_kind: "chunks",
            },
        )
    };
    let (order, converged) =
        crate::segment::merger::block_in_place_if_multithread(|| match rayon_pool {
            Some(pool) => pool.install(bisect),
            None => bisect(),
        });
    check_cancelled(cancellation)?;
    drop(fwd);
    if order.len() != num_chunks {
        return Err(crate::Error::Internal(format!(
            "text reorder of field '{field_name}' produced {} ids for {num_chunks} chunks",
            order.len()
        )));
    }
    let mut inverse = vec![u32::MAX; num_chunks];
    for (new, &old) in order.iter().enumerate() {
        let slot = inverse
            .get_mut(old as usize)
            .ok_or_else(|| crate::Error::Internal("text BP returned an out-of-range ID".into()))?;
        *slot = new as u32;
    }
    if inverse.contains(&u32::MAX) {
        return Err(crate::Error::Internal(format!(
            "text reorder of field '{field_name}' is not a permutation"
        )));
    }
    let identity = order
        .iter()
        .enumerate()
        .all(|(new, &old)| new == old as usize);
    log::info!(
        "[reorder_text] index={} field {}: BP over {} chunks, {} active terms ({} postings, budget_limited={}) in {:.1}s (converged={}, identity={})",
        schema.index_label(),
        field_name,
        num_chunks,
        compact,
        kept,
        budget_limited,
        started.elapsed().as_secs_f64(),
        converged,
        identity,
    );
    if identity && converged && !budget_limited {
        return Ok(None);
    }
    Ok(Some(TextReorderPlan {
        field,
        order,
        inverse,
        converged: converged && !budget_limited,
    }))
}

/// Rewrite the term dictionary, postings, positions and chunk maps of a
/// segment with the planned fields' virtual ids permuted.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn rewrite_text_files<D: Directory + DirectoryWriter>(
    dir: &D,
    reader: &SegmentReader,
    dst_files: &SegmentFiles,
    schema: &Arc<Schema>,
    plans: &[TextReorderPlan],
    posting_config: (
        crate::structures::IndexOptimization,
        PostingCodec,
        crate::structures::SSTableBlockSize,
    ),
    memory_budget: usize,
    cancellation: Option<&std::sync::atomic::AtomicBool>,
) -> Result<()> {
    check_chunk_map_budget(reader, plans, memory_budget)?;
    let (optimization, posting_codec, term_dict_block_size) = posting_config;
    let started = std::time::Instant::now();
    let by_field: FxHashMap<u32, &TextReorderPlan> =
        plans.iter().map(|plan| (plan.field.0, plan)).collect();
    let term_budget = memory_budget.saturating_sub(plan_bytes(plans));
    let merger = SegmentMerger::new(Arc::clone(schema))
        .with_posting_config(optimization, posting_codec)
        .with_term_dict_block_size(term_dict_block_size);
    let mut postings_out = OffsetWriter::new(dir.streaming_writer_cold(&dst_files.postings).await?);
    let mut positions_out =
        OffsetWriter::new(dir.streaming_writer_cold(&dst_files.positions).await?);
    let mut term_dict_out =
        OffsetWriter::new(dir.streaming_writer_cold(&dst_files.term_dict).await?);
    let mut term_dict = SSTableWriter::<&mut OffsetWriter, TermInfo>::with_config(
        &mut term_dict_out,
        crate::structures::SSTableWriterConfig {
            block_size: term_dict_block_size,
            ..crate::structures::SSTableWriterConfig::from_optimization(optimization)
        },
    );
    let mut sources: Vec<(usize, TermInfo, u32)> = Vec::with_capacity(1);
    // Single-source copies never mix representations; the counter stays 0.
    let mut posting_stats = crate::segment::merger::PostingMergeStats::default();
    let mut terms = 0usize;
    let mut reordered_terms = 0usize;

    let mut iter = reader.term_dict_iter();
    while let Some((key, info)) = iter.next().await.map_err(crate::Error::from)? {
        if cancellation.is_some_and(|c| c.load(std::sync::atomic::Ordering::Acquire)) {
            return Err(crate::Error::IndexClosed);
        }
        let field_id = if key.len() >= 4 {
            field_of(&key)
        } else {
            u32::MAX
        };
        let new_info = match by_field.get(&field_id) {
            Some(plan) => {
                reordered_terms += 1;
                reorder_term(
                    std::slice::from_ref(reader),
                    &[(0, info, 0)],
                    plan,
                    &mut postings_out,
                    &mut positions_out,
                    posting_codec,
                    term_budget,
                    cancellation,
                )
                .await?
            }
            None => {
                sources.clear();
                sources.push((0, info, 0));
                merger
                    .merge_term(
                        std::slice::from_ref(reader),
                        crate::Field(field_id),
                        &mut sources,
                        &mut postings_out,
                        &mut positions_out,
                        &mut posting_stats,
                    )
                    .await?
            }
        };
        term_dict
            .insert(&key, &new_info)
            .map_err(crate::Error::Io)?;
        terms += 1;
    }
    term_dict.finish().map_err(crate::Error::Io)?;
    let positions_bytes = positions_out.offset();
    postings_out.finish()?;
    term_dict_out.finish()?;
    if positions_bytes > 0 {
        positions_out.finish()?;
    } else {
        drop(positions_out);
        let _ = dir.delete(&dst_files.positions).await;
    }

    write_reordered_chunk_maps(
        dir,
        reader,
        dst_files,
        schema,
        plans,
        memory_budget,
        cancellation,
    )
    .await?;

    log::info!(
        "[reorder_text] index={} rewrote {} terms ({} reordered) in {:.1}s",
        schema.index_label(),
        terms,
        reordered_terms,
        started.elapsed().as_secs_f64(),
    );
    Ok(())
}
// Admit the chunk-map writer's peak scratch, including retained BP plans.
// Existing physical columns copy once; only one logical-slot permutation is sorted.
fn check_chunk_map_budget(
    reader: &SegmentReader,
    plans: &[TextReorderPlan],
    memory_budget: usize,
) -> Result<()> {
    let map_bytes: usize = reader
        .chunk_maps()
        .values()
        .map(|m| m.num_chunks() as usize * 8)
        .sum();
    let slot_bytes = reader
        .chunk_maps()
        .values()
        .map(|m| m.num_chunks() as usize * 4)
        .max()
        .unwrap_or(0);
    let plan_bytes = plan_bytes(plans);
    let required = map_bytes
        .saturating_add(slot_bytes)
        .saturating_add(plan_bytes);
    if required > memory_budget {
        return Err(crate::Error::Schema(format!(
            "text logical addressing needs {required} bytes of reorder scratch; increase bp-memory-budget-mb"
        )));
    }
    Ok(())
}

pub(crate) async fn write_reordered_chunk_maps<D: Directory + DirectoryWriter>(
    dir: &D,
    reader: &SegmentReader,
    dst_files: &SegmentFiles,
    schema: &Arc<Schema>,
    plans: &[TextReorderPlan],
    memory_budget: usize,
    cancellation: Option<&std::sync::atomic::AtomicBool>,
) -> Result<()> {
    check_chunk_map_budget(reader, plans, memory_budget)?;
    let by_field: FxHashMap<u32, &TextReorderPlan> = plans.iter().map(|p| (p.field.0, p)).collect();
    // Chunk maps and length columns: planned fields in the new order, the
    // rest verbatim.
    let mut chunk_fields: Vec<(u32, ChunkMapBuilder)> = Vec::new();
    for (field_id, map) in reader.chunk_maps() {
        if cancellation.is_some_and(|c| c.load(std::sync::atomic::Ordering::Acquire)) {
            return Err(crate::Error::IndexClosed);
        }
        let mut builder = ChunkMapBuilder::with_capacity(map.num_chunks() as usize);
        builder.set_document_units(map.is_document_map());
        match by_field.get(field_id) {
            Some(plan) => {
                for &old in &plan.order {
                    let (doc, ordinal) = map.resolve(old);
                    builder.push(doc, ordinal, map.length(old))?;
                }
            }
            None => {
                for vid in 0..map.num_chunks() {
                    let (doc, ordinal) = map.resolve(vid);
                    builder.push(doc, ordinal, map.length(vid))?;
                }
            }
        }
        builder.set_total_tokens(map.total_tokens());
        chunk_fields.push((*field_id, builder));
    }
    chunk_fields.sort_by_key(|(field_id, _)| *field_id);
    let mut norms: Vec<_> = schema
        .fields()
        .filter_map(|(field, _)| reader.doc_lengths(field).map(|lengths| (field.0, lengths)))
        .collect();
    norms.sort_unstable_by_key(|(field_id, _)| *field_id);
    let fields: Vec<(u32, &ChunkMapBuilder)> = chunk_fields
        .iter()
        .filter(|(_, builder)| !builder.is_empty())
        .map(|(field_id, builder)| (*field_id, builder))
        .collect();
    if !fields.is_empty() || !norms.is_empty() {
        let mut writer = dir.streaming_writer_cold(&dst_files.chunks).await?;
        write_chunk_maps_with_copied_norms(&mut *writer, &fields, &norms)
            .map_err(crate::Error::Io)?;
        writer.finish()?;
    }

    Ok(())
}

/// Rewrite one term of a planned field: postings sorted by the new virtual
/// ids, positions re-encoded in that order, block bounds from the new
/// lengths.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn reorder_term(
    readers: &[SegmentReader],
    sources: &[(usize, TermInfo, u32)],
    plan: &TextReorderPlan,
    postings_out: &mut OffsetWriter,
    positions_out: &mut OffsetWriter,
    posting_codec: PostingCodec,
    memory_budget: usize,
    cancellation: Option<&std::sync::atomic::AtomicBool>,
) -> Result<TermInfo> {
    // Source-local cursors survive sorting; positions remain in bounded readers.
    struct Entry {
        new: u32,
        old: u32,
        tf: u32,
        source: usize,
        cursor: u64,
    }
    check_cancelled(cancellation)?;
    let count = sources
        .iter()
        .try_fold(0u32, |count, (_, info, _)| {
            count.checked_add(info.doc_freq())
        })
        .ok_or_else(|| crate::Error::Corruption("reorder frequency overflow".into()))?;
    let fixed_bytes = (count as usize)
        .saturating_mul(std::mem::size_of::<Entry>())
        .saturating_add(64 * 1024)
        .saturating_add(
            sources
                .len()
                .saturating_mul(16 * 1024 + std::mem::size_of::<Option<PositionRangeSource>>()),
        );
    admit_scratch(fixed_bytes, memory_budget)?;
    // One input posting directory, one position directory per source, two outputs.
    let directory_budget = (memory_budget - fixed_bytes) / (sources.len() + 3);
    let mut entries = Vec::with_capacity(count as usize);
    let has_positions = sources
        .first()
        .is_some_and(|(_, info, _)| info.position_info().is_some());
    if sources
        .iter()
        .any(|(_, info, _)| info.position_info().is_some() != has_positions)
    {
        return Err(crate::Error::Corruption(
            "cannot reorder a term with inconsistent position data".into(),
        ));
    }
    let mut positions = Vec::with_capacity(sources.len());
    let (mut compact_postings, mut compact_positions, mut impact, mut ratio) =
        (true, false, false, false);
    let mut docs = Vec::with_capacity(crate::structures::POSTING_BLOCK_SIZE);
    let mut tfs = Vec::with_capacity(crate::structures::POSTING_BLOCK_SIZE);
    for (source, &(segment, ref info, base)) in sources.iter().enumerate() {
        check_cancelled(cancellation)?;
        let reader = &readers[segment];
        let units = reader
            .chunk_map(plan.field)
            .map_or(reader.num_docs(), |map| map.num_chunks());
        let remap = |old: u32| -> Result<u32> {
            if old >= units {
                return Err(crate::Error::Corruption(
                    "reordered posting exceeds its source field map".into(),
                ));
            }
            base.checked_add(old)
                .and_then(|id| plan.inverse.get(id as usize))
                .copied()
                .ok_or_else(|| {
                    crate::Error::Corruption("reordered posting exceeds its field map".into())
                })
        };
        let position_source = match info.position_info() {
            Some((offset, len)) => Some(
                PositionRangeSource::open(
                    reader.position_file_range(offset, len)?,
                    directory_budget,
                    cancellation,
                )
                .await?
                .with_repacked_blocks(),
            ),
            None => None,
        };
        compact_positions |= position_source
            .as_ref()
            .is_some_and(PositionRangeSource::is_compact);
        let start = entries.len();
        if let Some((ids, freqs)) = info.decode_inline() {
            compact_postings = false;
            for (old, tf) in ids.into_iter().zip(freqs) {
                entries.push(Entry {
                    new: remap(old)?,
                    old,
                    tf,
                    source,
                    cursor: 0,
                });
            }
        } else {
            let (offset, len) = info
                .external_info()
                .ok_or_else(|| crate::Error::Corruption("missing text postings".into()))?;
            let input = PostingBlockSource::open(
                reader.posting_file_range(offset, len)?,
                directory_budget,
                cancellation,
            )
            .await?;
            if input.doc_count() != info.doc_freq() || input.len() == 0 {
                return Err(crate::Error::Corruption(
                    "reorder posting count disagrees with dictionary".into(),
                ));
            }
            if input.has_positions() != has_positions {
                return Err(crate::Error::Corruption(
                    "reorder posting and position formats disagree".into(),
                ));
            }
            compact_postings &= input.is_compact();
            impact |= input.has_impact_bounds();
            ratio |= input.has_ratio_bounds();
            for block in 0..input.len() {
                check_cancelled(cancellation)?;
                let encoded = input.read_block(block).await?;
                let span = input.position_span(block)?;
                if !encoded.decode_block_into(0, &mut docs, &mut tfs)
                    || (entries.len() - start).saturating_add(docs.len()) > info.doc_freq() as usize
                {
                    return Err(crate::Error::Corruption(
                        "invalid reordered posting block".into(),
                    ));
                }
                let mut cursor = span.start;
                for (&old, &tf) in docs.iter().zip(&tfs) {
                    entries.push(Entry {
                        new: remap(old)?,
                        old,
                        tf,
                        source,
                        cursor,
                    });
                    cursor = cursor.checked_add(u64::from(tf)).ok_or_else(|| {
                        crate::Error::Corruption("reorder position overflow".into())
                    })?;
                }
                if input.has_positions() && cursor != span.end {
                    return Err(crate::Error::Corruption(
                        "reorder position cursor disagrees with frequencies".into(),
                    ));
                }
            }
            if position_source.as_ref().is_some_and(|p| {
                p.total_positions()
                    != input
                        .position_span(input.len() - 1)
                        .map_or(u64::MAX, |span| span.end)
            }) {
                return Err(crate::Error::Corruption(
                    "reorder position count mismatch".into(),
                ));
            }
        }
        if entries.len() - start != info.doc_freq() as usize {
            return Err(crate::Error::Corruption(
                "reorder posting count mismatch".into(),
            ));
        }
        positions.push(position_source);
    }
    entries.sort_unstable_by_key(|entry| entry.new);
    if !has_positions
        && let Some(inline) =
            TermInfo::try_inline_iter(entries.len(), entries.iter().map(|e| (e.new, e.tf)))
    {
        return Ok(inline);
    }
    let posting_offset = postings_out.offset();
    let position_offset = positions_out.offset();
    let mut output = PostingStreamWriter::new(
        &mut *postings_out,
        entries
            .len()
            .div_ceil(crate::structures::POSTING_BLOCK_SIZE),
        has_positions,
        posting_codec,
        directory_budget,
    )?;
    if compact_postings && posting_codec != PostingCodec::Pfor {
        output.enable_compact_headers()?;
    }
    if impact {
        output.enable_impact_bounds()?;
    } else if ratio {
        output.enable_ratio_bounds()?;
    }
    let mut position_output =
        PositionStreamEncoder::with_budget(&mut *positions_out, directory_budget, posting_codec);
    if compact_positions {
        position_output = position_output.with_compact_directory();
    }
    for (i, entry) in entries.iter().enumerate() {
        if i.is_multiple_of(4096) {
            check_cancelled(cancellation)?;
        }
        let reader = &readers[sources[entry.source].0];
        let length = match reader.chunk_map(plan.field) {
            Some(map) => map.length(entry.old),
            None => reader
                .doc_lengths(plan.field)
                .ok_or_else(|| crate::Error::Corruption("reordered field lost its lengths".into()))?
                .length(entry.old),
        };
        output.push(entry.new, entry.tf, length)?;
        if let Some(positions) = &mut positions[entry.source] {
            positions
                .append_doc(&mut position_output, entry.cursor, entry.tf, cancellation)
                .await?;
        }
    }
    let (docs, posting_len) = output.finish(cancellation)?;
    if !has_positions {
        return Ok(TermInfo::external(posting_offset, posting_len, docs));
    }
    let (_, position_len) = position_output.finish_cancellable(cancellation)?;
    Ok(TermInfo::external_with_positions(
        posting_offset,
        posting_len,
        docs,
        position_offset,
        position_len,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::{
        CountCollector, PhraseQuery, Query, TermQuery, TopKCollector, collect_segment,
    };
    use crate::structures::TermPositions;
    use crate::{Document, Index, IndexConfig, IndexWriter, RamDirectory};

    #[tokio::test]
    async fn merged_text_rgb_matches_copy_merge_then_standalone_bytes_and_results() {
        for codec in [
            PostingCodec::Rounded,
            PostingCodec::Packed,
            PostingCodec::Pfor,
            PostingCodec::Simd4x,
        ] {
            let schema = |mapped| {
                let mut b = Schema::builder();
                let plain = b.add_text_field("plain", true, true);
                let chunks = b.add_text_field("chunks", true, true);
                let other = b.add_text_field("other", true, true);
                b.set_multi(plain, true);
                b.set_chunked(chunks, true);
                for field in [plain, chunks] {
                    b.set_reorder(field, mapped);
                    b.set_positions(field, crate::dsl::PositionMode::TokenPosition);
                }
                b.set_fast(other, true);
                (b.build(), plain, chunks)
            };
            let (target, plain, chunks) = schema(true);
            let target = Arc::new(target);
            let mut sources = Vec::new();
            for source in 0..2 {
                let dir = RamDirectory::new();
                let config = IndexConfig {
                    num_threads: 1,
                    num_indexing_threads: 1,
                    posting_codec: Some(codec),
                    compact_text: source == 1,
                    quantized_norms: true,
                    posting_ratio_bounds: true,
                    posting_impact_bounds: source == 1,
                    merge_policy: Box::new(crate::NoMergePolicy),
                    ..Default::default()
                };
                let (source_schema, _, _) = schema(source == 1);
                let other = source_schema.get_field("other").unwrap();
                let mut writer = IndexWriter::create(dir.clone(), source_schema, config.clone())
                    .await
                    .unwrap();
                for doc in 0..512 {
                    let mut row = Document::new();
                    if doc % 13 != 0 {
                        row.add_text(
                            plain,
                            if doc % 2 == 0 {
                                "alpha beta alpha"
                            } else {
                                "gamma delta gamma"
                            },
                        );
                        if doc % 5 == 0 {
                            row.add_text(plain, "extra alpha");
                        }
                    }
                    if doc % 7 != 0 {
                        row.add_text(
                            chunks,
                            if doc % 3 == 0 {
                                "alpha beta"
                            } else {
                                "gamma delta"
                            },
                        );
                        if doc % 2 == 0 {
                            row.add_text(chunks, "extra alpha beta");
                        }
                    }
                    row.add_text(other, "unchanged token");
                    writer.add_document(row).unwrap();
                }
                writer.commit().await.unwrap();
                if source == 1 {
                    writer.reorder().await.unwrap();
                }
                writer.shutdown().await.unwrap();
                let index = Index::open(dir.clone(), config).await.unwrap();
                let reader = index.reader().await.unwrap();
                let searcher = reader.searcher().await.unwrap();
                let id = crate::segment::SegmentId(searcher.segment_readers()[0].meta().id);
                sources.push(
                    SegmentReader::open(&dir, id, Arc::new(schema(source == 1).0), 256)
                        .await
                        .unwrap(),
                );
            }
            let output = RamDirectory::new();
            let copy = crate::segment::SegmentId::new();
            let fused = crate::segment::SegmentId::new();
            let separate = crate::segment::SegmentId::new();
            let merger = || {
                SegmentMerger::new(Arc::clone(&target))
                    .with_posting_config(crate::structures::IndexOptimization::default(), codec)
            };
            merger().merge(&output, &sources, copy, None).await.unwrap();
            merger()
                .with_reorder_fields(true)
                .merge(&output, &sources, fused, None)
                .await
                .unwrap();
            crate::segment::reorder::reorder_segment(
                &output,
                &target,
                copy,
                separate,
                256,
                None,
                crate::segment::reorder::DEFAULT_MEMORY_BUDGET,
                crate::segment::BpBudget::full(),
                true,
                crate::segment::reorder::BpGranularity::Auto,
                crate::structures::IndexOptimization::default(),
                codec,
                None,
                crate::structures::SSTableBlockSize::default(),
                None,
                None,
                None,
            )
            .await
            .unwrap();
            let a = SegmentFiles::new(fused.0);
            let b = SegmentFiles::new(separate.0);
            for (a, b) in [
                (a.postings, b.postings),
                (a.positions, b.positions),
                (a.term_dict, b.term_dict),
                (a.chunks, b.chunks),
                (a.store, b.store),
                (a.fast, b.fast),
            ] {
                let actual = output
                    .open_read(&a)
                    .await
                    .unwrap()
                    .read_bytes()
                    .await
                    .unwrap();
                let expected = output
                    .open_read(&b)
                    .await
                    .unwrap()
                    .read_bytes()
                    .await
                    .unwrap();
                assert_eq!(
                    actual.as_slice(),
                    expected.as_slice(),
                    "codec={codec:?}, file={a:?}"
                );
            }
            let expected = SegmentReader::open(&output, copy, Arc::clone(&target), 256)
                .await
                .unwrap();
            let actual = SegmentReader::open(&output, fused, Arc::clone(&target), 256)
                .await
                .unwrap();
            let queries: Vec<Box<dyn Query>> = vec![
                Box::new(TermQuery::text(plain, "alpha")),
                Box::new(PhraseQuery::text(plain, "alpha beta")),
                Box::new(PhraseQuery::text(chunks, "alpha beta")),
                Box::new(
                    crate::query::BooleanQuery::new()
                        .must(TermQuery::text(plain, "alpha"))
                        .must_not(TermQuery::text(chunks, "gamma")),
                ),
            ];
            for query in queries {
                let mut snapshots = Vec::new();
                for reader in [&expected, &actual] {
                    let mut top = TopKCollector::with_positions(2000);
                    let mut count = CountCollector::new();
                    collect_segment(reader, query.as_ref(), &mut (&mut top, &mut count))
                        .await
                        .unwrap();
                    snapshots.push((
                        count.count(),
                        top.into_sorted_results()
                            .iter()
                            .map(|h| {
                                // Chunk ordinal lists preserve physical encounter order;
                                // compare their meaning across permutations.
                                let mut positions = h.positions.clone();
                                for (_, values) in &mut positions {
                                    values.sort_by_key(|p| p.position);
                                }
                                positions.sort_by_key(|(field, _)| *field);
                                (
                                    h.doc_id,
                                    h.score.to_bits(),
                                    serde_json::to_value(&positions).unwrap(),
                                )
                            })
                            .collect::<Vec<_>>(),
                    ));
                }
                assert_eq!(snapshots[0], snapshots[1], "codec={codec:?}, query={query}");
                let mut top = TopKCollector::new(10);
                crate::query::collect_segment_with_limit(&actual, query.as_ref(), &mut top, 10)
                    .await
                    .unwrap();
                let ranked: Vec<_> = top
                    .into_sorted_results()
                    .iter()
                    .map(|hit| (hit.doc_id, hit.score.to_bits()))
                    .collect();
                assert_eq!(
                    ranked,
                    snapshots[0]
                        .1
                        .iter()
                        .take(10)
                        .map(|hit| (hit.0, hit.1))
                        .collect::<Vec<_>>()
                );
            }
        }
    }

    #[tokio::test]
    async fn legacy_plain_reorder_requires_explicit_map_migration() {
        let schema = |reorder| {
            let mut builder = Schema::builder();
            let field = builder.add_text_field("text", true, false);
            builder.set_reorder(field, reorder);
            (builder.build(), field)
        };
        let (original, field) = schema(false);
        let dir = RamDirectory::new();
        let config = IndexConfig {
            num_threads: 1,
            num_indexing_threads: 1,
            merge_policy: Box::new(crate::NoMergePolicy),
            ..Default::default()
        };
        let mut writer = IndexWriter::create(dir.clone(), original, config.clone())
            .await
            .unwrap();
        let mut doc = Document::new();
        doc.add_text(field, "alpha beta");
        writer.add_document(doc).unwrap();
        writer.commit().await.unwrap();
        writer.shutdown().await.unwrap();
        let index = Index::open(dir, config).await.unwrap();
        let reader = index.reader().await.unwrap();
        let searcher = reader.searcher().await.unwrap();
        let error = plan_text_reorders(
            &searcher.segment_readers()[0],
            &schema(true).0,
            1024 * 1024,
            crate::segment::BpBudget::full(),
            None,
            None,
        )
        .await
        .err()
        .expect("legacy RGB must not silently skip a requested field");
        assert!(error.to_string().contains("document map"));
    }

    #[tokio::test]
    async fn reordering_preserves_compact_and_legacy_layouts_and_untouched_payloads() {
        for (compact, codec) in [false, true].into_iter().flat_map(|compact| {
            [
                PostingCodec::Rounded,
                PostingCodec::Packed,
                PostingCodec::Pfor,
                PostingCodec::Simd4x,
            ]
            .map(|codec| (compact, codec))
        }) {
            let mut schema = Schema::builder();
            let field = schema.add_text_field("text", true, false);
            let untouched = schema.add_text_field("other", true, false);
            schema.set_reorder(field, true);
            for f in [field, untouched] {
                schema.set_positions(f, crate::dsl::PositionMode::TokenPosition);
            }
            let config = IndexConfig {
                num_threads: 1,
                num_indexing_threads: 1,
                posting_codec: Some(codec),
                posting_impact_bounds: compact,
                compact_text: compact,
                quantized_norms: compact,
                posting_ratio_bounds: true,
                merge_policy: Box::new(crate::merge::NoMergePolicy),
                ..Default::default()
            };
            let dir = RamDirectory::new();
            let mut writer = IndexWriter::create(dir.clone(), schema.build(), config.clone())
                .await
                .unwrap();
            for i in 0..512 {
                let mut doc = Document::new();
                doc.add_text(
                    field,
                    if i % 2 == 0 {
                        "alpha beta alpha"
                    } else {
                        "gamma delta gamma"
                    },
                );
                doc.add_text(untouched, "fixed token fixed");
                writer.add_document(doc).unwrap();
            }
            writer.commit().await.unwrap();
            let queries: Vec<Box<dyn Query>> = vec![
                Box::new(TermQuery::text(field, "alpha")),
                Box::new(PhraseQuery::text(field, "alpha beta")),
            ];
            let mut baseline = None;
            let mut unchanged = None;
            let mut unchanged_norms = None;
            for reordered in [false, true] {
                if reordered {
                    writer.reorder().await.unwrap();
                }
                let index = Index::open(dir.clone(), config.clone()).await.unwrap();
                let reader = index.reader().await.unwrap();
                let searcher = reader.searcher().await.unwrap();
                let segment = &searcher.segment_readers()[0];
                if !reordered {
                    let cancelled = std::sync::atomic::AtomicBool::new(true);
                    assert!(matches!(
                        plan_text_reorders(
                            segment,
                            segment.schema(),
                            usize::MAX,
                            crate::segment::BpBudget::default(),
                            Some(&cancelled),
                            None
                        )
                        .await,
                        Err(crate::Error::IndexClosed)
                    ));
                    let error = plan_text_reorders(
                        segment,
                        segment.schema(),
                        1,
                        crate::segment::BpBudget::default(),
                        None,
                        None,
                    )
                    .await
                    .err()
                    .expect("the graph must reject insufficient scratch before allocation");
                    assert!(error.to_string().contains("budget"));
                }
                let lengths = segment.doc_lengths(untouched).unwrap();
                let norm_snapshot = (lengths.is_quantized(), lengths.length_bytes().to_vec());
                if let Some(expected) = &unchanged_norms {
                    assert_eq!(
                        &norm_snapshot, expected,
                        "untouched norm encoding must be copied"
                    );
                } else {
                    unchanged_norms = Some(norm_snapshot);
                }
                let mut payloads = Vec::new();
                for (key, info) in segment.all_terms().await.unwrap() {
                    let (offset, len) = info.external_info().unwrap();
                    let posting = segment.read_postings(offset, len).await.unwrap();
                    let list = BlockPostingList::deserialize(posting.as_slice()).unwrap();
                    let (offset, len) = info.position_info().unwrap();
                    let positions = segment
                        .read_position_bytes(offset, len)
                        .await
                        .unwrap()
                        .unwrap();
                    let parsed = TermPositions::open(positions.clone()).unwrap();
                    assert_eq!(
                        list.has_compact_headers(),
                        compact && codec != PostingCodec::Pfor,
                        "posting layout after reorder={reordered}"
                    );
                    assert_eq!(
                        parsed.has_compact_directory(),
                        compact,
                        "position layout after reorder={reordered}"
                    );
                    if field_of(&key) == untouched.0 {
                        payloads.push((
                            key,
                            posting.as_slice().to_vec(),
                            positions.as_slice().to_vec(),
                        ));
                    }
                }
                if let Some(expected) = &unchanged {
                    assert_eq!(&payloads, expected);
                } else {
                    unchanged = Some(payloads);
                }
                let mut snapshots = Vec::new();
                for query in &queries {
                    let mut top = TopKCollector::with_positions(512);
                    let mut count = CountCollector::new();
                    collect_segment(segment, query.as_ref(), &mut (&mut top, &mut count))
                        .await
                        .unwrap();
                    snapshots.push((
                        count.count(),
                        top.into_sorted_results()
                            .into_iter()
                            .map(|h| {
                                (
                                    h.doc_id,
                                    h.score.to_bits(),
                                    serde_json::to_value(h.positions).unwrap(),
                                )
                            })
                            .collect::<Vec<_>>(),
                    ));
                }
                if let Some(expected) = &baseline {
                    assert_eq!(&snapshots, expected);
                } else {
                    baseline = Some(snapshots);
                }
            }
            writer.shutdown().await.unwrap();
        }
    }
}
