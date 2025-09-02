//! Field-level BP reordering of chunked text fields.
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
//! Gated by the `reorder` schema attribute on a chunked text field.

use std::io::Write;
use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::Result;
use crate::directories::{Directory, DirectoryWriter, OwnedBytes};
use crate::dsl::{Field, FieldType, Schema};
use crate::segment::builder::graph_bisection::{
    BpProgressLabel, ForwardIndex, graph_bisection_with_progress,
};
use crate::segment::chunk_map::{ChunkMapBuilder, DocLengthsColumn, write_chunk_maps};
use crate::segment::reader::SegmentReader;
use crate::segment::types::SegmentFiles;
use crate::segment::{OffsetWriter, SegmentMerger};
use crate::structures::{
    BlockPostingList, PositionStreamEncoder, PostingCodec, PostingList, SSTableWriter, TERMINATED,
    TermInfo, TermPositions,
};

/// Minimum partition of the bisection: one posting block, the pruning unit.
const MIN_PARTITION: usize = crate::structures::POSTING_BLOCK_SIZE;
const BP_ITERATIONS: usize = 20;

/// A computed permutation of one chunked text field's virtual ids.
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

fn term_df(info: &TermInfo) -> u32 {
    match info {
        TermInfo::Inline { doc_freq, .. } => u32::from(*doc_freq),
        TermInfo::External { doc_freq, .. } => *doc_freq,
    }
}

/// Decode one term's postings into `(id, tf)` pairs.
async fn read_postings(reader: &SegmentReader, info: &TermInfo) -> Result<Vec<(u32, u32)>> {
    if let Some((ids, tfs)) = info.decode_inline() {
        return Ok(ids.into_iter().zip(tfs).collect());
    }
    let (offset, len) = info.external_info().ok_or_else(|| {
        crate::Error::Corruption("term has neither inline nor external postings".into())
    })?;
    let bytes = reader.read_postings(offset, len).await?;
    let list = BlockPostingList::deserialize(&bytes)?;
    let mut it = list.iterator();
    let mut out = Vec::with_capacity(list.doc_count() as usize);
    while it.doc() != TERMINATED {
        out.push((it.doc(), it.term_freq()));
        it.advance();
    }
    Ok(out)
}

/// Plan the reorder of every chunked text field carrying the `reorder`
/// attribute that has data in this segment.
pub(crate) async fn plan_text_reorders(
    reader: &SegmentReader,
    schema: &Schema,
    memory_budget: usize,
    bp_budget: crate::segment::BpBudget,
    cancellation: Option<&std::sync::atomic::AtomicBool>,
    rayon_pool: Option<Arc<rayon::ThreadPool>>,
) -> Result<Vec<TextReorderPlan>> {
    let mut plans = Vec::new();
    for (field, entry) in schema.fields() {
        if !(entry.reorder && entry.chunked && entry.field_type == FieldType::Text) {
            continue;
        }
        if reader.chunk_map(field).is_none() {
            continue;
        }
        if let Some(plan) = plan_field(
            reader,
            schema,
            field,
            memory_budget,
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
    reader: &SegmentReader,
    schema: &Schema,
    field: Field,
    memory_budget: usize,
    bp_budget: crate::segment::BpBudget,
    cancellation: Option<&std::sync::atomic::AtomicBool>,
    rayon_pool: Option<Arc<rayon::ThreadPool>>,
) -> Result<Option<TextReorderPlan>> {
    let started = std::time::Instant::now();
    let field_name = schema.get_field_name(field).unwrap_or("?");
    let num_chunks = reader.num_chunks(field) as usize;
    if num_chunks < 2 * MIN_PARTITION {
        return Ok(None);
    }
    let prefix = field.0.to_le_bytes();

    // Pass 1: document frequencies. Terms in one chunk add no edge; the
    // highest-df terms are dropped until the postings fit the memory budget
    // (12 B per posting: the (term, chunk) pairs plus the CSR).
    let mut dfs: Vec<u32> = Vec::new();
    let mut iter = reader.term_dict_iter();
    while let Some((key, info)) = iter.next().await.map_err(crate::Error::from)? {
        if key.len() < 4 || key[..4] != prefix {
            continue;
        }
        dfs.push(term_df(&info));
    }
    if dfs.is_empty() {
        return Ok(None);
    }
    let mut by_df: Vec<(u32, u32)> = dfs
        .iter()
        .enumerate()
        .filter(|(_, df)| **df >= 2)
        .map(|(ordinal, df)| (*df, ordinal as u32))
        .collect();
    by_df.sort_unstable();
    let max_postings = (memory_budget / 12).max(1) as u64;
    let mut kept = 0u64;
    let mut active = vec![u32::MAX; dfs.len()];
    let mut compact = 0u32;
    for &(df, ordinal) in &by_df {
        if kept + u64::from(df) > max_postings {
            break;
        }
        kept += u64::from(df);
        active[ordinal as usize] = compact;
        compact += 1;
    }
    let budget_limited = (compact as usize) < by_df.len();
    if compact == 0 {
        return Ok(None);
    }

    // Pass 2: (chunk, term) pairs, then a counting sort into CSR.
    let mut pairs: Vec<(u32, u32)> = Vec::with_capacity(kept as usize);
    let mut iter = reader.term_dict_iter();
    let mut ordinal = 0u32;
    while let Some((key, info)) = iter.next().await.map_err(crate::Error::from)? {
        if key.len() < 4 || key[..4] != prefix {
            continue;
        }
        let compact_id = active[ordinal as usize];
        ordinal += 1;
        if compact_id == u32::MAX {
            continue;
        }
        if cancellation.is_some_and(|c| c.load(std::sync::atomic::Ordering::Acquire)) {
            return Err(crate::Error::IndexClosed);
        }
        for (vid, _) in read_postings(reader, &info).await? {
            if (vid as usize) < num_chunks {
                pairs.push((vid, compact_id));
            }
        }
    }
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
    let fwd = ForwardIndex::from_csr(
        terms,
        offsets,
        compact as usize,
        memory_budget,
        budget_limited,
    );

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
    let (order, converged) = match rayon_pool {
        Some(pool) => pool.install(bisect),
        None => bisect(),
    };
    if order.len() != num_chunks {
        return Err(crate::Error::Internal(format!(
            "text reorder of field '{field_name}' produced {} ids for {num_chunks} chunks",
            order.len()
        )));
    }
    let mut inverse = vec![u32::MAX; num_chunks];
    for (new, &old) in order.iter().enumerate() {
        inverse[old as usize] = new as u32;
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
    if identity {
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
pub(crate) async fn rewrite_text_files<D: Directory + DirectoryWriter>(
    dir: &D,
    reader: &SegmentReader,
    dst_files: &SegmentFiles,
    schema: &Arc<Schema>,
    plans: &[TextReorderPlan],
    posting_config: (crate::structures::IndexOptimization, PostingCodec),
    cancellation: Option<&std::sync::atomic::AtomicBool>,
) -> Result<()> {
    let (optimization, posting_codec) = posting_config;
    let started = std::time::Instant::now();
    let by_field: FxHashMap<u32, &TextReorderPlan> =
        plans.iter().map(|plan| (plan.field.0, plan)).collect();
    // New-order lengths per planned field (bounds of the rewritten lists).
    let mut new_lengths: FxHashMap<u32, Vec<u32>> = FxHashMap::default();
    for plan in plans {
        let map = reader.chunk_map(plan.field).ok_or_else(|| {
            crate::Error::Corruption("planned text field lost its chunk map".into())
        })?;
        new_lengths.insert(
            plan.field.0,
            plan.order.iter().map(|&old| map.length(old)).collect(),
        );
    }

    let merger =
        SegmentMerger::new(Arc::clone(schema)).with_posting_config(optimization, posting_codec);
    let mut postings_out = OffsetWriter::new(dir.streaming_writer_cold(&dst_files.postings).await?);
    let mut positions_out =
        OffsetWriter::new(dir.streaming_writer_cold(&dst_files.positions).await?);
    let mut term_dict_out =
        OffsetWriter::new(dir.streaming_writer_cold(&dst_files.term_dict).await?);
    let mut term_dict = SSTableWriter::<&mut OffsetWriter, TermInfo>::with_config(
        &mut term_dict_out,
        crate::structures::SSTableWriterConfig::from_optimization(optimization),
    );
    let mut buf: Vec<u8> = Vec::new();
    let mut sources: Vec<(usize, TermInfo, u32)> = Vec::with_capacity(1);
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
                    reader,
                    &info,
                    plan,
                    &new_lengths[&field_id],
                    &mut postings_out,
                    &mut positions_out,
                    &mut buf,
                    posting_codec,
                )
                .await?
            }
            None => {
                sources.clear();
                sources.push((0, info, 0));
                merger
                    .merge_term(
                        std::slice::from_ref(reader),
                        &mut sources,
                        &mut postings_out,
                        &mut positions_out,
                        &mut buf,
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

    // Chunk maps and length columns: planned fields in the new order, the
    // rest verbatim.
    let mut chunk_fields: Vec<(u32, ChunkMapBuilder)> = Vec::new();
    for (field_id, map) in reader.chunk_maps() {
        let mut builder = ChunkMapBuilder::default();
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
        chunk_fields.push((*field_id, builder));
    }
    chunk_fields.sort_by_key(|(field_id, _)| *field_id);
    let mut norm_columns: Vec<(u32, Vec<u16>, u64)> = Vec::new();
    for (field, entry) in schema.fields() {
        if entry.chunked || entry.field_type != FieldType::Text {
            continue;
        }
        if let Some(lengths) = reader.doc_lengths(field) {
            let column: Vec<u16> = (0..lengths.num_docs())
                .map(|doc| lengths.length(doc).min(u32::from(u16::MAX)) as u16)
                .collect();
            norm_columns.push((field.0, column, lengths.total_tokens()));
        }
    }
    norm_columns.sort_by_key(|(field_id, _, _)| *field_id);
    let fields: Vec<(u32, &ChunkMapBuilder)> = chunk_fields
        .iter()
        .filter(|(_, builder)| !builder.is_empty())
        .map(|(field_id, builder)| (*field_id, builder))
        .collect();
    let norms: Vec<DocLengthsColumn<'_>> = norm_columns
        .iter()
        .map(|(field_id, lengths, total)| DocLengthsColumn {
            field_id: *field_id,
            lengths,
            total_tokens: *total,
        })
        .collect();
    if !fields.is_empty() || !norms.is_empty() {
        let mut writer = dir.streaming_writer_cold(&dst_files.chunks).await?;
        write_chunk_maps(&mut *writer, &fields, &norms).map_err(crate::Error::Io)?;
        writer.finish()?;
    }

    log::info!(
        "[reorder_text] index={} rewrote {} terms ({} reordered) in {:.1}s",
        schema.index_label(),
        terms,
        reordered_terms,
        started.elapsed().as_secs_f64(),
    );
    Ok(())
}

/// Rewrite one term of a planned field: postings sorted by the new virtual
/// ids, positions re-encoded in that order, block bounds from the new
/// lengths.
#[allow(clippy::too_many_arguments)]
async fn reorder_term(
    reader: &SegmentReader,
    info: &TermInfo,
    plan: &TextReorderPlan,
    new_lengths: &[u32],
    postings_out: &mut OffsetWriter,
    positions_out: &mut OffsetWriter,
    buf: &mut Vec<u8>,
    posting_codec: PostingCodec,
) -> Result<TermInfo> {
    // (new vid, tf, positions of the old vid)
    let mut entries: Vec<(u32, u32, Vec<u32>)> = Vec::new();
    let has_positions = info.position_info().is_some();
    let positions = match info.position_info() {
        Some((offset, len)) => {
            let bytes = reader
                .read_position_bytes(offset, len)
                .await?
                .ok_or_else(|| {
                    crate::Error::Corruption(
                        "term has positions but the segment has no file".into(),
                    )
                })?;
            Some(TermPositions::open(OwnedBytes::new(bytes))?)
        }
        None => None,
    };
    if let Some((ids, tfs)) = info.decode_inline() {
        for (old, tf) in ids.into_iter().zip(tfs) {
            entries.push((plan.inverse[old as usize], tf, Vec::new()));
        }
    } else {
        let (offset, len) = info.external_info().ok_or_else(|| {
            crate::Error::Corruption("term has neither inline nor external postings".into())
        })?;
        let bytes = reader.read_postings(offset, len).await?;
        let list = BlockPostingList::deserialize(&bytes)?;
        let mut it = list.iterator();
        let mut scratch = Vec::new();
        while it.doc() != TERMINATED {
            let old = it.doc();
            let tf = it.term_freq();
            let mut pos = Vec::new();
            if let Some(positions) = &positions
                && !positions.positions_into(old, it.position_cursor(), tf, &mut scratch, &mut pos)
            {
                return Err(crate::Error::Corruption(format!(
                    "positions of chunk {old} could not be read during reorder"
                )));
            }
            entries.push((plan.inverse[old as usize], tf, pos));
            it.advance();
        }
    }
    entries.sort_unstable_by_key(|(new, _, _)| *new);

    let mut list = PostingList::with_capacity(entries.len());
    for (new, tf, _) in &entries {
        list.push(*new, *tf);
    }
    if !has_positions
        && let Some(inline) = TermInfo::try_inline_iter(
            entries.len(),
            entries.iter().map(|(new, tf, _)| (*new, *tf)),
        )
    {
        return Ok(inline);
    }
    let length_of = |vid: u32| new_lengths.get(vid as usize).copied().unwrap_or(1);
    let block = BlockPostingList::from_posting_list_with_options(
        &list,
        has_positions,
        Some(&length_of),
        posting_codec,
    )?;
    buf.clear();
    block.serialize(buf)?;
    let posting_offset = postings_out.offset();
    postings_out.write_all(buf)?;
    let posting_len = buf.len() as u64;
    if !has_positions {
        return Ok(TermInfo::external(
            posting_offset,
            posting_len,
            entries.len() as u32,
        ));
    }
    let position_offset = positions_out.offset();
    let mut encoder = PositionStreamEncoder::new(&mut *positions_out);
    for (_, _, pos) in entries.iter_mut() {
        encoder.push_doc(pos)?;
    }
    let (_, position_len) = encoder.finish()?;
    Ok(TermInfo::external_with_positions(
        posting_offset,
        posting_len,
        list.doc_count(),
        position_offset,
        position_len,
    ))
}
