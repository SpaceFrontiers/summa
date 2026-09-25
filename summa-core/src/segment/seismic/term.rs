//! Coordinate-transposed nomination summaries, borrowed directly from the run.
use super::*;
use crate::structures::fast_field::bitpack_read;
#[cfg(any(feature = "native", feature = "wasm", test))]
use crate::structures::fast_field::{bitpack_write, bits_needed_u64};

#[cfg(any(feature = "native", feature = "wasm", test))]
const TERM_MAGIC: u32 = 0x34544d53;
// Paired experiments persist their layout; readers never depend on this choice.
#[cfg(any(feature = "native", feature = "wasm", test))]
const LOCAL_OCCURRENCES: bool = false;
const TERM_FOOTER: usize = 40;
const BLOCKED: u8 = 255;
const CLUSTER_ENTRY: usize = 12;

#[derive(Clone, Copy)]
struct Layout {
    clusters: usize,
    rows: usize,
    dimensions: usize,
    #[cfg(test)]
    occurrences: usize,
    limit: u32,
    dimension_bits: u8,
    offset_bits: u8,
    cluster_bits: u8,
    occurrence_mode: u8,
    row_offset: usize,
    dimension_offset: usize,
    end_offset: usize,
    id_offset: usize,
    value_offset: usize,
}
#[cfg(any(feature = "native", feature = "wasm", test))]
fn packed_bytes(count: usize, width: u8) -> Option<usize> {
    count.checked_mul(usize::from(width)).map(|n| n.div_ceil(8))
}
impl Layout {
    // Immutable term payloads are produced by our writer. Read their layout
    // directly; full payload validation belongs to writer/codec tests.
    fn read(bytes: &[u8]) -> Self {
        let f = &bytes[bytes.len() - TERM_FOOTER..];
        let clusters = u32_at(f, 4) as usize;
        let rows = u32_at(f, 8) as usize;
        let row_offset = clusters * CLUSTER_ENTRY;
        let dimension_offset = row_offset + rows * 4;
        let end_offset = dimension_offset + u32_at(f, 28) as usize;
        let id_offset = end_offset + u32_at(f, 32) as usize;
        Self {
            clusters,
            rows,
            dimensions: u32_at(f, 12) as usize,
            #[cfg(test)]
            occurrences: u32_at(f, 16) as usize,
            limit: u32_at(f, 20),
            dimension_bits: f[24],
            offset_bits: f[25],
            cluster_bits: f[26],
            occurrence_mode: f[27],
            row_offset,
            dimension_offset,
            end_offset,
            id_offset,
            value_offset: id_offset + u32_at(f, 36) as usize,
        }
    }
    #[cfg(test)]
    fn validate(bytes: &[u8]) -> Result<Self> {
        let footer = bytes
            .len()
            .checked_sub(TERM_FOOTER)
            .ok_or_else(|| corrupt("missing term footer"))?;
        let f = &bytes[footer..];
        let clusters = u32_at(f, 4) as usize;
        let rows = u32_at(f, 8) as usize;
        let dimensions = u32_at(f, 12) as usize;
        let occurrences = u32_at(f, 16) as usize;
        let limit = u32_at(f, 20);
        let dimension_bits = f[24];
        let offset_bits = f[25];
        let cluster_bits = f[26];
        let occurrence_mode = f[27];
        let id_bytes = u32_at(f, 36) as usize;
        if u32_at(f, 0) != TERM_MAGIC
            || occurrence_mode > 2
            || (occurrence_mode != 0 && (clusters > 64 || dimensions == 0))
            || (occurrence_mode == 0 && packed_bytes(occurrences, cluster_bits) != Some(id_bytes))
            || (occurrence_mode != 0 && id_bytes < dimensions.div_ceil(occurrences::GROUP) * 4)
            || clusters > rows
            || rows > 65_536
            || (dimension_bits > 32 && dimension_bits != BLOCKED)
            || (offset_bits > 32 && offset_bits != BLOCKED)
            || cluster_bits != bits_needed_u64(clusters.saturating_sub(1) as u64)
            || (dimension_bits == 0 && dimensions != limit as usize)
            || (dimension_bits != 0 && dimensions > limit as usize)
            || (offset_bits == 0 && occurrences != dimensions)
            || (offset_bits == 0 && dimensions != 0 && (clusters != 1 || dimension_bits == 0))
            || (clusters == 0 && (rows != 0 || dimensions != 0 || occurrences != 0))
        {
            return Err(corrupt("invalid term layout"));
        }
        let dimension_bytes = u32_at(f, 28) as usize;
        let end_bytes = u32_at(f, 32) as usize;
        if (dimension_bits != BLOCKED
            && packed_bytes(dimensions, dimension_bits) != Some(dimension_bytes))
            || (offset_bits != BLOCKED && packed_bytes(dimensions, offset_bits) != Some(end_bytes))
        {
            return Err(corrupt("invalid summary array lengths"));
        }
        let extent = || -> Option<_> {
            let row_offset = clusters.checked_mul(CLUSTER_ENTRY)?;
            let dimension_offset = row_offset.checked_add(rows.checked_mul(4)?)?;
            let end_offset = dimension_offset.checked_add(dimension_bytes)?;
            let id_offset = end_offset.checked_add(end_bytes)?;
            let value_offset = id_offset.checked_add(id_bytes)?;
            (value_offset.checked_add(occurrences)? == footer).then_some((
                row_offset,
                dimension_offset,
                end_offset,
                id_offset,
                value_offset,
            ))
        };
        let (row_offset, dimension_offset, end_offset, id_offset, value_offset) =
            extent().ok_or_else(|| corrupt("invalid term extents"))?;
        Ok(Self {
            clusters,
            rows,
            dimensions,
            occurrences,
            limit,
            dimension_bits,
            offset_bits,
            cluster_bits,
            occurrence_mode,
            row_offset,
            dimension_offset,
            end_offset,
            id_offset,
            value_offset,
        })
    }
}

#[derive(Clone, Copy)]
pub(crate) struct TermRun<'a> {
    bytes: &'a [u8],
    row_base: u32,
    layout: Layout,
}
impl<'a> TermRun<'a> {
    pub(super) fn new(bytes: &'a [u8], row_base: u32) -> Self {
        Self {
            bytes,
            row_base,
            layout: Layout::read(bytes),
        }
    }
    fn layout(self) -> Layout {
        self.layout
    }
    pub(crate) fn cluster_count(self) -> usize {
        self.layout.clusters
    }
    pub(super) fn nomination_count(self) -> usize {
        self.layout.rows
    }
    pub(crate) fn clusters(self) -> impl Iterator<Item = ClusterView<'a>> {
        let layout = self.layout();
        (0..layout.clusters).map(move |cluster| self.cluster_with_layout(cluster, layout))
    }
    #[cfg(test)]
    pub(crate) fn cluster(self, cluster: usize) -> ClusterView<'a> {
        self.cluster_with_layout(cluster, self.layout())
    }
    fn cluster_with_layout(self, cluster: usize, layout: Layout) -> ClusterView<'a> {
        assert!(cluster < layout.clusters);
        let start = if cluster == 0 {
            0
        } else {
            u32_at(self.bytes, (cluster - 1) * CLUSTER_ENTRY) as usize
        };
        let end = u32_at(self.bytes, cluster * CLUSTER_ENTRY) as usize;
        ClusterView {
            rows: &self.bytes[layout.row_offset + start * 4..layout.row_offset + end * 4],
            row_base: self.row_base,
        }
    }
    fn dimension(self, layout: Layout, i: usize) -> u32 {
        if layout.dimension_bits == 0 {
            i as u32
        } else if layout.dimension_bits == BLOCKED {
            crate::structures::monotone::Monotone::new(
                &self.bytes[layout.dimension_offset..layout.end_offset],
                layout.dimensions,
            )
            .get(i)
        } else {
            bitpack_read(
                &self.bytes[layout.dimension_offset..layout.end_offset],
                layout.dimension_bits,
                i,
            ) as u32
        }
    }
    fn end(self, layout: Layout, i: usize) -> usize {
        if layout.offset_bits == 0 {
            i + 1
        } else if layout.offset_bits == BLOCKED {
            crate::structures::monotone::Monotone::new(
                &self.bytes[layout.end_offset..layout.id_offset],
                layout.dimensions,
            )
            .get(i) as usize
        } else {
            bitpack_read(
                &self.bytes[layout.end_offset..layout.id_offset],
                layout.offset_bits,
                i,
            ) as usize
        }
    }
    fn decode_ends(self, block: usize, out: &mut [u32]) {
        let layout = self.layout;
        if layout.offset_bits == BLOCKED {
            crate::structures::monotone::Monotone::new(
                &self.bytes[layout.end_offset..layout.id_offset],
                layout.dimensions,
            )
            .decode_block(block, out);
        } else {
            for (i, value) in out.iter_mut().enumerate() {
                *value = self.end(layout, block * occurrences::GROUP + i) as u32;
            }
        }
    }
    /// Query coordinates are ascending, with duplicate coordinates retained.
    /// The caller admits the cluster count before allocating the score array.
    pub(crate) fn score_summaries(self, query: &[(u32, f32)], scores: &mut [f32]) {
        let layout = self.layout();
        assert_eq!(scores.len(), layout.clusters);
        scores.fill(0.0);
        let ids = &self.bytes[layout.id_offset..layout.value_offset];
        let values = &self.bytes[layout.value_offset..self.bytes.len() - TERM_FOOTER];
        let mut cached_block = usize::MAX;
        let mut ends = [0u32; occurrences::GROUP];
        let mut offsets = [0usize; occurrences::GROUP + 1];
        let mut rank_bytes = &[][..];
        let mut codes = &[][..];
        let mut group_start = 0;
        let mut decoded = [0u32; 64];
        for &(dimension, weight) in query {
            if weight == 0.0 || dimension >= layout.limit {
                continue;
            }
            let i = if layout.dimension_bits == 0 {
                dimension as usize
            } else if layout.dimension_bits == BLOCKED {
                let view = crate::structures::monotone::Monotone::new(
                    &self.bytes[layout.dimension_offset..layout.end_offset],
                    layout.dimensions,
                );
                let at = view.lower_bound(dimension);
                if at == layout.dimensions || view.get(at) != dimension {
                    continue;
                }
                at
            } else {
                let mut low = 0;
                let mut high = layout.dimensions;
                while low < high {
                    let mid = low + (high - low) / 2;
                    if self.dimension(layout, mid) < dimension {
                        low = mid + 1;
                    } else {
                        high = mid;
                    }
                }
                if low == layout.dimensions || self.dimension(layout, low) != dimension {
                    continue;
                }
                low
            };
            let mut contribute = |cluster: usize, code: u8| {
                let metadata = cluster * CLUSTER_ENTRY;
                let scale = f32::from_bits(u32_at(self.bytes, metadata + 4));
                let minimum = f32::from_bits(u32_at(self.bytes, metadata + 8));
                let value = f32::from(code) * scale + minimum;
                scores[cluster] = scores[cluster].algebraic_add(value.algebraic_mul(weight.abs()));
            };
            if layout.occurrence_mode == 0 {
                let start = if i == 0 { 0 } else { self.end(layout, i - 1) };
                let end = self.end(layout, i);
                for (position, &code) in values.iter().enumerate().take(end).skip(start) {
                    contribute(
                        bitpack_read(ids, layout.cluster_bits, position) as usize,
                        code,
                    );
                }
            } else {
                let block = i / occurrences::GROUP;
                if cached_block != block {
                    let count =
                        (layout.dimensions - block * occurrences::GROUP).min(occurrences::GROUP);
                    self.decode_ends(block, &mut ends[..count]);
                    group_start = if block == 0 {
                        0
                    } else {
                        self.end(layout, block * occurrences::GROUP - 1)
                    };
                    let len = occurrences::bit_offsets(
                        &ends[..count],
                        group_start as u32,
                        layout.clusters,
                        &mut offsets,
                    );
                    let at = layout.id_offset
                        + layout.dimensions.div_ceil(occurrences::GROUP) * 4
                        + u32_at(self.bytes, layout.id_offset + block * 4) as usize;
                    rank_bytes = &self.bytes[at..at + len];
                    codes = if layout.occurrence_mode == 2 {
                        &self.bytes[at + len..at + len + ends[count - 1] as usize - group_start]
                    } else {
                        &values[group_start..ends[count - 1] as usize]
                    };
                    cached_block = block;
                }
                let local = i % occurrences::GROUP;
                let start = if local == 0 {
                    group_start
                } else {
                    ends[local - 1] as usize
                };
                let end = ends[local] as usize;
                let rank = crate::structures::combination::read(
                    rank_bytes,
                    offsets[local],
                    (offsets[local + 1] - offsets[local]) as u8,
                );
                crate::structures::combination::decode(
                    rank,
                    layout.clusters,
                    &mut decoded[..end - start],
                );
                for (&cluster, &code) in decoded[..end - start]
                    .iter()
                    .zip(&codes[start - group_start..end - group_start])
                {
                    contribute(cluster as usize, code);
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct ClusterView<'a> {
    rows: &'a [u8],
    row_base: u32,
}
impl ClusterView<'_> {
    #[cfg(test)]
    pub(crate) fn row_count(&self) -> usize {
        self.rows.len() / 4
    }
    pub(crate) fn rows(&self) -> impl Iterator<Item = u32> + '_ {
        self.rows
            .chunks_exact(4)
            .map(|b| u32_at(b, 0) + self.row_base)
    }
}

#[cfg(test)]
fn admit(bytes: &[u8], expected_clusters: u32) -> Result<()> {
    let layout = Layout::validate(bytes)?;
    if layout.clusters != expected_clusters as usize {
        return Err(corrupt("term cluster count disagrees"));
    }
    for (tag, array) in [
        (
            layout.dimension_bits,
            &bytes[layout.dimension_offset..layout.end_offset],
        ),
        (
            layout.offset_bits,
            &bytes[layout.end_offset..layout.id_offset],
        ),
    ] {
        if tag == BLOCKED {
            crate::structures::monotone::admit(array, layout.dimensions)?;
        }
    }
    let run = TermRun {
        bytes,
        row_base: 0,
        layout,
    };
    let mut previous_row = 0;
    for cluster in 0..layout.clusters {
        let at = cluster * CLUSTER_ENTRY;
        let end = u32_at(bytes, at) as usize;
        let scale = f32::from_bits(u32_at(bytes, at + 4));
        let minimum = f32::from_bits(u32_at(bytes, at + 8));
        if end <= previous_row
            || end > layout.rows
            || !scale.is_finite()
            || scale < 0.0
            || !minimum.is_finite()
            || minimum < 0.0
        {
            return Err(corrupt("invalid cluster directory"));
        }
        previous_row = end;
    }
    if previous_row != layout.rows {
        return Err(corrupt("unowned nomination rows"));
    }
    let mut previous_dimension = None;
    let mut previous_end = 0;
    const BLOCK: usize = crate::structures::monotone::BLOCK;
    let mut dimensions = [0u32; BLOCK];
    let mut ends = [0u32; BLOCK];
    for start in (0..layout.dimensions).step_by(BLOCK) {
        let count = (layout.dimensions - start).min(BLOCK);
        if layout.dimension_bits == BLOCKED {
            crate::structures::monotone::Monotone::new(
                &bytes[layout.dimension_offset..layout.end_offset],
                layout.dimensions,
            )
            .decode_block(start / BLOCK, &mut dimensions[..count]);
        } else {
            for (i, dim) in dimensions[..count].iter_mut().enumerate() {
                *dim = run.dimension(layout, start + i);
            }
        }
        if layout.offset_bits == BLOCKED {
            crate::structures::monotone::Monotone::new(
                &bytes[layout.end_offset..layout.id_offset],
                layout.dimensions,
            )
            .decode_block(start / BLOCK, &mut ends[..count]);
        } else {
            for (i, end) in ends[..count].iter_mut().enumerate() {
                *end = run.end(layout, start + i) as u32;
            }
        }
        for (&dim, &end) in dimensions[..count].iter().zip(&ends[..count]) {
            let end = end as usize;
            if dim >= layout.limit
                || previous_dimension.is_some_and(|old| old >= dim)
                || end < previous_end
                || end > layout.occurrences
                || (layout.dimension_bits != 0 && end == previous_end)
            {
                return Err(corrupt("invalid summary directory"));
            }
            previous_dimension = Some(dim);
            previous_end = end;
        }
    }
    if previous_end != layout.occurrences {
        return Err(corrupt("unowned summary occurrences"));
    }
    let mut previous = 0;
    let mut payload_offset = 0;
    let groups = layout.dimensions.div_ceil(occurrences::GROUP);
    let mut offsets = [0usize; occurrences::GROUP + 1];
    for block in 0..groups {
        let count = (layout.dimensions - block * occurrences::GROUP).min(occurrences::GROUP);
        run.decode_ends(block, &mut ends[..count]);
        if layout.occurrence_mode == 0 {
            let ids = &bytes[layout.id_offset..layout.value_offset];
            for &end in &ends[..count] {
                let mut last = None;
                for i in previous..end as usize {
                    let id = bitpack_read(ids, layout.cluster_bits, i) as usize;
                    if id >= layout.clusters || last.is_some_and(|old| id <= old) {
                        return Err(corrupt("invalid summary cluster IDs"));
                    }
                    last = Some(id);
                }
                previous = end as usize;
            }
        } else {
            let len = occurrences::bit_offsets(
                &ends[..count],
                previous as u32,
                layout.clusters,
                &mut offsets,
            );
            let next = ends[count - 1] as usize;
            if u32_at(bytes, layout.id_offset + block * 4) as usize != payload_offset {
                return Err(corrupt("invalid summary subset checkpoint"));
            }
            let at = layout.id_offset + groups * 4 + payload_offset;
            let limit = if layout.occurrence_mode == 2 {
                bytes.len() - TERM_FOOTER
            } else {
                layout.value_offset
            };
            let block_len = len
                + if layout.occurrence_mode == 2 {
                    next - previous
                } else {
                    0
                };
            if at > limit || block_len > limit - at {
                return Err(corrupt("truncated summary subset block"));
            }
            let ranks = &bytes[at..at + len];
            for (i, &end) in ends[..count].iter().enumerate() {
                let rank = crate::structures::combination::read(
                    ranks,
                    offsets[i],
                    (offsets[i + 1] - offsets[i]) as u8,
                );
                if !crate::structures::combination::valid(
                    rank,
                    layout.clusters,
                    end as usize - previous,
                ) {
                    return Err(corrupt("invalid summary subset rank"));
                }
                previous = end as usize;
            }
            let padding = offsets[count] % 8;
            if padding != 0 && ranks[len - 1] >> padding != 0 {
                return Err(corrupt("invalid summary subset padding"));
            }
            payload_offset += block_len;
        }
    }
    if layout.occurrence_mode != 0 {
        let size = layout.value_offset - layout.id_offset
            + if layout.occurrence_mode == 2 {
                layout.occurrences
            } else {
                0
            };
        if groups * 4 + payload_offset != size {
            return Err(corrupt("unowned summary subset bytes"));
        }
    }
    Ok(())
}

#[cfg(any(feature = "native", feature = "wasm", test))]
fn encode_directory(values: &[u64], bits: &mut u8) -> Result<Vec<u8>> {
    let mut packed = Vec::new();
    bitpack_write(values, *bits, &mut packed);
    if *bits != 0 {
        let blocked = crate::structures::monotone::encode(values)?;
        if blocked.len() < packed.len() {
            *bits = BLOCKED;
            return Ok(blocked);
        }
    }
    Ok(packed)
}

/// One term's transient transpose. Occurrences never outnumber coordinates in
/// its selected rows; maintenance includes these buffers in its scratch charge.
#[cfg(any(feature = "native", feature = "wasm", test))]
#[derive(Default)]
pub(super) struct Encoder {
    rows: Vec<u32>,
    clusters: Vec<(u32, [u8; 8])>,
    occurrences: Vec<(u32, u32, u8)>,
}
#[cfg(any(feature = "native", feature = "wasm", test))]
impl Encoder {
    pub(super) fn push(&mut self, rows: &[u32], summary: &[(u32, f32)]) -> Result<()> {
        let cluster = self.clusters.len() as u32;
        self.rows.extend_from_slice(rows);
        let weights: Vec<_> = summary.iter().map(|p| p.1).collect();
        let encoded = crate::structures::postings::encode_sparse_weights(
            &weights,
            WeightQuantization::UInt8,
        )?;
        self.clusters
            .push((self.rows.len() as u32, encoded[..8].try_into().unwrap()));
        self.occurrences.extend(
            summary
                .iter()
                .zip(&encoded[8..])
                .map(|(&(dim, _), &code)| (dim, cluster, code)),
        );
        Ok(())
    }
    pub(super) fn finish(self) -> Result<(Vec<u8>, u32)> {
        self.finish_layout(LOCAL_OCCURRENCES)
    }
    fn finish_layout(mut self, local: bool) -> Result<(Vec<u8>, u32)> {
        let occurrence_count = u32::try_from(self.occurrences.len())
            .map_err(|_| corrupt("too many summary coordinates"))?;
        self.occurrences
            .sort_unstable_by_key(|&(dim, cluster, _)| (dim, cluster));
        let dimensions = self.occurrences.chunk_by(|a, b| a.0 == b.0).count();
        let mut dims = Vec::with_capacity(dimensions);
        let mut ends = Vec::with_capacity(dimensions);
        let mut end = 0u64;
        for group in self.occurrences.chunk_by(|a, b| a.0 == b.0) {
            dims.push(u64::from(group[0].0));
            end += group.len() as u64;
            ends.push(end);
        }
        let limit = dims.last().map_or(0, |&dim| dim + 1);
        let mut dimension_bits = bits_needed_u64(limit.saturating_sub(1)).max(1);
        let mut offset_bits = if self.clusters.len() == 1 {
            0
        } else {
            bits_needed_u64(end)
        };
        let sparse_bytes = packed_bytes(dimensions, dimension_bits)
            .and_then(|bytes| {
                packed_bytes(dimensions, offset_bits).and_then(|ends| bytes.checked_add(ends))
            })
            .ok_or_else(|| corrupt("summary directory exceeds address space"))?;
        let dense_bytes = packed_bytes(limit as usize, bits_needed_u64(end));
        if self.clusters.len() != 1
            && limit <= (dimensions as u64).saturating_mul(2)
            && dense_bytes.is_some_and(|bytes| bytes < sparse_bytes)
        {
            let mut dense_ends = Vec::with_capacity(limit as usize);
            let mut previous = 0;
            for (&dimension, &end) in dims.iter().zip(&ends) {
                dense_ends.resize(dimension as usize, previous);
                dense_ends.push(end);
                previous = end;
            }
            ends = dense_ends;
            dims.clear();
            dimension_bits = 0;
            offset_bits = bits_needed_u64(end);
        }
        if self.clusters.is_empty() {
            dimension_bits = 0;
        }
        let cluster_bits = bits_needed_u64(self.clusters.len().saturating_sub(1) as u64);
        let mut output = Vec::new();
        for &(row_end, ref quantizer) in &self.clusters {
            put32(&mut output, row_end)?;
            output.extend_from_slice(quantizer);
        }
        for &row in &self.rows {
            put32(&mut output, row)?;
        }
        let dimension_bytes = encode_directory(&dims, &mut dimension_bits)?;
        let end_bytes = encode_directory(&ends, &mut offset_bits)?;
        output.extend_from_slice(&dimension_bytes);
        output.extend_from_slice(&end_bytes);
        drop(dims);
        let ids: Vec<_> = self.occurrences.iter().map(|p| u64::from(p.1)).collect();
        let weights: Vec<_> = self.occurrences.iter().map(|p| p.2).collect();
        let mut id_bytes = Vec::new();
        bitpack_write(&ids, cluster_bits, &mut id_bytes);
        let mut occurrence_mode = 0;
        if self.clusters.len() <= 64 && !ends.is_empty() {
            let candidate = occurrences::encode(&ids, &ends, &weights, self.clusters.len(), local)?;
            let candidate_ids = candidate.len() - if local { weights.len() } else { 0 };
            if candidate_ids < id_bytes.len() {
                id_bytes = candidate;
                occurrence_mode = if local { 2 } else { 1 };
            }
        }
        let id_length = id_bytes.len()
            - if occurrence_mode == 2 {
                weights.len()
            } else {
                0
            };
        output.extend(id_bytes);
        if occurrence_mode != 2 {
            output.extend(weights);
        }
        for value in [
            TERM_MAGIC,
            self.clusters.len() as u32,
            self.rows.len() as u32,
            if dimension_bits == 0 {
                limit as u32
            } else {
                dimensions as u32
            },
            occurrence_count,
            u32::try_from(limit).map_err(|_| corrupt("summary dimension overflow"))?,
        ] {
            put32(&mut output, value)?;
        }
        output.extend_from_slice(&[dimension_bits, offset_bits, cluster_bits, occurrence_mode]);
        put32(
            &mut output,
            u32::try_from(dimension_bytes.len())
                .map_err(|_| corrupt("summary dimensions exceed address space"))?,
        )?;
        put32(
            &mut output,
            u32::try_from(end_bytes.len())
                .map_err(|_| corrupt("summary ends exceed address space"))?,
        )?;
        put32(
            &mut output,
            u32::try_from(id_length).map_err(|_| corrupt("summary IDs exceed address space"))?,
        )?;
        Ok((output, self.clusters.len() as u32))
    }
}

#[cfg(any(feature = "native", test))]
pub(super) fn remap_rows(bytes: &mut [u8], rows: &[u32]) {
    let layout = Layout::read(bytes);
    for encoded in bytes[layout.row_offset..layout.dimension_offset].chunks_exact_mut(4) {
        let row = rows[u32_at(encoded, 0) as usize];
        encoded.copy_from_slice(&row.to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_summaries(summaries: &[Vec<(u32, f32)>], query: &[(u32, f32)]) -> Vec<u8> {
        let mut encoder = Encoder::default();
        let mut expected = Vec::new();
        for (cluster, summary) in summaries.iter().enumerate() {
            encoder.push(&[cluster as u32], summary).unwrap();
            let weights: Vec<_> = summary.iter().map(|p| p.1).collect();
            let encoded = crate::structures::postings::encode_sparse_weights(
                &weights,
                WeightQuantization::UInt8,
            )
            .unwrap();
            let mut score = 0.0f32;
            for (i, &(dimension, _)) in summary.iter().enumerate() {
                let value = crate::structures::postings::decode_sparse_weight_at(
                    &encoded,
                    WeightQuantization::UInt8,
                    i,
                );
                for &(_, weight) in query.iter().filter(|&&(dim, _)| dim == dimension) {
                    score = score.algebraic_add(value.algebraic_mul(weight.abs()));
                }
            }
            expected.push(score);
        }
        let (bytes, clusters) = encoder.finish().unwrap();
        admit(&bytes, clusters).unwrap();
        let run = TermRun::new(&bytes, 7);
        let mut scores = vec![f32::NAN; run.cluster_count()];
        run.score_summaries(query, &mut scores);
        assert_eq!(scores, expected);
        for (cluster, view) in run.clusters().enumerate() {
            assert_eq!(view.rows().collect::<Vec<_>>(), vec![cluster as u32 + 7]);
            assert_eq!(run.cluster(cluster).row_count(), 1);
        }
        bytes
    }

    #[test]
    fn transposed_summaries_preserve_quantized_proxy_operations_for_signed_duplicates() {
        let summaries = vec![
            vec![(0, 0.25), (7, 0.9), (1 << 24, 1.5)],
            vec![(7, 2.0)],
            vec![(0, 0.0), (1 << 24, 0.7)],
        ];
        let query = [(0, -3.0), (7, 0.25), (7, -0.75), (9, 5.0), (1 << 24, -2.0)];
        let bytes = check_summaries(&summaries, &query);
        assert!(Layout::read(&bytes).dimension_bits > 16);
    }

    #[test]
    fn dense_summary_directory_and_non_byte_aligned_cluster_ids_match_scalar_codec() {
        let summaries: Vec<_> = (0..33)
            .map(|cluster| {
                (0..257)
                    .map(|dim| (dim, ((cluster + dim) % 19) as f32 * 0.125))
                    .collect()
            })
            .collect();
        let bytes = check_summaries(&summaries, &[(0, 1.0), (1, -2.0), (128, 1.5), (256, -0.5)]);
        let layout = Layout::read(&bytes);
        assert_eq!(layout.dimension_bits, 0);
        assert_eq!(layout.cluster_bits, 6);
        assert!(bytes.len() < 33 * 257 * 3);
    }

    #[test]
    fn single_cluster_and_empty_terms_omit_unnecessary_integer_arrays() {
        let bytes = check_summaries(&[vec![(7, 1.0), (65_537, 1.0)]], &[(7, -2.0)]);
        let layout = Layout::read(&bytes);
        assert_eq!(layout.offset_bits, 0);
        assert_eq!(layout.cluster_bits, 0);
        let empty = check_summaries(&[], &[(7, 1.0)]);
        assert_eq!(empty.len(), TERM_FOOTER);
    }

    #[test]
    fn compressed_directories_preserve_quantized_scores_and_nomination_bytes() {
        let summaries: Vec<_> = (0..33)
            .map(|cluster| {
                (0..519)
                    .map(|i| (1_000_000 + i * 7, ((i + cluster) % 31) as f32 * 0.125))
                    .collect()
            })
            .collect();
        let query = [
            (1_000_000, -0.5),
            (1_000_007, 2.0),
            (1_000_007, -0.3),
            (1_000_008, 8.0),
            (1_003_626, 0.25),
            (u32::MAX, 1.0),
        ];
        let bytes = check_summaries(&summaries, &query);
        let layout = Layout::read(&bytes);
        assert_eq!(layout.dimension_bits, BLOCKED);
        assert_eq!(layout.offset_bits, BLOCKED);
        let mut remapped = bytes.clone();
        remap_rows(&mut remapped, &(0..33).map(|i| i + 100).collect::<Vec<_>>());
        assert_eq!(
            &bytes[layout.dimension_offset..],
            &remapped[layout.dimension_offset..]
        );
        for start in [layout.dimension_offset, layout.end_offset] {
            let mut corrupt = bytes.clone();
            corrupt[start + 13] = 9;
            assert!(admit(&corrupt, 33).is_err());
        }
    }

    #[test]
    fn proxy_overflow_remains_eligible_and_nomination_rows_remap_without_summary_changes() {
        let mut bytes = check_summaries(&[vec![(0, f32::MAX)]], &[(0, 2.0)]);
        let layout = Layout::read(&bytes);
        let before = bytes[layout.dimension_offset..].to_vec();
        remap_rows(&mut bytes, &[123]);
        assert_eq!(&bytes[layout.dimension_offset..], before);
        let run = TermRun::new(&bytes, 4);
        assert_eq!(run.cluster(0).rows().collect::<Vec<_>>(), vec![127]);
    }

    #[test]
    fn subset_layouts_preserve_scores_codes_rows_and_reject_invalid_payloads() {
        let summaries: Vec<Vec<_>> = (0..33)
            .map(|cluster| {
                (0..519)
                    .filter(|dim| (dim + cluster) % 7 < 3)
                    .map(|dim| (dim * 7 + 1000, ((dim + cluster) % 19) as f32 * 0.25))
                    .collect()
            })
            .collect();
        let query = [
            (1000, -0.7),
            (1007, 1.0),
            (1007, -2.0),
            (1010, 3.0),
            (2806, 0.5),
            (4626, 1.1),
        ];
        let reference = check_summaries(&summaries, &query);
        let mut expected = vec![0.0; 33];
        TermRun::new(&reference, 0).score_summaries(&query, &mut expected);
        let mut sizes = Vec::new();
        for local in [false, true] {
            let mut encoder = Encoder::default();
            for (cluster, summary) in summaries.iter().enumerate() {
                encoder.push(&[cluster as u32], summary).unwrap();
            }
            let (bytes, count) = encoder.finish_layout(local).unwrap();
            sizes.push(bytes.len());
            let layout = Layout::read(&bytes);
            assert_eq!(layout.occurrence_mode, if local { 2 } else { 1 });
            admit(&bytes, count).unwrap();
            let mut actual = vec![f32::NAN; 33];
            TermRun::new(&bytes, 0).score_summaries(&query, &mut actual);
            assert_eq!(actual, expected);
            let mut remapped = bytes.clone();
            remap_rows(&mut remapped, &(100..133).collect::<Vec<_>>());
            assert_eq!(
                &bytes[layout.dimension_offset..],
                &remapped[layout.dimension_offset..]
            );
            for cut in [0, 1, layout.id_offset, layout.value_offset, bytes.len() - 1] {
                assert!(
                    std::panic::catch_unwind(|| admit(&bytes[..cut], count))
                        .unwrap()
                        .is_err()
                );
            }
            let mut invalid = bytes.clone();
            invalid[layout.id_offset] = 1;
            assert!(admit(&invalid, count).is_err());
            let mut invalid = bytes.clone();
            let first_rank = layout.id_offset + layout.dimensions.div_ceil(occurrences::GROUP) * 4;
            invalid[first_rank..first_rank + 8].fill(255);
            assert!(admit(&invalid, count).is_err());
        }
        assert_eq!(sizes[0], sizes[1]);
    }

    #[test]
    fn invalid_term_extents_and_cluster_directories_fail_admission() {
        let bytes = check_summaries(&[vec![(7, 1.0)], vec![(8, 2.0)]], &[]);
        let mut invalid = bytes.clone();
        invalid[0..4].copy_from_slice(&0u32.to_le_bytes());
        assert!(admit(&invalid, 2).is_err());
        let mut invalid = bytes.clone();
        let footer = invalid.len() - TERM_FOOTER;
        invalid[footer + 16..footer + 20].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(admit(&invalid, 2).is_err());
        assert!(admit(&bytes[..bytes.len() - 1], 2).is_err());
        assert!(admit(&bytes, 3).is_err());
    }
}
