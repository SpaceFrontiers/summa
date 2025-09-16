//! Portable Seismic build: bounded top postings, sampled geometric centroids,
//! and coordinate-wise maximum summaries cropped by retained energy.
//!
//! Adapted from the Seismic random-kmeans and energy-preserving summary approach.
//! Copyright (c) 2024 Rossano Venturini. MIT license; see LICENSE.seismic.
use super::*;
use rustc_hash::FxHashMap;
use std::collections::BTreeMap;
type Values = Vec<(u32, f32)>;
const ASSIGNMENT_COORDINATES: usize = 15;

/// Reused by every term's centroid assignment; no per-row heap allocations.
pub(super) struct AssignmentCoordinates {
    rows: Vec<[(u32, f32); ASSIGNMENT_COORDINATES]>,
}
impl AssignmentCoordinates {
    #[cfg(any(feature = "native", test))]
    pub(super) const BYTES_PER_ROW: usize =
        std::mem::size_of::<[(u32, f32); ASSIGNMENT_COORDINATES]>();

    pub(super) fn prepare(rows: &[Values]) -> Self {
        let rows = rows
            .iter()
            .map(|values| {
                let mut strongest = [(0, 0.0); ASSIGNMENT_COORDINATES];
                let mut len = 0;
                for &value in values {
                    let position = strongest[..len]
                        .partition_point(|current| assignment_order(current, &value).is_lt());
                    if position < ASSIGNMENT_COORDINATES {
                        let end = len.min(ASSIGNMENT_COORDINATES - 1);
                        strongest.copy_within(position..end, position + 1);
                        strongest[position] = value;
                        len = (len + 1).min(ASSIGNMENT_COORDINATES);
                    }
                }
                strongest
            })
            .collect();
        Self { rows }
    }

    fn row(&self, row: u32, coordinate_count: usize) -> &[(u32, f32)] {
        &self.rows[row as usize][..coordinate_count.min(ASSIGNMENT_COORDINATES)]
    }
}

fn assignment_order(a: &(u32, f32), b: &(u32, f32)) -> std::cmp::Ordering {
    b.1.abs().total_cmp(&a.1.abs()).then(a.0.cmp(&b.0))
}

fn nomination_order(a: &(u32, f32), b: &(u32, f32)) -> std::cmp::Ordering {
    b.1.total_cmp(&a.1).then(a.0.cmp(&b.0))
}

fn retain_nominations(candidates: &mut Vec<(u32, f32)>, keep: usize) {
    if keep < candidates.len() {
        candidates.select_nth_unstable_by(keep, nomination_order);
        candidates.truncate(keep);
    }
    candidates.sort_unstable_by(nomination_order);
}

#[cfg(test)]
pub(crate) fn build_blob(
    postings: FxHashMap<u32, Vec<(u32, u16, f32)>>,
    config: &SparseVectorConfig,
    writer: &mut impl SeismicWriter,
) -> Result<OutputLengths> {
    build_blob_with_keys(postings, &[], config, writer)
}

pub(crate) fn build_blob_with_keys(
    postings: FxHashMap<u32, Vec<(u32, u16, f32)>>,
    keys: &[(u32, u16)],
    config: &SparseVectorConfig,
    writer: &mut impl SeismicWriter,
) -> Result<OutputLengths> {
    config.seismic.validate().map_err(Error::Schema)?;
    let mut by_key: BTreeMap<LogicalUnit, Values> = BTreeMap::new();
    for &(doc, ordinal) in keys {
        by_key.entry(LogicalUnit { doc, ordinal }).or_default();
    }
    let mut observed = 0;
    for (dim, values) in postings {
        observed = observed.max(
            dim.checked_add(1)
                .ok_or_else(|| corrupt("dimension cannot be represented"))?,
        );
        for (doc, ordinal, weight) in values {
            if !weight.is_finite() {
                return Err(Error::Schema("non-finite sparse weight".into()));
            }
            by_key
                .entry(LogicalUnit { doc, ordinal })
                .or_default()
                .push((dim, weight));
        }
    }
    let dims = config.dims.unwrap_or(observed.max(1));
    if observed > dims {
        return Err(Error::Schema(
            "sparse dimension exceeds configured vocabulary".into(),
        ));
    }
    u32::try_from(by_key.len()).map_err(|_| corrupt("too many forward vectors"))?;
    let mut rows = Vec::with_capacity(by_key.len());
    let mut directory = Vec::with_capacity(by_key.len());
    let mut offset = 0u64;
    for (key, mut values) in by_key {
        values.sort_unstable_by_key(|p| p.0);
        let mut n = 0;
        for i in 0..values.len() {
            if n > 0 && values[n - 1].0 == values[i].0 {
                values[n - 1].1 += values[i].1;
            } else {
                values[n] = values[i];
                n += 1;
            }
        }
        values.truncate(n);
        if values.iter().any(|p| !p.1.is_finite()) {
            return Err(Error::Schema(
                "duplicate sparse weight sum overflows".into(),
            ));
        }
        let weights: Vec<_> = values.iter().map(|p| p.1).collect();
        let encoded = crate::structures::postings::encode_sparse_weights(
            &weights,
            config.weight_quantization,
        )?;
        let len = values
            .len()
            .checked_mul(4)
            .and_then(|n| n.checked_add(encoded.len()))
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| corrupt("forward vector too large"))?;
        let mut bytes = Vec::with_capacity(len as usize);
        for &(dim, _) in &values {
            put32(&mut bytes, dim)?;
        }
        bytes.extend_from_slice(&encoded);
        let quantized = ForwardVector::new(&bytes, values.len(), config.weight_quantization);
        for (dest, (_, weight)) in values.iter_mut().zip(quantized.iter()) {
            dest.1 = weight;
        }
        if values.iter().any(|p| !p.1.is_finite()) {
            return Err(Error::Schema(
                "sparse quantization overflows configured precision".into(),
            ));
        }
        writer.root().write_all(&bytes)?;
        directory.push((key, offset, len, values.len() as u32));
        offset += u64::from(len);
        rows.push(values);
    }
    finish_blob(rows, directory, offset, dims, config, writer)
}

pub(super) fn finish_blob(
    rows: Vec<Values>,
    directory: Vec<(LogicalUnit, u64, u32, u32)>,
    mut offset: u64,
    dims: u32,
    config: &SparseVectorConfig,
    writer: &mut impl SeismicWriter,
) -> Result<OutputLengths> {
    let count = u32::try_from(rows.len()).map_err(|_| corrupt("too many forward vectors"))?;
    let row_offset = offset;
    for &(key, start, len, n) in &directory {
        let root = writer.root();
        put32(root, key.doc)?;
        root.write_all(&key.ordinal.to_le_bytes())?;
        root.write_all(&[0; 2])?;
        put64(root, start)?;
        put32(root, len)?;
        put32(root, n)?;
        offset += ROW_ENTRY as u64;
    }
    drop(directory);
    let settings = SeismicIndex {
        bytes: OwnedBytes::new(Vec::new()),
        source_offset: 0,
        runs: Vec::new(),
        partitions: vec![None; PARTITIONS],
        dims,
        quantization: config.weight_quantization,
        postings: config.seismic.postings as u32,
        cluster_size: config.seismic.cluster_size as u32,
        summary_energy: config.seismic.summary_energy,
        rows: count,
        single_valued: true,
        pending_terms: 0,
        pending_bytes: 0,
        forward_entries: 0,
    };
    let root = finish_run(
        writer.root(),
        offset,
        row_offset,
        offset,
        0,
        &settings,
        None,
    )?;
    let mut lengths = OutputLengths {
        root,
        ..Default::default()
    };
    let mut terms: BTreeMap<u32, Vec<(u32, f32)>> = BTreeMap::new();
    for (row, vector) in rows.iter().enumerate() {
        for &(dim, weight) in vector {
            if weight != 0.0 {
                terms
                    .entry(dim)
                    .or_default()
                    .push((row as u32, weight.abs()));
            }
        }
    }
    let assignment = AssignmentCoordinates::prepare(&rows);
    let mut term_directory: [Vec<_>; PARTITIONS] = std::array::from_fn(|_| Vec::new());
    for (dim, mut candidates) in terms {
        let frequency = candidates.len() as u32;
        let mut keep = config.seismic.postings;
        if let Some(fraction) = config.pruning
            && candidates.len() >= config.min_terms
        {
            keep =
                keep.min(((candidates.len() as f64 * f64::from(fraction)).ceil() as usize).max(1));
        }
        retain_nominations(&mut candidates, keep);
        let selected: Vec<_> = candidates.into_iter().map(|p| p.0).collect();
        let (bytes, clusters) = encode_term(
            &selected,
            &rows,
            &assignment,
            config.seismic.cluster_size,
            config.seismic.summary_energy,
        )?;
        let part = dim as usize % PARTITIONS;
        term_directory[part].push((
            dim,
            clusters,
            lengths.partitions[part],
            bytes.len() as u64,
            frequency,
        ));
        writer.partition(part).write_all(&bytes)?;
        lengths.partitions[part] += bytes.len() as u64;
    }
    for (part, entries) in term_directory.iter().enumerate() {
        let writer = writer.partition(part);
        let term_offset = lengths.partitions[part];
        for &(dim, clusters, at, len, frequency) in entries {
            put32(writer, dim)?;
            put32(writer, clusters)?;
            put64(writer, at)?;
            put64(writer, len)?;
            put32(writer, 0)?;
            put32(
                writer,
                if config.pruning.is_some_and(|p| p < 1.0) {
                    0
                } else {
                    config.seismic.postings as u32
                },
            )?;
            put32(writer, frequency)?;
            put32(writer, 0)?;
        }
        lengths.partitions[part] = finish_run(
            writer,
            term_offset + entries.len() as u64 * TERM_ENTRY as u64,
            0,
            term_offset,
            entries.len() as u32,
            &settings,
            Some(part),
        )?;
    }
    Ok(lengths)
}

fn sample_hash(row: u32) -> u64 {
    let mut v = u64::from(row) ^ 1142;
    v = (v ^ (v >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    v = (v ^ (v >> 27)).wrapping_mul(0x94d049bb133111eb);
    v ^ (v >> 31)
}
/// Sample centroids deterministically, then assign by approximate sparse dot
/// product over the strongest fifteen coordinates, as in Seismic's inverted
/// centroid assignment. No corpus-dimensional scratch array is allocated.
pub(super) fn encode_term(
    selected: &[u32],
    rows: &[Values],
    assignment: &AssignmentCoordinates,
    target: usize,
    energy: f32,
) -> Result<(Vec<u8>, u32)> {
    encode_term_with_coordinates(selected, rows, target, energy, |row| {
        assignment.row(row, rows[row as usize].len())
    })
}

fn encode_term_with_coordinates<T: AsRef<[(u32, f32)]>>(
    selected: &[u32],
    rows: &[Values],
    target: usize,
    energy: f32,
    coordinates: impl Fn(u32) -> T,
) -> Result<(Vec<u8>, u32)> {
    if selected.is_empty() {
        return term::Encoder::default().finish();
    }
    let nclusters = selected.len().div_ceil(target.max(1));
    let mut seeds = selected.to_vec();
    seeds.sort_unstable_by_key(|&r| (sample_hash(r), r));
    seeds.truncate(nclusters);
    let mut inverted: FxHashMap<u32, Vec<(usize, f32)>> = FxHashMap::default();
    for (i, &row) in seeds.iter().enumerate() {
        for &(dim, w) in &rows[row as usize] {
            inverted.entry(dim).or_default().push((i, w.abs()));
        }
    }
    let seed_ids: FxHashMap<_, _> = seeds.iter().enumerate().map(|(i, &r)| (r, i)).collect();
    let mut clusters = vec![Vec::new(); nclusters];
    let mut scores = vec![0.0f64; nclusters];
    for &row in selected {
        let cluster = if let Some(&i) = seed_ids.get(&row) {
            i
        } else {
            scores.fill(0.0);
            let top = coordinates(row);
            for &(dim, weight) in top.as_ref() {
                if let Some(hits) = inverted.get(&dim) {
                    for &(i, w) in hits {
                        scores[i] += f64::from(weight.abs()) * f64::from(w);
                    }
                }
            }
            scores
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
                .unwrap()
                .0
        };
        clusters[cluster].push(row);
    }
    let mut encoder = term::Encoder::default();
    for cluster in clusters.into_iter().filter(|c| !c.is_empty()) {
        let mut summary: FxHashMap<u32, f32> = FxHashMap::default();
        for &row in &cluster {
            for &(dim, weight) in &rows[row as usize] {
                let max = summary.entry(dim).or_default();
                *max = max.max(weight.abs());
            }
        }
        let mut summary: Vec<_> = summary.into_iter().collect();
        summary.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let total: f64 = summary.iter().map(|p| f64::from(p.1)).sum();
        let target = total * f64::from(energy);
        let mut retained = 0.0;
        let mut keep = 0;
        while keep < summary.len() && retained < target {
            retained += f64::from(summary[keep].1);
            keep += 1;
        }
        summary.truncate(keep);
        summary.sort_unstable_by_key(|p| p.0);
        encoder.push(&cluster, &summary)?;
    }
    encoder.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signed_rows() -> Vec<Values> {
        [0, 1, 14, 15, 16, 31, 128, 257]
            .into_iter()
            .enumerate()
            .map(|(row, count)| {
                (0..count)
                    .map(|dim| {
                        let magnitude = ((dim * 17 + row * 3) % 11) as f32;
                        let weight = if (dim + row) % 2 == 0 {
                            magnitude
                        } else {
                            -magnitude
                        };
                        (dim as u32 * 65_537, weight)
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn cached_assignment_preserves_signed_coordinate_order_and_cluster_bytes() {
        let rows = signed_rows();
        let assignment = AssignmentCoordinates::prepare(&rows);
        assert_eq!(AssignmentCoordinates::BYTES_PER_ROW, 120);
        let original = |row: u32| {
            let mut top = rows[row as usize].clone();
            top.sort_unstable_by(assignment_order);
            top.truncate(ASSIGNMENT_COORDINATES);
            top
        };
        for row in 0..rows.len() as u32 {
            assert_eq!(assignment.row(row, rows[row as usize].len()), original(row));
        }
        for selected in [vec![], vec![0], vec![7, 3, 1, 6, 0, 4, 2, 5]] {
            for target in [1, 3, 64] {
                for energy in [0.1, 0.4, 1.0] {
                    assert_eq!(
                        encode_term(&selected, &rows, &assignment, target, energy).unwrap(),
                        encode_term_with_coordinates(&selected, &rows, target, energy, original)
                            .unwrap(),
                        "cached assignment changed cluster bytes"
                    );
                }
            }
        }
    }

    #[test]
    fn partitioned_nominations_match_full_sort_with_weight_ties() {
        let candidates: Vec<_> = (0..257).rev().map(|row| (row, (row % 7) as f32)).collect();
        for keep in [0, 1, 15, 64, 256, 257, 258] {
            let mut expected = candidates.clone();
            expected.sort_unstable_by(nomination_order);
            expected.truncate(keep);
            let mut actual = candidates.clone();
            retain_nominations(&mut actual, keep);
            assert_eq!(actual, expected);
        }
    }
}
