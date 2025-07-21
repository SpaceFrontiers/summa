//! Merge-native ANN segment format.
//!
//! ANN payloads are split into immutable cluster runs. A normal segment merge
//! copies the three run columns (doc IDs, ordinals, codes) byte-for-byte and
//! rewrites only the compact run directory with an adjusted document base.
//! No centroid assignment, payload deserialization, or reserialization occurs
//! on the merge path.

#[cfg(feature = "native")]
use std::cmp::Reverse;
#[cfg(feature = "native")]
use std::collections::BinaryHeap;
use std::io;
#[cfg(feature = "native")]
use std::io::Write;
use std::ops::Range;

#[cfg(feature = "native")]
use byteorder::WriteBytesExt;
use byteorder::{LittleEndian, ReadBytesExt};

use crate::directories::OwnedBytes;
use crate::dsl::IvfRoutingMode;
use crate::structures::IvfPqQueryPlan;
use crate::structures::vector::index::BoundedAnnCollector;
#[cfg(feature = "native")]
use crate::structures::{BinaryIvfIndex, IVFPQIndex};

const ANN_HEADER_MAGIC: u32 = 0x3152_4e41; // "ANR1"
const ANN_FOOTER_MAGIC: u32 = 0x3146_4e41; // "ANF1"
const ANN_DISK_VERSION: u16 = 1;
const ANN_HEADER_SIZE: usize = 56;
const ANN_RUN_SIZE: usize = 48;
const ANN_FOOTER_SIZE: usize = 24;
#[cfg(feature = "native")]
const COPY_CHUNK: usize = 8 * 1024 * 1024;
#[cfg(feature = "native")]
const PREFETCH_COALESCE_GAP: usize = 4 * 1024;
const BINARY_SCORE_BATCH: usize = 8_192;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AnnKind {
    IvfPq = 1,
    BinaryIvf = 2,
    /// TurboQuant flat scan: single logical cluster, block-packed codes.
    TqFlat = 3,
    /// Trained IVF router with TurboQuant-coded centroid residuals.
    IvfTq = 4,
}

impl AnnKind {
    fn from_u8(value: u8) -> io::Result<Self> {
        match value {
            1 => Ok(Self::IvfPq),
            2 => Ok(Self::BinaryIvf),
            3 => Ok(Self::TqFlat),
            4 => Ok(Self::IvfTq),
            _ => Err(invalid_data(format!("unknown ANN kind {value}"))),
        }
    }
}

/// Codes-column byte length for one run. TQ packs vectors into 16-lane
/// blocks (gammas + dimension-major nibbles), so its column is block-padded
/// rather than `count * code_size`.
fn expected_codes_column_len(kind: AnnKind, count: usize, code_size: usize) -> io::Result<usize> {
    match kind {
        AnnKind::IvfPq | AnnKind::BinaryIvf => count
            .checked_mul(code_size)
            .ok_or_else(|| invalid_data("ANN code column size overflows usize")),
        // Single source of truth for the block-packed layouts lives in tq.rs.
        AnnKind::TqFlat => {
            crate::structures::vector::quantization::tq_codes_column_len_checked(count, code_size)
                .ok_or_else(|| invalid_data("TQ code column size overflows usize"))
        }
        AnnKind::IvfTq => crate::structures::vector::quantization::tq_ivf_codes_column_len_checked(
            count, code_size,
        )
        .ok_or_else(|| invalid_data("IVF-TQ code column size overflows usize")),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AnnDiskHeader {
    pub kind: AnnKind,
    pub routing: IvfRoutingMode,
    pub dim: usize,
    pub code_size: usize,
    pub num_clusters: u32,
    pub quantizer_version: u64,
    pub codebook_version: u64,
    pub vector_count: usize,
}

#[derive(Debug)]
struct AnnRun {
    cluster_id: u32,
    doc_base: u32,
    max_doc_id: u32,
    count: usize,
    doc_ids: Range<usize>,
    ordinals: Range<usize>,
    codes: Range<usize>,
}

/// Mmap-backed searchable ANN payload. Only the fixed-size run directory is
/// heap-resident; all corpus-sized columns remain zero-copy file slices.
pub(crate) struct AnnDiskIndex {
    // Drop locks before the directory allocation they reference.
    #[cfg(feature = "native")]
    heap_pins: crate::segment::pin::HeapPinSet,
    raw: OwnedBytes,
    header: AnnDiskHeader,
    runs: Vec<AnnRun>,
}

impl AnnDiskIndex {
    pub(crate) fn open(
        raw: OwnedBytes,
        expected_kind: AnnKind,
        total_docs: u32,
    ) -> io::Result<Self> {
        if raw.len() < ANN_HEADER_SIZE + ANN_FOOTER_SIZE {
            return Err(invalid_data("ANN payload is shorter than header + footer"));
        }
        let bytes = raw.as_slice();
        let mut header_cursor = std::io::Cursor::new(&bytes[..ANN_HEADER_SIZE]);
        if header_cursor.read_u32::<LittleEndian>()? != ANN_HEADER_MAGIC {
            return Err(invalid_data("ANN payload has unsupported header magic"));
        }
        let kind = AnnKind::from_u8(header_cursor.read_u8()?)?;
        if kind != expected_kind {
            return Err(invalid_data(format!(
                "ANN payload kind {kind:?} does not match expected {expected_kind:?}"
            )));
        }
        let routing = routing_from_u8(header_cursor.read_u8()?)?;
        if header_cursor.read_u16::<LittleEndian>()? != ANN_DISK_VERSION {
            return Err(invalid_data("ANN payload has unsupported format version"));
        }
        let dim = header_cursor.read_u32::<LittleEndian>()? as usize;
        let code_size = header_cursor.read_u32::<LittleEndian>()? as usize;
        let num_clusters = header_cursor.read_u32::<LittleEndian>()?;
        if header_cursor.read_u32::<LittleEndian>()? != 0 {
            return Err(invalid_data("ANN header reserved field is non-zero"));
        }
        let quantizer_version = header_cursor.read_u64::<LittleEndian>()?;
        let codebook_version = header_cursor.read_u64::<LittleEndian>()?;
        let vector_count = usize::try_from(header_cursor.read_u64::<LittleEndian>()?)
            .map_err(|_| invalid_data("ANN vector count exceeds usize"))?;
        if header_cursor.read_u64::<LittleEndian>()? != 0 {
            return Err(invalid_data("ANN header tail is non-zero"));
        }
        let header = AnnDiskHeader {
            kind,
            routing,
            dim,
            code_size,
            num_clusters,
            quantizer_version,
            codebook_version,
            vector_count,
        };
        validate_header(&header)?;

        let footer_start = bytes.len() - ANN_FOOTER_SIZE;
        let mut footer_cursor = std::io::Cursor::new(&bytes[footer_start..]);
        let directory_offset = usize::try_from(footer_cursor.read_u64::<LittleEndian>()?)
            .map_err(|_| invalid_data("ANN directory offset exceeds usize"))?;
        let num_runs = usize::try_from(footer_cursor.read_u64::<LittleEndian>()?)
            .map_err(|_| invalid_data("ANN run count exceeds usize"))?;
        if footer_cursor.read_u32::<LittleEndian>()? != ANN_FOOTER_MAGIC
            || footer_cursor.read_u32::<LittleEndian>()? != u32::from(ANN_DISK_VERSION)
        {
            return Err(invalid_data("ANN payload has unsupported footer"));
        }
        if num_runs == 0 {
            return Err(invalid_data("ANN payload has no cluster runs"));
        }
        let directory_len = num_runs
            .checked_mul(ANN_RUN_SIZE)
            .ok_or_else(|| invalid_data("ANN directory size overflows usize"))?;
        if directory_offset < ANN_HEADER_SIZE
            || directory_offset.checked_add(directory_len) != Some(footer_start)
        {
            return Err(invalid_data("ANN directory does not end at the footer"));
        }

        let mut runs = Vec::with_capacity(num_runs);
        let mut directory_cursor = std::io::Cursor::new(&bytes[directory_offset..footer_start]);
        let mut previous_cluster = None;
        let mut counted_vectors = 0usize;
        for _ in 0..num_runs {
            let cluster_id = directory_cursor.read_u32::<LittleEndian>()?;
            let doc_base = directory_cursor.read_u32::<LittleEndian>()?;
            let count = directory_cursor.read_u32::<LittleEndian>()? as usize;
            let max_doc_id = directory_cursor.read_u32::<LittleEndian>()?;
            let doc_ids_offset = usize::try_from(directory_cursor.read_u64::<LittleEndian>()?)
                .map_err(|_| invalid_data("ANN doc-ID offset exceeds usize"))?;
            let ordinals_offset = usize::try_from(directory_cursor.read_u64::<LittleEndian>()?)
                .map_err(|_| invalid_data("ANN ordinal offset exceeds usize"))?;
            let codes_offset = usize::try_from(directory_cursor.read_u64::<LittleEndian>()?)
                .map_err(|_| invalid_data("ANN code offset exceeds usize"))?;
            let codes_len = usize::try_from(directory_cursor.read_u64::<LittleEndian>()?)
                .map_err(|_| invalid_data("ANN code length exceeds usize"))?;
            if count == 0
                || cluster_id >= num_clusters
                || previous_cluster.is_some_and(|previous| previous > cluster_id)
                || doc_base
                    .checked_add(max_doc_id)
                    .is_none_or(|doc_id| doc_id >= total_docs)
            {
                return Err(invalid_data("ANN run metadata is invalid"));
            }
            previous_cluster = Some(cluster_id);
            let doc_ids_len = count
                .checked_mul(std::mem::size_of::<u32>())
                .ok_or_else(|| invalid_data("ANN doc-ID column size overflows usize"))?;
            let ordinals_len = count
                .checked_mul(std::mem::size_of::<u16>())
                .ok_or_else(|| invalid_data("ANN ordinal column size overflows usize"))?;
            let expected_codes_len = expected_codes_column_len(kind, count, code_size)?;
            let doc_ids_end = doc_ids_offset
                .checked_add(doc_ids_len)
                .ok_or_else(|| invalid_data("ANN doc-ID range overflows usize"))?;
            let ordinals_end = ordinals_offset
                .checked_add(ordinals_len)
                .ok_or_else(|| invalid_data("ANN ordinal range overflows usize"))?;
            let codes_end = codes_offset
                .checked_add(codes_len)
                .ok_or_else(|| invalid_data("ANN code range overflows usize"))?;
            if doc_ids_offset < ANN_HEADER_SIZE
                || ordinals_offset != doc_ids_end
                || codes_offset != ordinals_end
                || codes_len != expected_codes_len
                || codes_end > directory_offset
            {
                return Err(invalid_data("ANN run columns are not contiguous/in bounds"));
            }
            runs.push(AnnRun {
                cluster_id,
                doc_base,
                max_doc_id,
                count,
                doc_ids: doc_ids_offset..doc_ids_end,
                ordinals: ordinals_offset..ordinals_end,
                codes: codes_offset..codes_end,
            });
            counted_vectors = counted_vectors
                .checked_add(count)
                .ok_or_else(|| invalid_data("ANN run vector count overflows usize"))?;
        }
        let mut payload_order: Vec<usize> = (0..runs.len()).collect();
        payload_order.sort_unstable_by_key(|&index| runs[index].doc_ids.start);
        let mut expected_payload_offset = ANN_HEADER_SIZE;
        for index in payload_order {
            let run = &runs[index];
            if run.doc_ids.start != expected_payload_offset {
                return Err(invalid_data("ANN payload runs overlap or contain gaps"));
            }
            expected_payload_offset = run.codes.end;
        }
        if expected_payload_offset != directory_offset || counted_vectors != vector_count {
            return Err(invalid_data(
                "ANN payload coverage/vector count is inconsistent",
            ));
        }

        // Clustered queries visit a small set of runs at unrelated offsets.
        // Disable default mmap readahead for those corpus-sized payloads:
        // without this, each small run can pull in ~128 KiB and amplify
        // cold-query IO by an order of magnitude. Their search methods issue
        // exact WILLNEED ranges before scoring. Flat TQ deliberately scans its
        // sole cluster and therefore retains a sequential access policy.
        #[cfg(feature = "native")]
        raw.madvise_range(
            ANN_HEADER_SIZE..directory_offset,
            if kind == AnnKind::TqFlat {
                libc::MADV_SEQUENTIAL
            } else {
                libc::MADV_RANDOM
            },
        );

        Ok(Self {
            #[cfg(feature = "native")]
            heap_pins: Default::default(),
            raw,
            header,
            runs,
        })
    }

    pub(crate) fn header(&self) -> &AnnDiskHeader {
        &self.header
    }

    pub(crate) fn estimated_heap_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.runs.capacity() * std::mem::size_of::<AnnRun>()
    }

    #[cfg(feature = "native")]
    pub(crate) fn pin_lookup_directory(
        &mut self,
        mode: crate::segment::pin::PinMode,
        remaining: &mut u64,
        report: &mut crate::segment::pin::PinReport,
    ) {
        let before = self.heap_pins.report();
        self.heap_pins
            .pin_slice(&self.runs, "ANN cluster-run directory", mode, remaining);
        let after = self.heap_pins.report();
        report.intended_bytes += after.intended_bytes - before.intended_bytes;
        report.pinned_bytes += after.pinned_bytes - before.pinned_bytes;
        report.skipped_budget_bytes += after.skipped_budget_bytes - before.skipped_budget_bytes;
        report.failed_bytes += after.failed_bytes - before.failed_bytes;
        report.heap_copy_bytes += after.heap_copy_bytes - before.heap_copy_bytes;
    }

    fn cluster_runs(&self, cluster_id: u32) -> &[AnnRun] {
        let start = self.runs.partition_point(|run| run.cluster_id < cluster_id);
        let end = self
            .runs
            .partition_point(|run| run.cluster_id <= cluster_id);
        &self.runs[start..end]
    }

    /// Prefetch exactly the mmap ranges needed by the selected IVF leaves.
    ///
    /// A pure-copy merge preserves each source payload as one physical extent,
    /// so runs for one logical cluster can be far apart. Sorting by file offset
    /// lets us coalesce overlaps and page-near ranges without reading through
    /// unrelated clusters.
    #[cfg(feature = "native")]
    fn prefetch_cluster_runs(&self, cluster_ids: &[u32]) {
        if cluster_ids.is_empty() || !self.raw.is_mmap() {
            return;
        }
        let mut ranges = Vec::with_capacity(cluster_ids.len());
        for &cluster_id in cluster_ids {
            ranges.extend(
                self.cluster_runs(cluster_id)
                    .iter()
                    .map(|run| run.doc_ids.start..run.codes.end),
            );
        }
        coalesce_prefetch_ranges(&mut ranges);
        for range in ranges {
            self.raw.madvise_range(range, libc::MADV_WILLNEED);
        }
    }

    pub(crate) fn search_ivf_pq_distinct(
        &self,
        k: usize,
        plan: &IvfPqQueryPlan,
    ) -> io::Result<Vec<(u32, u16, f32)>> {
        let mut collector = BoundedAnnCollector::<true, false>::new(k);
        #[cfg(feature = "native")]
        self.prefetch_cluster_runs(&plan.cluster_ids);
        let bytes = self.raw.as_slice();
        for (cluster_id, distance_table) in plan.cluster_distance_tables() {
            for run in self.cluster_runs(cluster_id) {
                for index in 0..run.count {
                    let code_start = run.codes.start + index * self.header.code_size;
                    let code = &bytes[code_start..code_start + self.header.code_size];
                    let distance = distance_table.compute_distance(code);
                    collector.insert(
                        run_doc_id(bytes, run, index)?,
                        read_u16(bytes, run.ordinals.start + index * 2),
                        distance,
                    );
                }
            }
        }
        Ok(collector.into_sorted_results())
    }

    /// Score every TQ block against the query plan and keep the top `k`
    /// distinct documents by estimated similarity.
    pub(crate) fn search_tq_distinct(
        &self,
        k: usize,
        plan: &crate::structures::TqQueryPlan,
    ) -> io::Result<Vec<(u32, u16, f32)>> {
        use crate::structures::vector::quantization::{TQ_BLOCK_LANES, tq_block_bytes};

        if plan.padded_dim() != self.header.code_size * 2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "TQ query plan does not match the payload dimension",
            ));
        }
        let block_bytes = tq_block_bytes(self.header.code_size);
        let bytes = self.raw.as_slice();
        let mut collector = BoundedAnnCollector::<true, true>::new(k);
        let mut scores = [0.0f32; TQ_BLOCK_LANES];
        for run in &self.runs {
            let codes = &bytes[run.codes.clone()];
            for (block_index, block) in codes.chunks_exact(block_bytes).enumerate() {
                crate::structures::vector::quantization::tq_score_block(plan, block, &mut scores);
                let lane_base = block_index * TQ_BLOCK_LANES;
                let lanes = TQ_BLOCK_LANES.min(run.count.saturating_sub(lane_base));
                for (lane, &score) in scores.iter().enumerate().take(lanes) {
                    let index = lane_base + lane;
                    collector.insert(
                        run_doc_id(bytes, run, index)?,
                        read_u16(bytes, run.ordinals.start + index * 2),
                        score,
                    );
                }
            }
        }
        Ok(collector.into_sorted_results())
    }

    /// Score the probed IVF-TQ leaves and keep the top `k` distinct
    /// documents by estimated similarity (`⟨q̂,c⟩ + scale·⟨q̂,r̂⟩`).
    pub(crate) fn search_ivf_tq_distinct(
        &self,
        k: usize,
        plan: &crate::structures::TqIvfQueryPlan,
    ) -> io::Result<Vec<(u32, u16, f32)>> {
        use crate::structures::vector::quantization::{
            TQ_BLOCK_LANES, tq_ivf_block_bytes, tq_score_ivf_block,
        };

        let tq_plan = plan.tq_plan();
        if tq_plan.padded_dim() != self.header.code_size * 2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "IVF-TQ query plan does not match the payload dimension",
            ));
        }
        #[cfg(feature = "native")]
        self.prefetch_cluster_runs(&plan.cluster_ids);
        let block_bytes = tq_ivf_block_bytes(self.header.code_size);
        let bytes = self.raw.as_slice();
        let mut collector = BoundedAnnCollector::<true, true>::new(k);
        let mut scores = [0.0f32; TQ_BLOCK_LANES];
        for (cluster_id, cluster_dot) in plan.cluster_dots() {
            for run in self.cluster_runs(cluster_id) {
                let codes = &bytes[run.codes.clone()];
                for (block_index, block) in codes.chunks_exact(block_bytes).enumerate() {
                    tq_score_ivf_block(tq_plan, block, cluster_dot, &mut scores);
                    let lane_base = block_index * TQ_BLOCK_LANES;
                    let lanes = TQ_BLOCK_LANES.min(run.count.saturating_sub(lane_base));
                    for (lane, &score) in scores.iter().enumerate().take(lanes) {
                        let index = lane_base + lane;
                        collector.insert(
                            run_doc_id(bytes, run, index)?,
                            read_u16(bytes, run.ordinals.start + index * 2),
                            score,
                        );
                    }
                }
            }
        }
        Ok(collector.into_sorted_results())
    }

    pub(crate) fn search_binary_clusters<const BY_DOCUMENT: bool>(
        &self,
        query: &[u8],
        k: usize,
        cluster_ids: &[u32],
    ) -> io::Result<Vec<(u32, u16, f32)>> {
        let mut collector = BoundedAnnCollector::<BY_DOCUMENT, true>::new(k);
        if query.len() != self.header.code_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "binary ANN query has the wrong byte length",
            ));
        }
        #[cfg(feature = "native")]
        self.prefetch_cluster_runs(cluster_ids);
        let bytes = self.raw.as_slice();
        let mut scores = vec![0.0f32; BINARY_SCORE_BATCH.min(self.header.vector_count)];
        for &cluster_id in cluster_ids {
            for run in self.cluster_runs(cluster_id) {
                score_binary_run(
                    bytes,
                    run,
                    query,
                    self.header.dim,
                    self.header.code_size,
                    &mut scores,
                    &mut collector,
                )?;
            }
        }
        Ok(collector.into_sorted_results())
    }
}

#[cfg(feature = "native")]
fn coalesce_prefetch_ranges(ranges: &mut Vec<Range<usize>>) {
    if ranges.len() < 2 {
        return;
    }
    ranges.sort_unstable_by_key(|range| range.start);
    let mut output_len = 1usize;
    for input_index in 1..ranges.len() {
        let next_start = ranges[input_index].start;
        let next_end = ranges[input_index].end;
        let previous = &mut ranges[output_len - 1];
        if next_start <= previous.end.saturating_add(PREFETCH_COALESCE_GAP) {
            previous.end = previous.end.max(next_end);
        } else {
            ranges[output_len] = next_start..next_end;
            output_len += 1;
        }
    }
    ranges.truncate(output_len);
}

fn score_binary_run<const BY_DOCUMENT: bool>(
    bytes: &[u8],
    run: &AnnRun,
    query: &[u8],
    dim_bits: usize,
    code_size: usize,
    scores: &mut [f32],
    collector: &mut BoundedAnnCollector<BY_DOCUMENT, true>,
) -> io::Result<()> {
    for batch_start in (0..run.count).step_by(BINARY_SCORE_BATCH) {
        let batch_count = BINARY_SCORE_BATCH.min(run.count - batch_start);
        let code_start = run.codes.start + batch_start * code_size;
        let code_end = code_start + batch_count * code_size;
        crate::structures::simd::batch_hamming_scores(
            query,
            &bytes[code_start..code_end],
            code_size,
            dim_bits,
            &mut scores[..batch_count],
        );
        for (batch_index, &score) in scores.iter().enumerate().take(batch_count) {
            let index = batch_start + batch_index;
            collector.insert(
                run_doc_id(bytes, run, index)?,
                read_u16(bytes, run.ordinals.start + index * 2),
                score,
            );
        }
    }
    Ok(())
}

#[cfg(feature = "native")]
pub(crate) fn write_built_ivf_pq(
    index: &IVFPQIndex,
    num_clusters: u32,
    writer: &mut (impl Write + ?Sized),
) -> io::Result<u64> {
    let mut clusters: Vec<_> = index.clusters.iter().collect();
    clusters.sort_unstable_by_key(|(cluster_id, _)| **cluster_id);
    let runs: Vec<_> = clusters
        .into_iter()
        .map(|(&cluster_id, cluster)| BuildRun {
            cluster_id,
            doc_ids: &cluster.doc_ids,
            ordinals: &cluster.ordinals,
            codes: &cluster.codes,
        })
        .collect();
    write_built_runs(
        AnnDiskHeader {
            kind: AnnKind::IvfPq,
            routing: index.config.routing,
            dim: index.config.dim,
            code_size: index.config.code_size,
            num_clusters,
            quantizer_version: index.centroids_version,
            codebook_version: index.codebook_version,
            vector_count: index.len(),
        },
        &runs,
        writer,
    )
}

#[cfg(feature = "native")]
pub(crate) fn write_built_binary_ivf(
    index: &BinaryIvfIndex,
    routing: IvfRoutingMode,
    writer: &mut (impl Write + ?Sized),
) -> io::Result<u64> {
    let runs: Vec<_> = index
        .clusters
        .iter()
        .map(|(cluster_id, cluster)| BuildRun {
            cluster_id: *cluster_id,
            doc_ids: &cluster.doc_ids,
            ordinals: &cluster.ordinals,
            codes: &cluster.codes,
        })
        .collect();
    write_built_runs(
        AnnDiskHeader {
            kind: AnnKind::BinaryIvf,
            routing,
            dim: index.dim_bits,
            code_size: index.dim_bits.div_ceil(8),
            num_clusters: index.num_clusters,
            quantizer_version: index.quantizer_version,
            codebook_version: 0,
            vector_count: index.len(),
        },
        &runs,
        writer,
    )
}

/// Serialize a populated IVF-TQ build: one run per non-empty leaf, codes
/// block-packed per run (scales + gammas + dimension-major nibbles).
#[cfg(feature = "native")]
pub(crate) fn write_built_ivf_tq(
    index: &crate::structures::IvfTqIndex,
    num_clusters: u32,
    writer: &mut (impl Write + ?Sized),
) -> io::Result<u64> {
    use crate::structures::vector::quantization::{TQ_BLOCK_LANES, tq_pack_ivf_block};

    let codec = index.codec();
    let padded_dim = codec.padded_dim();
    let mut clusters: Vec<_> = index.clusters.iter().collect();
    clusters.sort_unstable_by_key(|(cluster_id, _)| **cluster_id);
    let packed: Vec<(u32, Vec<u8>)> = clusters
        .iter()
        .map(|&(&cluster_id, cluster)| {
            let mut codes = Vec::with_capacity(
                crate::structures::vector::quantization::tq_ivf_codes_column_len_checked(
                    cluster.doc_ids.len(),
                    codec.code_size(),
                )
                .unwrap_or_default(),
            );
            for block_start in (0..cluster.doc_ids.len()).step_by(TQ_BLOCK_LANES) {
                let lanes = TQ_BLOCK_LANES.min(cluster.doc_ids.len() - block_start);
                let rows: Vec<&[u8]> = (0..lanes)
                    .map(|lane| {
                        let row = block_start + lane;
                        &cluster.rows[row * padded_dim..(row + 1) * padded_dim]
                    })
                    .collect();
                tq_pack_ivf_block(
                    &rows,
                    &cluster.scales[block_start..block_start + lanes],
                    &cluster.gammas[block_start..block_start + lanes],
                    padded_dim,
                    &mut codes,
                );
            }
            (cluster_id, codes)
        })
        .collect();
    let runs: Vec<_> = clusters
        .iter()
        .zip(&packed)
        .map(|(&(&cluster_id, cluster), (packed_id, codes))| {
            debug_assert_eq!(cluster_id, *packed_id);
            BuildRun {
                cluster_id,
                doc_ids: &cluster.doc_ids,
                ordinals: &cluster.ordinals,
                codes,
            }
        })
        .collect();
    write_built_runs(
        AnnDiskHeader {
            kind: AnnKind::IvfTq,
            routing: index.routing,
            dim: index.dim,
            code_size: codec.code_size(),
            num_clusters,
            quantizer_version: index.centroids_version,
            codebook_version: codec.fingerprint(),
            vector_count: index.len(),
        },
        &runs,
        writer,
    )
}

/// Serialize a populated TQ builder as a single-run payload.
#[cfg(feature = "native")]
pub(crate) fn write_built_tq_flat(
    builder: &crate::structures::TqFlatBuilder,
    writer: &mut (impl Write + ?Sized),
) -> io::Result<u64> {
    let codec = builder.codec();
    let runs = [BuildRun {
        cluster_id: 0,
        doc_ids: &builder.doc_ids,
        ordinals: &builder.ordinals,
        codes: &builder.codes,
    }];
    write_built_runs(
        AnnDiskHeader {
            kind: AnnKind::TqFlat,
            routing: IvfRoutingMode::Flat,
            dim: codec.dim(),
            code_size: codec.code_size(),
            num_clusters: 1,
            quantizer_version: codec.fingerprint(),
            codebook_version: 0,
            vector_count: builder.len(),
        },
        &runs,
        writer,
    )
}

#[cfg(feature = "native")]
struct BuildRun<'a> {
    cluster_id: u32,
    doc_ids: &'a [u32],
    ordinals: &'a [u16],
    codes: &'a [u8],
}

#[cfg(feature = "native")]
struct RunRecord {
    cluster_id: u32,
    doc_base: u32,
    count: u32,
    max_doc_id: u32,
    doc_ids_offset: u64,
    ordinals_offset: u64,
    codes_offset: u64,
    codes_len: u64,
}

#[cfg(feature = "native")]
fn write_built_runs(
    header: AnnDiskHeader,
    runs: &[BuildRun<'_>],
    writer: &mut (impl Write + ?Sized),
) -> io::Result<u64> {
    if runs.is_empty() || header.vector_count == 0 {
        return Err(invalid_data("cannot write an empty ANN payload"));
    }
    validate_header(&header)?;
    write_header(writer, &header)?;
    let mut offset = ANN_HEADER_SIZE as u64;
    let mut records = Vec::with_capacity(runs.len());
    let mut counted = 0usize;
    let mut scratch = Vec::new();
    let mut previous_cluster = None;
    for run in runs {
        let count = run.doc_ids.len();
        if count == 0
            || run.cluster_id >= header.num_clusters
            || previous_cluster.is_some_and(|cluster| cluster >= run.cluster_id)
            || run.ordinals.len() != count
            || run.codes.len() != expected_codes_column_len(header.kind, count, header.code_size)?
        {
            return Err(invalid_data("ANN build run columns are inconsistent"));
        }
        previous_cluster = Some(run.cluster_id);
        let count_u32 = u32::try_from(count)
            .map_err(|_| invalid_data("ANN cluster run exceeds u32 vectors"))?;
        let max_doc_id = run.doc_ids.iter().copied().max().unwrap_or(0);
        let doc_ids_offset = offset;
        write_u32_column(writer, run.doc_ids, &mut scratch)?;
        offset = offset
            .checked_add(
                u64::try_from(count)
                    .ok()
                    .and_then(|count| count.checked_mul(4))
                    .ok_or_else(|| invalid_data("ANN doc-ID output size overflows u64"))?,
            )
            .ok_or_else(|| invalid_data("ANN output offset overflow"))?;
        let ordinals_offset = offset;
        write_u16_column(writer, run.ordinals, &mut scratch)?;
        offset = offset
            .checked_add(
                u64::try_from(count)
                    .ok()
                    .and_then(|count| count.checked_mul(2))
                    .ok_or_else(|| invalid_data("ANN ordinal output size overflows u64"))?,
            )
            .ok_or_else(|| invalid_data("ANN output offset overflow"))?;
        let codes_offset = offset;
        writer.write_all(run.codes)?;
        offset = offset
            .checked_add(
                u64::try_from(run.codes.len())
                    .map_err(|_| invalid_data("ANN code output size exceeds u64"))?,
            )
            .ok_or_else(|| invalid_data("ANN output offset overflow"))?;
        records.push(RunRecord {
            cluster_id: run.cluster_id,
            doc_base: 0,
            count: count_u32,
            max_doc_id,
            doc_ids_offset,
            ordinals_offset,
            codes_offset,
            codes_len: u64::try_from(run.codes.len())
                .map_err(|_| invalid_data("ANN code output size exceeds u64"))?,
        });
        counted = counted
            .checked_add(count)
            .ok_or_else(|| invalid_data("ANN vector count overflow"))?;
    }
    if counted != header.vector_count {
        return Err(invalid_data("ANN header/build vector counts disagree"));
    }
    finish_layout(writer, offset, &records)
}

/// Pure-copy normal merge. Corpus-sized source columns are never decoded or
/// rewritten; only this compact directory is regenerated with adjusted bases.
#[cfg(feature = "native")]
pub(crate) fn write_merged_ann(
    sources: &[(&AnnDiskIndex, u32)],
    writer: &mut (impl Write + ?Sized),
) -> io::Result<u64> {
    let Some((first, _)) = sources.first() else {
        return Err(invalid_data("cannot merge an empty ANN source list"));
    };
    let mut header = first.header.clone();
    header.vector_count = 0;
    for &(source, _) in sources {
        if !headers_compatible(&first.header, &source.header) {
            return Err(invalid_data(
                "ANN merge sources use incompatible generations",
            ));
        }
        header.vector_count = header
            .vector_count
            .checked_add(source.header.vector_count)
            .ok_or_else(|| invalid_data("merged ANN vector count overflows usize"))?;
    }
    validate_header(&header)?;
    write_header(writer, &header)?;
    let mut offset = ANN_HEADER_SIZE as u64;
    let run_capacity = sources.iter().try_fold(0usize, |count, (source, _)| {
        count
            .checked_add(source.runs.len())
            .ok_or_else(|| invalid_data("merged ANN run count overflows usize"))
    })?;
    let mut output_payload_starts = Vec::with_capacity(sources.len());
    for &(source, _) in sources {
        let payload_end = source
            .runs
            .iter()
            .map(|run| run.codes.end)
            .max()
            .ok_or_else(|| invalid_data("ANN source has no payload runs"))?;
        let output_payload_start = offset;
        output_payload_starts.push(output_payload_start);
        copy_range(writer, &source.raw, ANN_HEADER_SIZE..payload_end)?;
        offset = checked_advance(offset, payload_end - ANN_HEADER_SIZE)?;
    }

    // Every source directory is already cluster-sorted. Merge those compact
    // directories directly into the output with O(source count) heap memory;
    // the corpus payload extents above remain untouched and source-contiguous.
    let directory_offset = offset;
    let mut pending = BinaryHeap::with_capacity(sources.len());
    for (source_index, (source, _)) in sources.iter().enumerate() {
        pending.push(Reverse((source.runs[0].cluster_id, source_index, 0usize)));
    }
    let mut written_runs = 0usize;
    while let Some(Reverse((_, source_index, run_index))) = pending.pop() {
        let (source, segment_base) = sources[source_index];
        let run = &source.runs[run_index];
        write_run_record(
            writer,
            &RunRecord {
                cluster_id: run.cluster_id,
                doc_base: run
                    .doc_base
                    .checked_add(segment_base)
                    .ok_or_else(|| invalid_data("merged ANN document base overflows u32"))?,
                count: u32::try_from(run.count)
                    .map_err(|_| invalid_data("ANN source run exceeds u32 vectors"))?,
                max_doc_id: run.max_doc_id,
                doc_ids_offset: relocate_payload_offset(
                    output_payload_starts[source_index],
                    run.doc_ids.start,
                )?,
                ordinals_offset: relocate_payload_offset(
                    output_payload_starts[source_index],
                    run.ordinals.start,
                )?,
                codes_offset: relocate_payload_offset(
                    output_payload_starts[source_index],
                    run.codes.start,
                )?,
                codes_len: u64::try_from(run.codes.len())
                    .map_err(|_| invalid_data("ANN source code length exceeds u64"))?,
            },
        )?;
        written_runs = written_runs
            .checked_add(1)
            .ok_or_else(|| invalid_data("merged ANN run count overflows usize"))?;
        let next_run_index = run_index + 1;
        if let Some(next_run) = source.runs.get(next_run_index) {
            pending.push(Reverse((next_run.cluster_id, source_index, next_run_index)));
        }
    }
    if written_runs != run_capacity {
        return Err(invalid_data("merged ANN directory lost source runs"));
    }
    finish_footer(writer, directory_offset, written_runs)
}

#[cfg(feature = "native")]
fn relocate_payload_offset(output_payload_start: u64, source_offset: usize) -> io::Result<u64> {
    let relative = source_offset
        .checked_sub(ANN_HEADER_SIZE)
        .ok_or_else(|| invalid_data("ANN source offset precedes its payload"))?;
    output_payload_start
        .checked_add(
            u64::try_from(relative)
                .map_err(|_| invalid_data("ANN source relative offset exceeds u64"))?,
        )
        .ok_or_else(|| invalid_data("merged ANN payload offset overflows u64"))
}

#[cfg(feature = "native")]
fn headers_compatible(left: &AnnDiskHeader, right: &AnnDiskHeader) -> bool {
    left.kind == right.kind
        && left.routing == right.routing
        && left.dim == right.dim
        && left.code_size == right.code_size
        && left.num_clusters == right.num_clusters
        && left.quantizer_version == right.quantizer_version
        && left.codebook_version == right.codebook_version
}

#[cfg(feature = "native")]
fn finish_layout(
    writer: &mut (impl Write + ?Sized),
    directory_offset: u64,
    records: &[RunRecord],
) -> io::Result<u64> {
    for record in records {
        write_run_record(writer, record)?;
    }
    finish_footer(writer, directory_offset, records.len())
}

#[cfg(feature = "native")]
fn write_run_record(writer: &mut (impl Write + ?Sized), record: &RunRecord) -> io::Result<()> {
    writer.write_u32::<LittleEndian>(record.cluster_id)?;
    writer.write_u32::<LittleEndian>(record.doc_base)?;
    writer.write_u32::<LittleEndian>(record.count)?;
    writer.write_u32::<LittleEndian>(record.max_doc_id)?;
    writer.write_u64::<LittleEndian>(record.doc_ids_offset)?;
    writer.write_u64::<LittleEndian>(record.ordinals_offset)?;
    writer.write_u64::<LittleEndian>(record.codes_offset)?;
    writer.write_u64::<LittleEndian>(record.codes_len)?;
    Ok(())
}

#[cfg(feature = "native")]
fn finish_footer(
    writer: &mut (impl Write + ?Sized),
    directory_offset: u64,
    num_records: usize,
) -> io::Result<u64> {
    writer.write_u64::<LittleEndian>(directory_offset)?;
    writer.write_u64::<LittleEndian>(
        u64::try_from(num_records).map_err(|_| invalid_data("ANN run count exceeds u64"))?,
    )?;
    writer.write_u32::<LittleEndian>(ANN_FOOTER_MAGIC)?;
    writer.write_u32::<LittleEndian>(u32::from(ANN_DISK_VERSION))?;
    let tail_size = num_records
        .checked_mul(ANN_RUN_SIZE)
        .and_then(|size| size.checked_add(ANN_FOOTER_SIZE))
        .and_then(|size| u64::try_from(size).ok())
        .ok_or_else(|| invalid_data("ANN final tail size overflows u64"))?;
    directory_offset
        .checked_add(tail_size)
        .ok_or_else(|| invalid_data("ANN final size overflows u64"))
}

#[cfg(feature = "native")]
fn write_header(writer: &mut (impl Write + ?Sized), header: &AnnDiskHeader) -> io::Result<()> {
    writer.write_u32::<LittleEndian>(ANN_HEADER_MAGIC)?;
    writer.write_u8(header.kind as u8)?;
    writer.write_u8(routing_to_u8(header.routing))?;
    writer.write_u16::<LittleEndian>(ANN_DISK_VERSION)?;
    writer.write_u32::<LittleEndian>(
        u32::try_from(header.dim).map_err(|_| invalid_data("ANN dimension exceeds u32"))?,
    )?;
    writer.write_u32::<LittleEndian>(
        u32::try_from(header.code_size).map_err(|_| invalid_data("ANN code size exceeds u32"))?,
    )?;
    writer.write_u32::<LittleEndian>(header.num_clusters)?;
    writer.write_u32::<LittleEndian>(0)?;
    writer.write_u64::<LittleEndian>(header.quantizer_version)?;
    writer.write_u64::<LittleEndian>(header.codebook_version)?;
    writer.write_u64::<LittleEndian>(
        u64::try_from(header.vector_count)
            .map_err(|_| invalid_data("ANN vector count exceeds u64"))?,
    )?;
    writer.write_u64::<LittleEndian>(0)?;
    Ok(())
}

#[cfg(feature = "native")]
fn write_u32_column(
    writer: &mut (impl Write + ?Sized),
    values: &[u32],
    scratch: &mut Vec<u8>,
) -> io::Result<()> {
    for chunk in values.chunks(64 * 1024) {
        scratch.clear();
        scratch.reserve(chunk.len() * 4);
        for value in chunk {
            scratch.extend_from_slice(&value.to_le_bytes());
        }
        writer.write_all(scratch)?;
    }
    Ok(())
}

#[cfg(feature = "native")]
fn write_u16_column(
    writer: &mut (impl Write + ?Sized),
    values: &[u16],
    scratch: &mut Vec<u8>,
) -> io::Result<()> {
    for chunk in values.chunks(64 * 1024) {
        scratch.clear();
        scratch.reserve(chunk.len() * 2);
        for value in chunk {
            scratch.extend_from_slice(&value.to_le_bytes());
        }
        writer.write_all(scratch)?;
    }
    Ok(())
}

#[cfg(feature = "native")]
fn copy_range(
    writer: &mut (impl Write + ?Sized),
    bytes: &OwnedBytes,
    range: Range<usize>,
) -> io::Result<()> {
    if range.is_empty() {
        return Ok(());
    }
    let range_end = range.end;
    let mut chunk_start = range.start;
    let first_end = chunk_start.saturating_add(COPY_CHUNK).min(range_end);
    bytes.madvise_range(chunk_start..first_end, libc::MADV_WILLNEED);
    while chunk_start < range_end {
        let chunk_end = chunk_start.saturating_add(COPY_CHUNK).min(range_end);
        let next_end = chunk_end.saturating_add(COPY_CHUNK).min(range_end);
        if chunk_end < next_end {
            // Keep one bounded window of IO in flight while the current
            // window is copied. The query mapping remains MADV_RANDOM.
            bytes.madvise_range(chunk_end..next_end, libc::MADV_WILLNEED);
        }
        writer.write_all(&bytes.as_slice()[chunk_start..chunk_end])?;
        chunk_start = chunk_end;
    }
    Ok(())
}

#[cfg(feature = "native")]
fn checked_advance(offset: u64, length: usize) -> io::Result<u64> {
    offset
        .checked_add(
            u64::try_from(length).map_err(|_| invalid_data("ANN copy length exceeds u64"))?,
        )
        .ok_or_else(|| invalid_data("ANN output offset overflows u64"))
}

fn validate_header(header: &AnnDiskHeader) -> io::Result<()> {
    if header.dim == 0
        || header.code_size == 0
        || header.num_clusters == 0
        || header.quantizer_version == 0
        || header.vector_count == 0
        || (header.kind == AnnKind::IvfPq && header.codebook_version == 0)
        || (header.kind == AnnKind::BinaryIvf
            && (header.codebook_version != 0
                || !header.dim.is_multiple_of(8)
                || header.code_size != header.dim.div_ceil(8)))
        || (header.kind == AnnKind::TqFlat
            && (header.codebook_version != 0
                || header.num_clusters != 1
                || header.routing != IvfRoutingMode::Flat
                || header.code_size * 2
                    != crate::structures::vector::quantization::tq_padded_dim(header.dim)))
        // IVF-TQ: quantizer_version is the trained centroid generation and
        // codebook_version carries the (nonzero) TQ codec fingerprint.
        || (header.kind == AnnKind::IvfTq
            && (header.codebook_version == 0
                || header.code_size * 2
                    != crate::structures::vector::quantization::tq_padded_dim(header.dim)))
    {
        return Err(invalid_data("ANN header contains invalid metadata"));
    }
    Ok(())
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn read_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap())
}

fn run_doc_id(bytes: &[u8], run: &AnnRun, index: usize) -> io::Result<u32> {
    let local_doc_id = read_u32(bytes, run.doc_ids.start + index * 4);
    if local_doc_id > run.max_doc_id {
        return Err(invalid_data(
            "ANN run contains a document above its declared maximum",
        ));
    }
    run.doc_base
        .checked_add(local_doc_id)
        .ok_or_else(|| invalid_data("ANN run document ID overflows u32"))
}

#[cfg(feature = "native")]
fn routing_to_u8(routing: IvfRoutingMode) -> u8 {
    match routing {
        IvfRoutingMode::Auto => 0,
        IvfRoutingMode::Flat => 1,
        IvfRoutingMode::TwoLevel => 2,
        IvfRoutingMode::Hnsw => 3,
    }
}

fn routing_from_u8(value: u8) -> io::Result<IvfRoutingMode> {
    match value {
        0 => Ok(IvfRoutingMode::Auto),
        1 => Ok(IvfRoutingMode::Flat),
        2 => Ok(IvfRoutingMode::TwoLevel),
        3 => Ok(IvfRoutingMode::Hnsw),
        _ => Err(invalid_data(format!("unknown ANN routing mode {value}"))),
    }
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

#[cfg(all(test, feature = "native"))]
mod tests {
    use super::*;

    fn binary_header(vector_count: usize) -> AnnDiskHeader {
        AnnDiskHeader {
            kind: AnnKind::BinaryIvf,
            routing: IvfRoutingMode::Hnsw,
            dim: 8,
            code_size: 1,
            num_clusters: 2,
            quantizer_version: 42,
            codebook_version: 0,
            vector_count,
        }
    }

    fn payload_end(index: &AnnDiskIndex) -> usize {
        index.runs.iter().map(|run| run.codes.end).max().unwrap()
    }

    #[test]
    fn ann_prefetch_ranges_are_sorted_and_only_merge_page_near_extents() {
        let mut ranges = vec![
            15_000..16_000,
            0..1_000,
            9_000..10_000,
            1_000..2_000,
            7_000..8_000,
        ];
        coalesce_prefetch_ranges(&mut ranges);
        assert_eq!(ranges, [0..2_000, 7_000..10_000, 15_000..16_000]);
    }

    #[test]
    fn normal_merge_copies_ann_payload_columns_byte_for_byte() {
        let first_doc_0 = [0u32];
        let first_doc_1 = [1u32];
        let first_ord_0 = [0u16];
        let first_ord_1 = [2u16];
        let first_code_0 = [0x00u8];
        let first_code_1 = [0xffu8];
        let first_runs = [
            BuildRun {
                cluster_id: 0,
                doc_ids: &first_doc_0,
                ordinals: &first_ord_0,
                codes: &first_code_0,
            },
            BuildRun {
                cluster_id: 1,
                doc_ids: &first_doc_1,
                ordinals: &first_ord_1,
                codes: &first_code_1,
            },
        ];
        let mut first_bytes = Vec::new();
        write_built_runs(binary_header(2), &first_runs, &mut first_bytes).unwrap();
        let first = AnnDiskIndex::open(OwnedBytes::new(first_bytes.clone()), AnnKind::BinaryIvf, 2)
            .unwrap();

        let second_docs = [0u32, 1u32];
        let second_ords = [1u16, 0u16];
        let second_codes = [0x0fu8, 0xf0u8];
        let second_runs = [BuildRun {
            cluster_id: 0,
            doc_ids: &second_docs,
            ordinals: &second_ords,
            codes: &second_codes,
        }];
        let mut second_bytes = Vec::new();
        write_built_runs(binary_header(2), &second_runs, &mut second_bytes).unwrap();
        let second =
            AnnDiskIndex::open(OwnedBytes::new(second_bytes.clone()), AnnKind::BinaryIvf, 2)
                .unwrap();

        let mut merged_bytes = Vec::new();
        write_merged_ann(&[(&first, 0), (&second, 2)], &mut merged_bytes).unwrap();
        let merged =
            AnnDiskIndex::open(OwnedBytes::new(merged_bytes.clone()), AnnKind::BinaryIvf, 4)
                .unwrap();

        let mut expected_payload = first_bytes[ANN_HEADER_SIZE..payload_end(&first)].to_vec();
        expected_payload.extend_from_slice(&second_bytes[ANN_HEADER_SIZE..payload_end(&second)]);
        assert_eq!(
            &merged_bytes[ANN_HEADER_SIZE..payload_end(&merged)],
            expected_payload.as_slice(),
            "normal merge must not decode or rewrite any corpus-sized ANN column",
        );

        let mut docs: Vec<u32> = merged
            .search_binary_clusters::<false>(&[0], 4, &[0, 1])
            .unwrap()
            .into_iter()
            .map(|result| result.0)
            .collect();
        docs.sort_unstable();
        assert_eq!(docs, [0, 1, 2, 3]);

        // A merged source's directory is cluster-sorted while its payload is
        // source-order. A later merge must follow physical offsets and still
        // preserve every source column byte-for-byte.
        let mut second_merge_bytes = Vec::new();
        write_merged_ann(&[(&merged, 0), (&first, 4)], &mut second_merge_bytes).unwrap();
        let second_merge = AnnDiskIndex::open(
            OwnedBytes::new(second_merge_bytes.clone()),
            AnnKind::BinaryIvf,
            6,
        )
        .unwrap();
        let mut expected_second_payload =
            merged_bytes[ANN_HEADER_SIZE..payload_end(&merged)].to_vec();
        expected_second_payload
            .extend_from_slice(&first_bytes[ANN_HEADER_SIZE..payload_end(&first)]);
        assert_eq!(
            &second_merge_bytes[ANN_HEADER_SIZE..payload_end(&second_merge)],
            expected_second_payload.as_slice(),
        );
        let mut docs: Vec<u32> = second_merge
            .search_binary_clusters::<false>(&[0], 6, &[0, 1])
            .unwrap()
            .into_iter()
            .map(|result| result.0)
            .collect();
        docs.sort_unstable();
        assert_eq!(docs, [0, 1, 2, 3, 4, 5]);
    }

    fn build_tq_payload(dim: usize, count: usize, seed: u64) -> (Vec<u8>, Vec<Vec<f32>>) {
        let codec = std::sync::Arc::new(crate::structures::TqCodec::new(dim));
        let mut builder = crate::structures::TqFlatBuilder::new(std::sync::Arc::clone(&codec));
        let mut state = seed;
        let mut vectors = Vec::new();
        let mut flat = Vec::new();
        for _ in 0..count {
            let vector: Vec<f32> = (0..dim)
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5
                })
                .collect();
            flat.extend_from_slice(&vector);
            vectors.push(vector);
        }
        let labels: Vec<(u32, u16)> = (0..count).map(|index| (index as u32, 0)).collect();
        builder.add_batch(&labels, &flat).unwrap();
        builder.finish();
        let mut bytes = Vec::new();
        write_built_tq_flat(&builder, &mut bytes).unwrap();
        (bytes, vectors)
    }

    #[test]
    fn tq_payload_roundtrip_search_and_pure_copy_merge() {
        let dim = 20; // pads to 32; exercises padding + partial final block
        let count = 21;
        let (bytes, vectors) = build_tq_payload(dim, count, 42);
        let index = AnnDiskIndex::open(
            OwnedBytes::new(bytes.clone()),
            AnnKind::TqFlat,
            count as u32,
        )
        .unwrap();
        assert_eq!(index.header().vector_count, count);

        // The stored estimate must rank an exact-duplicate query's own row first.
        let codec = crate::structures::TqCodec::new(dim);
        for target in [0usize, 7, 20] {
            let plan = crate::structures::TqQueryPlan::build(&codec, &vectors[target]);
            let results = index.search_tq_distinct(3, &plan).unwrap();
            assert_eq!(
                results[0].0, target as u32,
                "query duplicating vector {target} must rank it first: {results:?}"
            );
        }

        // Ordinary merge must not decode or rewrite the corpus columns.
        let (second_bytes, _) = build_tq_payload(dim, 5, 77);
        let second =
            AnnDiskIndex::open(OwnedBytes::new(second_bytes.clone()), AnnKind::TqFlat, 5).unwrap();
        let mut merged_bytes = Vec::new();
        write_merged_ann(&[(&index, 0), (&second, count as u32)], &mut merged_bytes).unwrap();
        let merged = AnnDiskIndex::open(
            OwnedBytes::new(merged_bytes.clone()),
            AnnKind::TqFlat,
            count as u32 + 5,
        )
        .unwrap();
        let mut expected_payload = bytes[ANN_HEADER_SIZE..payload_end(&index)].to_vec();
        expected_payload.extend_from_slice(&second_bytes[ANN_HEADER_SIZE..payload_end(&second)]);
        assert_eq!(
            &merged_bytes[ANN_HEADER_SIZE..payload_end(&merged)],
            expected_payload.as_slice(),
            "TQ merge must be a pure byte copy of the source columns",
        );
        let plan = crate::structures::TqQueryPlan::build(&codec, &vectors[7]);
        let results = merged.search_tq_distinct(1, &plan).unwrap();
        assert_eq!(results[0].0, 7, "merged payload must keep doc bases");
    }

    #[test]
    fn open_rejects_tq_payload_with_inconsistent_geometry() {
        let (bytes, _) = build_tq_payload(20, 4, 9);
        // code_size (header bytes 12..16) is P/2 = 16 for dim 20; corrupt to 15.
        let mut corrupted = bytes.clone();
        corrupted[12..16].copy_from_slice(&15u32.to_le_bytes());
        assert!(
            AnnDiskIndex::open(OwnedBytes::new(corrupted), AnnKind::TqFlat, 4).is_err(),
            "TQ header with code_size != padded_dim/2 must be refused"
        );

        // A truncated codes column (not block-padded) must also be refused.
        let (short_bytes, _) = build_tq_payload(20, 4, 9);
        let mut wrong_kind = short_bytes.clone();
        wrong_kind[4] = AnnKind::IvfPq as u8;
        assert!(
            AnnDiskIndex::open(OwnedBytes::new(wrong_kind), AnnKind::IvfPq, 4).is_err(),
            "TQ block-padded columns must not validate under another kind"
        );
        assert!(AnnDiskIndex::open(OwnedBytes::new(bytes), AnnKind::TqFlat, 4).is_ok());
    }

    #[test]
    fn open_rejects_old_or_out_of_range_ann_payloads() {
        let mut legacy = vec![0u8; ANN_HEADER_SIZE + ANN_FOOTER_SIZE];
        legacy[..4].copy_from_slice(b"old!");
        assert!(AnnDiskIndex::open(OwnedBytes::new(legacy), AnnKind::BinaryIvf, 1).is_err());

        let docs = [0u32];
        let ordinals = [0u16];
        let codes = [0u8];
        let runs = [BuildRun {
            cluster_id: 0,
            doc_ids: &docs,
            ordinals: &ordinals,
            codes: &codes,
        }];
        let mut bytes = Vec::new();
        write_built_runs(binary_header(1), &runs, &mut bytes).unwrap();
        let footer = bytes.len() - ANN_FOOTER_SIZE;
        let directory = usize::try_from(u64::from_le_bytes(
            bytes[footer..footer + 8].try_into().unwrap(),
        ))
        .unwrap();
        bytes[directory + 12..directory + 16].copy_from_slice(&10u32.to_le_bytes());
        assert!(AnnDiskIndex::open(OwnedBytes::new(bytes), AnnKind::BinaryIvf, 1).is_err());
    }
}
