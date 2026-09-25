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

use crate::observe::DenseAnnScanStats;
#[cfg(feature = "native")]
use crate::structures::BinaryIvfIndex;
use crate::structures::vector::index::{BoundedAnnCollector, BoundedUniqueAnnCollector};

/// Combined binary candidates: the retained documents, plus the exact
/// per-ordinal `(doc_id, ordinal, score)` scores behind them.
type CombinedBinaryCandidates = (Vec<AnnDocumentCandidate>, Vec<(u32, u16, f32)>);
/// Distinct-key scan output: `(doc_id, ordinal, score)` hits plus scan diagnostics.
type ScanResults = (Vec<(u32, u16, f32)>, DenseAnnScanStats);

/// Top-k sink shared by the TQ lane loops: the deduplicating collector for
/// multi-valued or SOAR-spilled payloads, the heap-only collector when every
/// `(doc_id, ordinal)` key is unique by construction (single-valued field and
/// exactly one posting per stored vector).
trait IvfTqTopK: Sized + Send {
    fn with_k(k: usize) -> Self;
    fn insert(&mut self, doc_id: u32, ordinal: u16, score: f32);
    /// Conservative k-th best score once `k` entries are held.
    fn pruning_threshold(&self) -> Option<f32>;
    fn non_finite_dropped(&self) -> usize;
    #[cfg(feature = "native")]
    fn merge_from(&mut self, other: Self);
    fn into_sorted_results(self) -> Vec<(u32, u16, f32)>;
}

impl IvfTqTopK for BoundedAnnCollector<true, true> {
    #[inline]
    fn with_k(k: usize) -> Self {
        Self::new(k)
    }
    #[inline]
    fn insert(&mut self, doc_id: u32, ordinal: u16, score: f32) {
        BoundedAnnCollector::insert(self, doc_id, ordinal, score)
    }
    #[inline]
    fn pruning_threshold(&self) -> Option<f32> {
        BoundedAnnCollector::pruning_threshold(self)
    }
    #[inline]
    fn non_finite_dropped(&self) -> usize {
        BoundedAnnCollector::non_finite_dropped(self)
    }
    #[cfg(feature = "native")]
    #[inline]
    fn merge_from(&mut self, other: Self) {
        BoundedAnnCollector::merge_from(self, other)
    }
    #[inline]
    fn into_sorted_results(self) -> Vec<(u32, u16, f32)> {
        BoundedAnnCollector::into_sorted_results(self)
    }
}

impl IvfTqTopK for BoundedUniqueAnnCollector<true> {
    #[inline]
    fn with_k(k: usize) -> Self {
        Self::new(k)
    }
    #[inline]
    fn insert(&mut self, doc_id: u32, ordinal: u16, score: f32) {
        BoundedUniqueAnnCollector::insert(self, doc_id, ordinal, score)
    }
    #[inline]
    fn pruning_threshold(&self) -> Option<f32> {
        BoundedUniqueAnnCollector::pruning_threshold(self)
    }
    #[inline]
    fn non_finite_dropped(&self) -> usize {
        BoundedUniqueAnnCollector::non_finite_dropped(self)
    }
    #[cfg(feature = "native")]
    #[inline]
    fn merge_from(&mut self, other: Self) {
        BoundedUniqueAnnCollector::merge_from(self, other)
    }
    #[inline]
    fn into_sorted_results(self) -> Vec<(u32, u16, f32)> {
        BoundedUniqueAnnCollector::into_sorted_results(self)
    }
}

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
/// Upper bound on the TQ leaf estimate `est⟨q̂,r̂⟩` for a unit residual
/// direction: `base ≤ ‖recon‖ ≈ 1` plus the QJL term `≤ √(π/2)·γ ≤ 1.26·γ`
/// with `γ < 1`, kept with slack so pruning can never drop a candidate the
/// unpruned scan would have kept (pinned by a test).
const TQ_PRUNE_ESTIMATE_BOUND: f32 = 1.3;
/// Flat TQ scans fan out across Rayon above this vector count. Rayon folds
/// chunks into one collector per worker before reducing those collectors, so
/// temporary top-k memory is bounded by the active worker count rather than
/// the number of chunks. Small segments stay sequential because fan-out
/// overhead would dominate.
#[cfg(feature = "native")]
const TQ_PARALLEL_SCAN_MIN_VECTORS: usize = 65_536;
#[cfg(feature = "native")]
const TQ_PARALLEL_SCAN_CHUNK_BLOCKS: usize = 512;
/// IVF-TQ scans are already parallel across segments. Fan out inside one
/// segment only when the selected leaves contain enough postings to amortize
/// Rayon scheduling and worker-local top-k state.
const IVF_PARALLEL_SCAN_MIN_POSTINGS: usize = 65_536;
const IVF_TQ_PARALLEL_SCAN_CHUNK_BLOCKS: usize = 512;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AnnKind {
    // Discriminant 1 was IVF-PQ, removed after IVF-TQ superseded it
    // (docs/turboquant-quantization.md). Never reuse it.
    BinaryIvf = 2,
    /// TurboQuant flat scan: single logical cluster, block-packed codes.
    TqFlat = 3,
    /// Trained IVF router with TurboQuant-coded centroid residuals.
    IvfTq = 4,
    /// ScaNN float leaves encoded with 4-bit asymmetric hashes.
    ScannAh = 5,
    /// ScaNN binary leaves retaining exact packed vectors for Hamming scan.
    ScannBinary = 6,
}

impl AnnKind {
    fn from_u8(value: u8) -> io::Result<Self> {
        match value {
            1 => Err(invalid_data(
                "ANN kind 1 (IVF-PQ) is no longer supported; recreate the index \
                 with `ivf_tq` and reindex",
            )),
            2 => Ok(Self::BinaryIvf),
            3 => Ok(Self::TqFlat),
            4 => Ok(Self::IvfTq),
            5 => Ok(Self::ScannAh),
            6 => Ok(Self::ScannBinary),
            _ => Err(invalid_data(format!("unknown ANN kind {value}"))),
        }
    }
}

/// Codes-column byte length for one run. TQ packs vectors into 16-lane
/// blocks (gammas + dimension-major nibbles), so its column is block-padded
/// rather than `count * code_size`.
fn expected_codes_column_len(
    kind: AnnKind,
    count: usize,
    dim: usize,
    code_size: usize,
) -> io::Result<usize> {
    match kind {
        AnnKind::BinaryIvf | AnnKind::ScannBinary => count
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
        // For ScaNN AH, `code_size` stores dimensions-per-block. This is the
        // one encoding parameter not derivable from the fixed ANN header;
        // the actual byte length remains count-dependent because complete
        // 32-row FastScan blocks interleave their nibbles.
        AnnKind::ScannAh => crate::structures::vector::scann::ScannEncoding::AsymmetricHash {
            dimensions_per_block: u16::try_from(code_size)
                .map_err(|_| invalid_data("ScaNN AH block dimension exceeds u16"))?,
            bits_per_code: 4,
        }
        .leaf_code_bytes(
            u32::try_from(dim).map_err(|_| invalid_data("ScaNN dimension exceeds u32"))?,
            count,
        )
        .map_err(|error| invalid_data(error.to_string())),
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

#[derive(Clone, Copy)]
struct IvfTqScanTask<'a> {
    run: &'a AnnRun,
    cluster_dot: f32,
    first_block: usize,
    last_block: usize,
}

#[derive(Clone, Copy)]
struct BinaryScanTask<'a> {
    run: &'a AnnRun,
    first_index: usize,
    count: usize,
}

/// Mmap-backed searchable ANN payload. Only the fixed-size run directory is
/// heap-resident; all corpus-sized columns remain zero-copy file slices.
#[derive(Clone)]
pub(crate) struct AnnDiskIndex {
    pub(crate) alive_docs: Option<std::sync::Arc<crate::query::DocBitset>>,
    // Drop locks before the directory allocation they reference.
    #[cfg(feature = "native")]
    heap_pins: std::sync::Arc<crate::segment::pin::HeapPinSet>,
    raw: OwnedBytes,
    header: AnnDiskHeader,
    runs: std::sync::Arc<Vec<AnnRun>>,
}

/// Cheap structural health of one ANN payload, computed from the in-memory
/// run directory in O(runs) — payload bytes are never touched.
///
/// Two production failure modes motivated every field here (see
/// `docs/diagnostics.md`): a 31%-of-vectors leaf built from degenerate
/// embeddings, and probe read-amplification from byte-copy merges that leave
/// one logical cluster scattered across many physical extents.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnnHealth {
    /// Vectors across all runs.
    pub vectors: u64,
    /// Distinct cluster IDs holding at least one posting.
    pub clusters_nonempty: u32,
    /// Codebook size (`num_clusters` from the header).
    pub clusters_total: u32,
    /// Run-directory entries; each is one physical extent on disk.
    pub runs: u32,
    /// Vectors in the most populated cluster, with its ID.
    pub largest_cluster: u32,
    pub largest_cluster_vectors: u64,
    /// Faiss `imbalance_factor` over non-empty clusters:
    /// `K · Σ nᵢ² / N²`. 1.0 is perfectly balanced; a value of γ means a
    /// fixed-`nprobe` probe computes γ× the distances of the balanced
    /// baseline in expectation.
    pub imbalance: f64,
    /// Bytes of the codes columns (what a probe of every leaf would read).
    pub payload_bytes: u64,
}

impl AnnHealth {
    /// Physical extents per non-empty cluster. 1.0 after a rebuild; each
    /// byte-copy merge multiplies it, and every extent is a potential seek
    /// when the index is cold.
    pub fn fragmentation(&self) -> f64 {
        if self.clusters_nonempty == 0 {
            return 0.0;
        }
        f64::from(self.runs) / f64::from(self.clusters_nonempty)
    }

    /// Share of all vectors held by the single largest cluster.
    pub fn largest_cluster_share(&self) -> f64 {
        if self.vectors == 0 {
            return 0.0;
        }
        self.largest_cluster_vectors as f64 / self.vectors as f64
    }
}

/// `largest_cluster_share` above this, on at least [`ANN_SKEW_WARN_MIN_VECTORS`]
/// vectors, is a scan cliff worth a warning: the production incident value was
/// 0.31, and a healthy 100k+-cluster codebook sits orders of magnitude lower.
const ANN_SKEW_WARN_SHARE: f64 = 0.05;
const ANN_SKEW_WARN_MIN_VECTORS: u64 = 100_000;
/// A rebuilt segment has fragmentation 1.0; a single 32-way byte-copy merge
/// can reach 32. Warn once probes pay ~an order of magnitude extra seeks.
const ANN_FRAGMENTATION_WARN: f64 = 8.0;

/// A document selected by a combiner-aware compressed scan.
///
/// This intentionally has no ordinal: `score` combines every value belonging
/// to the document and must never be reused as an individual vector score by
/// an exact reranker.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct AnnDocumentCandidate {
    pub(crate) doc_id: u32,
    pub(crate) score: f32,
}

/// Temporary admission view for one exact-code span; no vector bytes are decoded.
pub(super) struct ExactVectorSpan<'a> {
    docs: &'a [u8],
    ordinals: &'a [u8],
    doc_base: u32,
}

impl ExactVectorSpan<'_> {
    pub(super) fn validate_label(&self, row: u32, doc: u32, ordinal: u16) -> io::Result<()> {
        let row = row as usize;
        if row >= self.docs.len() / 4 {
            return Err(invalid_data("vector lookup row is out of range"));
        }
        if read_u32(self.docs, row * 4).checked_add(self.doc_base) != Some(doc)
            || read_u16(self.ordinals, row * 2) != ordinal
        {
            return Err(invalid_data("vector location labels disagree with ANN row"));
        }
        Ok(())
    }
}

impl AnnDiskIndex {
    /// Share the existing immutable payload owner with the exact-vector view.
    /// In particular, HTTP readers must not fetch the same ANN codes twice.
    pub(crate) fn exact_vectors_handle(&self) -> crate::directories::FileHandle {
        crate::directories::FileHandle::from_bytes(self.raw.clone())
    }

    #[cfg(feature = "native")]
    pub(crate) fn copied_payload_bytes(&self) -> usize {
        self.runs
            .iter()
            .map(|run| run.codes.end)
            .max()
            .expect("ANN has runs")
            - ANN_HEADER_SIZE
    }

    /// Upper bound for binary coalescing directories, span metadata, and
    /// document-rebase scratch. The lookup sorter has a separate budget.
    #[cfg(feature = "native")]
    pub(crate) fn binary_compaction_scratch_bytes(&self) -> io::Result<usize> {
        self.runs
            .len()
            .checked_mul(2 * (std::mem::size_of::<RunRecord>() + std::mem::size_of::<(u64, u32)>()))
            .and_then(|bytes| bytes.checked_add(DOC_ID_REWRITE_CHUNK * 4 + 4096))
            .ok_or_else(|| invalid_data("binary compaction scratch size overflow"))
    }

    pub(super) fn exact_location_spans(
        &self,
        spans: impl Iterator<Item = (u64, u32)>,
    ) -> io::Result<Vec<ExactVectorSpan<'_>>> {
        if !matches!(self.header.kind, AnnKind::BinaryIvf | AnnKind::ScannBinary) {
            return Err(invalid_data(
                "exact-vector lookup requires exact binary ANN codes",
            ));
        }
        let width = self.header.code_size;
        let mut runs: Vec<_> = self.runs.iter().collect();
        runs.sort_unstable_by_key(|run| run.codes.start);
        #[cfg(feature = "native")]
        let mut ordinal_prefetch_bytes = 8 * 1024 * 1024usize;
        #[cfg(feature = "native")]
        let prefetch_page_size = usize::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) })
            .ok()
            .filter(|&page| page > 0);
        spans
            .map(|(offset, count)| {
                let offset = usize::try_from(offset)
                    .map_err(|_| invalid_data("vector location exceeds address space"))?;
                let i = runs.partition_point(|run| run.codes.start <= offset);
                let run = i
                    .checked_sub(1)
                    .and_then(|i| runs.get(i))
                    .ok_or_else(|| invalid_data("vector location precedes ANN codes"))?;
                let relative = offset - run.codes.start;
                let bytes = (count as usize).checked_mul(width);
                if !relative.is_multiple_of(width)
                    || bytes
                        .and_then(|bytes| offset.checked_add(bytes))
                        .is_none_or(|end| end > run.codes.end)
                {
                    return Err(invalid_data(
                        "vector location span is not within ANN code rows",
                    ));
                }
                let row = relative / width;
                let end = row + count as usize;
                // Code references require ordinal labels too. Bound eager
                // metadata I/O independently of corpus size.
                #[cfg(feature = "native")]
                if let Some(page) = prefetch_page_size
                    && ordinal_prefetch_bytes > 0
                {
                    let start = run.ordinals.start + row * 2;
                    let len = count as usize * 2;
                    let address = self.raw.as_slice().as_ptr() as usize + start;
                    let touched = (address % page + len).div_ceil(page) * page;
                    if touched <= ordinal_prefetch_bytes {
                        self.raw
                            .madvise_range(start..start + len, libc::MADV_WILLNEED);
                        ordinal_prefetch_bytes -= touched;
                    }
                }
                Ok(ExactVectorSpan {
                    docs: &self.raw[run.doc_ids.start + row * 4..run.doc_ids.start + end * 4],
                    ordinals: &self.raw[run.ordinals.start + row * 2..run.ordinals.start + end * 2],
                    doc_base: run.doc_base,
                })
            })
            .collect()
    }

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
        check_layout_revision(kind, header_cursor.read_u32::<LittleEndian>()?)?;
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
            let expected_codes_len = expected_codes_column_len(kind, count, dim, code_size)?;
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
        for &index in &payload_order {
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
        // Validate every doc-ID column once so the scan loops can resolve
        // IDs with plain reads (`run_doc_id_validated`) instead of a per-lane
        // bounds/overflow check. A corrupt column fails loud here.
        for &index in &payload_order {
            let run = &runs[index];
            let column = &bytes[run.doc_ids.clone()];
            if let Some(local_doc_id) = column
                .chunks_exact(std::mem::size_of::<u32>())
                .map(|entry| u32::from_le_bytes([entry[0], entry[1], entry[2], entry[3]]))
                .find(|&local_doc_id| local_doc_id > run.max_doc_id)
            {
                return Err(invalid_data(format!(
                    "ANN run for cluster {} contains local document {local_doc_id} above its declared maximum {}",
                    run.cluster_id, run.max_doc_id
                )));
            }
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
            alive_docs: None,
            #[cfg(feature = "native")]
            heap_pins: Default::default(),
            raw,
            header,
            runs: std::sync::Arc::new(runs),
        })
    }

    /// Structural health from the run directory alone. O(runs), no payload
    /// reads — safe to call at every open.
    pub(crate) fn health(&self) -> AnnHealth {
        let mut vectors = 0u64;
        let mut clusters_nonempty = 0u32;
        let mut payload_bytes = 0u64;
        let mut largest = (0u32, 0u64);
        let mut sum_squares = 0f64;
        // Runs are sorted by cluster ID, so one pass groups them.
        let mut index = 0usize;
        while index < self.runs.len() {
            let cluster_id = self.runs[index].cluster_id;
            let mut cluster_vectors = 0u64;
            while index < self.runs.len() && self.runs[index].cluster_id == cluster_id {
                let run = &self.runs[index];
                cluster_vectors += run.count as u64;
                payload_bytes += (run.codes.end - run.codes.start) as u64;
                index += 1;
            }
            vectors += cluster_vectors;
            clusters_nonempty += 1;
            sum_squares += (cluster_vectors as f64) * (cluster_vectors as f64);
            if cluster_vectors > largest.1 {
                largest = (cluster_id, cluster_vectors);
            }
        }
        let imbalance = if vectors == 0 || clusters_nonempty == 0 {
            0.0
        } else {
            f64::from(clusters_nonempty) * sum_squares / ((vectors as f64) * (vectors as f64))
        };
        AnnHealth {
            vectors,
            clusters_nonempty,
            clusters_total: self.header.num_clusters,
            runs: self.runs.len() as u32,
            largest_cluster: largest.0,
            largest_cluster_vectors: largest.1,
            imbalance,
            payload_bytes,
        }
    }

    /// Log this payload's health, warning on the two known cliff shapes.
    ///
    /// Called once per segment open; the caller supplies identity because the
    /// payload itself does not know its index or field.
    pub(crate) fn report_health(&self, index_label: &str, field_id: u32, segment_id: u128) {
        let health = self.health();
        let share = health.largest_cluster_share();
        let fragmentation = health.fragmentation();
        log::info!(
            "[ann_health] index={index_label} field={field_id} segment={segment_id:016x}: \
             vectors={} clusters={}/{} runs={} fragmentation={fragmentation:.2} \
             imbalance={:.2} largest_leaf={:.2}% payload={}",
            health.vectors,
            health.clusters_nonempty,
            health.clusters_total,
            health.runs,
            health.imbalance,
            100.0 * share,
            crate::format_bytes(health.payload_bytes),
        );
        crate::observe::ann_health(
            index_label,
            field_id,
            health.imbalance,
            fragmentation,
            share,
        );
        if share >= ANN_SKEW_WARN_SHARE && health.vectors >= ANN_SKEW_WARN_MIN_VECTORS {
            log::warn!(
                "[ann_health] index={index_label} field={field_id} segment={segment_id:016x}: \
                 leaf {} holds {:.1}% of {} vectors — every query probing it scans that leaf \
                 in full; degenerate embeddings collapse into one leaf exactly like this",
                health.largest_cluster,
                100.0 * share,
                health.vectors,
            );
        }
        if fragmentation >= ANN_FRAGMENTATION_WARN {
            log::warn!(
                "[ann_health] index={index_label} field={field_id} segment={segment_id:016x}: \
                 {fragmentation:.1} extents per probed cluster ({} runs / {} clusters) — \
                 cold probes can require multiple seeks; byte-copy merges preserve source extents",
                health.runs,
                health.clusters_nonempty,
            );
        }
    }

    pub(crate) fn header(&self) -> &AnnDiskHeader {
        &self.header
    }

    /// Refuse to pair a segment payload with any global ScaNN model other
    /// than the exact generation that encoded it.
    pub(crate) fn validate_scann_generation(
        &self,
        config: &crate::structures::vector::scann::ScannConfig,
        generation: u64,
        artifact_id: u64,
    ) -> io::Result<()> {
        use crate::structures::vector::scann::ScannEncoding;

        let expected_kind = match config.encoding {
            ScannEncoding::AsymmetricHash { .. } => AnnKind::ScannAh,
            ScannEncoding::BinaryHamming => AnnKind::ScannBinary,
        };
        if self.header.kind != expected_kind
            || self.header.routing != IvfRoutingMode::Flat
            || self.header.dim != config.dimension as usize
            || self.header.num_clusters != config.num_leaves
            || self.header.quantizer_version != generation
            || self.header.codebook_version != artifact_id
        {
            return Err(invalid_data(
                "ScaNN ANN payload does not match the global trained generation",
            ));
        }
        match config.encoding {
            ScannEncoding::AsymmetricHash {
                dimensions_per_block,
                bits_per_code: 4,
            } if self.header.code_size == usize::from(dimensions_per_block) => Ok(()),
            ScannEncoding::BinaryHamming
                if self.header.code_size == config.dimension as usize / 8 =>
            {
                Ok(())
            }
            _ => Err(invalid_data(
                "ScaNN ANN payload encoding does not match the global trained artifact",
            )),
        }
    }

    /// Validate physical leaf postings against the logical flat-vector count.
    /// Float ScaNN and primary-only binary ScaNN are exact. Binary spilling
    /// may add at most one posting per logical vector, with target-fraction
    /// policies retaining a stricter segment-local cap.
    #[cfg(feature = "native")]
    pub(crate) fn validate_scann_posting_count(
        &self,
        logical_vectors: usize,
        soar: Option<&crate::structures::SoarConfig>,
    ) -> io::Result<()> {
        let spill_budget = match self.header.kind {
            AnnKind::ScannAh => 0,
            AnnKind::ScannBinary
                if self.header.num_clusters > 1
                    && soar.is_some_and(|config| config.num_secondary > 0) =>
            {
                match soar.and_then(crate::structures::SoarConfig::calibration_target) {
                    Some(target_fraction) => {
                        (logical_vectors as f64 * f64::from(target_fraction)).floor() as usize
                    }
                    None => logical_vectors,
                }
            }
            AnnKind::ScannBinary => 0,
            _ => {
                return Err(invalid_data(
                    "posting-count validation requires a ScaNN ANN payload",
                ));
            }
        };
        let maximum = logical_vectors
            .checked_add(spill_budget)
            .ok_or_else(|| invalid_data("ScaNN posting-count bound overflows usize"))?;
        if self.header.vector_count < logical_vectors || self.header.vector_count > maximum {
            return Err(invalid_data(format!(
                "ScaNN ANN payload has {} physical postings for {logical_vectors} logical vectors; expected {logical_vectors}..={maximum}",
                self.header.vector_count,
            )));
        }
        Ok(())
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
        let Some(pins) = std::sync::Arc::get_mut(&mut self.heap_pins) else {
            log::warn!("[pin] ANN directory already shared across reader generations");
            return;
        };
        // Keep the allocation alive until the last shared pin guard is dropped.
        pins.retain_owner(std::sync::Arc::clone(&self.runs));
        let before = pins.report();
        pins.pin_slice(&self.runs, "ANN cluster-run directory", mode, remaining);
        let after = pins.report();
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

    #[cfg(feature = "native")]
    fn ivf_tq_scan_tasks<'a>(
        &'a self,
        plan: &'a crate::structures::TqIvfQueryPlan,
        block_bytes: usize,
        chunk_blocks: usize,
    ) -> Vec<IvfTqScanTask<'a>> {
        debug_assert!(chunk_blocks > 0);
        let mut tasks = Vec::new();
        for (cluster_id, cluster_dot) in plan.cluster_dots() {
            for run in self.cluster_runs(cluster_id) {
                let block_count = run.codes.len() / block_bytes;
                for first_block in (0..block_count).step_by(chunk_blocks) {
                    tasks.push(IvfTqScanTask {
                        run,
                        cluster_dot,
                        first_block,
                        last_block: (first_block + chunk_blocks).min(block_count),
                    });
                }
            }
        }
        tasks
    }

    #[cfg(feature = "native")]
    fn binary_scan_tasks<'a>(&'a self, cluster_ids: &[u32]) -> Vec<BinaryScanTask<'a>> {
        let mut tasks = Vec::new();
        for &cluster_id in cluster_ids {
            for run in self.cluster_runs(cluster_id) {
                for first_index in (0..run.count).step_by(BINARY_SCORE_BATCH) {
                    tasks.push(BinaryScanTask {
                        run,
                        first_index,
                        count: BINARY_SCORE_BATCH.min(run.count - first_index),
                    });
                }
            }
        }
        tasks
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

    /// Score a flat TQ payload while every value of a document is still in
    /// hand, combine those approximate scores, and retain document-level
    /// top-k. TQ build runs preserve `(doc_id, ordinal)` input order, so the
    /// scratch space is bounded by one document plus the retained heap.
    pub(crate) fn search_tq_combined_documents(
        &self,
        k: usize,
        plan: &crate::structures::TqQueryPlan,
        combiner: crate::query::MultiValueCombiner,
    ) -> io::Result<Vec<AnnDocumentCandidate>> {
        use crate::structures::vector::quantization::{TQ_BLOCK_LANES, tq_block_bytes};

        combiner
            .validate()
            .map_err(|message| io::Error::new(io::ErrorKind::InvalidInput, message))?;
        if plan.padded_dim() != self.header.code_size * 2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "TQ query plan does not match the payload dimension",
            ));
        }
        if k == 0 {
            return Ok(Vec::new());
        }

        let block_bytes = tq_block_bytes(self.header.code_size);
        let bytes = self.raw.as_slice();
        let mut top_documents = BoundedAnnCollector::<true, true>::new(k);
        let mut ordinal_scores = Vec::new();
        let mut scores = [0.0f32; TQ_BLOCK_LANES];

        for run in self.runs.iter() {
            let mut current_doc = None;
            let codes = &bytes[run.codes.clone()];
            for (block_index, block) in codes.chunks_exact(block_bytes).enumerate() {
                crate::structures::vector::quantization::tq_score_block(plan, block, &mut scores);
                let lane_base = block_index * TQ_BLOCK_LANES;
                let lanes = TQ_BLOCK_LANES.min(run.count.saturating_sub(lane_base));
                for (lane, &score) in scores.iter().enumerate().take(lanes) {
                    let index = lane_base + lane;
                    let doc_id = run_doc_id_validated(bytes, run, index);
                    if self
                        .alive_docs
                        .as_ref()
                        .is_some_and(|bits| !bits.contains(doc_id))
                    {
                        continue;
                    }
                    if current_doc.is_some_and(|previous| doc_id < previous) {
                        return Err(invalid_data("flat TQ run is not grouped by document ID"));
                    }
                    if current_doc.is_some_and(|previous| doc_id != previous) {
                        retain_combined_document(
                            &mut top_documents,
                            current_doc.expect("current document is present"),
                            &ordinal_scores,
                            combiner,
                        );
                        ordinal_scores.clear();
                    }
                    current_doc = Some(doc_id);
                    if score.is_finite() {
                        ordinal_scores.push((
                            u32::from(read_u16(bytes, run.ordinals.start + index * 2)),
                            score,
                        ));
                    }
                }
            }
            if let Some(doc_id) = current_doc {
                retain_combined_document(&mut top_documents, doc_id, &ordinal_scores, combiner);
                ordinal_scores.clear();
            }
        }

        Ok(top_documents
            .into_sorted_results()
            .into_iter()
            .map(|(doc_id, _, score)| AnnDocumentCandidate { doc_id, score })
            .collect())
    }

    /// Score every posting in the probed IVF-TQ leaves, deduplicate SOAR
    /// assignments by `(doc_id, ordinal)`, combine every approximate ordinal
    /// present in the probed leaves, and retain at most `k` documents.
    ///
    /// Unlike the Max path, this deliberately performs no individual-score
    /// pruning: an ordinal that cannot enter vector top-k may still change a
    /// Sum, Avg, LogSumExp, or WeightedTopK document score.
    #[cfg(test)]
    pub(crate) fn search_ivf_tq_combined_documents(
        &self,
        k: usize,
        plan: &crate::structures::TqIvfQueryPlan,
        combiner: crate::query::MultiValueCombiner,
    ) -> io::Result<Vec<AnnDocumentCandidate>> {
        self.search_ivf_tq_combined_documents_with_stats(k, plan, combiner)
            .map(|(candidates, _)| candidates)
    }

    /// [`Self::search_ivf_tq_combined_documents`] plus the scan diagnostics:
    /// every probed posting is materialized here, so `posting_count` is the
    /// buffer this path allocates.
    pub(crate) fn search_ivf_tq_combined_documents_with_stats(
        &self,
        k: usize,
        plan: &crate::structures::TqIvfQueryPlan,
        combiner: crate::query::MultiValueCombiner,
    ) -> io::Result<(Vec<AnnDocumentCandidate>, DenseAnnScanStats)> {
        self.search_ivf_tq_combined_documents_impl(
            k,
            plan,
            combiner,
            IVF_PARALLEL_SCAN_MIN_POSTINGS,
            IVF_TQ_PARALLEL_SCAN_CHUNK_BLOCKS,
        )
    }

    #[cfg(test)]
    fn search_ivf_tq_combined_documents_with_tuning(
        &self,
        k: usize,
        plan: &crate::structures::TqIvfQueryPlan,
        combiner: crate::query::MultiValueCombiner,
        parallel_min_postings: usize,
        parallel_chunk_blocks: usize,
    ) -> io::Result<Vec<AnnDocumentCandidate>> {
        self.search_ivf_tq_combined_documents_impl(
            k,
            plan,
            combiner,
            parallel_min_postings,
            parallel_chunk_blocks,
        )
        .map(|(candidates, _)| candidates)
    }

    fn search_ivf_tq_combined_documents_impl(
        &self,
        k: usize,
        plan: &crate::structures::TqIvfQueryPlan,
        combiner: crate::query::MultiValueCombiner,
        parallel_min_postings: usize,
        parallel_chunk_blocks: usize,
    ) -> io::Result<(Vec<AnnDocumentCandidate>, DenseAnnScanStats)> {
        use crate::structures::vector::quantization::{
            TQ_BLOCK_LANES, tq_ivf_block_bytes, tq_score_ivf_block,
        };

        validate_combined_search(combiner)?;
        let tq_plan = plan.tq_plan();
        self.validate_ivf_tq_query_plan(plan)?;
        #[cfg(not(feature = "native"))]
        let _ = (parallel_min_postings, parallel_chunk_blocks);
        if k == 0 {
            return Ok((Vec::new(), DenseAnnScanStats::default()));
        }
        #[cfg(feature = "native")]
        self.prefetch_cluster_runs(&plan.cluster_ids);

        let block_bytes = tq_ivf_block_bytes(self.header.code_size);
        let bytes = self.raw.as_slice();
        let posting_count = probed_posting_count(self, &plan.cluster_ids)?;
        let mut stats = DenseAnnScanStats {
            posting_count,
            ..DenseAnnScanStats::default()
        };

        #[cfg(feature = "native")]
        if rayon::current_num_threads() > 1 && posting_count >= parallel_min_postings {
            use rayon::prelude::*;
            let tasks = self.ivf_tq_scan_tasks(plan, block_bytes, parallel_chunk_blocks);
            let (ordinal_scores, scored_blocks, non_finite) = tasks
                .par_iter()
                .fold(
                    || {
                        (
                            Vec::with_capacity(parallel_chunk_blocks * TQ_BLOCK_LANES),
                            0usize,
                            0usize,
                        )
                    },
                    |(mut ordinal_scores, scored, non_finite), task| {
                        let (task_scored, task_non_finite) = score_ivf_tq_combined_blocks(
                            self.alive_docs.as_deref(),
                            bytes,
                            tq_plan,
                            block_bytes,
                            *task,
                            &mut ordinal_scores,
                        );
                        (
                            ordinal_scores,
                            scored + task_scored,
                            non_finite + task_non_finite,
                        )
                    },
                )
                .reduce(
                    || (Vec::new(), 0usize, 0usize),
                    |(mut left, left_scored, left_non_finite),
                     (mut right, right_scored, right_non_finite)| {
                        left.append(&mut right);
                        (
                            left,
                            left_scored + right_scored,
                            left_non_finite + right_non_finite,
                        )
                    },
                );
            stats.scored_blocks = scored_blocks;
            stats.non_finite_dropped = non_finite;
            stats.parallel = true;
            return Ok((combine_scored_ordinals(ordinal_scores, k, combiner), stats));
        }

        let mut ordinal_scores = Vec::new();
        ordinal_scores
            .try_reserve_exact(posting_count)
            .map_err(|_| invalid_data("IVF-TQ combined score buffer allocation failed"))?;
        let mut scores = [0.0f32; TQ_BLOCK_LANES];
        for (cluster_id, cluster_dot) in plan.cluster_dots() {
            for run in self.cluster_runs(cluster_id) {
                let codes = &bytes[run.codes.clone()];
                for (block_index, block) in codes.chunks_exact(block_bytes).enumerate() {
                    tq_score_ivf_block(tq_plan, block, cluster_dot, &mut scores);
                    stats.scored_blocks += 1;
                    let lane_base = block_index * TQ_BLOCK_LANES;
                    let lanes = TQ_BLOCK_LANES.min(run.count.saturating_sub(lane_base));
                    for (lane, &score) in scores.iter().enumerate().take(lanes) {
                        if !score.is_finite() {
                            stats.non_finite_dropped += 1;
                            continue;
                        }
                        let index = lane_base + lane;
                        let doc_id = run_doc_id_validated(bytes, run, index);
                        if self
                            .alive_docs
                            .as_ref()
                            .is_some_and(|bits| !bits.contains(doc_id))
                        {
                            continue;
                        }
                        ordinal_scores.push((
                            doc_id,
                            read_u16(bytes, run.ordinals.start + index * 2),
                            score,
                        ));
                    }
                }
            }
        }
        Ok((combine_scored_ordinals(ordinal_scores, k, combiner), stats))
    }

    /// Score packed codes in the selected binary-IVF leaves, deduplicate
    /// repeated `(doc_id, ordinal)` assignments, combine per document, and
    /// retain at most `k` approximate document candidates.
    ///
    /// The second return value carries the per-ordinal scores behind the
    /// retained documents. Binary leaves store the original packed codes, so
    /// those scores are *exact*, and reranking can reuse them instead of
    /// re-reading the same codes from flat storage.
    pub(crate) fn search_binary_combined_documents(
        &self,
        k: usize,
        query: &[u8],
        cluster_ids: &[u32],
        combiner: crate::query::MultiValueCombiner,
    ) -> io::Result<CombinedBinaryCandidates> {
        self.search_binary_combined_documents_with_tuning(
            k,
            query,
            cluster_ids,
            combiner,
            IVF_PARALLEL_SCAN_MIN_POSTINGS,
        )
    }

    /// Score float ScaNN AH codes in the routed leaves and retain approximate
    /// document candidates for the shared exact-flat reranker.
    pub(crate) fn search_scann_ah_combined_documents(
        &self,
        k: usize,
        query: &crate::structures::vector::scann::FloatScannQuery,
        combiner: crate::query::MultiValueCombiner,
    ) -> io::Result<Vec<AnnDocumentCandidate>> {
        self.search_scann_ah_combined_documents_with_tuning(
            k,
            query,
            combiner,
            IVF_PARALLEL_SCAN_MIN_POSTINGS,
        )
    }

    /// Bounded float ScaNN leaf scan.
    ///
    /// The FastScan lookup table is built once per query (it lives in the
    /// shared `FloatScannQuery`), the kernel is resolved once, and scoring
    /// never materializes more than the probed posting count. `Max` streams
    /// every lane into a document-level top-k collector and uses its k-th
    /// best score as an integer lane cutoff (FAISS-style), so rows that cannot
    /// enter the top-k are rejected before any float conversion or doc-ID
    /// decode. Other combiners need every ordinal of a document and keep the
    /// compact `(doc, ordinal, score)` path, now reserved up front and fanned
    /// out across Rayon above the same posting threshold as the binary scan.
    fn search_scann_ah_combined_documents_with_tuning(
        &self,
        k: usize,
        query: &crate::structures::vector::scann::FloatScannQuery,
        combiner: crate::query::MultiValueCombiner,
        parallel_min_postings: usize,
    ) -> io::Result<Vec<AnnDocumentCandidate>> {
        validate_combined_search(combiner)?;
        if self.header.kind != AnnKind::ScannAh {
            return Err(invalid_data(
                "ScaNN AH query used with a different ANN payload",
            ));
        }
        #[cfg(not(feature = "native"))]
        let _ = parallel_min_postings;
        if k == 0 {
            return Ok(Vec::new());
        }
        #[cfg(feature = "native")]
        self.prefetch_cluster_runs(query.routed_leaves());
        let blocks = self.header.dim.div_ceil(self.header.code_size);
        if query.fast_scan().blocks() != blocks {
            return Err(invalid_data(format!(
                "ScaNN query has {} AH blocks but the payload stores {blocks}",
                query.fast_scan().blocks()
            )));
        }
        let geometry = ScannLeafGeometry {
            packed_block_bytes: query.fast_scan().packed_block_bytes(),
            packed_row_bytes: blocks.div_ceil(2),
        };
        let bytes = self.raw.as_slice();
        let posting_count = probed_posting_count(self, query.routed_leaves())?;

        if matches!(combiner, crate::query::MultiValueCombiner::Max) {
            #[cfg(feature = "native")]
            if rayon::current_num_threads() > 1 && posting_count >= parallel_min_postings {
                use rayon::prelude::*;
                let tasks = self.scann_scan_tasks(query, SCANN_PARALLEL_SCAN_CHUNK_ROWS);
                let collector = tasks
                    .par_iter()
                    .try_fold(
                        || BoundedAnnCollector::<true, true>::new(k),
                        |mut collector, task| {
                            scan_scann_ah_task(
                                self.alive_docs.as_deref(),
                                bytes,
                                query,
                                geometry,
                                *task,
                                &mut collector,
                            )?;
                            Ok::<_, io::Error>(collector)
                        },
                    )
                    .try_reduce(
                        || BoundedAnnCollector::<true, true>::new(k),
                        |mut left, right| {
                            left.merge_from(right);
                            Ok(left)
                        },
                    )?;
                return Ok(scann_document_candidates(collector));
            }
            let mut collector = BoundedAnnCollector::<true, true>::new(k);
            for (leaf, centroid_dot) in query.routed() {
                for run in self.cluster_runs(leaf) {
                    scan_scann_ah_task(
                        self.alive_docs.as_deref(),
                        bytes,
                        query,
                        geometry,
                        ScannScanTask {
                            run,
                            centroid_dot,
                            first_row: 0,
                            count: run.count,
                        },
                        &mut collector,
                    )?;
                }
            }
            return Ok(scann_document_candidates(collector));
        }

        #[cfg(feature = "native")]
        if rayon::current_num_threads() > 1 && posting_count >= parallel_min_postings {
            use rayon::prelude::*;
            let tasks = self.scann_scan_tasks(query, SCANN_PARALLEL_SCAN_CHUNK_ROWS);
            let ordinal_scores = tasks
                .par_iter()
                .try_fold(
                    || Vec::with_capacity(SCANN_PARALLEL_SCAN_CHUNK_ROWS),
                    |mut ordinal_scores, task| {
                        scan_scann_ah_task(
                            self.alive_docs.as_deref(),
                            bytes,
                            query,
                            geometry,
                            *task,
                            &mut ordinal_scores,
                        )?;
                        Ok::<_, io::Error>(ordinal_scores)
                    },
                )
                .try_reduce(Vec::new, |mut left, mut right| {
                    left.append(&mut right);
                    Ok(left)
                })?;
            return Ok(combine_scored_ordinals(ordinal_scores, k, combiner));
        }
        let mut ordinal_scores = Vec::new();
        ordinal_scores
            .try_reserve_exact(posting_count)
            .map_err(|_| invalid_data("ScaNN combined score buffer allocation failed"))?;
        for (leaf, centroid_dot) in query.routed() {
            for run in self.cluster_runs(leaf) {
                scan_scann_ah_task(
                    self.alive_docs.as_deref(),
                    bytes,
                    query,
                    geometry,
                    ScannScanTask {
                        run,
                        centroid_dot,
                        first_row: 0,
                        count: run.count,
                    },
                    &mut ordinal_scores,
                )?;
            }
        }
        Ok(combine_scored_ordinals(ordinal_scores, k, combiner))
    }

    /// Split the routed ScaNN leaves into FastScan-group-aligned row ranges.
    /// The task that reaches a run's end also owns its row-major tail.
    #[cfg(feature = "native")]
    fn scann_scan_tasks<'a>(
        &'a self,
        query: &crate::structures::vector::scann::FloatScannQuery,
        chunk_rows: usize,
    ) -> Vec<ScannScanTask<'a>> {
        debug_assert!(chunk_rows.is_multiple_of(crate::structures::vector::scann::FAST_SCAN_LANES));
        let mut tasks = Vec::new();
        for (leaf, centroid_dot) in query.routed() {
            for run in self.cluster_runs(leaf) {
                for first_row in (0..run.count).step_by(chunk_rows) {
                    tasks.push(ScannScanTask {
                        run,
                        centroid_dot,
                        first_row,
                        count: chunk_rows.min(run.count - first_row),
                    });
                }
            }
        }
        tasks
    }

    fn search_binary_combined_documents_with_tuning(
        &self,
        k: usize,
        query: &[u8],
        cluster_ids: &[u32],
        combiner: crate::query::MultiValueCombiner,
        parallel_min_postings: usize,
    ) -> io::Result<CombinedBinaryCandidates> {
        validate_combined_search(combiner)?;
        #[cfg(not(feature = "native"))]
        let _ = parallel_min_postings;
        if query.len() != self.header.code_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "binary ANN query has the wrong byte length",
            ));
        }
        if k == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        #[cfg(feature = "native")]
        self.prefetch_cluster_runs(cluster_ids);

        let bytes = self.raw.as_slice();
        let posting_count = probed_posting_count(self, cluster_ids)?;
        #[cfg(feature = "native")]
        if rayon::current_num_threads() > 1 && posting_count >= parallel_min_postings {
            use rayon::prelude::*;
            let tasks = self.binary_scan_tasks(cluster_ids);
            let (ordinal_scores, _) = tasks
                .par_iter()
                .try_fold(
                    || {
                        (
                            Vec::with_capacity(BINARY_SCORE_BATCH),
                            vec![0.0f32; BINARY_SCORE_BATCH],
                        )
                    },
                    |(mut ordinal_scores, mut scores), task| {
                        score_binary_task(
                            self.alive_docs.as_deref(),
                            bytes,
                            query,
                            self.header.dim,
                            self.header.code_size,
                            *task,
                            &mut scores,
                            &mut ordinal_scores,
                        )?;
                        Ok::<_, io::Error>((ordinal_scores, scores))
                    },
                )
                .try_reduce(
                    || (Vec::new(), Vec::new()),
                    |(mut left, scores), (mut right, _)| {
                        left.append(&mut right);
                        Ok((left, scores))
                    },
                )?;
            return Ok(combine_scored_ordinals_retaining(
                ordinal_scores,
                k,
                combiner,
            ));
        }

        let mut score_batch = vec![0.0f32; BINARY_SCORE_BATCH.min(self.header.vector_count)];
        let mut ordinal_scores = Vec::new();
        ordinal_scores
            .try_reserve_exact(posting_count)
            .map_err(|_| invalid_data("binary combined score buffer allocation failed"))?;
        score_binary_cluster_runs(
            self,
            bytes,
            query,
            cluster_ids,
            &mut score_batch,
            &mut ordinal_scores,
        )?;
        Ok(combine_scored_ordinals_retaining(
            ordinal_scores,
            k,
            combiner,
        ))
    }

    /// Score every TQ block against the query plan and keep the top `k`
    /// distinct documents by estimated similarity.
    #[cfg(test)]
    pub(crate) fn search_tq_distinct(
        &self,
        k: usize,
        plan: &crate::structures::TqQueryPlan,
    ) -> io::Result<Vec<(u32, u16, f32)>> {
        self.search_tq_distinct_generic::<BoundedAnnCollector<true, true>>(k, plan)
            .map(|(results, _)| results)
    }

    /// [`Self::search_tq_distinct`] plus scan diagnostics. `unique_keys`
    /// selects the heap-only collector, which is only sound when every
    /// `(doc_id, ordinal)` occurs once (single-valued field, one posting per
    /// stored vector) — the caller checks both against flat metadata.
    pub(crate) fn search_tq_distinct_with_stats(
        &self,
        k: usize,
        plan: &crate::structures::TqQueryPlan,
        unique_keys: bool,
    ) -> io::Result<ScanResults> {
        if unique_keys {
            self.search_tq_distinct_generic::<BoundedUniqueAnnCollector<true>>(k, plan)
        } else {
            self.search_tq_distinct_generic::<BoundedAnnCollector<true, true>>(k, plan)
        }
    }

    fn search_tq_distinct_generic<C: IvfTqTopK>(
        &self,
        k: usize,
        plan: &crate::structures::TqQueryPlan,
    ) -> io::Result<ScanResults> {
        use crate::structures::vector::quantization::{
            TQ_BLOCK_LANES, tq_block_bytes, tq_score_block,
        };

        if plan.padded_dim() != self.header.code_size * 2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "TQ query plan does not match the payload dimension",
            ));
        }
        let block_bytes = tq_block_bytes(self.header.code_size);
        let bytes = self.raw.as_slice();
        let mut stats = DenseAnnScanStats {
            posting_count: self.header.vector_count,
            ..DenseAnnScanStats::default()
        };

        // Score a block, then offer only lanes that can still enter top-k.
        // `insert` re-checks, so a stale threshold is merely conservative;
        // NaN fails the comparison and reaches `insert`, which counts it.
        let score_lanes = |collector: &mut C,
                           run: &AnnRun,
                           block_index: usize,
                           scores: &[f32; TQ_BLOCK_LANES]| {
            let lane_base = block_index * TQ_BLOCK_LANES;
            let lanes = TQ_BLOCK_LANES.min(run.count.saturating_sub(lane_base));
            let mut threshold = collector.pruning_threshold();
            for (lane, &score) in scores.iter().enumerate().take(lanes) {
                if threshold.is_some_and(|threshold| score < threshold) {
                    continue;
                }
                let index = lane_base + lane;
                let doc_id = run_doc_id_validated(bytes, run, index);
                if self
                    .alive_docs
                    .as_ref()
                    .is_some_and(|bits| !bits.contains(doc_id))
                {
                    continue;
                }
                collector.insert(
                    doc_id,
                    read_u16(bytes, run.ordinals.start + index * 2),
                    score,
                );
                threshold = collector.pruning_threshold();
            }
        };

        // Large flat scans are CPU-bound on the LUT16 kernel. Fold chunks into
        // worker-local collectors and reduce them directly: materializing one
        // top-k Vec per chunk would make temporary memory O(chunks * k) on
        // large segments instead of O(workers * k).
        #[cfg(feature = "native")]
        if self.header.vector_count >= TQ_PARALLEL_SCAN_MIN_VECTORS {
            use rayon::prelude::*;
            let (collector, scored_blocks) = self
                .runs
                .par_iter()
                .flat_map(|run| {
                    let codes = &bytes[run.codes.clone()];
                    let blocks = codes.len() / block_bytes;
                    (0..blocks.div_ceil(TQ_PARALLEL_SCAN_CHUNK_BLOCKS))
                        .into_par_iter()
                        .map(move |chunk| (run, chunk * TQ_PARALLEL_SCAN_CHUNK_BLOCKS, blocks))
                })
                .fold(
                    || (C::with_k(k), 0usize),
                    |(mut collector, scored), (run, first_block, total_blocks)| {
                        let codes = &bytes[run.codes.clone()];
                        let last_block =
                            (first_block + TQ_PARALLEL_SCAN_CHUNK_BLOCKS).min(total_blocks);
                        let mut scores = [0.0f32; TQ_BLOCK_LANES];
                        for block_index in first_block..last_block {
                            let block = &codes[block_index * block_bytes..][..block_bytes];
                            tq_score_block(plan, block, &mut scores);
                            score_lanes(&mut collector, run, block_index, &scores);
                        }
                        (collector, scored + (last_block - first_block))
                    },
                )
                .reduce(
                    || (C::with_k(k), 0usize),
                    |(mut collector, left_scored), (partial, right_scored)| {
                        collector.merge_from(partial);
                        (collector, left_scored + right_scored)
                    },
                );
            stats.scored_blocks = scored_blocks;
            stats.non_finite_dropped = collector.non_finite_dropped();
            stats.parallel = true;
            return Ok((collector.into_sorted_results(), stats));
        }

        let mut collector = C::with_k(k);
        let mut scores = [0.0f32; TQ_BLOCK_LANES];
        for run in self.runs.iter() {
            let codes = &bytes[run.codes.clone()];
            for (block_index, block) in codes.chunks_exact(block_bytes).enumerate() {
                tq_score_block(plan, block, &mut scores);
                stats.scored_blocks += 1;
                score_lanes(&mut collector, run, block_index, &scores);
            }
        }
        stats.non_finite_dropped = collector.non_finite_dropped();
        Ok((collector.into_sorted_results(), stats))
    }

    /// Score the probed IVF-TQ leaves and keep the top `k` distinct
    /// documents by estimated cosine similarity
    /// (`⟨q̂,c⟩ + scale·⟨q̂,r̂⟩`).
    ///
    /// Blocks whose best possible estimate cannot beat the running k-th score
    /// terminate their run. The supported cosine generation guarantees that
    /// residual scales are stored in descending order.
    #[cfg(test)]
    pub(crate) fn search_ivf_tq_distinct(
        &self,
        k: usize,
        plan: &crate::structures::TqIvfQueryPlan,
    ) -> io::Result<Vec<(u32, u16, f32)>> {
        self.search_ivf_tq_distinct_generic::<BoundedAnnCollector<true, true>>(
            k,
            plan,
            IVF_PARALLEL_SCAN_MIN_POSTINGS,
            IVF_TQ_PARALLEL_SCAN_CHUNK_BLOCKS,
        )
        .map(|(results, _)| results)
    }

    /// [`Self::search_ivf_tq_distinct`] plus scan diagnostics. `unique_keys`
    /// selects the heap-only collector; it is only sound when the field is
    /// single-valued *and* the payload holds exactly one posting per stored
    /// vector (no SOAR spill) — the caller checks both against flat metadata.
    pub(crate) fn search_ivf_tq_distinct_with_stats(
        &self,
        k: usize,
        plan: &crate::structures::TqIvfQueryPlan,
        unique_keys: bool,
    ) -> io::Result<ScanResults> {
        if unique_keys {
            self.search_ivf_tq_distinct_generic::<BoundedUniqueAnnCollector<true>>(
                k,
                plan,
                IVF_PARALLEL_SCAN_MIN_POSTINGS,
                IVF_TQ_PARALLEL_SCAN_CHUNK_BLOCKS,
            )
        } else {
            self.search_ivf_tq_distinct_generic::<BoundedAnnCollector<true, true>>(
                k,
                plan,
                IVF_PARALLEL_SCAN_MIN_POSTINGS,
                IVF_TQ_PARALLEL_SCAN_CHUNK_BLOCKS,
            )
        }
    }

    #[cfg(test)]
    fn search_ivf_tq_distinct_with_tuning(
        &self,
        k: usize,
        plan: &crate::structures::TqIvfQueryPlan,
        parallel_min_postings: usize,
        parallel_chunk_blocks: usize,
    ) -> io::Result<Vec<(u32, u16, f32)>> {
        self.search_ivf_tq_distinct_generic::<BoundedAnnCollector<true, true>>(
            k,
            plan,
            parallel_min_postings,
            parallel_chunk_blocks,
        )
        .map(|(results, _)| results)
    }

    fn search_ivf_tq_distinct_generic<C: IvfTqTopK>(
        &self,
        k: usize,
        plan: &crate::structures::TqIvfQueryPlan,
        parallel_min_postings: usize,
        parallel_chunk_blocks: usize,
    ) -> io::Result<ScanResults> {
        use crate::structures::vector::quantization::tq_ivf_block_bytes;

        let tq_plan = plan.tq_plan();
        self.validate_ivf_tq_query_plan(plan)?;
        #[cfg(not(feature = "native"))]
        let _ = (parallel_min_postings, parallel_chunk_blocks);
        if k == 0 {
            return Ok((Vec::new(), DenseAnnScanStats::default()));
        }
        #[cfg(feature = "native")]
        self.prefetch_cluster_runs(&plan.cluster_ids);
        let block_bytes = tq_ivf_block_bytes(self.header.code_size);
        let bytes = self.raw.as_slice();
        let posting_count = probed_posting_count(self, &plan.cluster_ids)?;
        let mut stats = DenseAnnScanStats {
            posting_count,
            ..DenseAnnScanStats::default()
        };

        // Score one chunk first so every worker starts with a top-k backed by
        // real documents. Its k-th score is therefore a safe pruning floor.
        // Chunking also exposes parallelism when a skewed leaf is one large
        // physical run. Nested Rayon stays on the caller's bounded search pool.
        #[cfg(feature = "native")]
        if rayon::current_num_threads() > 1 && posting_count >= parallel_min_postings {
            use rayon::prelude::*;
            let tasks = self.ivf_tq_scan_tasks(plan, block_bytes, parallel_chunk_blocks);

            if let Some((pilot, remaining)) = tasks.split_first() {
                let mut pilot_collector = C::with_k(k);
                let (pilot_pruned, pilot_scored) = score_ivf_tq_blocks(
                    self.alive_docs.as_deref(),
                    bytes,
                    tq_plan,
                    block_bytes,
                    *pilot,
                    &mut pilot_collector,
                );
                let pilot_non_finite = pilot_collector.non_finite_dropped();
                let seed = pilot_collector.into_sorted_results();
                let seeded_collector = || {
                    let mut collector = C::with_k(k);
                    for &(doc_id, ordinal, score) in &seed {
                        collector.insert(doc_id, ordinal, score);
                    }
                    collector
                };
                let (collector, pruned_blocks, scored_blocks) = remaining
                    .par_iter()
                    .fold(
                        || (seeded_collector(), 0usize, 0usize),
                        |(mut collector, pruned, scored), task| {
                            let (task_pruned, task_scored) = score_ivf_tq_blocks(
                                self.alive_docs.as_deref(),
                                bytes,
                                tq_plan,
                                block_bytes,
                                *task,
                                &mut collector,
                            );
                            (collector, pruned + task_pruned, scored + task_scored)
                        },
                    )
                    .reduce(
                        || (seeded_collector(), 0usize, 0usize),
                        |(mut collector, left_pruned, left_scored),
                         (partial, right_pruned, right_scored)| {
                            collector.merge_from(partial);
                            (
                                collector,
                                left_pruned + right_pruned,
                                left_scored + right_scored,
                            )
                        },
                    );
                stats.pruned_blocks = pilot_pruned + pruned_blocks;
                stats.scored_blocks = pilot_scored + scored_blocks;
                stats.non_finite_dropped = pilot_non_finite + collector.non_finite_dropped();
                stats.parallel = true;
                log_ivf_tq_pruning(stats.pruned_blocks, stats.scored_blocks, true);
                return Ok((collector.into_sorted_results(), stats));
            }
        }

        let mut collector = C::with_k(k);
        for (cluster_id, cluster_dot) in plan.cluster_dots() {
            for run in self.cluster_runs(cluster_id) {
                let block_count = run.codes.len() / block_bytes;
                let (run_pruned, run_scored) = score_ivf_tq_blocks(
                    self.alive_docs.as_deref(),
                    bytes,
                    tq_plan,
                    block_bytes,
                    IvfTqScanTask {
                        run,
                        cluster_dot,
                        first_block: 0,
                        last_block: block_count,
                    },
                    &mut collector,
                );
                stats.pruned_blocks += run_pruned;
                stats.scored_blocks += run_scored;
            }
        }
        stats.non_finite_dropped = collector.non_finite_dropped();
        log_ivf_tq_pruning(stats.pruned_blocks, stats.scored_blocks, false);
        Ok((collector.into_sorted_results(), stats))
    }

    fn validate_ivf_tq_query_plan(
        &self,
        plan: &crate::structures::TqIvfQueryPlan,
    ) -> io::Result<()> {
        if self.header.kind != AnnKind::IvfTq
            || !crate::structures::is_ivf_tq_cosine_generation(self.header.quantizer_version)
        {
            return Err(invalid_data(
                "legacy raw IVF-TQ payloads cannot be searched; rebuild the index",
            ));
        }
        if plan.tq_plan().padded_dim() != self.header.code_size * 2
            || plan.quantizer_version != self.header.quantizer_version
            || plan.fingerprint != self.header.codebook_version
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "IVF-TQ query plan does not match the payload generation",
            ));
        }
        Ok(())
    }

    pub(crate) fn search_binary_clusters<const BY_DOCUMENT: bool>(
        &self,
        query: &[u8],
        k: usize,
        cluster_ids: &[u32],
    ) -> io::Result<Vec<(u32, u16, f32)>> {
        self.search_binary_clusters_with_tuning::<BY_DOCUMENT>(
            query,
            k,
            cluster_ids,
            IVF_PARALLEL_SCAN_MIN_POSTINGS,
        )
    }

    fn search_binary_clusters_with_tuning<const BY_DOCUMENT: bool>(
        &self,
        query: &[u8],
        k: usize,
        cluster_ids: &[u32],
        parallel_min_postings: usize,
    ) -> io::Result<Vec<(u32, u16, f32)>> {
        #[cfg(not(feature = "native"))]
        let _ = parallel_min_postings;
        if query.len() != self.header.code_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "binary ANN query has the wrong byte length",
            ));
        }
        if k == 0 {
            return Ok(Vec::new());
        }
        #[cfg(feature = "native")]
        self.prefetch_cluster_runs(cluster_ids);
        let bytes = self.raw.as_slice();

        // A probe plan returns each leaf once. IVF has one posting per vector;
        // binary ScaNN may intentionally spill a vector into a second leaf.
        debug_assert!(
            BY_DOCUMENT || self.header.kind == AnnKind::ScannBinary || {
                let mut seen = rustc_hash::FxHashSet::default();
                cluster_ids
                    .iter()
                    .all(|cluster_id| seen.insert(*cluster_id))
            },
            "an IVF probe plan must not repeat a cluster",
        );

        #[cfg(feature = "native")]
        if rayon::current_num_threads() > 1
            && probed_posting_count(self, cluster_ids)? >= parallel_min_postings
        {
            use rayon::prelude::*;
            let tasks = self.binary_scan_tasks(cluster_ids);
            let (collector, _) = tasks
                .par_iter()
                .try_fold(
                    || {
                        (
                            BoundedAnnCollector::<BY_DOCUMENT, true>::new(k),
                            vec![0.0f32; BINARY_SCORE_BATCH],
                        )
                    },
                    |(mut collector, mut scores), task| {
                        score_binary_task(
                            self.alive_docs.as_deref(),
                            bytes,
                            query,
                            self.header.dim,
                            self.header.code_size,
                            *task,
                            &mut scores,
                            &mut collector,
                        )?;
                        Ok::<_, io::Error>((collector, scores))
                    },
                )
                .try_reduce(
                    || (BoundedAnnCollector::<BY_DOCUMENT, true>::new(k), Vec::new()),
                    |(mut collector, scores), (partial, _)| {
                        collector.merge_from(partial);
                        Ok((collector, scores))
                    },
                )?;
            return Ok(collector.into_sorted_results());
        }

        let mut scores = vec![0.0f32; BINARY_SCORE_BATCH.min(self.header.vector_count)];
        if BY_DOCUMENT || self.header.kind == AnnKind::ScannBinary {
            let mut collector = BoundedAnnCollector::<BY_DOCUMENT, true>::new(k);
            score_binary_cluster_runs(
                self,
                bytes,
                query,
                cluster_ids,
                &mut scores,
                &mut collector,
            )?;
            return Ok(collector.into_sorted_results());
        }

        // Avoid the deduplication hash map in the serial single-value path.
        let mut collector = BoundedUniqueAnnCollector::<true>::new(k);
        score_binary_cluster_runs(self, bytes, query, cluster_ids, &mut scores, &mut collector)?;
        Ok(collector.into_sorted_results())
    }
}

/// Score one contiguous block range from an IVF-TQ run. A task may start in
/// the middle of a run because residual scales descend across the complete
/// run: the first block is still an upper bound for the remainder of the task.
///
/// Returns `(pruned_blocks, scored_blocks)`.
fn score_ivf_tq_blocks<C: IvfTqTopK>(
    visibility: Option<&crate::query::DocBitset>,
    bytes: &[u8],
    plan: &crate::structures::TqQueryPlan,
    block_bytes: usize,
    task: IvfTqScanTask<'_>,
    collector: &mut C,
) -> (usize, usize) {
    use crate::structures::vector::quantization::{TQ_BLOCK_LANES, tq_score_ivf_block};

    let run = task.run;
    let codes = &bytes[run.codes.clone()];
    let mut scores = [0.0f32; TQ_BLOCK_LANES];
    let mut scored_blocks = 0usize;
    let mut threshold = collector.pruning_threshold();
    for block_index in task.first_block..task.last_block {
        let block = &codes[block_index * block_bytes..][..block_bytes];
        if let Some(threshold) = threshold
            && task.cluster_dot + tq_ivf_block_max_scale(block) * TQ_PRUNE_ESTIMATE_BOUND
                <= threshold
        {
            return (task.last_block - block_index, scored_blocks);
        }
        scored_blocks += 1;
        tq_score_ivf_block(plan, block, task.cluster_dot, &mut scores);
        let lane_base = block_index * TQ_BLOCK_LANES;
        let lanes = TQ_BLOCK_LANES.min(run.count.saturating_sub(lane_base));
        for (lane, &score) in scores.iter().enumerate().take(lanes) {
            // Reject non-competitive lanes before touching the doc-ID and
            // ordinal columns. Strict `<` leaves exact ties to `insert`, which
            // breaks them by key exactly as before; NaN fails the comparison
            // and reaches `insert`, which counts it.
            if threshold.is_some_and(|threshold| score < threshold) {
                continue;
            }
            let index = lane_base + lane;
            let doc_id = run_doc_id_validated(bytes, run, index);
            if visibility.is_some_and(|bits| !bits.contains(doc_id)) {
                continue;
            }
            collector.insert(
                doc_id,
                read_u16(bytes, run.ordinals.start + index * 2),
                score,
            );
            threshold = collector.pruning_threshold();
        }
    }
    (0, scored_blocks)
}

/// Returns `(scored_blocks, non_finite_dropped)`.
#[cfg(feature = "native")]
fn score_ivf_tq_combined_blocks(
    visibility: Option<&crate::query::DocBitset>,
    bytes: &[u8],
    plan: &crate::structures::TqQueryPlan,
    block_bytes: usize,
    task: IvfTqScanTask<'_>,
    ordinal_scores: &mut Vec<(u32, u16, f32)>,
) -> (usize, usize) {
    use crate::structures::vector::quantization::{TQ_BLOCK_LANES, tq_score_ivf_block};

    let run = task.run;
    let codes = &bytes[run.codes.clone()];
    let mut scores = [0.0f32; TQ_BLOCK_LANES];
    let mut non_finite = 0usize;
    for block_index in task.first_block..task.last_block {
        let block = &codes[block_index * block_bytes..][..block_bytes];
        tq_score_ivf_block(plan, block, task.cluster_dot, &mut scores);
        let lane_base = block_index * TQ_BLOCK_LANES;
        let lanes = TQ_BLOCK_LANES.min(run.count.saturating_sub(lane_base));
        for (lane, &score) in scores.iter().enumerate().take(lanes) {
            if !score.is_finite() {
                non_finite += 1;
                continue;
            }
            let index = lane_base + lane;
            let doc_id = run_doc_id_validated(bytes, run, index);
            if visibility.is_some_and(|bits| !bits.contains(doc_id)) {
                continue;
            }
            ordinal_scores.push((
                doc_id,
                read_u16(bytes, run.ordinals.start + index * 2),
                score,
            ));
        }
    }
    (task.last_block - task.first_block, non_finite)
}

fn log_ivf_tq_pruning(pruned_blocks: usize, scored_blocks: usize, parallel: bool) {
    if pruned_blocks > 0 {
        log::debug!(
            "[search_ivf_tq] pruned {pruned_blocks} of {} blocks via scale bounds ({})",
            pruned_blocks + scored_blocks,
            if parallel { "parallel" } else { "serial" },
        );
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

trait AnnScoreSink {
    fn insert_score(&mut self, doc_id: u32, ordinal: u16, score: f32);

    /// Conservative k-th best score, or no bound when every ordinal is needed
    /// for document combination (Sum/Avg/LogSumExp/WeightedTopK).
    fn prune_threshold(&self) -> Option<f32> {
        None
    }
}

impl<const BY_DOCUMENT: bool> AnnScoreSink for BoundedAnnCollector<BY_DOCUMENT, true> {
    #[inline]
    fn prune_threshold(&self) -> Option<f32> {
        self.pruning_threshold()
    }

    #[inline]
    fn insert_score(&mut self, doc_id: u32, ordinal: u16, score: f32) {
        self.insert(doc_id, ordinal, score);
    }
}

impl AnnScoreSink for BoundedUniqueAnnCollector<true> {
    #[inline]
    fn prune_threshold(&self) -> Option<f32> {
        self.pruning_threshold()
    }

    #[inline]
    fn insert_score(&mut self, doc_id: u32, ordinal: u16, score: f32) {
        self.insert(doc_id, ordinal, score);
    }
}

impl AnnScoreSink for Vec<(u32, u16, f32)> {
    #[inline]
    fn insert_score(&mut self, doc_id: u32, ordinal: u16, score: f32) {
        if score.is_finite() {
            self.push((doc_id, ordinal, score));
        }
    }
}

fn probed_posting_count(index: &AnnDiskIndex, cluster_ids: &[u32]) -> io::Result<usize> {
    cluster_ids.iter().try_fold(0usize, |count, &cluster_id| {
        index
            .cluster_runs(cluster_id)
            .iter()
            .try_fold(count, |count, run| {
                count
                    .checked_add(run.count)
                    .ok_or_else(|| invalid_data("ANN probed posting count overflows usize"))
            })
    })
}

fn score_binary_cluster_runs(
    index: &AnnDiskIndex,
    bytes: &[u8],
    query: &[u8],
    cluster_ids: &[u32],
    scores: &mut [f32],
    collector: &mut impl AnnScoreSink,
) -> io::Result<()> {
    for &cluster_id in cluster_ids {
        for run in index.cluster_runs(cluster_id) {
            score_binary_run(
                index.alive_docs.as_deref(),
                bytes,
                run,
                query,
                index.header.dim,
                index.header.code_size,
                scores,
                collector,
            )?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn score_binary_run(
    visibility: Option<&crate::query::DocBitset>,
    bytes: &[u8],
    run: &AnnRun,
    query: &[u8],
    dim_bits: usize,
    code_size: usize,
    scores: &mut [f32],
    collector: &mut impl AnnScoreSink,
) -> io::Result<()> {
    for batch_start in (0..run.count).step_by(BINARY_SCORE_BATCH) {
        score_binary_task(
            visibility,
            bytes,
            query,
            dim_bits,
            code_size,
            BinaryScanTask {
                run,
                first_index: batch_start,
                count: BINARY_SCORE_BATCH.min(run.count - batch_start),
            },
            scores,
            collector,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn score_binary_task(
    visibility: Option<&crate::query::DocBitset>,
    bytes: &[u8],
    query: &[u8],
    dim_bits: usize,
    code_size: usize,
    task: BinaryScanTask<'_>,
    scores: &mut [f32],
    collector: &mut impl AnnScoreSink,
) -> io::Result<()> {
    let code_start = task.run.codes.start + task.first_index * code_size;
    let code_end = code_start + task.count * code_size;
    crate::structures::simd::batch_hamming_scores(
        query,
        &bytes[code_start..code_end],
        code_size,
        dim_bits,
        &mut scores[..task.count],
    );
    const THRESHOLD_REFRESH_ROWS: usize = 64;

    // Refresh once per small block. The threshold can only improve while
    // collecting, so an older bound merely admits extra candidates. Strict
    // comparison preserves equal-score document/ordinal tie-breaking.
    for (block_index, block) in scores[..task.count]
        .chunks(THRESHOLD_REFRESH_ROWS)
        .enumerate()
    {
        let threshold = collector.prune_threshold().unwrap_or(f32::NEG_INFINITY);
        for (lane, &score) in block.iter().enumerate() {
            if score < threshold {
                continue;
            }
            let index = task.first_index + block_index * THRESHOLD_REFRESH_ROWS + lane;
            let doc_id = run_doc_id_validated(bytes, task.run, index);
            if visibility.is_some_and(|bits| !bits.contains(doc_id)) {
                continue;
            }
            collector.insert_score(
                doc_id,
                read_u16(bytes, task.run.ordinals.start + index * 2),
                score,
            );
        }
    }
    Ok(())
}

#[inline]
fn retain_combined_document(
    collector: &mut BoundedAnnCollector<true, true>,
    doc_id: u32,
    ordinal_scores: &[(u32, f32)],
    combiner: crate::query::MultiValueCombiner,
) {
    if !ordinal_scores.is_empty() {
        collector.insert(doc_id, 0, combiner.combine(ordinal_scores));
    }
}

fn validate_combined_search(combiner: crate::query::MultiValueCombiner) -> io::Result<()> {
    combiner
        .validate()
        .map_err(|message| io::Error::new(io::ErrorKind::InvalidInput, message))
}

/// Total non-finite leaf scores dropped by `combine_scored_ordinals_retaining`
/// (also exported as `summa_ann_non_finite_scores_dropped_total`).
pub(crate) static NON_FINITE_ANN_SCORES_DROPPED: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static NON_FINITE_ANN_SCORE_DROP_EVENTS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Rows per parallel ScaNN scan task: 512 FastScan groups, matching the
/// IVF-TQ chunking so worker-local top-k state stays small.
#[cfg(feature = "native")]
const SCANN_PARALLEL_SCAN_CHUNK_ROWS: usize =
    512 * crate::structures::vector::scann::FAST_SCAN_LANES;

/// Byte geometry of one ScaNN AH run under a fixed block count.
#[derive(Clone, Copy)]
struct ScannLeafGeometry {
    /// Bytes per complete 32-row FastScan group.
    packed_block_bytes: usize,
    /// Bytes per row-major tail row.
    packed_row_bytes: usize,
}

/// One FastScan-aligned row range of a routed ScaNN leaf run.
#[derive(Clone, Copy)]
struct ScannScanTask<'a> {
    run: &'a AnnRun,
    centroid_dot: f32,
    first_row: usize,
    count: usize,
}

/// Score one row range of a ScaNN AH run. Complete 32-row groups go through
/// the integer FastScan kernel; the run's row-major tail (only reached by the
/// task that ends at `run.count`) uses the float packed-row scorer.
fn scan_scann_ah_task(
    visibility: Option<&crate::query::DocBitset>,
    bytes: &[u8],
    query: &crate::structures::vector::scann::FloatScannQuery,
    geometry: ScannLeafGeometry,
    task: ScannScanTask<'_>,
    sink: &mut impl AnnScoreSink,
) -> io::Result<()> {
    use crate::structures::vector::scann::FAST_SCAN_LANES;

    let run = task.run;
    let codes = &bytes[run.codes.clone()];
    let fast = query.fast_scan();
    let full_rows = run.count / FAST_SCAN_LANES * FAST_SCAN_LANES;
    let end = task.first_row + task.count;
    debug_assert!(task.first_row.is_multiple_of(FAST_SCAN_LANES) || task.first_row >= full_rows);
    let mut lanes = [0i32; FAST_SCAN_LANES];
    let mut row = task.first_row;
    while row < full_rows.min(end) {
        let start = (row / FAST_SCAN_LANES) * geometry.packed_block_bytes;
        fast.accumulate_block(
            &codes[start..start + geometry.packed_block_bytes],
            &mut lanes,
        )
        .map_err(|error| invalid_data(error.to_string()))?;
        let cutoff = sink
            .prune_threshold()
            .map(|threshold| fast.lane_cutoff(threshold, task.centroid_dot));
        for (lane, &sum) in lanes.iter().enumerate() {
            if cutoff.is_some_and(|cutoff| sum <= cutoff) {
                continue;
            }
            let index = row + lane;
            let doc_id = run_doc_id(bytes, run, index)?;
            if visibility.is_some_and(|bits| !bits.contains(doc_id)) {
                continue;
            }
            sink.insert_score(
                doc_id,
                read_u16(bytes, run.ordinals.start + index * 2),
                fast.lane_score(sum, task.centroid_dot),
            );
        }
        row += FAST_SCAN_LANES;
    }
    let tail_bytes = (full_rows / FAST_SCAN_LANES) * geometry.packed_block_bytes;
    while row < end {
        let start = tail_bytes + (row - full_rows) * geometry.packed_row_bytes;
        let score = query
            .ah_query()
            .score_packed(
                &codes[start..start + geometry.packed_row_bytes],
                task.centroid_dot,
            )
            .map_err(|error| invalid_data(error.to_string()))?;
        let doc_id = run_doc_id(bytes, run, row)?;
        if visibility.is_some_and(|bits| !bits.contains(doc_id)) {
            row += 1;
            continue;
        }
        sink.insert_score(doc_id, read_u16(bytes, run.ordinals.start + row * 2), score);
        row += 1;
    }
    Ok(())
}

fn scann_document_candidates(
    collector: BoundedAnnCollector<true, true>,
) -> Vec<AnnDocumentCandidate> {
    collector
        .into_sorted_results()
        .into_iter()
        .map(|(doc_id, _, score)| AnnDocumentCandidate { doc_id, score })
        .collect()
}

/// Sort arbitrary probed-run output into complete documents, retain the best
/// estimate for every duplicated `(doc_id, ordinal)` SOAR assignment, then
/// apply the requested combiner. The corpus-dependent scratch is one compact
/// tuple per probed posting and never expands to unprobed leaves or raw
/// vectors; final retained state is O(k).
fn combine_scored_ordinals(
    scores: Vec<(u32, u16, f32)>,
    k: usize,
    combiner: crate::query::MultiValueCombiner,
) -> Vec<AnnDocumentCandidate> {
    combine_scored_ordinals_retaining(scores, k, combiner).0
}

/// As [`combine_scored_ordinals`], additionally returning the deduplicated
/// per-ordinal scores of the retained documents, sorted by `(doc_id, ordinal)`.
///
/// Only callers whose leaf scores are exact may reuse those numbers; TQ block
/// scores are estimates and must still be reranked against raw vectors.
fn combine_scored_ordinals_retaining(
    mut scores: Vec<(u32, u16, f32)>,
    k: usize,
    combiner: crate::query::MultiValueCombiner,
) -> CombinedBinaryCandidates {
    if k == 0 || scores.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let scored = scores.len();
    scores.retain(|entry| entry.2.is_finite());
    let dropped = scored - scores.len();
    if dropped > 0 {
        // A non-finite leaf score means a degenerate stored vector or
        // codebook; dropping it is the right ranking decision but never a
        // silent one.
        NON_FINITE_ANN_SCORES_DROPPED
            .fetch_add(dropped as u64, std::sync::atomic::Ordering::Relaxed);
        crate::observe::ann_non_finite_scores_dropped(dropped as u64);
        crate::structures::vector::scann::warn_rate_limited(
            &NON_FINITE_ANN_SCORE_DROP_EVENTS,
            |events| {
                format!(
                    "[ann] dropped {dropped} of {scored} non-finite leaf scores before combining \
                     ({events} scans affected so far; total dropped {})",
                    NON_FINITE_ANN_SCORES_DROPPED.load(std::sync::atomic::Ordering::Relaxed)
                )
            },
        );
    }
    let by_document_ordinal = |left: &(u32, u16, f32), right: &(u32, u16, f32)| {
        left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1))
    };
    #[cfg(feature = "native")]
    if rayon::current_num_threads() > 1 && scores.len() >= IVF_PARALLEL_SCAN_MIN_POSTINGS {
        use rayon::prelude::*;
        scores.par_sort_unstable_by(by_document_ordinal);
    } else {
        scores.sort_unstable_by(by_document_ordinal);
    }
    #[cfg(not(feature = "native"))]
    scores.sort_unstable_by(by_document_ordinal);

    // Compact duplicates in place. IVF-TQ SOAR copies can have slightly
    // different residual estimates; preserving the highest one matches the
    // legacy Max collector's duplicate semantics without double-counting it
    // for additive combiners.
    let mut unique_len = 0usize;
    for read_index in 0..scores.len() {
        let candidate = scores[read_index];
        if unique_len > 0
            && scores[unique_len - 1].0 == candidate.0
            && scores[unique_len - 1].1 == candidate.1
        {
            if candidate.2.total_cmp(&scores[unique_len - 1].2).is_gt() {
                scores[unique_len - 1].2 = candidate.2;
            }
        } else {
            scores[unique_len] = candidate;
            unique_len += 1;
        }
    }
    scores.truncate(unique_len);

    let mut top_documents = BoundedAnnCollector::<true, true>::new(k);
    let mut current_doc = None;
    let mut ordinal_scores = Vec::new();
    for &(doc_id, ordinal, score) in &scores {
        if current_doc.is_some_and(|current| current != doc_id) {
            retain_combined_document(
                &mut top_documents,
                current_doc.expect("current document is present"),
                &ordinal_scores,
                combiner,
            );
            ordinal_scores.clear();
        }
        current_doc = Some(doc_id);
        ordinal_scores.push((u32::from(ordinal), score));
    }
    if let Some(doc_id) = current_doc {
        retain_combined_document(&mut top_documents, doc_id, &ordinal_scores, combiner);
    }

    let candidates: Vec<AnnDocumentCandidate> = top_documents
        .into_sorted_results()
        .into_iter()
        .map(|(doc_id, _, score)| AnnDocumentCandidate { doc_id, score })
        .collect();

    // Narrow the per-ordinal scores to the retained documents. `scores` is
    // already sorted by document, so this is one pass over a sorted ID set
    // rather than a hash lookup per posting.
    let mut retained_ids: Vec<u32> = candidates
        .iter()
        .map(|candidate| candidate.doc_id)
        .collect();
    retained_ids.sort_unstable();
    let mut retained = Vec::with_capacity(scores.len().min(retained_ids.len().saturating_mul(4)));
    let mut cursor = 0usize;
    for entry in scores {
        while cursor < retained_ids.len() && retained_ids[cursor] < entry.0 {
            cursor += 1;
        }
        if retained_ids.get(cursor) == Some(&entry.0) {
            retained.push(entry);
        }
    }
    (candidates, retained)
}

#[cfg(feature = "native")]
fn record_exact_locations(
    locations: &mut Option<&mut crate::segment::vector_locations::ExactLocations>,
    offset: u64,
    count: usize,
    labels: impl Fn(usize) -> io::Result<(u32, u16)>,
    cancellation: Option<&std::sync::atomic::AtomicBool>,
) -> io::Result<()> {
    let Some(locations) = locations else {
        return Ok(());
    };
    let span = locations.span(offset, count)?;
    for row in 0..count {
        if row.is_multiple_of(4096)
            && cancellation.is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Relaxed))
        {
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "ANN locations cancelled",
            ));
        }
        let (doc, ordinal) = labels(row)?;
        let at = (u64::from(span) << 32) | row as u64;
        locations.push(doc, ordinal, at)?;
    }
    Ok(())
}

#[cfg(feature = "native")]
pub(crate) fn write_built_binary_ivf(
    index: &BinaryIvfIndex,
    routing: IvfRoutingMode,
    writer: &mut (impl Write + ?Sized),
    locations: Option<&mut crate::segment::vector_locations::ExactLocations>,
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
        locations,
    )
}

/// Serialize ScaNN segment-local leaf runs into Summa's merge-native ANN
/// format. The global model is not embedded: its generation and content
/// fingerprint occupy the header's compatibility slots, so an ordinary merge
/// can reject mixed models before copying any corpus bytes.
#[cfg(feature = "native")]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn write_built_scann(
    payload: &crate::structures::vector::scann::ScannSegmentPayload,
    writer: &mut (impl Write + ?Sized),
    mut locations: Option<&mut crate::segment::vector_locations::ExactLocations>,
) -> io::Result<u64> {
    use crate::structures::vector::scann::ScannEncoding;

    let (kind, code_size) = match payload.encoding {
        ScannEncoding::AsymmetricHash {
            dimensions_per_block,
            bits_per_code: 4,
        } => (AnnKind::ScannAh, usize::from(dimensions_per_block)),
        ScannEncoding::AsymmetricHash { .. } => {
            return Err(invalid_data(
                "ScaNN ANN disk format supports only 4-bit AH codes",
            ));
        }
        ScannEncoding::BinaryHamming => (
            AnnKind::ScannBinary,
            usize::try_from(payload.dimension)
                .map_err(|_| invalid_data("ScaNN dimension exceeds usize"))?
                .div_ceil(8),
        ),
    };
    let vector_count = payload.runs().iter().try_fold(0usize, |total, run| {
        total
            .checked_add(run.row_count as usize)
            .ok_or_else(|| invalid_data("ScaNN vector count overflows usize"))
    })?;
    let header = AnnDiskHeader {
        kind,
        // ScaNN owns its multi-level routing geometry in the global artifact;
        // Flat is the reserved sentinel in the shared ANN header.
        routing: IvfRoutingMode::Flat,
        dim: usize::try_from(payload.dimension)
            .map_err(|_| invalid_data("ScaNN dimension exceeds usize"))?,
        code_size,
        num_clusters: payload.num_leaves,
        quantizer_version: payload.generation,
        codebook_version: payload.artifact_id,
        vector_count,
    };
    validate_header(&header)?;
    write_header(writer, &header)?;

    let mut offset = ANN_HEADER_SIZE as u64;
    let mut records = Vec::with_capacity(payload.runs().len());
    let mut previous_leaf = None;
    for run in payload.runs() {
        let count = run.row_count as usize;
        let expected_docs = count
            .checked_mul(4)
            .ok_or_else(|| invalid_data("ScaNN doc-ID column size overflows usize"))?;
        let expected_ordinals = count
            .checked_mul(2)
            .ok_or_else(|| invalid_data("ScaNN ordinal column size overflows usize"))?;
        if count == 0
            || run.leaf_id >= header.num_clusters
            || previous_leaf.is_some_and(|leaf| leaf > run.leaf_id)
            || run.doc_ids_le.len() != expected_docs
            || run.ordinals_le.len() != expected_ordinals
            || run.codes.len()
                != expected_codes_column_len(header.kind, count, header.dim, header.code_size)?
        {
            return Err(invalid_data("ScaNN ANN leaf run columns are inconsistent"));
        }
        previous_leaf = Some(run.leaf_id);
        let max_doc_id = run
            .doc_ids_le
            .chunks_exact(4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
            .max()
            .unwrap_or(0);
        if run
            .doc_base
            .checked_add(max_doc_id)
            .is_none_or(|doc_id| doc_id >= payload.doc_count)
        {
            return Err(invalid_data(
                "ScaNN ANN leaf run document ID is out of range",
            ));
        }

        let doc_ids_offset = offset;
        writer.write_all(&run.doc_ids_le)?;
        offset = checked_advance(offset, run.doc_ids_le.len())?;
        let ordinals_offset = offset;
        writer.write_all(&run.ordinals_le)?;
        offset = checked_advance(offset, run.ordinals_le.len())?;
        let codes_offset = offset;
        record_exact_locations(
            &mut locations,
            codes_offset,
            count,
            |row| {
                Ok((
                    read_u32(&run.doc_ids_le, row * 4) + run.doc_base,
                    read_u16(&run.ordinals_le, row * 2),
                ))
            },
            None,
        )?;
        writer.write_all(&run.codes)?;
        offset = checked_advance(offset, run.codes.len())?;
        records.push(RunRecord {
            cluster_id: run.leaf_id,
            doc_base: run.doc_base,
            count: run.row_count,
            max_doc_id,
            doc_ids_offset,
            ordinals_offset,
            codes_offset,
            codes_len: u64::try_from(run.codes.len())
                .map_err(|_| invalid_data("ScaNN code output size exceeds u64"))?,
        });
    }
    if records.is_empty() {
        return Err(invalid_data("cannot write an empty ScaNN ANN payload"));
    }
    finish_layout(writer, offset, &records)
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

    if !crate::structures::is_ivf_tq_cosine_generation(index.centroids_version) {
        return Err(invalid_data(
            "legacy raw IVF-TQ generations cannot be serialized; rebuild the index",
        ));
    }
    let codec = index.codec();
    let padded_dim = codec.padded_dim();
    let mut clusters: Vec<_> = index.clusters.iter().collect();
    clusters.sort_unstable_by_key(|(cluster_id, _)| **cluster_id);
    // Emit every cluster in descending residual-scale order: block maxima
    // then decrease monotonically, so the scan's per-block score bound can
    // stop a run at the first block that cannot beat the running k-th score.
    struct PackedCluster {
        cluster_id: u32,
        doc_ids: Vec<u32>,
        ordinals: Vec<u16>,
        codes: Vec<u8>,
    }
    let packed: Vec<PackedCluster> = clusters
        .iter()
        .map(|&(&cluster_id, cluster)| {
            let count = cluster.doc_ids.len();
            let mut order: Vec<usize> = (0..count).collect();
            order.sort_by(|&a, &b| {
                cluster.scales[b]
                    .total_cmp(&cluster.scales[a])
                    .then_with(|| a.cmp(&b))
            });
            let doc_ids: Vec<u32> = order.iter().map(|&i| cluster.doc_ids[i]).collect();
            let ordinals: Vec<u16> = order.iter().map(|&i| cluster.ordinals[i]).collect();
            let scales: Vec<f32> = order.iter().map(|&i| cluster.scales[i]).collect();
            let gammas: Vec<f32> = order.iter().map(|&i| cluster.gammas[i]).collect();
            let mut codes = Vec::with_capacity(
                crate::structures::vector::quantization::tq_ivf_codes_column_len_checked(
                    count,
                    codec.code_size(),
                )
                .unwrap_or_default(),
            );
            for block_start in (0..count).step_by(TQ_BLOCK_LANES) {
                let lanes = TQ_BLOCK_LANES.min(count - block_start);
                let rows: Vec<&[u8]> = order[block_start..block_start + lanes]
                    .iter()
                    .map(|&row| &cluster.rows[row * padded_dim..(row + 1) * padded_dim])
                    .collect();
                tq_pack_ivf_block(
                    &rows,
                    &scales[block_start..block_start + lanes],
                    &gammas[block_start..block_start + lanes],
                    padded_dim,
                    &mut codes,
                );
            }
            PackedCluster {
                cluster_id,
                doc_ids,
                ordinals,
                codes,
            }
        })
        .collect();
    let runs: Vec<_> = packed
        .iter()
        .map(|cluster| BuildRun {
            cluster_id: cluster.cluster_id,
            doc_ids: &cluster.doc_ids,
            ordinals: &cluster.ordinals,
            codes: &cluster.codes,
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
        None,
    )
}

/// View a populated TQ builder as a single extra merge run (cluster 0).
#[cfg(feature = "native")]
pub(crate) fn tq_builder_extra_run(builder: &crate::structures::TqFlatBuilder) -> BuildRun<'_> {
    BuildRun {
        cluster_id: 0,
        doc_ids: &builder.doc_ids,
        ordinals: &builder.ordinals,
        codes: &builder.codes,
    }
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
        None,
    )
}

#[cfg(feature = "native")]
pub(crate) struct BuildRun<'a> {
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
    mut locations: Option<&mut crate::segment::vector_locations::ExactLocations>,
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
            || run.codes.len()
                != expected_codes_column_len(header.kind, count, header.dim, header.code_size)?
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
        record_exact_locations(
            &mut locations,
            codes_offset,
            count,
            |row| Ok((run.doc_ids[row], run.ordinals[row])),
            None,
        )?;
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
#[cfg(all(feature = "native", test))]
pub(crate) fn write_merged_ann(
    sources: &[(&AnnDiskIndex, u32)],
    writer: &mut (impl Write + ?Sized),
) -> io::Result<u64> {
    write_merged_ann_impl(sources, &[], writer, None)
}

#[cfg(feature = "native")]
pub(crate) fn write_merged_ann_cancellable(
    sources: &[(&AnnDiskIndex, u32)],
    writer: &mut (impl Write + ?Sized),
    cancellation: Option<&std::sync::atomic::AtomicBool>,
) -> io::Result<u64> {
    write_merged_ann_impl(sources, &[], writer, cancellation)
}

/// Physical extents per non-empty cluster the byte-copy merge of `sources`
/// would produce.
///
/// Byte-copy preserves every source run, so the merged fragmentation is
/// `total runs / distinct non-empty clusters` — computable exactly from the
/// in-memory directories before writing a byte. Float AH uses this to select
/// leaf-wise compaction; binary merge reports fragmentation and copies extents.
#[cfg(feature = "native")]
pub(crate) fn predicted_merge_fragmentation(sources: &[(&AnnDiskIndex, u32)]) -> f64 {
    let mut total_runs = 0usize;
    // Distinct clusters via a k-way sorted walk over the (already
    // cluster-sorted) directories — no allocation proportional to clusters.
    let mut cursors: Vec<std::iter::Peekable<std::slice::Iter<'_, AnnRun>>> = sources
        .iter()
        .map(|(source, _)| {
            total_runs += source.runs.len();
            source.runs.iter().peekable()
        })
        .collect();
    let mut distinct = 0usize;
    while let Some(cluster) = cursors
        .iter_mut()
        .filter_map(|cursor| cursor.peek().map(|run| run.cluster_id))
        .min()
    {
        distinct += 1;
        for cursor in &mut cursors {
            while cursor.peek().is_some_and(|run| run.cluster_id == cluster) {
                cursor.next();
            }
        }
    }
    if distinct == 0 {
        0.0
    } else {
        total_runs as f64 / distinct as f64
    }
}

/// Doc IDs rewritten per scratch flush during compaction (256 KiB of u32s).
#[cfg(feature = "native")]
const DOC_ID_REWRITE_CHUNK: usize = 64 * 1024;

/// Cluster-major compacting merge for exact-binary and ScaNN-AH payloads.
///
/// The byte-copy merge keeps each source payload as one physical extent, so a
/// logical cluster's postings scatter across up to `sources.len()` extents —
/// and another factor per earlier merge generation. Every extent is a
/// potential seek when the index is cold; production measured the array
/// IOPS-bound at 32 KB/read from exactly this. This writer instead gathers
/// each cluster's runs from all sources and emits **one contiguous run per
/// cluster**, restoring the freshly-built layout (fragmentation 1.0).
///
/// Cost: the same total payload bytes the byte-copy merge already streams,
/// plus one `u32` add per posting — document IDs are rewritten absolute
/// (`doc_base = 0`) because runs from different sources cannot share a single
/// directory entry otherwise. Ordinals and binary codes are copied verbatim.
/// ScaNN AH codes are decoded one row at a time into a single 32-row scratch
/// block and repacked, because FastScan block/tail boundaries are run-local.
///
/// TQ payloads also pack codes into fixed-lane blocks, but their quantized
/// representation is intentionally outside this ScaNN compactor; TQ merges
/// stay byte-copy.
///
/// Normal float-AH merges use this when fragmented. Binary normal merges
/// retain source runs so their exact-vector lookup rows remain copyable.
/// Measured (interleaved best-of-3, pre-faulted buffers, aarch64): byte-copy
/// 38.0 GiB/s vs compaction 32.4 GiB/s — ~17% more CPU on a stage that is a
/// rounding error of merge wall-clock (a production dense stage is ~0.7s of
/// a 20s+ merge), so there is no threshold below which byte-copy is worth
/// the fragmentation it leaves behind.
#[cfg(feature = "native")]
pub(crate) fn write_compacted_ann_cancellable(
    sources: &[(&AnnDiskIndex, u32)],
    writer: &mut (impl Write + ?Sized),
    cancellation: Option<&std::sync::atomic::AtomicBool>,
    mut locations: Option<&mut crate::segment::vector_locations::ExactLocations>,
) -> io::Result<u64> {
    let Some((first, _)) = sources.first() else {
        return Err(invalid_data("cannot compact an empty ANN source list"));
    };
    if !matches!(
        first.header.kind,
        AnnKind::BinaryIvf | AnnKind::ScannBinary | AnnKind::ScannAh
    ) {
        return Err(invalid_data(
            "ANN run compaction is only defined for exact binary and ScaNN AH payloads",
        ));
    }
    let mut header = first.header.clone();
    header.vector_count = 0;
    for &(source, _) in sources {
        if !headers_compatible(&first.header, &source.header) {
            return Err(invalid_data(
                "ANN compaction sources use incompatible generations",
            ));
        }
        header.vector_count = header
            .vector_count
            .checked_add(source.header.vector_count)
            .ok_or_else(|| invalid_data("compacted ANN vector count overflows usize"))?;
    }
    validate_header(&header)?;
    write_header(writer, &header)?;

    let code_size = header.code_size;
    let mut offset = ANN_HEADER_SIZE as u64;
    let mut records: Vec<RunRecord> = Vec::new();
    let mut scratch = Vec::new();
    let mut cursors: Vec<usize> = vec![0; sources.len()];

    loop {
        if cancellation.is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Relaxed)) {
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "ANN compaction cancelled",
            ));
        }
        // Next cluster = minimum un-consumed cluster ID across sources.
        let Some(cluster_id) = sources
            .iter()
            .zip(&cursors)
            .filter_map(|((source, _), &cursor)| source.runs.get(cursor).map(|run| run.cluster_id))
            .min()
        else {
            break;
        };

        // Pass 1: doc IDs, rewritten absolute. Sources are visited in
        // segment order and each source's same-cluster runs in directory
        // order, which is ascending document ranges — so the output column
        // stays sorted like a built segment's.
        let mut count = 0usize;
        let mut max_doc_id = 0u32;
        let doc_ids_offset = offset;
        for (source_index, &(source, segment_base)) in sources.iter().enumerate() {
            let mut cursor = cursors[source_index];
            while let Some(run) = source
                .runs
                .get(cursor)
                .filter(|run| run.cluster_id == cluster_id)
            {
                let base = run
                    .doc_base
                    .checked_add(segment_base)
                    .ok_or_else(|| invalid_data("compacted ANN document base overflows u32"))?;
                let bytes = source.raw.as_slice();
                // Chunked rewrite: peak scratch stays at 256 KiB no matter how
                // large the run — the production incident had a single run of
                // 20M postings, and buffering it whole would be an 80 MB spike
                // in the middle of a merge.
                for chunk_start in (0..run.count).step_by(DOC_ID_REWRITE_CHUNK) {
                    let chunk_end = (chunk_start + DOC_ID_REWRITE_CHUNK).min(run.count);
                    scratch.clear();
                    scratch.reserve((chunk_end - chunk_start) * 4);
                    for index in chunk_start..chunk_end {
                        let doc_id = run_doc_id_with_base(bytes, run, index, base)?;
                        max_doc_id = max_doc_id.max(doc_id);
                        scratch.extend_from_slice(&doc_id.to_le_bytes());
                    }
                    writer.write_all(&scratch)?;
                    if cancellation
                        .is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Relaxed))
                    {
                        return Err(io::Error::new(
                            io::ErrorKind::Interrupted,
                            "ANN compaction cancelled",
                        ));
                    }
                }
                offset = checked_advance(offset, run.count * 4)?;
                count = count
                    .checked_add(run.count)
                    .ok_or_else(|| invalid_data("compacted ANN run count overflows usize"))?;
                cursor += 1;
            }
        }

        // Pass 2: ordinals, verbatim.
        let ordinals_offset = offset;
        for (source_index, &(source, _)) in sources.iter().enumerate() {
            let mut cursor = cursors[source_index];
            while let Some(run) = source
                .runs
                .get(cursor)
                .filter(|run| run.cluster_id == cluster_id)
            {
                copy_range(writer, &source.raw, run.ordinals.clone(), cancellation)?;
                offset = checked_advance(offset, run.ordinals.len())?;
                cursor += 1;
            }
        }

        // Pass 3: codes. Exact binary rows concatenate directly. ScaNN AH
        // blocks are run-relative, so repack a continuous output stream.
        let codes_offset = offset;
        if header.kind == AnnKind::ScannAh {
            let blocks = header.dim.div_ceil(code_size);
            let lanes = crate::structures::vector::scann::FAST_SCAN_LANES;
            let mut unpacked = Vec::with_capacity(lanes * blocks);
            let mut packed = Vec::with_capacity(blocks * lanes / 2);
            for (source_index, &(source, _)) in sources.iter().enumerate() {
                let mut cursor = cursors[source_index];
                while let Some(run) = source
                    .runs
                    .get(cursor)
                    .filter(|run| run.cluster_id == cluster_id)
                {
                    let bytes = &source.raw.as_slice()[run.codes.clone()];
                    for row in 0..run.count {
                        unpack_scann_ah_row(bytes, run.count, blocks, row, &mut unpacked)?;
                        if unpacked.len() == lanes * blocks {
                            packed.clear();
                            crate::structures::vector::scann::pack_fast_scan_block(
                                &unpacked,
                                blocks,
                                &mut packed,
                            )
                            .map_err(|error| invalid_data(error.to_string()))?;
                            writer.write_all(&packed)?;
                            offset = checked_advance(offset, packed.len())?;
                            unpacked.clear();
                            if cancellation
                                .is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Relaxed))
                            {
                                return Err(io::Error::new(
                                    io::ErrorKind::Interrupted,
                                    "ANN compaction cancelled",
                                ));
                            }
                        }
                    }
                    cursor += 1;
                }
            }
            // One compacted run has one row-major tail, not one tail per
            // source run. Pack each remaining row's adjacent block nibbles.
            if !unpacked.is_empty() {
                packed.clear();
                for row in unpacked.chunks_exact(blocks) {
                    for pair in row.chunks(2) {
                        packed.push(pair[0] | (pair.get(1).copied().unwrap_or(0) << 4));
                    }
                }
                writer.write_all(&packed)?;
                offset = checked_advance(offset, packed.len())?;
            }
        } else {
            for (source_index, &(source, _)) in sources.iter().enumerate() {
                let mut cursor = cursors[source_index];
                while let Some(run) = source
                    .runs
                    .get(cursor)
                    .filter(|run| run.cluster_id == cluster_id)
                {
                    let base = run
                        .doc_base
                        .checked_add(sources[source_index].1)
                        .ok_or_else(|| invalid_data("ANN document base overflow"))?;
                    record_exact_locations(
                        &mut locations,
                        offset,
                        run.count,
                        |row| {
                            Ok((
                                run_doc_id_with_base(source.raw.as_slice(), run, row, base)?,
                                read_u16(source.raw.as_slice(), run.ordinals.start + row * 2),
                            ))
                        },
                        cancellation,
                    )?;
                    copy_range(writer, &source.raw, run.codes.clone(), cancellation)?;
                    offset = checked_advance(offset, run.codes.len())?;
                    cursor += 1;
                }
            }
        }
        let expected_codes_len =
            expected_codes_column_len(header.kind, count, header.dim, header.code_size)?;
        if offset - codes_offset != expected_codes_len as u64 {
            return Err(invalid_data("compacted ANN code column length mismatch"));
        }

        // Consume this cluster's runs from every cursor.
        for (source_index, &(source, _)) in sources.iter().enumerate() {
            while source
                .runs
                .get(cursors[source_index])
                .is_some_and(|run| run.cluster_id == cluster_id)
            {
                cursors[source_index] += 1;
            }
        }

        records.push(RunRecord {
            cluster_id,
            doc_base: 0,
            count: u32::try_from(count)
                .map_err(|_| invalid_data("compacted ANN run exceeds u32 vectors"))?,
            max_doc_id,
            doc_ids_offset,
            ordinals_offset,
            codes_offset,
            codes_len: u64::try_from(expected_codes_column_len(
                header.kind,
                count,
                header.dim,
                code_size,
            )?)
            .map_err(|_| invalid_data("compacted ANN code length exceeds u64"))?,
        });
    }

    if records.is_empty() {
        return Err(invalid_data("cannot compact an ANN payload with no runs"));
    }
    finish_layout(writer, offset, &records)
}

/// Append one ScaNN-AH row as unpacked 4-bit block codes. Complete 32-row
/// groups use the FastScan v2 block-pair layout (`packed_code_position` is
/// the single definition); each run's final partial group is row-major.
#[cfg(feature = "native")]
fn unpack_scann_ah_row(
    codes: &[u8],
    count: usize,
    blocks: usize,
    row: usize,
    output: &mut Vec<u8>,
) -> io::Result<()> {
    use crate::structures::vector::scann::{
        FAST_SCAN_LANES, packed_block_bytes, packed_code_position,
    };

    let lanes = FAST_SCAN_LANES;
    let full_rows = count / lanes * lanes;
    let full_block_bytes = packed_block_bytes(blocks)
        .ok_or_else(|| invalid_data("ScaNN AH block size overflows usize"))?;
    let tail_row_bytes = blocks.div_ceil(2);
    for block in 0..blocks {
        let (byte_offset, high) = if row < full_rows {
            let (offset, high) = packed_code_position(row % lanes, block);
            ((row / lanes) * full_block_bytes + offset, high)
        } else {
            (
                (full_rows / lanes) * full_block_bytes
                    + (row - full_rows) * tail_row_bytes
                    + block / 2,
                !block.is_multiple_of(2),
            )
        };
        let byte = *codes
            .get(byte_offset)
            .ok_or_else(|| invalid_data("ScaNN AH row exceeds its code column"))?;
        output.push(if high { byte >> 4 } else { byte & 0x0f });
    }
    Ok(())
}

/// [`run_doc_id`] against an explicit base, for rewriting IDs absolute.
#[cfg(feature = "native")]
fn run_doc_id_with_base(bytes: &[u8], run: &AnnRun, index: usize, base: u32) -> io::Result<u32> {
    let local_doc_id = read_u32(bytes, run.doc_ids.start + index * 4);
    if local_doc_id > run.max_doc_id {
        return Err(invalid_data(
            "ANN run contains a document above its declared maximum",
        ));
    }
    base.checked_add(local_doc_id)
        .ok_or_else(|| invalid_data("compacted ANN document ID overflows u32"))
}

/// [`write_merged_ann`] plus freshly built runs appended to the payload —
/// used when some merge sources predate the field's current format and were
/// re-encoded while every compatible source is still byte-copied.
#[cfg(feature = "native")]
pub(crate) fn write_merged_ann_with_extra(
    sources: &[(&AnnDiskIndex, u32)],
    extra: &[BuildRun<'_>],
    writer: &mut (impl Write + ?Sized),
    cancellation: Option<&std::sync::atomic::AtomicBool>,
) -> io::Result<u64> {
    write_merged_ann_impl(sources, extra, writer, cancellation)
}

#[cfg(feature = "native")]
fn write_merged_ann_impl(
    sources: &[(&AnnDiskIndex, u32)],
    extra: &[BuildRun<'_>],
    writer: &mut (impl Write + ?Sized),
    cancellation: Option<&std::sync::atomic::AtomicBool>,
) -> io::Result<u64> {
    let Some((first, _)) = sources.first() else {
        return Err(invalid_data("cannot merge an empty ANN source list"));
    };
    if first.header.kind == AnnKind::IvfTq
        && !crate::structures::is_ivf_tq_cosine_generation(first.header.quantizer_version)
    {
        return Err(invalid_data(
            "legacy raw IVF-TQ generations cannot be merged; rebuild the index",
        ));
    }
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
    for run in extra {
        if run.doc_ids.is_empty()
            || run.cluster_id >= header.num_clusters
            || run.ordinals.len() != run.doc_ids.len()
            || run.codes.len()
                != expected_codes_column_len(
                    header.kind,
                    run.doc_ids.len(),
                    header.dim,
                    header.code_size,
                )?
        {
            return Err(invalid_data("extra ANN merge run columns are inconsistent"));
        }
        header.vector_count = header
            .vector_count
            .checked_add(run.doc_ids.len())
            .ok_or_else(|| invalid_data("merged ANN vector count overflows usize"))?;
    }
    validate_header(&header)?;
    write_header(writer, &header)?;
    let mut offset = ANN_HEADER_SIZE as u64;
    let run_capacity = sources.iter().try_fold(extra.len(), |count, (source, _)| {
        count
            .checked_add(source.runs.len())
            .ok_or_else(|| invalid_data("merged ANN run count overflows usize"))
    })?;
    let mut output_payload_starts = Vec::with_capacity(sources.len());
    for &(source, _) in sources {
        let payload_end = ANN_HEADER_SIZE + source.copied_payload_bytes();
        let output_payload_start = offset;
        output_payload_starts.push(output_payload_start);
        copy_range(
            writer,
            &source.raw,
            ANN_HEADER_SIZE..payload_end,
            cancellation,
        )?;
        offset = checked_advance(offset, payload_end - ANN_HEADER_SIZE)?;
    }

    // Extra runs' columns are appended after the copied extents so the
    // payload region stays contiguous for open()'s coverage validation.
    let mut extra_records = Vec::with_capacity(extra.len());
    let mut scratch = Vec::new();
    for run in extra {
        let count = run.doc_ids.len();
        let doc_ids_offset = offset;
        write_u32_column(writer, run.doc_ids, &mut scratch)?;
        offset = checked_advance(offset, count * 4)?;
        let ordinals_offset = offset;
        write_u16_column(writer, run.ordinals, &mut scratch)?;
        offset = checked_advance(offset, count * 2)?;
        let codes_offset = offset;
        writer.write_all(run.codes)?;
        offset = checked_advance(offset, run.codes.len())?;
        extra_records.push(RunRecord {
            cluster_id: run.cluster_id,
            doc_base: 0,
            count: u32::try_from(count)
                .map_err(|_| invalid_data("extra ANN run exceeds u32 vectors"))?,
            max_doc_id: run.doc_ids.iter().copied().max().unwrap_or(0),
            doc_ids_offset,
            ordinals_offset,
            codes_offset,
            codes_len: u64::try_from(run.codes.len())
                .map_err(|_| invalid_data("extra ANN code length exceeds u64"))?,
        });
    }

    // Every source directory is already cluster-sorted. Merge those compact
    // directories (plus the extra records) directly into the output with
    // O(source count) heap memory; the corpus payload extents above remain
    // untouched and source-contiguous. Extra records use pseudo source index
    // `sources.len()` so ties stay deterministic.
    let directory_offset = offset;
    let mut pending = BinaryHeap::with_capacity(sources.len() + 1);
    for (source_index, (source, _)) in sources.iter().enumerate() {
        pending.push(Reverse((source.runs[0].cluster_id, source_index, 0usize)));
    }
    if let Some(first_extra) = extra_records.first() {
        pending.push(Reverse((first_extra.cluster_id, sources.len(), 0usize)));
    }
    let mut written_runs = 0usize;
    while let Some(Reverse((_, source_index, run_index))) = pending.pop() {
        if source_index == sources.len() {
            write_run_record(writer, &extra_records[run_index])?;
            written_runs += 1;
            if let Some(next) = extra_records.get(run_index + 1) {
                pending.push(Reverse((next.cluster_id, source_index, run_index + 1)));
            }
            continue;
        }
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

/// Code-layout revision stamped into the header's former reserved `u32`.
///
/// Kinds whose leaf encoding changed shape without changing byte length need
/// an explicit gate: a FastScan v1 (`ScannAh`) column has exactly the v2
/// length for even block counts, so nothing else in the container would
/// notice a stale generation (docs/fast-scan-layout-v2.md). Every other kind
/// keeps the field at zero, preserving the old "reserved must be zero" check.
fn expected_layout_revision(kind: AnnKind) -> u32 {
    match kind {
        AnnKind::ScannAh => u32::from(crate::structures::vector::scann::FAST_SCAN_LAYOUT_VERSION),
        AnnKind::BinaryIvf | AnnKind::TqFlat | AnnKind::IvfTq | AnnKind::ScannBinary => 0,
    }
}

fn check_layout_revision(kind: AnnKind, found: u32) -> io::Result<()> {
    let expected = expected_layout_revision(kind);
    if found == expected {
        return Ok(());
    }
    Err(invalid_data(match kind {
        AnnKind::ScannAh => format!(
            "ScaNN AH payload uses FastScan layout revision {found}; this build reads revision \
             {expected} (docs/fast-scan-layout-v2.md). Rebuild the float ScaNN generation for \
             this field before searching it"
        ),
        _ => format!("ANN header reserved field is non-zero ({found}) for kind {kind:?}"),
    }))
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
    writer.write_u32::<LittleEndian>(expected_layout_revision(header.kind))?;
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
    cancellation: Option<&std::sync::atomic::AtomicBool>,
) -> io::Result<()> {
    if range.is_empty() {
        return Ok(());
    }
    let range_end = range.end;
    let mut chunk_start = range.start;
    let first_end = chunk_start.saturating_add(COPY_CHUNK).min(range_end);
    bytes.madvise_range(chunk_start..first_end, libc::MADV_WILLNEED);
    while chunk_start < range_end {
        if cancellation
            .is_some_and(|cancelled| cancelled.load(std::sync::atomic::Ordering::Relaxed))
        {
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "ANN merge copy cancelled",
            ));
        }
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
    if header.kind == AnnKind::IvfTq
        && !crate::structures::is_ivf_tq_cosine_generation(header.quantizer_version)
    {
        return Err(invalid_data(
            "IVF-TQ payload uses an unsupported legacy generation; rebuild the index",
        ));
    }
    if header.dim == 0
        || header.code_size == 0
        || header.num_clusters == 0
        || header.quantizer_version == 0
        || header.vector_count == 0
        || (header.kind == AnnKind::BinaryIvf
            && (header.codebook_version != 0
                || !header.dim.is_multiple_of(8)
                || header.code_size != header.dim.div_ceil(8)))
        || (header.kind == AnnKind::ScannBinary
            && (header.codebook_version == 0
                || header.routing != IvfRoutingMode::Flat
                || !header.dim.is_multiple_of(8)
                || header.code_size != header.dim / 8))
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
        || (header.kind == AnnKind::ScannAh
            && (header.codebook_version == 0
                || header.routing != IvfRoutingMode::Flat
                || u16::try_from(header.code_size).is_err()
                || u32::try_from(header.dim).is_err()
                || crate::structures::vector::scann::ScannEncoding::AsymmetricHash {
                    dimensions_per_block: header.code_size as u16,
                    bits_per_code: 4,
                }
                .row_code_bytes(u32::try_from(header.dim).unwrap_or(0))
                .is_err()))
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

#[inline]
fn tq_ivf_block_max_scale(block: &[u8]) -> f32 {
    // Supported cosine-generation writers sort the complete run by descending
    // residual scale, so lane zero is both this block's maximum and an upper
    // bound for every following block.
    f32::from_le_bytes(
        block[..size_of::<f32>()]
            .try_into()
            .expect("scale is one f32"),
    )
}

/// Doc ID for `index` in a run whose doc-ID column was validated at
/// [`AnnDiskIndex::open`]: every local ID is `<= max_doc_id` and
/// `doc_base + max_doc_id < total_docs`, so the sum cannot overflow. The scan
/// loops use this; [`run_doc_id`] remains the checked form for merge/rewrite
/// paths and tests.
#[inline]
fn run_doc_id_validated(bytes: &[u8], run: &AnnRun, index: usize) -> u32 {
    debug_assert!(index < run.count);
    run.doc_base + read_u32(bytes, run.doc_ids.start + index * 4)
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

    #[test]
    fn ann_visibility_clones_share_run_allocations_and_pin_lifetime() {
        use std::sync::Arc;
        let runs = [BuildRun {
            cluster_id: 0,
            doc_ids: &[0, 1],
            ordinals: &[0, 0],
            codes: &[0, 255],
        }];
        let mut bytes = Vec::new();
        write_built_runs(binary_header(2), &runs, &mut bytes, None).unwrap();
        let mut original =
            AnnDiskIndex::open(OwnedBytes::new(bytes), AnnKind::BinaryIvf, 2).unwrap();
        let mut report = crate::segment::pin::PinReport::default();
        original.pin_lookup_directory(crate::segment::pin::PinMode::Copy, &mut 1024, &mut report);
        assert!(report.pinned_bytes > 0);
        let mut changed = original.clone();
        let mut alive = crate::query::DocBitset::all(2);
        alive.clear(0);
        changed.alive_docs = Some(Arc::new(alive));
        assert!(original.alive_docs.is_none());
        assert!(Arc::ptr_eq(&original.runs, &changed.runs));
        assert!(Arc::ptr_eq(&original.heap_pins, &changed.heap_pins));
        assert_eq!(
            original.raw.as_slice().as_ptr(),
            changed.raw.as_slice().as_ptr()
        );
        let runs = Arc::downgrade(&original.runs);
        let pins = Arc::downgrade(&original.heap_pins);
        drop(original);
        assert!(runs.upgrade().is_some());
        assert!(pins.upgrade().is_some());
        assert_eq!(changed.runs.len(), 1);
        drop(changed);
        assert!(pins.upgrade().is_none());
        assert!(runs.upgrade().is_none());
    }

    /// Compaction must produce a payload indistinguishable from a fresh
    /// build: one run per cluster, fragmentation 1.0, absolute doc IDs — and
    /// return exactly the results the byte-copy merge of the same sources
    /// returns, for every cluster.
    #[test]
    fn compacted_merge_matches_byte_copy_and_resets_fragmentation() {
        // Segment A: clusters {0: 3 docs, 5: 2 docs}. Segment B: {0: 2, 2: 1}.
        // Merging A+B and then merging that result with A again produces
        // multi-generation fragmentation (cluster 0 in three extents).
        let a0_docs = [0u32, 1, 2];
        let a0_ords = [0u16, 0, 1];
        let a0_codes = [0x11u8, 0x22, 0x33];
        let a5_docs = [3u32, 4];
        let a5_ords = [0u16; 2];
        let a5_codes = [0x44u8, 0x55];
        let a_runs = [
            BuildRun {
                cluster_id: 0,
                doc_ids: &a0_docs,
                ordinals: &a0_ords,
                codes: &a0_codes,
            },
            BuildRun {
                cluster_id: 5,
                doc_ids: &a5_docs,
                ordinals: &a5_ords,
                codes: &a5_codes,
            },
        ];
        let mut header = binary_header(5);
        header.num_clusters = 8;
        let mut a_bytes = Vec::new();
        write_built_runs(header.clone(), &a_runs, &mut a_bytes, None).unwrap();
        let a = AnnDiskIndex::open(OwnedBytes::new(a_bytes), AnnKind::BinaryIvf, 5).unwrap();

        let b0_docs = [0u32, 2];
        let b0_ords = [1u16, 0];
        let b0_codes = [0x66u8, 0x77];
        let b2_docs = [1u32];
        let b2_ords = [0u16];
        let b2_codes = [0x88u8];
        let b_runs = [
            BuildRun {
                cluster_id: 0,
                doc_ids: &b0_docs,
                ordinals: &b0_ords,
                codes: &b0_codes,
            },
            BuildRun {
                cluster_id: 2,
                doc_ids: &b2_docs,
                ordinals: &b2_ords,
                codes: &b2_codes,
            },
        ];
        let mut header_b = binary_header(3);
        header_b.num_clusters = 8;
        let mut b_bytes = Vec::new();
        write_built_runs(header_b, &b_runs, &mut b_bytes, None).unwrap();
        let b = AnnDiskIndex::open(OwnedBytes::new(b_bytes), AnnKind::BinaryIvf, 3).unwrap();

        // Generation 1: byte-copy A (docs 0..5) + B (docs 5..8).
        let mut gen1_bytes = Vec::new();
        write_merged_ann(&[(&a, 0), (&b, 5)], &mut gen1_bytes).unwrap();
        let gen1 = AnnDiskIndex::open(OwnedBytes::new(gen1_bytes), AnnKind::BinaryIvf, 8).unwrap();

        // Generation 2 sources: gen1 (docs 0..8) + A again (docs 8..13).
        let sources: [(&AnnDiskIndex, u32); 2] = [(&gen1, 0), (&a, 8)];
        let predicted = predicted_merge_fragmentation(&sources);
        // 4 runs (gen1) + 2 runs (a) over 3 distinct clusters {0, 2, 5}.
        assert!((predicted - 2.0).abs() < 1e-9, "{predicted}");

        let mut copied_bytes = Vec::new();
        write_merged_ann(&sources, &mut copied_bytes).unwrap();
        let copied =
            AnnDiskIndex::open(OwnedBytes::new(copied_bytes), AnnKind::BinaryIvf, 13).unwrap();
        let mut compacted_bytes = Vec::new();
        write_compacted_ann_cancellable(&sources, &mut compacted_bytes, None, None).unwrap();
        let compacted =
            AnnDiskIndex::open(OwnedBytes::new(compacted_bytes), AnnKind::BinaryIvf, 13).unwrap();

        // Byte-copy carries the fragmentation forward; compaction resets it.
        let copied_health = copied.health();
        let compacted_health = compacted.health();
        assert!((copied_health.fragmentation() - 2.0).abs() < 1e-9);
        assert!((compacted_health.fragmentation() - 1.0).abs() < 1e-9);
        assert_eq!(compacted_health.runs, 3, "one run per non-empty cluster");
        assert_eq!(copied_health.vectors, compacted_health.vectors);
        assert_eq!(copied_health.payload_bytes, compacted_health.payload_bytes);
        assert_eq!(
            copied_health.largest_cluster_vectors,
            compacted_health.largest_cluster_vectors
        );

        // Every cluster returns identical (doc, ordinal, score) results.
        for cluster in 0..8u32 {
            let query = [0x5Au8];
            let from_copy = copied
                .search_binary_clusters::<false>(&query, 16, &[cluster])
                .unwrap();
            let from_compact = compacted
                .search_binary_clusters::<false>(&query, 16, &[cluster])
                .unwrap();
            assert_eq!(from_copy, from_compact, "cluster {cluster} diverged");
        }

        // Doc IDs are absolute now: every directory entry has doc_base 0.
        assert!(compacted.runs.iter().all(|run| run.doc_base == 0));

        // A compacted payload is indistinguishable from a built one, so it
        // must remain a valid source for future ordinary byte-copy merges.
        let mut generation3 = Vec::new();
        write_merged_ann(&[(&compacted, 0), (&b, 13)], &mut generation3).unwrap();
        let generation3 =
            AnnDiskIndex::open(OwnedBytes::new(generation3), AnnKind::BinaryIvf, 16).unwrap();
        assert_eq!(generation3.health().vectors, 16);
        // And the third A copy's docs landed at offset 8.
        let all: Vec<(u32, u16, f32)> = compacted
            .search_binary_clusters::<false>(&[0x5A], 32, &[0, 2, 5])
            .unwrap();
        let mut docs: Vec<u32> = all.iter().map(|&(doc, _, _)| doc).collect();
        docs.sort_unstable();
        assert_eq!(docs, (0..=12).collect::<Vec<u32>>());
    }

    #[test]
    fn compacted_scann_ah_repacks_blocks_across_run_boundaries() {
        use crate::structures::vector::scann::{FAST_SCAN_LANES, pack_fast_scan_block};

        const DIM: usize = 8;
        const DIMS_PER_BLOCK: usize = 2;
        const BLOCKS: usize = DIM / DIMS_PER_BLOCK;

        let make_rows = |start: usize, count: usize| -> Vec<u8> {
            (0..count)
                .flat_map(|row| {
                    (0..BLOCKS).map(move |block| ((start + row * 3 + block * 5) & 0x0f) as u8)
                })
                .collect()
        };
        let encode = |rows: &[u8]| -> Vec<u8> {
            let count = rows.len() / BLOCKS;
            let full_rows = count / FAST_SCAN_LANES * FAST_SCAN_LANES;
            let mut codes = Vec::new();
            for group in rows[..full_rows * BLOCKS].chunks_exact(FAST_SCAN_LANES * BLOCKS) {
                pack_fast_scan_block(group, BLOCKS, &mut codes).unwrap();
            }
            for row in rows[full_rows * BLOCKS..].chunks_exact(BLOCKS) {
                for pair in row.chunks(2) {
                    codes.push(pair[0] | (pair.get(1).copied().unwrap_or(0) << 4));
                }
            }
            codes
        };
        let make_source = |rows: &[u8]| {
            let count = rows.len() / BLOCKS;
            let docs: Vec<u32> = (0..count as u32).collect();
            let ordinals = vec![0u16; count];
            let codes = encode(rows);
            let run = BuildRun {
                cluster_id: 0,
                doc_ids: &docs,
                ordinals: &ordinals,
                codes: &codes,
            };
            let header = AnnDiskHeader {
                kind: AnnKind::ScannAh,
                routing: IvfRoutingMode::Flat,
                dim: DIM,
                code_size: DIMS_PER_BLOCK,
                num_clusters: 1,
                quantizer_version: 41,
                codebook_version: 73,
                vector_count: count,
            };
            let mut bytes = Vec::new();
            write_built_runs(header, &[run], &mut bytes, None).unwrap();
            AnnDiskIndex::open(OwnedBytes::new(bytes), AnnKind::ScannAh, count as u32).unwrap()
        };

        // The first source has a complete FastScan group plus a tail; the
        // second tail completes a new cross-source group plus an output tail.
        let left_rows = make_rows(1, 35);
        let right_rows = make_rows(9, 30);
        let left = make_source(&left_rows);
        let right = make_source(&right_rows);
        let mut bytes = Vec::new();
        write_compacted_ann_cancellable(&[(&left, 0), (&right, 35)], &mut bytes, None, None)
            .unwrap();
        let compacted = AnnDiskIndex::open(OwnedBytes::new(bytes), AnnKind::ScannAh, 65).unwrap();

        assert_eq!(compacted.runs.len(), 1);
        assert!((compacted.health().fragmentation() - 1.0).abs() < 1e-9);
        let run = &compacted.runs[0];
        let codes = &compacted.raw.as_slice()[run.codes.clone()];
        let mut decoded = Vec::new();
        for row in 0..run.count {
            unpack_scann_ah_row(codes, run.count, BLOCKS, row, &mut decoded).unwrap();
        }
        let mut expected = left_rows;
        expected.extend_from_slice(&right_rows);
        assert_eq!(decoded, expected);
        for row in 0..65 {
            assert_eq!(
                run_doc_id(compacted.raw.as_slice(), run, row).unwrap(),
                row as u32
            );
        }
    }

    /// Throughput comparison, prod-shaped: 320-byte codes, 4 sources.
    /// Ignored: run with `cargo test --release -- --ignored ann_merge_throughput --nocapture`.
    #[test]
    #[ignore]
    fn ann_merge_throughput_byte_copy_vs_compaction() {
        let code_size = 320usize;
        let clusters = 4_096u32;
        let vectors_per_source = 262_144usize;
        let sources_count = 4usize;

        let mut sources_bytes = Vec::new();
        for source_index in 0..sources_count {
            let mut per_cluster: Vec<(Vec<u32>, Vec<u16>, Vec<u8>)> = Vec::new();
            let vectors_per_cluster = vectors_per_source / clusters as usize;
            let mut doc = 0u32;
            for cluster in 0..clusters {
                let mut docs = Vec::with_capacity(vectors_per_cluster);
                let mut ords = Vec::with_capacity(vectors_per_cluster);
                let mut codes = Vec::with_capacity(vectors_per_cluster * code_size);
                for _ in 0..vectors_per_cluster {
                    docs.push(doc);
                    ords.push(0u16);
                    codes.extend(std::iter::repeat_n(
                        (doc ^ cluster ^ source_index as u32) as u8,
                        code_size,
                    ));
                    doc += 1;
                }
                per_cluster.push((docs, ords, codes));
            }
            let runs: Vec<BuildRun<'_>> = per_cluster
                .iter()
                .enumerate()
                .map(|(cluster, (docs, ords, codes))| BuildRun {
                    cluster_id: cluster as u32,
                    doc_ids: docs,
                    ordinals: ords,
                    codes,
                })
                .collect();
            let header = AnnDiskHeader {
                kind: AnnKind::BinaryIvf,
                routing: IvfRoutingMode::Hnsw,
                dim: code_size * 8,
                code_size,
                num_clusters: clusters,
                quantizer_version: 42,
                codebook_version: 0,
                vector_count: vectors_per_source,
            };
            let mut bytes = Vec::new();
            write_built_runs(header, &runs, &mut bytes, None).unwrap();
            sources_bytes.push(bytes);
        }
        let sources_open: Vec<AnnDiskIndex> = sources_bytes
            .iter()
            .map(|bytes| {
                AnnDiskIndex::open(
                    OwnedBytes::new(bytes.clone()),
                    AnnKind::BinaryIvf,
                    (vectors_per_source * sources_count) as u32,
                )
                .unwrap()
            })
            .collect();
        let sources: Vec<(&AnnDiskIndex, u32)> = sources_open
            .iter()
            .enumerate()
            .map(|(index, source)| (source, (index * vectors_per_source) as u32))
            .collect();
        let payload_bytes = sources_bytes.iter().map(Vec::len).sum::<usize>();

        // Fault the output buffer in before timing anything: the first pass
        // over a fresh Vec pays demand paging + kernel page zeroing for the
        // whole capacity, which the earlier version of this bench silently
        // charged to whichever writer ran first (making compaction look 2×
        // faster than the byte-copy purely by running second).
        let mut out = vec![0u8; payload_bytes + (1 << 20)];
        out.clear();
        // Interleave rounds and keep the best of each so neither path is
        // systematically first.
        let mut copy_secs = f64::INFINITY;
        let mut compact_secs = f64::INFINITY;
        let mut compacted_bytes = Vec::new();
        for _ in 0..3 {
            out.clear();
            let start = std::time::Instant::now();
            write_merged_ann(&sources, &mut out).unwrap();
            copy_secs = copy_secs.min(start.elapsed().as_secs_f64());

            out.clear();
            let start = std::time::Instant::now();
            write_compacted_ann_cancellable(&sources, &mut out, None, None).unwrap();
            compact_secs = compact_secs.min(start.elapsed().as_secs_f64());
            compacted_bytes = out.clone();
        }
        let compacted = AnnDiskIndex::open(
            OwnedBytes::new(compacted_bytes),
            AnnKind::BinaryIvf,
            (vectors_per_source * sources_count) as u32,
        )
        .unwrap();
        assert!((compacted.health().fragmentation() - 1.0).abs() < 1e-9);

        let gib = payload_bytes as f64 / (1u64 << 30) as f64;
        println!(
            "ann merge {:.2} GiB: byte-copy {:.3}s ({:.2} GiB/s), compaction {:.3}s \
             ({:.2} GiB/s), overhead {:.1}%",
            gib,
            copy_secs,
            gib / copy_secs,
            compact_secs,
            gib / compact_secs,
            100.0 * (compact_secs - copy_secs) / copy_secs,
        );
    }

    /// Compaction is undefined for block-packed TQ codes and must refuse.
    #[test]
    fn compaction_refuses_non_binary_payloads() {
        // A binary payload whose header is rewritten to the TQ kind would not
        // validate, so exercise the guard through the real gate: any source
        // list whose first header is not BinaryIvf is refused before a byte
        // is written. Reuse a TQ payload from the flat-TQ writer used by the
        // pruning tests.
        let codec = std::sync::Arc::new(crate::structures::TqCodec::new(8));
        let mut builder = crate::structures::TqFlatBuilder::new(codec);
        builder
            .add_batch(
                &[(0, 0), (1, 0)],
                &[
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
                    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ],
            )
            .unwrap();
        builder.finish();
        let mut bytes = Vec::new();
        write_built_tq_flat(&builder, &mut bytes).unwrap();
        let disk = AnnDiskIndex::open(OwnedBytes::new(bytes), AnnKind::TqFlat, 2).unwrap();
        let error = write_compacted_ann_cancellable(&[(&disk, 0)], &mut Vec::new(), None, None)
            .expect_err("TQ payloads must not compact");
        assert!(error.to_string().contains("binary"), "{error}");
    }

    /// Health math on a hand-built payload: two runs of one cluster (a
    /// byte-copy merge shape) plus one dominant leaf, checked against the
    /// Faiss imbalance definition computed by hand.
    #[test]
    fn ann_health_measures_skew_and_fragmentation() {
        // Segment A: 6 vectors in cluster 0 and 2 in cluster 3; segment B: 2
        // more in cluster 0. A byte-copy merge preserves each source extent,
        // producing the fragmented shape (two physical runs for cluster 0)
        // that build alone can never emit.
        let a0_docs = [0u32, 1, 2, 3, 4, 5];
        let a0_ords = [0u16; 6];
        let a0_codes = [0xAAu8; 6];
        let a3_docs = [6u32, 7];
        let a3_ords = [0u16; 2];
        let a3_codes = [0x0Fu8; 2];
        let a_runs = [
            BuildRun {
                cluster_id: 0,
                doc_ids: &a0_docs,
                ordinals: &a0_ords,
                codes: &a0_codes,
            },
            BuildRun {
                cluster_id: 3,
                doc_ids: &a3_docs,
                ordinals: &a3_ords,
                codes: &a3_codes,
            },
        ];
        let mut header = binary_header(8);
        header.num_clusters = 8;
        let mut a_bytes = Vec::new();
        write_built_runs(header, &a_runs, &mut a_bytes, None).unwrap();
        let a = AnnDiskIndex::open(OwnedBytes::new(a_bytes), AnnKind::BinaryIvf, 8).unwrap();

        let b0_docs = [0u32, 1];
        let b0_ords = [0u16; 2];
        let b0_codes = [0xBBu8; 2];
        let b_runs = [BuildRun {
            cluster_id: 0,
            doc_ids: &b0_docs,
            ordinals: &b0_ords,
            codes: &b0_codes,
        }];
        let mut header_b = binary_header(2);
        header_b.num_clusters = 8;
        let mut b_bytes = Vec::new();
        write_built_runs(header_b, &b_runs, &mut b_bytes, None).unwrap();
        let b = AnnDiskIndex::open(OwnedBytes::new(b_bytes), AnnKind::BinaryIvf, 2).unwrap();

        let mut merged_bytes = Vec::new();
        write_merged_ann(&[(&a, 0), (&b, 8)], &mut merged_bytes).unwrap();
        let disk =
            AnnDiskIndex::open(OwnedBytes::new(merged_bytes), AnnKind::BinaryIvf, 10).unwrap();

        let health = disk.health();
        assert_eq!(health.vectors, 10);
        assert_eq!(health.clusters_nonempty, 2);
        assert_eq!(health.clusters_total, 8);
        assert_eq!(health.runs, 3);
        assert_eq!(health.largest_cluster, 0);
        assert_eq!(health.largest_cluster_vectors, 8);
        assert!((health.largest_cluster_share() - 0.8).abs() < 1e-9);
        // 3 runs over 2 non-empty clusters.
        assert!((health.fragmentation() - 1.5).abs() < 1e-9);
        // Faiss: K * sum(n_i^2) / N^2 = 2 * (64 + 4) / 100 = 1.36
        assert!(
            (health.imbalance - 1.36).abs() < 1e-9,
            "{}",
            health.imbalance
        );
        // codes columns: 6 + 2 + 2 bytes at code_size 1.
        assert_eq!(health.payload_bytes, 10);
    }

    #[test]
    fn ann_health_is_balanced_at_one() {
        let docs: Vec<Vec<u32>> = (0..4).map(|c| vec![c * 2, c * 2 + 1]).collect();
        let ords = [0u16; 2];
        let codes = [0x55u8; 2];
        let runs: Vec<BuildRun<'_>> = docs
            .iter()
            .enumerate()
            .map(|(cluster, doc_ids)| BuildRun {
                cluster_id: cluster as u32,
                doc_ids,
                ordinals: &ords,
                codes: &codes,
            })
            .collect();
        let mut header = binary_header(8);
        header.num_clusters = 4;
        let mut bytes = Vec::new();
        write_built_runs(header, &runs, &mut bytes, None).unwrap();
        let disk = AnnDiskIndex::open(OwnedBytes::new(bytes), AnnKind::BinaryIvf, 8).unwrap();
        let health = disk.health();
        assert!((health.imbalance - 1.0).abs() < 1e-9);
        assert!((health.fragmentation() - 1.0).abs() < 1e-9);
        assert!((health.largest_cluster_share() - 0.25).abs() < 1e-9);
    }

    #[tokio::test]
    async fn repeated_binary_merge_copies_codes_and_lookup_rows_verbatim() {
        use crate::directories::FileHandle;
        use crate::segment::vector_data::LazyFlatVectorData;
        use crate::segment::vector_locations::{ExactLocations, write_copied_locations};

        async fn open(
            bytes: Vec<u8>,
            map: Vec<u8>,
            kind: AnnKind,
            docs: u32,
        ) -> (AnnDiskIndex, LazyFlatVectorData) {
            let bytes = OwnedBytes::new(bytes);
            let ann = AnnDiskIndex::open(bytes.clone(), kind, docs).unwrap();
            let flat = LazyFlatVectorData::open_indirect(
                FileHandle::from_bytes(OwnedBytes::new(map)),
                &ann,
                docs,
            )
            .await
            .unwrap();
            (ann, flat)
        }
        // ScaNN includes a SOAR secondary assignment for (4, 1). It must
        // retain a single logical value while both ANN copies remain intact.
        for kind in [AnnKind::BinaryIvf, AnnKind::ScannBinary] {
            let mut header = binary_header(if kind == AnnKind::ScannBinary { 5 } else { 4 });
            header.kind = kind;
            if kind == AnnKind::ScannBinary {
                header.routing = IvfRoutingMode::Flat;
                header.codebook_version = 73;
                header.num_clusters = 3;
            }
            let runs = [
                BuildRun {
                    cluster_id: 0,
                    doc_ids: &[4, 1],
                    ordinals: &[1, 0],
                    codes: &[41, 10],
                },
                BuildRun {
                    cluster_id: 1,
                    doc_ids: &[4, 0],
                    ordinals: &[0, 0],
                    codes: &[40, 0],
                },
                BuildRun {
                    cluster_id: 2,
                    doc_ids: &[4],
                    ordinals: &[1],
                    codes: &[41],
                },
            ];
            let mut locations = ExactLocations::default();
            let mut bytes = Vec::new();
            write_built_runs(
                header,
                &runs[..if kind == AnnKind::ScannBinary { 3 } else { 2 }],
                &mut bytes,
                Some(&mut locations),
            )
            .unwrap();
            let mut map = Vec::new();
            locations
                .write(8, 4, bytes.len() as u64, &mut map, None)
                .unwrap();
            // Bad addresses must fail opening, before any query can read a
            // different logical vector or an out-of-range ANN byte.
            let mut bad_map = map.clone();
            bad_map[38..46].copy_from_slice(&u64::MAX.to_le_bytes());
            let raw = OwnedBytes::new(bytes.clone());
            let ann = AnnDiskIndex::open(raw.clone(), kind, 6).unwrap();
            assert!(
                LazyFlatVectorData::open_indirect(
                    FileHandle::from_bytes(OwnedBytes::new(bad_map)),
                    &ann,
                    6,
                )
                .await
                .is_err()
            );
            for mutation in 0..4 {
                let mut bad = map.clone();
                match mutation {
                    0 => bad[38..46].copy_from_slice(&u64::from(u32::MAX).to_le_bytes()),
                    1 => bad[36..38].copy_from_slice(&u16::MAX.to_le_bytes()),
                    // A span cannot start in the header or cross run columns.
                    2 => bad[88..96].copy_from_slice(&0u64.to_le_bytes()),
                    _ => bad[96..100].copy_from_slice(&6u32.to_le_bytes()),
                }
                assert!(
                    LazyFlatVectorData::open_indirect(
                        FileHandle::from_bytes(OwnedBytes::new(bad)),
                        &ann,
                        6,
                    )
                    .await
                    .is_err(),
                    "corrupt lookup mutation {mutation}"
                );
            }
            let (source, exact) = open(bytes, map, kind, 6).await;
            assert_eq!(
                exact.read_vectors_batch(0, 4).await.unwrap().as_slice(),
                &[0, 10, 40, 41]
            );
            assert_eq!(exact.get_doc_id(2), (4, 0));
            assert_eq!(exact.get_doc_id(3), (4, 1));
            let mut merged: Option<(AnnDiskIndex, LazyFlatVectorData)> = None;
            for generation in 1..=3u32 {
                let (left_ann, left_exact) = merged
                    .as_ref()
                    .map(|(ann, exact)| (ann, exact))
                    .unwrap_or((&source, &exact));
                let left_docs = generation * 6;
                let mut bytes = Vec::new();
                write_merged_ann(&[(left_ann, 0), (&source, left_docs)], &mut bytes).unwrap();
                let expected_payload = [
                    &left_ann.raw[ANN_HEADER_SIZE..payload_end(left_ann)],
                    &source.raw[ANN_HEADER_SIZE..payload_end(&source)],
                ]
                .concat();
                assert_eq!(
                    &bytes[ANN_HEADER_SIZE..ANN_HEADER_SIZE + expected_payload.len()],
                    expected_payload
                );
                let mut map = Vec::new();
                write_copied_locations(
                    &[
                        (left_exact, 0, 0),
                        (&exact, left_docs, left_ann.copied_payload_bytes() as u64),
                    ],
                    8,
                    (generation as usize + 1) * 4,
                    bytes.len() as u64,
                    &mut map,
                    None,
                )
                .unwrap();
                let expected_rows = [left_exact.location_rows(), exact.location_rows()].concat();
                let (next_ann, next_exact) = open(bytes, map, kind, left_docs + 6).await;
                assert_eq!(next_exact.location_rows(), expected_rows);
                let expected_codes = [0, 10, 40, 41].repeat(generation as usize + 1);
                assert_eq!(
                    next_exact
                        .read_vectors_batch(0, next_exact.num_vectors)
                        .await
                        .unwrap()
                        .as_slice(),
                    expected_codes
                );
                #[cfg(feature = "sync")]
                assert_eq!(
                    next_exact
                        .read_vectors_batch_sync(0, next_exact.num_vectors)
                        .unwrap()
                        .as_slice(),
                    expected_codes
                );
                for i in 0..next_exact.num_vectors {
                    assert_eq!(
                        next_exact.get_doc_id(i),
                        (
                            6 * (i / 4) as u32 + [0, 1, 4, 4][i % 4],
                            [0, 0, 0, 1][i % 4]
                        )
                    );
                }
                // Own the result between iterations; lookup/codes refer to
                // immutable byte owners and do not depend on source lifetimes.
                merged = Some((next_ann, next_exact));
            }
            let (merged_ann, merged_exact) = merged.unwrap();
            let mut coalesced = Vec::new();
            let mut locations = ExactLocations::with_budget(128 * 1024, None).unwrap();
            write_compacted_ann_cancellable(
                &[(&merged_ann, 0)],
                &mut coalesced,
                None,
                Some(&mut locations),
            )
            .unwrap();
            let mut map = Vec::new();
            locations
                .write(
                    8,
                    merged_exact.num_vectors,
                    coalesced.len() as u64,
                    &mut map,
                    None,
                )
                .unwrap();
            let (ann, exact) = open(coalesced, map, kind, 24).await;
            assert_eq!(ann.health().fragmentation(), 1.0);
            assert_eq!(
                ann.header.quantizer_version,
                merged_ann.header.quantizer_version
            );
            assert_eq!(
                ann.header.codebook_version,
                merged_ann.header.codebook_version
            );
            assert_eq!(exact.num_vectors, merged_exact.num_vectors);
            for row in 0..exact.num_vectors {
                assert_eq!(exact.get_doc_id(row), merged_exact.get_doc_id(row));
                assert_eq!(
                    exact.read_vectors_batch(row, 1).await.unwrap().as_slice(),
                    merged_exact
                        .read_vectors_batch(row, 1)
                        .await
                        .unwrap()
                        .as_slice()
                );
            }
            let cancelled = std::sync::atomic::AtomicBool::new(true);
            let error = write_compacted_ann_cancellable(
                &[(&merged_ann, 0)],
                &mut Vec::new(),
                Some(&cancelled),
                Some(&mut ExactLocations::default()),
            )
            .unwrap_err();
            assert_eq!(error.kind(), io::ErrorKind::Interrupted);
        }
    }

    #[tokio::test]
    async fn exact_vector_lookup_rejects_short_metadata_reads() {
        use crate::directories::FileHandle;
        use crate::segment::vector_data::LazyFlatVectorData;
        use crate::segment::vector_locations::ExactLocations;
        let mut locations = ExactLocations::default();
        let mut bytes = Vec::new();
        write_built_runs(
            binary_header(1),
            &[BuildRun {
                cluster_id: 0,
                doc_ids: &[0],
                ordinals: &[0],
                codes: &[42],
            }],
            &mut bytes,
            Some(&mut locations),
        )
        .unwrap();
        let mut map = Vec::new();
        locations
            .write(8, 1, bytes.len() as u64, &mut map, None)
            .unwrap();
        let ann = AnnDiskIndex::open(OwnedBytes::new(bytes), AnnKind::BinaryIvf, 1).unwrap();
        let map = OwnedBytes::new(map);
        let truncated = FileHandle::lazy(
            map.len() as u64 + 1,
            std::sync::Arc::new(move |_| {
                let map = map.clone();
                Box::pin(async move { Ok(map) })
            }),
        );
        assert_eq!(
            LazyFlatVectorData::open_indirect(truncated, &ann, 1)
                .await
                .unwrap_err()
                .kind(),
            io::ErrorKind::UnexpectedEof
        );
    }

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
    fn binary_pruning_preserves_ties_deletions_and_complete_ordinal_scores() {
        // More than one score batch, with decreasing IDs: later equal-score
        // candidates must replace earlier ones, including across run boundaries.
        let count = BINARY_SCORE_BATCH + 137;
        let docs: Vec<u32> = (0..count as u32).rev().map(|id| id / 2).collect();
        let ordinals: Vec<u16> = (0..count as u32).rev().map(|id| (id % 2) as u16).collect();
        let codes: Vec<u8> = (0..count).map(|i| (i % 256) as u8).collect();
        let mut bytes = Vec::new();
        write_built_runs(
            binary_header(count),
            &[
                BuildRun {
                    cluster_id: 0,
                    doc_ids: &docs[..count - 73],
                    ordinals: &ordinals[..count - 73],
                    codes: &codes[..count - 73],
                },
                BuildRun {
                    cluster_id: 1,
                    doc_ids: &docs[count - 73..],
                    ordinals: &ordinals[count - 73..],
                    codes: &codes[count - 73..],
                },
            ],
            &mut bytes,
            None,
        )
        .unwrap();
        let total_docs = count.div_ceil(2) as u32;
        let mut disk =
            AnnDiskIndex::open(OwnedBytes::new(bytes), AnnKind::BinaryIvf, total_docs).unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        for deleted in [false, true] {
            disk.alive_docs = deleted.then(|| {
                std::sync::Arc::new(crate::query::DocBitset::from_predicate(total_docs, &|id| {
                    id % 3 != 0
                }))
            });
            for query in [[0u8], [0x55], [0xff]] {
                let mut all = Vec::new();
                let mut scratch = vec![0.0; BINARY_SCORE_BATCH];
                score_binary_cluster_runs(
                    &disk,
                    disk.raw.as_slice(),
                    &query,
                    &[0, 1],
                    &mut scratch,
                    &mut all,
                )
                .unwrap();
                let expected_all: Vec<_> = docs
                    .iter()
                    .zip(&ordinals)
                    .zip(&codes)
                    .filter_map(|((&doc, &ordinal), &code)| {
                        (!deleted || doc % 3 != 0).then_some((
                            doc,
                            ordinal,
                            1.0 - (query[0] ^ code).count_ones() as f32 / 8.0,
                        ))
                    })
                    .collect();
                assert_eq!(
                    all, expected_all,
                    "combined scoring must keep every live ordinal"
                );
                for k in [0, 1, 10, 100, count + 1] {
                    let mut vectors = BoundedUniqueAnnCollector::<true>::new(k);
                    let mut documents = BoundedAnnCollector::<true, true>::new(k);
                    for &(doc, ordinal, score) in &all {
                        vectors.insert(doc, ordinal, score);
                        documents.insert(doc, ordinal, score);
                    }
                    let vectors = vectors.into_sorted_results();
                    let documents = documents.into_sorted_results();
                    for parallel_threshold in [usize::MAX, 1] {
                        let actual_vectors = pool
                            .install(|| {
                                disk.search_binary_clusters_with_tuning::<false>(
                                    &query,
                                    k,
                                    &[0, 1],
                                    parallel_threshold,
                                )
                            })
                            .unwrap();
                        let actual_documents = pool
                            .install(|| {
                                disk.search_binary_clusters_with_tuning::<true>(
                                    &query,
                                    k,
                                    &[0, 1],
                                    parallel_threshold,
                                )
                            })
                            .unwrap();
                        assert_eq!(actual_vectors, vectors);
                        assert_eq!(actual_documents, documents);
                    }
                }
            }
        }
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
        write_built_runs(binary_header(2), &first_runs, &mut first_bytes, None).unwrap();
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
        write_built_runs(binary_header(2), &second_runs, &mut second_bytes, None).unwrap();
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
        let serial = merged
            .search_binary_clusters_with_tuning::<false>(&[0], 4, &[0, 1], usize::MAX)
            .unwrap();
        let parallel = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| merged.search_binary_clusters_with_tuning::<false>(&[0], 4, &[0, 1], 1))
            .unwrap();
        assert_eq!(parallel, serial, "parallel binary top-k changed results");

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

    #[test]
    fn legacy_ivf_tq_payload_is_rejected_while_opening() {
        let dim = 8;
        let marked_version = crate::structures::mark_ivf_tq_cosine_generation(7);
        let centroids = crate::structures::CoarseCentroids {
            num_clusters: 1,
            dim,
            centroids: vec![0.0; dim],
            version: marked_version,
            soar_config: None,
            routing_index: None,
        };
        let mut bytes = crate::segment::ann_build::build_ivf_tq(
            dim,
            IvfRoutingMode::Flat,
            &centroids,
            &[(0, 0)],
            &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        .unwrap();

        // The fixed header stores quantizer_version at bytes 24..32.
        bytes[24..32].copy_from_slice(&7u64.to_le_bytes());
        let error = AnnDiskIndex::open(OwnedBytes::new(bytes), AnnKind::IvfTq, 1)
            .err()
            .expect("legacy IVF-TQ payload must fail while opening")
            .to_string();
        assert!(error.contains("unsupported legacy generation"), "{error}");
    }

    #[test]
    fn binary_combined_search_deduplicates_soar_and_bounds_document_results() {
        // doc 0 / ordinal 0 occurs in both leaves, as it can under SOAR.
        // Without `(doc, ordinal)` dedup it would beat doc 1 for both Sum and
        // default LogSumExp; with correct dedup doc 1's two distinct values win.
        let cluster_0_docs = [0u32, 0, 1];
        let cluster_0_ordinals = [0u16, 1, 0];
        let cluster_0_codes = [0x00u8, 0xff, 0x03];
        let cluster_1_docs = [0u32, 1, 2];
        let cluster_1_ordinals = [0u16, 1, 0];
        let cluster_1_codes = [0x00u8, 0x0c, 0xf0];
        let runs = [
            BuildRun {
                cluster_id: 0,
                doc_ids: &cluster_0_docs,
                ordinals: &cluster_0_ordinals,
                codes: &cluster_0_codes,
            },
            BuildRun {
                cluster_id: 1,
                doc_ids: &cluster_1_docs,
                ordinals: &cluster_1_ordinals,
                codes: &cluster_1_codes,
            },
        ];
        let mut bytes = Vec::new();
        write_built_runs(binary_header(6), &runs, &mut bytes, None).unwrap();
        let disk = AnnDiskIndex::open(OwnedBytes::new(bytes), AnnKind::BinaryIvf, 3).unwrap();
        let parallel_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let serial_documents = disk
            .search_binary_clusters_with_tuning::<true>(&[0], 3, &[0, 1], usize::MAX)
            .unwrap();
        let parallel_documents = parallel_pool
            .install(|| disk.search_binary_clusters_with_tuning::<true>(&[0], 3, &[0, 1], 1))
            .unwrap();
        assert_eq!(parallel_documents, serial_documents);

        // Under Sum the SOAR duplicate decides the winner outright: doc 0
        // counted twice (2.0) would beat doc 1's two distinct values (1.5);
        // deduplicated, doc 1 wins.
        let sum = crate::query::MultiValueCombiner::Sum;
        let (result, probed) = disk
            .search_binary_combined_documents(1, &[0], &[0, 1], sum)
            .unwrap();
        let parallel_sum = parallel_pool
            .install(|| disk.search_binary_combined_documents_with_tuning(1, &[0], &[0, 1], sum, 1))
            .unwrap();
        assert_eq!(parallel_sum, (result.clone(), probed.clone()));
        assert_eq!(result.len(), 1, "combined search must honor k");
        assert_eq!(
            result[0].doc_id, 1,
            "SOAR duplicate changed {sum:?} ranking: {result:?}",
        );
        // Exact leaf scores are handed back only for the retained document,
        // deduplicated and sorted, so reranking can skip re-reading them.
        assert_eq!(
            probed
                .iter()
                .map(|&(doc_id, ordinal, _)| (doc_id, ordinal))
                .collect::<Vec<_>>(),
            vec![(1, 0), (1, 1)],
            "{sum:?}",
        );

        // Under the default smooth-max combiner doc 0's perfect ordinal wins
        // regardless of the duplicate, so dedup is pinned through the exact
        // score instead: it must equal the combiner over the two *distinct*
        // ordinals (1.0 and 0.0) — a double-counted (doc 0, ordinal 0) would
        // inflate it.
        let smooth_max = crate::query::MultiValueCombiner::default();
        let (result, probed) = disk
            .search_binary_combined_documents(1, &[0], &[0, 1], smooth_max)
            .unwrap();
        let parallel_smooth_max = parallel_pool
            .install(|| {
                disk.search_binary_combined_documents_with_tuning(1, &[0], &[0, 1], smooth_max, 1)
            })
            .unwrap();
        assert_eq!(parallel_smooth_max, (result.clone(), probed.clone()));
        assert_eq!(result.len(), 1, "combined search must honor k");
        assert_eq!(result[0].doc_id, 0, "{result:?}");
        let expected = smooth_max.combine(&[(0, 1.0), (1, 0.0)]);
        assert!(
            (result[0].score - expected).abs() < 1e-6,
            "SOAR duplicate leaked into {smooth_max:?}: got {}, expected {expected}",
            result[0].score,
        );
        assert_eq!(
            probed
                .iter()
                .map(|&(doc_id, ordinal, _)| (doc_id, ordinal))
                .collect::<Vec<_>>(),
            vec![(0, 0), (0, 1)],
            "{smooth_max:?}",
        );

        let (top_two, probed) = disk
            .search_binary_combined_documents(
                2,
                &[0],
                &[0, 1],
                crate::query::MultiValueCombiner::Sum,
            )
            .unwrap();
        assert_eq!(
            top_two
                .iter()
                .map(|candidate| candidate.doc_id)
                .collect::<Vec<_>>(),
            vec![1, 0],
        );
        assert_eq!(top_two.len(), 2, "full probing must still return at most k");
        // doc 0 ordinal 0 was probed twice (a SOAR duplicate) and must appear
        // once, with the two retained documents in ascending order.
        assert_eq!(
            probed
                .iter()
                .map(|&(doc_id, ordinal, _)| (doc_id, ordinal))
                .collect::<Vec<_>>(),
            vec![(0, 0), (0, 1), (1, 0), (1, 1)],
        );
    }

    #[test]
    fn combined_ordinal_reduction_handles_out_of_order_runs_for_every_combiner() {
        let out_of_order_with_duplicate = vec![
            (7, 1, 0.4),
            (3, 0, 0.8),
            (7, 0, 0.6),
            (3, 1, 0.2),
            (7, 1, 0.5), // higher SOAR estimate replaces 0.4, never adds to it
        ];
        for combiner in [
            crate::query::MultiValueCombiner::Max,
            crate::query::MultiValueCombiner::Sum,
            crate::query::MultiValueCombiner::Avg,
            crate::query::MultiValueCombiner::default(),
            crate::query::MultiValueCombiner::WeightedTopK { k: 2, decay: 0.7 },
        ] {
            let actual = combine_scored_ordinals(out_of_order_with_duplicate.clone(), 2, combiner);
            let mut expected = vec![
                AnnDocumentCandidate {
                    doc_id: 3,
                    score: combiner.combine(&[(0, 0.8), (1, 0.2)]),
                },
                AnnDocumentCandidate {
                    doc_id: 7,
                    score: combiner.combine(&[(0, 0.6), (1, 0.5)]),
                },
            ];
            expected.sort_unstable_by(|left, right| {
                right
                    .score
                    .total_cmp(&left.score)
                    .then_with(|| left.doc_id.cmp(&right.doc_id))
            });
            assert_eq!(actual, expected, "combiner {combiner:?}");
        }
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
    fn tq_combined_scan_ranks_complete_documents_instead_of_individual_values() {
        let dim = 8;
        let codec = std::sync::Arc::new(crate::structures::TqCodec::new(dim));
        let mut builder = crate::structures::TqFlatBuilder::new(std::sync::Arc::clone(&codec));
        let labels = [(0u32, 0u16), (1, 0), (1, 1)];
        let vectors = [
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // doc 0: best single value
            0.8, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // doc 1: two good values
            0.8, -0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        builder.add_batch(&labels, &vectors).unwrap();
        builder.finish();
        let mut bytes = Vec::new();
        write_built_tq_flat(&builder, &mut bytes).unwrap();
        let disk = AnnDiskIndex::open(OwnedBytes::new(bytes), AnnKind::TqFlat, 2).unwrap();
        let query = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let plan = crate::structures::TqQueryPlan::build(&codec, &query);

        let max = disk.search_tq_distinct(1, &plan).unwrap();
        let sum = disk
            .search_tq_combined_documents(1, &plan, crate::query::MultiValueCombiner::Sum)
            .unwrap();
        let default_combiner = disk
            .search_tq_combined_documents(1, &plan, crate::query::MultiValueCombiner::default())
            .unwrap();
        assert_eq!(max[0].0, 0, "fixture must favor doc 0 by Max: {max:?}");
        assert_eq!(
            sum[0].doc_id, 1,
            "two complete values must make doc 1 win by Sum: {sum:?}"
        );
        // The default combiner is a smooth maximum: doc 0's single best value
        // (1.0) outranks doc 1's two 0.8 values — value count alone no longer
        // wins. Sum above is what pins that both of doc 1's values were seen.
        assert_eq!(
            default_combiner[0].doc_id, 0,
            "default smooth-max must follow the best value: {default_combiner:?}"
        );
        assert!(sum[0].score > max[0].2);

        // Pure-copy upgrade merges may append a re-encoded older segment
        // after copied runs, so global run order need not follow doc IDs.
        // Documents remain complete within each run and must still combine.
        let (single_bytes, _) = build_tq_payload(dim, 1, 91);
        let single = AnnDiskIndex::open(OwnedBytes::new(single_bytes), AnnKind::TqFlat, 1).unwrap();
        let mut merged_bytes = Vec::new();
        write_merged_ann(&[(&disk, 1), (&single, 0)], &mut merged_bytes).unwrap();
        let merged = AnnDiskIndex::open(OwnedBytes::new(merged_bytes), AnnKind::TqFlat, 3).unwrap();
        let merged_sum = merged
            .search_tq_combined_documents(1, &plan, crate::query::MultiValueCombiner::Sum)
            .unwrap();
        assert_eq!(merged_sum[0].doc_id, 2);
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

    /// The scan loops read doc IDs without a per-lane maximum check, so the
    /// open-time column validation is what keeps a corrupt payload from
    /// producing out-of-range document IDs.
    /// A FastScan v1 `ScannAh` column has the same byte length as v2 for even
    /// block counts, so the container header must carry the layout revision
    /// and refuse stale generations loudly instead of misreading nibbles.
    #[test]
    fn open_refuses_scann_ah_payload_with_pre_v2_fast_scan_layout() {
        use crate::structures::vector::scann::{
            FAST_SCAN_LANES, FAST_SCAN_LAYOUT_VERSION, pack_fast_scan_block,
        };
        const BLOCKS: usize = 4;
        const DIMS_PER_BLOCK: usize = 2;
        const DIM: usize = BLOCKS * DIMS_PER_BLOCK;
        // Header layout: magic u32, kind u8, routing u8, version u16, dim u32,
        // code_size u32, num_clusters u32, then the layout-revision u32.
        const REVISION_OFFSET: usize = 20;

        let count = FAST_SCAN_LANES;
        let rows: Vec<u8> = (0..count * BLOCKS).map(|i| (i % 16) as u8).collect();
        let mut codes = Vec::new();
        pack_fast_scan_block(&rows, BLOCKS, &mut codes).unwrap();
        let docs: Vec<u32> = (0..count as u32).collect();
        let ordinals = vec![0u16; count];
        let run = BuildRun {
            cluster_id: 0,
            doc_ids: &docs,
            ordinals: &ordinals,
            codes: &codes,
        };
        let header = AnnDiskHeader {
            kind: AnnKind::ScannAh,
            routing: IvfRoutingMode::Flat,
            dim: DIM,
            code_size: DIMS_PER_BLOCK,
            num_clusters: 1,
            quantizer_version: 1,
            codebook_version: 1,
            vector_count: count,
        };
        let mut bytes = Vec::new();
        write_built_runs(header, &[run], &mut bytes, None).unwrap();

        let stamped = u32::from_le_bytes(
            bytes[REVISION_OFFSET..REVISION_OFFSET + 4]
                .try_into()
                .unwrap(),
        );
        assert_eq!(stamped, u32::from(FAST_SCAN_LAYOUT_VERSION));
        AnnDiskIndex::open(
            OwnedBytes::new(bytes.clone()),
            AnnKind::ScannAh,
            count as u32,
        )
        .expect("current layout revision opens");

        // A v1 build left the field at zero.
        let mut legacy = bytes;
        legacy[REVISION_OFFSET..REVISION_OFFSET + 4].copy_from_slice(&0u32.to_le_bytes());
        let error =
            match AnnDiskIndex::open(OwnedBytes::new(legacy), AnnKind::ScannAh, count as u32) {
                Ok(_) => panic!("a pre-v2 FastScan payload must be refused at open"),
                Err(error) => error,
            };
        let message = error.to_string();
        assert!(
            message.contains("FastScan layout revision 0") && message.contains("Rebuild"),
            "unexpected error: {message}"
        );

        // Every other kind still requires the field to be zero.
        let (tq, _) = build_tq_payload(8, 40, 5);
        let mut tq_bad = tq;
        tq_bad[REVISION_OFFSET..REVISION_OFFSET + 4].copy_from_slice(&7u32.to_le_bytes());
        let error = AnnDiskIndex::open(OwnedBytes::new(tq_bad), AnnKind::TqFlat, 40)
            .err()
            .expect("non-zero reserved field must be refused");
        assert!(
            error.to_string().contains("reserved field is non-zero"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn open_rejects_doc_id_column_above_run_maximum() {
        let dim = 8;
        let count = 40;
        let (bytes, _) = build_tq_payload(dim, count, 5);
        let disk = AnnDiskIndex::open(
            OwnedBytes::new(bytes.clone()),
            AnnKind::TqFlat,
            count as u32,
        )
        .unwrap();
        let run = &disk.runs[0];
        // Corrupt one local doc ID to exceed the run's declared maximum
        // without touching the directory (so every other check still passes).
        let mut corrupt = bytes;
        let offset = run.doc_ids.start + 3 * 4;
        corrupt[offset..offset + 4].copy_from_slice(&(run.max_doc_id + 1).to_le_bytes());
        let error =
            match AnnDiskIndex::open(OwnedBytes::new(corrupt), AnnKind::TqFlat, count as u32) {
                Ok(_) => panic!("corrupt doc-ID column must fail at open"),
                Err(error) => error,
            };
        assert!(
            error.to_string().contains("above its declared maximum"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn tq_parallel_fold_matches_a_sequential_scan() {
        use crate::structures::vector::quantization::{
            TQ_BLOCK_LANES, tq_block_bytes, tq_score_block,
        };

        // Exactly the production fan-out threshold exercises the parallel
        // fold/reduce path without relying on a test-only configuration.
        let dim = 8;
        let count = TQ_PARALLEL_SCAN_MIN_VECTORS;
        let (bytes, vectors) = build_tq_payload(dim, count, 87);
        let disk =
            AnnDiskIndex::open(OwnedBytes::new(bytes), AnnKind::TqFlat, count as u32).unwrap();
        let codec = crate::structures::TqCodec::new(dim);
        let plan = crate::structures::TqQueryPlan::build(&codec, &vectors[count / 3]);
        let k = 31;

        let block_bytes = tq_block_bytes(disk.header().code_size);
        assert_eq!(
            disk.runs.len(),
            1,
            "the regression must parallelize chunks inside one run"
        );
        assert!(
            disk.runs[0].codes.len() / block_bytes > TQ_PARALLEL_SCAN_CHUNK_BLOCKS,
            "the single run must span multiple parallel chunks"
        );
        let raw = disk.raw.as_slice();
        let mut reference = BoundedAnnCollector::<true, true>::new(k);
        let mut scores = [0.0f32; TQ_BLOCK_LANES];
        for run in disk.runs.iter() {
            let codes = &raw[run.codes.clone()];
            for (block_index, block) in codes.chunks_exact(block_bytes).enumerate() {
                tq_score_block(&plan, block, &mut scores);
                let lane_base = block_index * TQ_BLOCK_LANES;
                let lanes = TQ_BLOCK_LANES.min(run.count.saturating_sub(lane_base));
                for (lane, &score) in scores.iter().enumerate().take(lanes) {
                    let index = lane_base + lane;
                    reference.insert(
                        run_doc_id(raw, run, index).unwrap(),
                        read_u16(raw, run.ordinals.start + index * 2),
                        score,
                    );
                }
            }
        }

        assert_eq!(
            disk.search_tq_distinct(k, &plan).unwrap(),
            reference.into_sorted_results(),
        );
    }

    #[test]
    fn ivf_tq_scale_pruning_matches_the_unpruned_scan() {
        use crate::structures::vector::ivf::{CoarseCentroids, CoarseConfig};
        use crate::structures::vector::quantization::{
            TQ_BLOCK_LANES, tq_ivf_block_bytes, tq_score_ivf_block,
        };
        use crate::structures::{IvfTqIndex, TqCodec, TqIvfEncodeScratch, TqIvfQueryPlan};

        let dim = 32;
        let count = 400usize;
        let codec = std::sync::Arc::new(TqCodec::new(dim));
        let mut state = 5u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        };
        let vectors: Vec<Vec<f32>> = (0..count)
            .map(|_| {
                let mut v: Vec<f32> = (0..dim).map(|_| next()).collect();
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                v.iter_mut().for_each(|x| *x /= norm);
                v
            })
            .collect();
        let mut centroids = CoarseCentroids::train(&CoarseConfig::new(dim, 8), &vectors, "test");
        centroids.version = crate::structures::mark_ivf_tq_cosine_generation(centroids.version);
        let mut index = IvfTqIndex::new(
            dim,
            crate::dsl::IvfRoutingMode::Flat,
            centroids.version,
            std::sync::Arc::clone(&codec),
        );
        let mut scratch = TqIvfEncodeScratch::default();
        for (i, vector) in vectors.iter().enumerate() {
            index.add_vector(
                &centroids,
                (i / 2) as u32,
                (i % 2) as u16,
                vector,
                &mut scratch,
            );
        }
        let mut bytes = Vec::new();
        write_built_ivf_tq(&index, centroids.num_clusters, &mut bytes).unwrap();
        let disk =
            AnnDiskIndex::open(OwnedBytes::new(bytes), AnnKind::IvfTq, count as u32).unwrap();

        // The supported cosine generation promises descending residual scales,
        // enabling the reader to terminate the run at the first losing block.
        let block_bytes = tq_ivf_block_bytes(disk.header().code_size);
        let raw = disk.raw.as_slice();
        for run in disk.runs.iter() {
            let codes = &raw[run.codes.clone()];
            let mut previous_scale = f32::INFINITY;
            for (block_index, block) in codes.chunks_exact(block_bytes).enumerate() {
                let lane_base = block_index * TQ_BLOCK_LANES;
                let lanes = TQ_BLOCK_LANES.min(run.count.saturating_sub(lane_base));
                let mut block_scales = block[..TQ_BLOCK_LANES * size_of::<f32>()]
                    .chunks_exact(size_of::<f32>())
                    .take(lanes)
                    .map(|lane| f32::from_le_bytes(lane.try_into().unwrap()));
                let first_scale = block_scales.next().unwrap();
                assert_eq!(tq_ivf_block_max_scale(block), first_scale);
                assert!(first_scale <= previous_scale);
                previous_scale = first_scale;
                for scale in block_scales {
                    assert!(scale <= previous_scale);
                    previous_scale = scale;
                }
            }
        }
        let k = 10;
        let parallel_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        for query_seed in [1u64, 9, 42] {
            let mut qstate = query_seed;
            let mut qnext = move || {
                qstate = qstate
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((qstate >> 33) as f32 / (1u64 << 31) as f32) - 0.5
            };
            let query: Vec<f32> = (0..dim).map(|_| qnext()).collect();
            let plan = TqIvfQueryPlan::build(
                &centroids,
                &codec,
                &query,
                8,
                crate::dsl::IvfRoutingMode::Flat,
            );
            // Unpruned reference: score every block of every probed run with
            // the identical kernel and collector.
            let mut reference = BoundedAnnCollector::<true, true>::new(k);
            let mut unpruned_scores = Vec::new();
            let mut scores = [0.0f32; TQ_BLOCK_LANES];
            for (cluster_id, cluster_dot) in plan.cluster_dots() {
                for run in disk.cluster_runs(cluster_id) {
                    let codes = &raw[run.codes.clone()];
                    for (block_index, block) in codes.chunks_exact(block_bytes).enumerate() {
                        tq_score_ivf_block(plan.tq_plan(), block, cluster_dot, &mut scores);
                        let lane_base = block_index * TQ_BLOCK_LANES;
                        let lanes = TQ_BLOCK_LANES.min(run.count.saturating_sub(lane_base));
                        for (lane, &score) in scores.iter().enumerate().take(lanes) {
                            let idx = lane_base + lane;
                            reference.insert(
                                run_doc_id(raw, run, idx).unwrap(),
                                read_u16(raw, run.ordinals.start + idx * 2),
                                score,
                            );
                            unpruned_scores.push((
                                run_doc_id(raw, run, idx).unwrap(),
                                read_u16(raw, run.ordinals.start + idx * 2),
                                score,
                            ));
                        }
                    }
                }
            }

            let pruned = disk.search_ivf_tq_distinct(k, &plan).unwrap();
            let forced_parallel = parallel_pool.install(|| {
                // Small chunks force several tasks from this compact test
                // payload instead of allocating a production-sized one.
                disk.search_ivf_tq_distinct_with_tuning(k, &plan, 1, 4)
                    .unwrap()
            });
            let reference = reference.into_sorted_results();
            assert_eq!(
                pruned, reference,
                "scale-bound pruning must not change the estimated top-k (seed {query_seed})"
            );
            assert_eq!(
                forced_parallel, reference,
                "parallel IVF-TQ pruning must match the unpruned top-k (seed {query_seed})"
            );
            for combiner in [
                crate::query::MultiValueCombiner::Max,
                crate::query::MultiValueCombiner::Sum,
                crate::query::MultiValueCombiner::Avg,
                crate::query::MultiValueCombiner::default(),
                crate::query::MultiValueCombiner::WeightedTopK { k: 3, decay: 0.7 },
            ] {
                let expected = combine_scored_ordinals(unpruned_scores.clone(), k, combiner);
                let combined = disk
                    .search_ivf_tq_combined_documents(k, &plan, combiner)
                    .unwrap();
                let parallel_combined = parallel_pool
                    .install(|| {
                        disk.search_ivf_tq_combined_documents_with_tuning(k, &plan, combiner, 1, 4)
                    })
                    .unwrap();
                assert_eq!(
                    combined, expected,
                    "combined IVF-TQ scan diverged from the unpruned reference \
                     for {combiner:?} (seed {query_seed})",
                );
                assert_eq!(
                    parallel_combined, expected,
                    "parallel combined IVF-TQ scan diverged from the unpruned reference \
                     for {combiner:?} (seed {query_seed})",
                );
                assert!(combined.len() <= k);
            }
        }
    }

    /// Focused single-segment scan benchmark for the adaptive IVF-TQ fan-out.
    /// Run with:
    ///
    /// `cargo test --release -p summa-core ivf_tq_parallel_scan_benchmark -- --ignored --nocapture`
    ///
    /// `IVF_SCAN_BENCH_DOCS`, `IVF_SCAN_BENCH_DIM`, `IVF_SCAN_BENCH_ITERS`,
    /// and `IVF_SCAN_BENCH_THREADS` override the defaults.
    #[test]
    #[ignore]
    fn ivf_tq_parallel_scan_benchmark() {
        use crate::structures::vector::ivf::CoarseCentroids;
        use crate::structures::{IvfTqIndex, TqCodec, TqIvfEncodeScratch, TqIvfQueryPlan};

        fn env_usize(name: &str, default: usize) -> usize {
            std::env::var(name)
                .ok()
                .map(|value| {
                    value
                        .parse::<usize>()
                        .unwrap_or_else(|_| panic!("{name} must be a positive integer"))
                })
                .unwrap_or(default)
                .max(1)
        }

        fn median_ms(samples: &mut [f64]) -> f64 {
            samples.sort_unstable_by(f64::total_cmp);
            samples[samples.len() / 2]
        }

        let dim = env_usize("IVF_SCAN_BENCH_DIM", 128);
        let count = env_usize("IVF_SCAN_BENCH_DOCS", 262_144);
        let iterations = env_usize("IVF_SCAN_BENCH_ITERS", 30);
        let threads = env_usize("IVF_SCAN_BENCH_THREADS", num_cpus::get().min(8));
        let codec = std::sync::Arc::new(TqCodec::new(dim));
        let version = crate::structures::mark_ivf_tq_cosine_generation(0x51ca_0001);
        let mut centroid = vec![0.0f32; dim];
        centroid[0] = 1.0;
        let centroids = CoarseCentroids {
            num_clusters: 1,
            dim,
            centroids: centroid,
            version,
            soar_config: None,
            routing_index: None,
        };
        let mut index = IvfTqIndex::new(
            dim,
            crate::dsl::IvfRoutingMode::Flat,
            version,
            std::sync::Arc::clone(&codec),
        );
        let mut scratch = TqIvfEncodeScratch::default();
        let mut state = 0x5eed_u64;
        let mut vector = vec![0.0f32; dim];
        for posting_index in 0..count {
            for value in &mut vector {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                *value = ((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5;
            }
            let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
            vector.iter_mut().for_each(|value| *value /= norm);
            index.add_vector(
                &centroids,
                u32::try_from(posting_index / 2).unwrap(),
                u16::try_from(posting_index % 2).unwrap(),
                &vector,
                &mut scratch,
            );
        }
        let mut bytes = Vec::new();
        write_built_ivf_tq(&index, 1, &mut bytes).unwrap();
        let disk = AnnDiskIndex::open(
            OwnedBytes::new(bytes),
            AnnKind::IvfTq,
            u32::try_from(count.div_ceil(2)).unwrap(),
        )
        .unwrap();
        let query = vec![1.0f32; dim];
        let plan = TqIvfQueryPlan::build(
            &centroids,
            &codec,
            &query,
            1,
            crate::dsl::IvfRoutingMode::Flat,
        );
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let serial_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();

        let serial = pool
            .install(|| disk.search_ivf_tq_distinct_with_tuning(20, &plan, usize::MAX, 512))
            .unwrap();
        let parallel = pool
            .install(|| disk.search_ivf_tq_distinct_with_tuning(20, &plan, 1, 512))
            .unwrap();
        assert_eq!(parallel, serial);

        let mut serial_ms = Vec::with_capacity(iterations);
        let mut parallel_ms = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let start = std::time::Instant::now();
            std::hint::black_box(
                pool.install(|| {
                    disk.search_ivf_tq_distinct_with_tuning(20, &plan, usize::MAX, 512)
                })
                .unwrap(),
            );
            serial_ms.push(start.elapsed().as_secs_f64() * 1_000.0);

            let start = std::time::Instant::now();
            std::hint::black_box(
                pool.install(|| disk.search_ivf_tq_distinct_with_tuning(20, &plan, 1, 512))
                    .unwrap(),
            );
            parallel_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
        }
        let serial_p50 = median_ms(&mut serial_ms);
        let parallel_p50 = median_ms(&mut parallel_ms);
        println!(
            "IVF-TQ distinct scan: postings={count} dim={dim} threads={threads} \
             serial_p50={serial_p50:.3}ms parallel_p50={parallel_p50:.3}ms speedup={:.2}x",
            serial_p50 / parallel_p50,
        );

        let combiner = crate::query::MultiValueCombiner::Sum;
        let serial_combined = serial_pool
            .install(|| {
                disk.search_ivf_tq_combined_documents_with_tuning(20, &plan, combiner, 1, 512)
            })
            .unwrap();
        let parallel_combined = pool
            .install(|| {
                disk.search_ivf_tq_combined_documents_with_tuning(20, &plan, combiner, 1, 512)
            })
            .unwrap();
        assert_eq!(parallel_combined, serial_combined);
        let mut serial_combined_ms = Vec::with_capacity(iterations);
        let mut parallel_combined_ms = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let start = std::time::Instant::now();
            std::hint::black_box(
                serial_pool
                    .install(|| {
                        disk.search_ivf_tq_combined_documents_with_tuning(
                            20, &plan, combiner, 1, 512,
                        )
                    })
                    .unwrap(),
            );
            serial_combined_ms.push(start.elapsed().as_secs_f64() * 1_000.0);

            let start = std::time::Instant::now();
            std::hint::black_box(
                pool.install(|| {
                    disk.search_ivf_tq_combined_documents_with_tuning(20, &plan, combiner, 1, 512)
                })
                .unwrap(),
            );
            parallel_combined_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
        }
        let serial_combined_p50 = median_ms(&mut serial_combined_ms);
        let parallel_combined_p50 = median_ms(&mut parallel_combined_ms);
        println!(
            "IVF-TQ combined scan: postings={count} dim={dim} threads={threads} \
             serial_p50={serial_combined_p50:.3}ms parallel_p50={parallel_combined_p50:.3}ms \
             speedup={:.2}x",
            serial_combined_p50 / parallel_combined_p50,
        );
    }

    /// Focused binary-IVF benchmark for single-value top-k and multi-value
    /// combined scoring. Environment overrides use the `BINARY_SCAN_BENCH_*`
    /// prefix with `POSTINGS`, `DIM_BITS`, `ITERS`, and `THREADS` suffixes.
    #[test]
    #[ignore]
    fn binary_ivf_parallel_scan_benchmark() {
        fn env_usize(name: &str, default: usize) -> usize {
            std::env::var(name)
                .ok()
                .map(|value| value.parse::<usize>().unwrap())
                .unwrap_or(default)
                .max(1)
        }

        fn median_ms(samples: &mut [f64]) -> f64 {
            samples.sort_unstable_by(f64::total_cmp);
            samples[samples.len() / 2]
        }

        let count = env_usize("BINARY_SCAN_BENCH_POSTINGS", 131_072);
        let dim_bits = env_usize("BINARY_SCAN_BENCH_DIM_BITS", 2_048);
        assert!(dim_bits.is_multiple_of(8));
        let code_size = dim_bits / 8;
        let iterations = env_usize("BINARY_SCAN_BENCH_ITERS", 30);
        let threads = env_usize("BINARY_SCAN_BENCH_THREADS", num_cpus::get().min(8));
        let mut state = 0xb1a4_5eed_u64;
        let mut codes = vec![0u8; count * code_size];
        for code in &mut codes {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *code = (state >> 56) as u8;
        }
        let query = vec![0xa5u8; code_size];
        let unique_docs: Vec<u32> = (0..count)
            .map(|index| u32::try_from(index).unwrap())
            .collect();
        let unique_ordinals = vec![0u16; count];
        let multi_docs: Vec<u32> = (0..count)
            .map(|index| u32::try_from(index / 2).unwrap())
            .collect();
        let multi_ordinals: Vec<u16> = (0..count)
            .map(|index| u16::try_from(index % 2).unwrap())
            .collect();
        let header = AnnDiskHeader {
            kind: AnnKind::BinaryIvf,
            routing: IvfRoutingMode::Flat,
            dim: dim_bits,
            code_size,
            num_clusters: 1,
            quantizer_version: 0xb1a4,
            codebook_version: 0,
            vector_count: count,
        };
        let mut unique_bytes = Vec::new();
        write_built_runs(
            header.clone(),
            &[BuildRun {
                cluster_id: 0,
                doc_ids: &unique_docs,
                ordinals: &unique_ordinals,
                codes: &codes,
            }],
            &mut unique_bytes,
            None,
        )
        .unwrap();
        let unique_disk = AnnDiskIndex::open(
            OwnedBytes::new(unique_bytes),
            AnnKind::BinaryIvf,
            u32::try_from(count).unwrap(),
        )
        .unwrap();
        let mut multi_bytes = Vec::new();
        write_built_runs(
            header,
            &[BuildRun {
                cluster_id: 0,
                doc_ids: &multi_docs,
                ordinals: &multi_ordinals,
                codes: &codes,
            }],
            &mut multi_bytes,
            None,
        )
        .unwrap();
        let multi_disk = AnnDiskIndex::open(
            OwnedBytes::new(multi_bytes),
            AnnKind::BinaryIvf,
            u32::try_from(count.div_ceil(2)).unwrap(),
        )
        .unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let serial_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();

        let serial = pool
            .install(|| {
                unique_disk.search_binary_clusters_with_tuning::<false>(
                    &query,
                    20,
                    &[0],
                    usize::MAX,
                )
            })
            .unwrap();
        let parallel = pool
            .install(|| {
                unique_disk.search_binary_clusters_with_tuning::<false>(&query, 20, &[0], 1)
            })
            .unwrap();
        assert_eq!(parallel, serial);
        let mut serial_ms = Vec::with_capacity(iterations);
        let mut parallel_ms = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let start = std::time::Instant::now();
            std::hint::black_box(
                pool.install(|| {
                    unique_disk.search_binary_clusters_with_tuning::<false>(
                        &query,
                        20,
                        &[0],
                        usize::MAX,
                    )
                })
                .unwrap(),
            );
            serial_ms.push(start.elapsed().as_secs_f64() * 1_000.0);

            let start = std::time::Instant::now();
            std::hint::black_box(
                pool.install(|| {
                    unique_disk.search_binary_clusters_with_tuning::<false>(&query, 20, &[0], 1)
                })
                .unwrap(),
            );
            parallel_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
        }
        let serial_p50 = median_ms(&mut serial_ms);
        let parallel_p50 = median_ms(&mut parallel_ms);
        println!(
            "binary-IVF distinct scan: postings={count} dim_bits={dim_bits} threads={threads} \
             serial_p50={serial_p50:.3}ms parallel_p50={parallel_p50:.3}ms speedup={:.2}x",
            serial_p50 / parallel_p50,
        );

        let combiner = crate::query::MultiValueCombiner::Sum;
        let serial_combined = serial_pool
            .install(|| {
                multi_disk.search_binary_combined_documents_with_tuning(
                    20,
                    &query,
                    &[0],
                    combiner,
                    1,
                )
            })
            .unwrap();
        let parallel_combined = pool
            .install(|| {
                multi_disk.search_binary_combined_documents_with_tuning(
                    20,
                    &query,
                    &[0],
                    combiner,
                    1,
                )
            })
            .unwrap();
        assert_eq!(parallel_combined, serial_combined);
        let mut serial_combined_ms = Vec::with_capacity(iterations);
        let mut parallel_combined_ms = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let start = std::time::Instant::now();
            std::hint::black_box(
                serial_pool
                    .install(|| {
                        multi_disk.search_binary_combined_documents_with_tuning(
                            20,
                            &query,
                            &[0],
                            combiner,
                            1,
                        )
                    })
                    .unwrap(),
            );
            serial_combined_ms.push(start.elapsed().as_secs_f64() * 1_000.0);

            let start = std::time::Instant::now();
            std::hint::black_box(
                pool.install(|| {
                    multi_disk.search_binary_combined_documents_with_tuning(
                        20,
                        &query,
                        &[0],
                        combiner,
                        1,
                    )
                })
                .unwrap(),
            );
            parallel_combined_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
        }
        let serial_combined_p50 = median_ms(&mut serial_combined_ms);
        let parallel_combined_p50 = median_ms(&mut parallel_combined_ms);
        println!(
            "binary-IVF combined scan: postings={count} dim_bits={dim_bits} threads={threads} \
             serial_p50={serial_combined_p50:.3}ms parallel_p50={parallel_combined_p50:.3}ms \
             speedup={:.2}x",
            serial_combined_p50 / parallel_combined_p50,
        );
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

        // A block-padded TQ column must not validate under another kind.
        let (short_bytes, _) = build_tq_payload(20, 4, 9);
        let mut wrong_kind = short_bytes.clone();
        wrong_kind[4] = AnnKind::BinaryIvf as u8;
        assert!(
            AnnDiskIndex::open(OwnedBytes::new(wrong_kind), AnnKind::BinaryIvf, 4).is_err(),
            "TQ block-padded columns must not validate under another kind"
        );

        // The retired IVF-PQ discriminant must be refused loudly.
        let (legacy_bytes, _) = build_tq_payload(20, 4, 9);
        let mut legacy_kind = legacy_bytes.clone();
        legacy_kind[4] = 1;
        let Err(error) = AnnDiskIndex::open(OwnedBytes::new(legacy_kind), AnnKind::TqFlat, 4)
        else {
            panic!("retired IVF-PQ payloads must not open");
        };
        assert!(
            error.to_string().contains("IVF-PQ"),
            "error must name the retired format: {error}"
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
        write_built_runs(binary_header(1), &runs, &mut bytes, None).unwrap();
        let footer = bytes.len() - ANN_FOOTER_SIZE;
        let directory = usize::try_from(u64::from_le_bytes(
            bytes[footer..footer + 8].try_into().unwrap(),
        ))
        .unwrap();
        bytes[directory + 12..directory + 16].copy_from_slice(&10u32.to_le_bytes());
        assert!(AnnDiskIndex::open(OwnedBytes::new(bytes), AnnKind::BinaryIvf, 1).is_err());
    }

    fn scann_binary_artifact(
        generation: u64,
    ) -> crate::structures::vector::scann::ScannTrainedArtifact {
        use crate::structures::vector::scann::{
            ScannConfig, ScannEncoding, ScannRoutingLevel, ScannTrainedArtifact,
        };
        ScannTrainedArtifact::new(
            generation,
            100_000,
            ScannConfig {
                dimension: 16,
                tree_levels: 1,
                num_leaves: 2,
                encoding: ScannEncoding::BinaryHamming,
            },
            vec![ScannRoutingLevel {
                centroid_count: 2,
                centroid_codes: vec![0, 0, 0xff, 0xff],
                minimums: Vec::new(),
                steps: Vec::new(),
                child_offsets: Vec::new(),
            }],
            None,
        )
        .unwrap()
    }

    #[tokio::test]
    async fn scann_binary_ann_round_trip_validates_exact_global_generation() {
        use crate::structures::vector::scann::{
            ScannEncoding, ScannLeafRun, ScannSegmentPayload, ScannTrainedArtifactView,
        };

        let artifact = scann_binary_artifact(41);
        let primary = ScannLeafRun::from_rows(
            0,
            0,
            &[0, 1],
            &[0, 2],
            vec![0x12, 0x34, 0xab, 0xcd],
            ScannEncoding::BinaryHamming,
            16,
        )
        .unwrap();
        let secondary = ScannLeafRun::from_rows(
            1,
            0,
            &[0],
            &[0],
            vec![0x12, 0x34],
            ScannEncoding::BinaryHamming,
            16,
        )
        .unwrap();
        let payload = ScannSegmentPayload::new(&artifact, 2, vec![primary, secondary]).unwrap();
        let mut bytes = Vec::new();
        let mut locations = crate::segment::vector_locations::ExactLocations::default();
        write_built_scann(&payload, &mut bytes, Some(&mut locations)).unwrap();
        let mut map = Vec::new();
        locations
            .write(16, 2, bytes.len() as u64, &mut map, None)
            .unwrap();

        let disk = AnnDiskIndex::open(OwnedBytes::new(bytes), AnnKind::ScannBinary, 2).unwrap();
        assert_eq!(disk.header.quantizer_version, artifact.generation);
        assert_eq!(disk.header.codebook_version, artifact.artifact_id);
        assert_eq!(disk.header.vector_count, 3);
        use crate::directories::FileHandle;
        use crate::segment::vector_data::LazyFlatVectorData;
        let exact = LazyFlatVectorData::open_indirect(
            FileHandle::from_bytes(OwnedBytes::new(map)),
            &disk,
            2,
        )
        .await
        .unwrap();
        assert_eq!(
            exact.read_vectors_batch(0, 2).await.unwrap().as_slice(),
            &[0x12, 0x34, 0xab, 0xcd]
        );
        assert_eq!(exact.get_doc_id(1), (1, 2));
        let rows = crate::segment::row_map::RowMap::new(2, 1, |doc| doc == 1, 1024).unwrap();
        let mut locations = crate::segment::vector_locations::ExactLocations::default();
        let mut compacted = Vec::new();
        write_live_ann(
            &disk,
            &rows,
            &mut compacted,
            1024 * 1024,
            None,
            Some(&mut locations),
        )
        .unwrap();
        let mut map = Vec::new();
        locations
            .write(16, 1, compacted.len() as u64, &mut map, None)
            .unwrap();
        let compacted =
            AnnDiskIndex::open(OwnedBytes::new(compacted), AnnKind::ScannBinary, 1).unwrap();
        let live = LazyFlatVectorData::open_indirect(
            FileHandle::from_bytes(OwnedBytes::new(map)),
            &compacted,
            1,
        )
        .await
        .unwrap();
        assert_eq!(live.get_doc_id(0), (0, 2));
        assert_eq!(
            live.read_vectors_batch(0, 1).await.unwrap().as_slice(),
            &[0xab, 0xcd]
        );
        let artifact_bytes = artifact.to_bytes().unwrap();
        let view = ScannTrainedArtifactView::parse(&artifact_bytes).unwrap();
        disk.validate_scann_generation(&view.config, view.generation, view.artifact_id)
            .unwrap();
        let selective = crate::structures::SoarConfig::new().target_spill_fraction(0.5);
        disk.validate_scann_posting_count(2, Some(&selective))
            .unwrap();
        assert!(
            disk.validate_scann_posting_count(2, None)
                .unwrap_err()
                .to_string()
                .contains("physical postings")
        );
        let too_small_budget = crate::structures::SoarConfig::new().target_spill_fraction(0.49);
        assert!(
            disk.validate_scann_posting_count(2, Some(&too_small_budget))
                .is_err()
        );

        let serial = disk
            .search_binary_clusters_with_tuning::<false>(&[0x12, 0x34], 2, &[0, 1], usize::MAX)
            .unwrap();
        let parallel = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap()
            .install(|| {
                disk.search_binary_clusters_with_tuning::<false>(&[0x12, 0x34], 2, &[0, 1], 1)
            })
            .unwrap();
        assert_eq!(parallel, serial);
        assert_eq!(serial.len(), 2, "secondary posting duplicated a result");
        assert_eq!(serial[0], (0, 0, 1.0));

        let other = scann_binary_artifact(42);
        let other_bytes = other.to_bytes().unwrap();
        let other_view = ScannTrainedArtifactView::parse(&other_bytes).unwrap();
        let error = disk
            .validate_scann_generation(
                &other_view.config,
                other_view.generation,
                other_view.artifact_id,
            )
            .unwrap_err();
        assert!(error.to_string().contains("global trained generation"));
    }

    #[test]
    fn scann_ah_ann_round_trip_preserves_fastscan_tail_geometry() {
        use crate::structures::vector::scann::{
            ScannAhCodebook, ScannConfig, ScannEncoding, ScannLeafRun, ScannRoutingLevel,
            ScannSegmentPayload, ScannTrainedArtifact, ScannTrainedArtifactView,
        };

        let encoding = ScannEncoding::AsymmetricHash {
            dimensions_per_block: 2,
            bits_per_code: 4,
        };
        let artifact = ScannTrainedArtifact::new(
            51,
            100_000,
            ScannConfig {
                dimension: 8,
                tree_levels: 1,
                num_leaves: 2,
                encoding,
            },
            vec![ScannRoutingLevel {
                centroid_count: 2,
                centroid_codes: vec![0; 16],
                minimums: vec![0.0; 8],
                steps: vec![1.0; 8],
                child_offsets: Vec::new(),
            }],
            Some(ScannAhCodebook {
                dimensions_per_block: 2,
                centers_per_block: 16,
                centers: vec![0.0; 4 * 16 * 2],
            }),
        )
        .unwrap();
        let docs: Vec<u32> = (0..33).collect();
        let ordinals = vec![0u16; docs.len()];
        let codes = vec![0x5a; encoding.leaf_code_bytes(8, docs.len()).unwrap()];
        let run = ScannLeafRun::from_rows(0, 0, &docs, &ordinals, codes, encoding, 8).unwrap();
        let payload = ScannSegmentPayload::new(&artifact, 33, vec![run]).unwrap();
        let mut bytes = Vec::new();
        write_built_scann(&payload, &mut bytes, None).unwrap();

        let disk = AnnDiskIndex::open(OwnedBytes::new(bytes), AnnKind::ScannAh, 33).unwrap();
        assert_eq!(
            disk.runs[0].codes.len(),
            encoding.leaf_code_bytes(8, 33).unwrap()
        );
        let artifact_bytes = artifact.to_bytes().unwrap();
        let view = ScannTrainedArtifactView::parse(&artifact_bytes).unwrap();
        disk.validate_scann_generation(&view.config, view.generation, view.artifact_id)
            .unwrap();
    }

    #[test]
    fn scann_binary_merge_copies_codes_and_rebases_without_retraining() {
        use crate::structures::vector::scann::{ScannEncoding, ScannLeafRun, ScannSegmentPayload};

        let artifact = scann_binary_artifact(61);
        let make = |codes: Vec<u8>| {
            let secondary_code = codes[..2].to_vec();
            let primary = ScannLeafRun::from_rows(
                0,
                0,
                &[0, 1],
                &[0, 0],
                codes,
                ScannEncoding::BinaryHamming,
                16,
            )
            .unwrap();
            let secondary = ScannLeafRun::from_rows(
                1,
                0,
                &[0],
                &[0],
                secondary_code,
                ScannEncoding::BinaryHamming,
                16,
            )
            .unwrap();
            let payload = ScannSegmentPayload::new(&artifact, 2, vec![primary, secondary]).unwrap();
            let mut bytes = Vec::new();
            write_built_scann(&payload, &mut bytes, None).unwrap();
            AnnDiskIndex::open(OwnedBytes::new(bytes), AnnKind::ScannBinary, 2).unwrap()
        };
        let left = make(vec![1, 2, 3, 4]);
        let right = make(vec![5, 6, 7, 8]);
        let left_codes = left.raw.as_slice()[left.runs[0].codes.clone()].to_vec();
        let right_codes = right.raw.as_slice()[right.runs[0].codes.clone()].to_vec();

        let mut merged_bytes = Vec::new();
        write_merged_ann(&[(&left, 0), (&right, 2)], &mut merged_bytes).unwrap();
        let merged =
            AnnDiskIndex::open(OwnedBytes::new(merged_bytes), AnnKind::ScannBinary, 4).unwrap();
        assert_eq!(merged.header.vector_count, 6);
        merged
            .validate_scann_posting_count(
                4,
                Some(&crate::structures::SoarConfig::new().target_spill_fraction(0.5)),
            )
            .unwrap();
        assert_eq!(merged.runs[0].doc_base, 0);
        assert_eq!(merged.runs[1].doc_base, 2);
        assert_eq!(
            &merged.raw.as_slice()[merged.runs[0].codes.clone()],
            left_codes
        );
        assert_eq!(
            &merged.raw.as_slice()[merged.runs[1].codes.clone()],
            right_codes
        );
        assert_eq!(
            merged
                .search_binary_clusters::<false>(&[0, 0], 4, &[0, 1])
                .unwrap()
                .len(),
            4,
            "merge exposed primary and secondary postings as separate results",
        );

        let other_artifact = scann_binary_artifact(62);
        let other_run = ScannLeafRun::from_rows(
            0,
            0,
            &[0],
            &[0],
            vec![9, 10],
            ScannEncoding::BinaryHamming,
            16,
        )
        .unwrap();
        let other_payload = ScannSegmentPayload::new(&other_artifact, 1, vec![other_run]).unwrap();
        let mut other_bytes = Vec::new();
        write_built_scann(&other_payload, &mut other_bytes, None).unwrap();
        let other =
            AnnDiskIndex::open(OwnedBytes::new(other_bytes), AnnKind::ScannBinary, 1).unwrap();
        let error = write_merged_ann(&[(&left, 0), (&other, 2)], &mut Vec::new()).unwrap_err();
        assert!(error.to_string().contains("incompatible generations"));
    }
}

#[cfg(feature = "native")]
fn copy_selected_column(
    writer: &mut (impl Write + ?Sized),
    bytes: &[u8],
    width: usize,
    rows: impl Iterator<Item = usize>,
    check: &impl Fn() -> io::Result<()>,
) -> io::Result<()> {
    let mut live = rows.peekable();
    while let Some(first) = live.next() {
        check()?;
        let mut end = first + 1;
        while end - first < 4096 && live.peek() == Some(&end) {
            live.next();
            end += 1;
        }
        for chunk in bytes[first * width..end * width].chunks(4 * 1024 * 1024) {
            check()?;
            writer.write_all(chunk)?;
        }
    }
    Ok(())
}

/// Remove document assignments while preserving every encoded survivor and
/// its global artifacts. Run-local packed lanes are repacked without training.
#[cfg(feature = "native")]
pub(crate) fn write_live_ann(
    source: &AnnDiskIndex,
    rows: &super::row_map::RowMap,
    writer: &mut (impl Write + ?Sized),
    budget: usize,
    cancellation: Option<&std::sync::atomic::AtomicBool>,
    mut locations: Option<&mut crate::segment::vector_locations::ExactLocations>,
) -> io::Result<u64> {
    let check = || {
        if cancellation.is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire)) {
            Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "ANN row compaction cancelled",
            ))
        } else {
            Ok(())
        }
    };
    if source
        .runs
        .len()
        .saturating_mul(128)
        .saturating_add(source.header.dim.saturating_mul(128))
        > budget
    {
        return Err(io::Error::other(
            "ANN row compaction exceeds scratch budget",
        ));
    }
    let raw = source.raw.as_slice();
    let mut counts = Vec::with_capacity(source.runs.len());
    let mut header = source.header.clone();
    header.vector_count = 0;
    for run in source.runs.iter() {
        check()?;
        let mut count = 0usize;
        for i in 0..run.count {
            if i.is_multiple_of(4096) {
                check()?;
            }
            count += usize::from(
                rows.get(run_doc_id_with_base(raw, run, i, run.doc_base)?)
                    .is_some(),
            );
        }
        header.vector_count = header
            .vector_count
            .checked_add(count)
            .ok_or_else(|| invalid_data("ANN compacted assignment count overflow"))?;
        counts.push(count);
    }
    if header.vector_count == 0 {
        return Ok(0);
    }
    write_header(writer, &header)?;
    let mut offset = ANN_HEADER_SIZE as u64;
    let mut records = Vec::with_capacity(source.runs.len());
    for (run, count) in source.runs.iter().zip(counts) {
        check()?;
        if count == 0 {
            continue;
        }
        // Input labels were validated at open; this iterator cannot invent IDs.
        let kept = || {
            (0..run.count)
                .take_while(|row| !row.is_multiple_of(4096) || check().is_ok())
                .filter(|&row| {
                    rows.get(
                        run_doc_id_with_base(raw, run, row, run.doc_base)
                            .expect("validated ANN label"),
                    )
                    .is_some()
                })
        };
        let doc_ids_offset = offset;
        let mut max_doc_id = 0;
        for old in kept() {
            if old.is_multiple_of(4096) {
                check()?;
            }
            let new = rows
                .get(run_doc_id_with_base(raw, run, old, run.doc_base)?)
                .unwrap();
            writer.write_all(&new.to_le_bytes())?;
            max_doc_id = max_doc_id.max(new);
        }
        offset = checked_advance(offset, count * 4)?;
        let ordinals_offset = offset;
        copy_selected_column(writer, &raw[run.ordinals.clone()], 2, kept(), &check)?;
        offset = checked_advance(offset, count * 2)?;
        let codes_offset = offset;
        if let Some(map) = locations.as_deref_mut() {
            let span = map.span(codes_offset, count)?;
            for (new_row, old) in kept().enumerate() {
                let doc = rows
                    .get(run_doc_id_with_base(raw, run, old, run.doc_base)?)
                    .unwrap();
                map.push(
                    doc,
                    read_u16(raw, run.ordinals.start + old * 2),
                    (u64::from(span) << 32) | new_row as u64,
                )?;
            }
            check()?;
        }
        let codes = &raw[run.codes.clone()];
        if count == run.count {
            for chunk in codes.chunks(4 * 1024 * 1024) {
                check()?;
                writer.write_all(chunk)?;
            }
        } else {
            match header.kind {
                AnnKind::BinaryIvf | AnnKind::ScannBinary => {
                    copy_selected_column(writer, codes, header.code_size, kept(), &check)?
                }
                AnnKind::TqFlat | AnnKind::IvfTq => {
                    crate::structures::vector::quantization::tq_repack_rows(
                        codes,
                        header.code_size,
                        header.kind == AnnKind::IvfTq,
                        kept(),
                        writer,
                    )?;
                }
                AnnKind::ScannAh => {
                    let blocks = header.dim.div_ceil(header.code_size);
                    let lanes = crate::structures::vector::scann::FAST_SCAN_LANES;
                    let mut unpacked = Vec::with_capacity(lanes * blocks);
                    let mut packed = Vec::new();
                    let mut live = kept();
                    let mut selected = [0usize; crate::structures::vector::scann::FAST_SCAN_LANES];
                    loop {
                        check()?;
                        let mut count = 0;
                        for slot in &mut selected {
                            let Some(row) = live.next() else {
                                break;
                            };
                            *slot = row;
                            count += 1;
                        }
                        if count == 0 {
                            break;
                        }
                        if count == lanes
                            && selected[0].is_multiple_of(lanes)
                            && selected.windows(2).all(|pair| pair[1] == pair[0] + 1)
                        {
                            let width =
                                crate::structures::vector::scann::packed_block_bytes(blocks)
                                    .ok_or_else(|| {
                                        invalid_data("ScaNN AH block size overflows usize")
                                    })?;
                            let at = selected[0] / lanes * width;
                            writer.write_all(&codes[at..at + width])?;
                            continue;
                        }
                        unpacked.clear();
                        for &old in &selected[..count] {
                            unpack_scann_ah_row(codes, run.count, blocks, old, &mut unpacked)?;
                        }
                        if count == lanes {
                            packed.clear();
                            crate::structures::vector::scann::pack_fast_scan_block(
                                &unpacked,
                                blocks,
                                &mut packed,
                            )
                            .map_err(|error| invalid_data(error.to_string()))?;
                            writer.write_all(&packed)?;
                        } else {
                            for row in unpacked.chunks_exact(blocks) {
                                for pair in row.chunks(2) {
                                    writer.write_all(&[
                                        pair[0] | (pair.get(1).copied().unwrap_or(0) << 4)
                                    ])?;
                                }
                            }
                        }
                    }
                }
            }
        }
        let codes_len =
            expected_codes_column_len(header.kind, count, header.dim, header.code_size)?;
        offset = checked_advance(offset, codes_len)?;
        records.push(RunRecord {
            cluster_id: run.cluster_id,
            doc_base: 0,
            count: count as u32,
            max_doc_id,
            doc_ids_offset,
            ordinals_offset,
            codes_offset,
            codes_len: codes_len as u64,
        });
    }
    // The survivor iterator stops at cancellation, including inside repackers
    // that consume it without an error channel. Never finish a truncated run.
    check()?;
    finish_layout(writer, offset, &records)
}

#[cfg(all(test, feature = "native"))]
#[test]
fn deleting_ann_rows_preserves_all_code_formats_generations_and_ordinals() {
    use crate::structures::vector::quantization::{tq_pack_block, tq_pack_ivf_block};
    use crate::structures::vector::scann::pack_fast_scan_block;
    for kind in [
        AnnKind::BinaryIvf,
        AnnKind::ScannBinary,
        AnnKind::ScannAh,
        AnnKind::TqFlat,
        AnnKind::IvfTq,
    ] {
        let encode = |indexes: &[usize]| {
            let mut out = Vec::new();
            match kind {
                AnnKind::BinaryIvf | AnnKind::ScannBinary => {
                    out.extend(indexes.iter().map(|&i| (i * 7) as u8));
                }
                AnnKind::TqFlat | AnnKind::IvfTq => {
                    for chunk in indexes.chunks(16) {
                        let rows: Vec<Vec<u8>> = chunk
                            .iter()
                            .map(|&i| (0..8).map(|d| ((i * 3 + d * 5) & 15) as u8).collect())
                            .collect();
                        let refs: Vec<&[u8]> = rows.iter().map(Vec::as_slice).collect();
                        let gammas: Vec<f32> =
                            chunk.iter().map(|&i| 0.1 + i as f32 * 0.005).collect();
                        if kind == AnnKind::TqFlat {
                            tq_pack_block(&refs, &gammas, 8, &mut out);
                        } else {
                            let scales: Vec<f32> =
                                chunk.iter().map(|&i| 1.0 / (i + 1) as f32).collect();
                            tq_pack_ivf_block(&refs, &scales, &gammas, 8, &mut out);
                        }
                    }
                }
                AnnKind::ScannAh => {
                    for chunk in indexes.chunks(32) {
                        let rows: Vec<u8> = chunk
                            .iter()
                            .flat_map(|&i| (0..4).map(move |d| ((i * 3 + d * 5) & 15) as u8))
                            .collect();
                        if chunk.len() == 32 {
                            pack_fast_scan_block(&rows, 4, &mut out).unwrap();
                        } else {
                            out.extend(rows.chunks(2).map(|p| p[0] | (p[1] << 4)));
                        }
                    }
                }
            }
            out
        };
        let indexes: Vec<usize> = (0..65).collect();
        let docs: Vec<u32> = indexes.iter().map(|i| (i / 2) as u32).collect();
        let ords: Vec<u16> = indexes.iter().map(|i| (i % 2) as u16).collect();
        let codes = encode(&indexes);
        let fingerprint = crate::structures::vector::quantization::tq_expected_fingerprint(8);
        let header = AnnDiskHeader {
            kind,
            routing: IvfRoutingMode::Flat,
            dim: 8,
            code_size: match kind {
                AnnKind::BinaryIvf | AnnKind::ScannBinary => 1,
                AnnKind::ScannAh => 2,
                _ => 4,
            },
            num_clusters: 1,
            quantizer_version: match kind {
                AnnKind::TqFlat => fingerprint,
                AnnKind::IvfTq => crate::structures::mark_ivf_tq_cosine_generation(42),
                _ => 42,
            },
            codebook_version: match kind {
                AnnKind::BinaryIvf | AnnKind::TqFlat => 0,
                AnnKind::IvfTq => fingerprint,
                _ => 73,
            },
            vector_count: indexes.len(),
        };
        let mut source_bytes = Vec::new();
        write_built_runs(
            header.clone(),
            &[BuildRun {
                cluster_id: 0,
                doc_ids: &docs,
                ordinals: &ords,
                codes: &codes,
            }],
            &mut source_bytes,
            None,
        )
        .unwrap();
        let source = AnnDiskIndex::open(OwnedBytes::new(source_bytes), kind, 33).unwrap();
        for pattern in 0..5 {
            let rows = super::row_map::RowMap::new(
                33,
                33,
                |doc| match pattern {
                    0 => doc % 3 != 1,
                    1 => doc >= 16,
                    2 => !(8..16).contains(&doc),
                    3 => doc != 0,
                    _ => true,
                },
                4096,
            )
            .unwrap();
            let survivors: Vec<usize> = indexes
                .iter()
                .copied()
                .filter(|&i| rows.get(docs[i]).is_some())
                .collect();
            let kept_docs: Vec<u32> = survivors
                .iter()
                .map(|&i| rows.get(docs[i]).unwrap())
                .collect();
            let kept_ords: Vec<u16> = survivors.iter().map(|&i| ords[i]).collect();
            let kept_codes = encode(&survivors);
            let expected_header = AnnDiskHeader {
                vector_count: survivors.len(),
                ..header.clone()
            };
            let mut expected = Vec::new();
            write_built_runs(
                expected_header,
                &[BuildRun {
                    cluster_id: 0,
                    doc_ids: &kept_docs,
                    ordinals: &kept_ords,
                    codes: &kept_codes,
                }],
                &mut expected,
                None,
            )
            .unwrap();
            let mut actual = Vec::new();
            write_live_ann(&source, &rows, &mut actual, 1024 * 1024, None, None).unwrap();
            assert_eq!(
                actual, expected,
                "{kind:?}: compaction changed surviving codes or global identity"
            );
            AnnDiskIndex::open(OwnedBytes::new(actual), kind, rows.len()).unwrap();
            if kind == AnnKind::TqFlat {
                struct CancelDuringLabels<'a> {
                    flag: &'a std::sync::atomic::AtomicBool,
                    written: usize,
                }
                impl std::io::Write for CancelDuringLabels<'_> {
                    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                        self.written += bytes.len();
                        if self.written > ANN_HEADER_SIZE {
                            self.flag.store(true, std::sync::atomic::Ordering::Release);
                        }
                        Ok(bytes.len())
                    }
                    fn flush(&mut self) -> io::Result<()> {
                        Ok(())
                    }
                }
                let flag = std::sync::atomic::AtomicBool::new(false);
                let mut writer = CancelDuringLabels {
                    flag: &flag,
                    written: 0,
                };
                let error =
                    write_live_ann(&source, &rows, &mut writer, 1024 * 1024, Some(&flag), None)
                        .expect_err(
                            "cancelled ANN iterator reported a truncated output as success",
                        );
                assert_eq!(error.kind(), io::ErrorKind::Interrupted);
            }
        }
    }
}

#[cfg(all(test, feature = "native"))]
#[tokio::test]
async fn compacted_aligned_scann_groups_preserve_odd_block_padding() {
    let encode = |first: usize| {
        let mut bytes = Vec::new();
        for group in (first..96).step_by(32) {
            let rows: Vec<_> = (group..group + 32)
                .flat_map(|row| (0..3).map(move |block| ((row + block) % 16) as u8))
                .collect();
            crate::structures::vector::scann::pack_fast_scan_block(&rows, 3, &mut bytes).unwrap();
        }
        bytes
    };
    let header = AnnDiskHeader {
        kind: AnnKind::ScannAh,
        routing: IvfRoutingMode::Flat,
        dim: 9,
        code_size: 3,
        num_clusters: 1,
        quantizer_version: 42,
        codebook_version: 73,
        vector_count: 96,
    };
    let docs: Vec<_> = (0..96).collect();
    let ordinals = vec![0; 96];
    let codes = encode(0);
    let mut bytes = Vec::new();
    write_built_runs(
        header.clone(),
        &[BuildRun {
            cluster_id: 0,
            doc_ids: &docs,
            ordinals: &ordinals,
            codes: &codes,
        }],
        &mut bytes,
        None,
    )
    .unwrap();
    let source = AnnDiskIndex::open(OwnedBytes::new(bytes), AnnKind::ScannAh, 96).unwrap();
    let rows = super::row_map::RowMap::new(96, 64, |doc| doc >= 32, 4096).unwrap();
    let mut actual = Vec::new();
    write_live_ann(&source, &rows, &mut actual, 1024 * 1024, None, None).unwrap();
    let mut expected = Vec::new();
    write_built_runs(
        AnnDiskHeader {
            vector_count: 64,
            ..header
        },
        &[BuildRun {
            cluster_id: 0,
            doc_ids: &docs[..64],
            ordinals: &ordinals[..64],
            codes: &encode(32),
        }],
        &mut expected,
        None,
    )
    .unwrap();
    assert_eq!(actual, expected);
    AnnDiskIndex::open(OwnedBytes::new(actual), AnnKind::ScannAh, 64).unwrap();
}
