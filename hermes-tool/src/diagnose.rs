//! `hermes-tool diagnose` — active index diagnostics.
//!
//! The cheap report (no flags) reads only structures the searcher already
//! parses at open: run directories, TOCs, and metadata. It is safe against a
//! live index. Payload-reading checks are gated behind explicit flags, the
//! way Lucene gates CheckIndex's exhaustive mode behind `-slow` and
//! Elasticsearch gates its disk-usage API behind `run_expensive_tasks`.
//!
//! See `docs/diagnostics.md` for metric definitions and thresholds.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use hermes_core::MmapDirectory;
use hermes_core::segment::{AnnHealth, SegmentReader, VectorIndex};
use hermes_core::{Field, IndexConfig};
use serde::Serialize;

/// A whole-index diagnostic report; `--json` serializes exactly this.
#[derive(Serialize)]
pub struct Report {
    pub index: String,
    pub segments: Vec<SegmentReport>,
    pub fields: BTreeMap<String, FieldAggregate>,
}

#[derive(Serialize)]
pub struct SegmentReport {
    pub id: String,
    pub docs: u32,
    pub files: BTreeMap<String, u64>,
    pub text_fields: Vec<TextFieldReport>,
    pub term_dict: TermDictReport,
    pub sparse_fields: Vec<SparseFieldReport>,
    pub dense_fields: Vec<DenseFieldReport>,
    pub fast_fields: Vec<FastFieldColumnReport>,
    pub store: StoreReport,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub term_scan: Option<TermScanReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub residency: Option<BTreeMap<String, Residency>>,
}

#[derive(Serialize)]
pub struct TextFieldReport {
    pub field: String,
    pub docs_with_field: u32,
    pub avg_tokens_per_doc: f32,
}

/// One-pass term-dictionary scan (`--terms N`).
///
/// The doc-frequency distribution is the shape WAND/MaxScore effectiveness
/// depends on; the inline ratio shows how many rare terms skip a postings
/// read entirely; the top-1% share flags stopword bloat.
#[derive(Serialize)]
pub struct TermScanReport {
    pub terms: u64,
    pub inline_terms: u64,
    pub p50_doc_freq: u32,
    pub p99_doc_freq: u32,
    pub max_doc_freq: u32,
    /// Share of external postings bytes held by the top 1% of terms.
    pub top_1pct_postings_share: f64,
    pub postings_bytes: u64,
    pub positions_bytes: u64,
    pub top_terms: Vec<TopTerm>,
}

#[derive(Serialize)]
pub struct TopTerm {
    pub field: String,
    pub term: String,
    pub doc_freq: u32,
    pub posting_bytes: u64,
}

/// `--sparse-stats`: per-dimension posting distribution for a BMP field.
#[derive(Serialize)]
pub struct SparseDimReport {
    pub nonzero_dims: u32,
    pub declared_dims: u32,
    pub p50_postings_per_dim: u64,
    pub p99_postings_per_dim: u64,
    pub max_postings_per_dim: u64,
    pub top_1pct_share: f64,
    /// Postings clipped to the u8 impact ceiling — weight-quantization loss.
    pub saturated_impacts: u64,
    pub saturation_rate: f64,
    pub top_dims: Vec<(u32, u64)>,
}

#[derive(Serialize)]
pub struct FastFieldColumnReport {
    pub field: String,
    pub column_type: String,
    pub docs: u32,
    pub multi: bool,
    pub disk_bytes: u64,
}

#[derive(Serialize)]
pub struct StoreReport {
    pub bytes: u64,
    pub blocks: usize,
    pub avg_docs_per_block: f64,
    pub avg_block_bytes: f64,
    pub bytes_per_doc: f64,
}

#[derive(Serialize)]
pub struct TermDictReport {
    pub terms: u64,
    pub blocks: usize,
    pub bloom_bytes: usize,
    pub dictionary_bytes: usize,
}

#[derive(Serialize)]
pub struct SparseFieldReport {
    pub field: String,
    pub format: &'static str,
    pub vectors: u64,
    pub postings: u64,
    pub postings_per_vector: f64,
    /// BMP only: blocks and the virtual-doc padding the block grid carries.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blocks: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub padding_ratio: Option<f64>,
    /// `--sparse-stats`: per-dimension distribution (O(postings) scan).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dims_detail: Option<SparseDimReport>,
}

#[derive(Serialize)]
pub struct DenseFieldReport {
    pub field: String,
    pub kind: &'static str,
    pub flat_vectors: usize,
    pub flat_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ann: Option<AnnReport>,
    /// `--sample`: payload scan results.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample: Option<SampleReport>,
    /// `--probe-cost`: expected per-query I/O at the given nprobe.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub probe_cost: Option<ProbeCostReport>,
}

#[derive(Serialize)]
pub struct AnnReport {
    pub vectors: u64,
    pub clusters_nonempty: u32,
    pub clusters_total: u32,
    pub runs: u32,
    pub fragmentation: f64,
    pub imbalance: f64,
    pub largest_leaf_share: f64,
    pub payload_bytes: u64,
}

impl From<AnnHealth> for AnnReport {
    fn from(health: AnnHealth) -> Self {
        Self {
            vectors: health.vectors,
            clusters_nonempty: health.clusters_nonempty,
            clusters_total: health.clusters_total,
            runs: health.runs,
            fragmentation: health.fragmentation(),
            imbalance: health.imbalance,
            largest_leaf_share: health.largest_cluster_share(),
            payload_bytes: health.payload_bytes,
        }
    }
}

/// Degenerate-vector scan over a deterministic sample of flat storage.
///
/// This is the check that catches the constant-embedding failure modes: an
/// upstream producer emitting all-zero (or, from signbit-packed NaN,
/// all-ones) vectors that carry no signal and collapse into a single IVF
/// leaf.
#[derive(Serialize)]
pub struct SampleReport {
    pub sampled: usize,
    pub all_zero: usize,
    /// Binary codes with every bit set — the saturated twin of `all_zero`.
    pub all_ones: usize,
    /// Binary codes: fraction of bits set, averaged over the sample. Healthy
    /// sign-quantized embeddings sit near 0.5.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean_bit_fraction: Option<f64>,
    /// Float vectors: rows containing NaN or ±inf.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub non_finite: Option<usize>,
}

#[derive(Serialize)]
pub struct ProbeCostReport {
    pub nprobe: usize,
    /// Balanced-baseline bytes: nprobe × payload / clusters.
    pub expected_bytes: u64,
    /// Baseline × imbalance — what a random query pays in expectation.
    pub expected_bytes_imbalance_adjusted: u64,
    /// Physical extents touched ≈ seeks on a cold index.
    pub expected_extents: f64,
}

#[derive(Serialize, Clone, Copy)]
pub struct Residency {
    pub resident_bytes: u64,
    pub file_bytes: u64,
}

#[derive(Serialize, Default)]
pub struct FieldAggregate {
    pub vectors: u64,
    pub payload_bytes: u64,
    pub runs: u64,
    pub clusters_nonempty: u64,
    pub worst_leaf_share: f64,
    pub all_zero_sampled: usize,
    pub all_ones_sampled: usize,
    pub sampled: usize,
}

pub struct DiagnoseOptions {
    pub index: PathBuf,
    pub json: bool,
    pub sample: Option<usize>,
    pub probe_cost: Option<usize>,
    pub residency: bool,
    pub terms: Option<usize>,
    pub sparse_stats: bool,
}

pub async fn diagnose(options: DiagnoseOptions) -> Result<()> {
    let report = build_report(&options).await?;
    if options.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_human(&report);
    }
    Ok(())
}

/// Build the full report; separated from printing so tests can assert on it.
pub async fn build_report(options: &DiagnoseOptions) -> Result<Report> {
    // MmapDirectory, matching hermes-server: zero-copy reads, sync access
    // for the BMP loader, and the mapping `--residency` inspects.
    let dir = MmapDirectory::new(&options.index);
    let index = hermes_core::Index::open(dir, IndexConfig::default())
        .await
        .context("opening index")?;
    let schema = index.schema().clone();
    let segments = index.segment_readers().await?;

    let mut report = Report {
        index: schema.index_label().to_string(),
        segments: Vec::with_capacity(segments.len()),
        fields: BTreeMap::new(),
    };

    for segment in segments.iter() {
        let mut segment_report =
            diagnose_segment(options, &schema, segment, &mut report.fields).await?;
        if let Some(top) = options.terms {
            segment_report.term_scan = Some(scan_term_dict(segment, &schema, top).await?);
        }
        report.segments.push(segment_report);
    }
    Ok(report)
}

async fn diagnose_segment(
    options: &DiagnoseOptions,
    schema: &hermes_core::Schema,
    segment: &SegmentReader,
    aggregates: &mut BTreeMap<String, FieldAggregate>,
) -> Result<SegmentReport> {
    let meta = segment.meta();
    let field_name = |field_id: u32| -> String {
        schema
            .get_field_name(Field(field_id))
            .unwrap_or("?")
            .to_string()
    };

    // Per-file-kind sizes: the on-disk footprint the ES disk-usage API
    // popularized. Stat is authoritative and works for every file kind.
    let segment_files = hermes_core::segment::SegmentFiles::new(meta.id);
    let mut files = BTreeMap::new();
    let mut store_bytes = 0u64;
    for (kind, path) in [
        ("terms", &segment_files.term_dict),
        ("postings", &segment_files.postings),
        ("positions", &segment_files.positions),
        ("store", &segment_files.store),
        ("sparse", &segment_files.sparse),
        ("vectors", &segment_files.vectors),
        ("fast", &segment_files.fast),
    ] {
        let size = std::fs::metadata(options.index.join(path))
            .map(|meta| meta.len())
            .unwrap_or(0);
        if kind == "store" {
            store_bytes = size;
        }
        if size > 0 {
            files.insert(kind.to_string(), size);
        }
    }

    // Full-text: BM25F stats carried in segment metadata.
    let mut text_fields = Vec::new();
    for (&field_id, field_stats) in &meta.field_stats {
        text_fields.push(TextFieldReport {
            field: field_name(field_id),
            docs_with_field: field_stats.doc_count,
            avg_tokens_per_doc: field_stats.avg_field_len(),
        });
    }
    text_fields.sort_by(|a, b| a.field.cmp(&b.field));

    let dict_stats = segment.term_dict_stats();
    let term_dict = TermDictReport {
        terms: dict_stats.num_entries,
        blocks: dict_stats.num_blocks,
        bloom_bytes: dict_stats.bloom_filter_size,
        dictionary_bytes: dict_stats.dictionary_size,
    };

    // Sparse: MaxScore and BMP formats.
    let mut sparse_fields = Vec::new();
    for (&field_id, sparse) in segment.sparse_indexes() {
        let vectors = u64::from(sparse.total_vectors);
        let postings = sparse.total_postings();
        sparse_fields.push(SparseFieldReport {
            field: field_name(field_id),
            format: "maxscore",
            vectors,
            postings,
            postings_per_vector: ratio(postings, vectors),
            blocks: None,
            padding_ratio: None,
            dims_detail: None,
        });
    }
    for (&field_id, bmp) in segment.bmp_indexes() {
        let vectors = u64::from(bmp.total_vectors);
        let postings = bmp.total_postings();
        let dims_detail = options.sparse_stats.then(|| {
            let stats = bmp.dim_stats(10);
            SparseDimReport {
                nonzero_dims: stats.nonzero_dims,
                declared_dims: stats.declared_dims,
                p50_postings_per_dim: stats.p50_postings_per_dim,
                p99_postings_per_dim: stats.p99_postings_per_dim,
                max_postings_per_dim: stats.max_postings_per_dim,
                top_1pct_share: stats.top_1pct_share,
                saturated_impacts: stats.saturated_impacts,
                saturation_rate: if stats.total_postings == 0 {
                    0.0
                } else {
                    stats.saturated_impacts as f64 / stats.total_postings as f64
                },
                top_dims: stats.top_dims,
            }
        });
        // Virtual docs are padded to whole blocks; padding is pure grid
        // overhead scanned by every query.
        let padding = if bmp.num_virtual_docs == 0 {
            0.0
        } else {
            1.0 - (vectors as f64 / f64::from(bmp.num_virtual_docs))
        };
        sparse_fields.push(SparseFieldReport {
            field: field_name(field_id),
            format: "bmp",
            vectors,
            postings,
            postings_per_vector: ratio(postings, vectors),
            blocks: Some(bmp.num_blocks),
            padding_ratio: Some(padding),
            dims_detail,
        });
    }
    sparse_fields.sort_by(|a, b| a.field.cmp(&b.field));

    // Dense: flat storage plus ANN health, with opt-in payload scans.
    let mut dense_fields = Vec::new();
    for (&field_id, flat) in segment.flat_vectors() {
        let name = field_name(field_id);
        let kind = match segment.vector_indexes().get(&field_id) {
            Some(VectorIndex::BinaryIvf(_)) => "binary_ivf",
            Some(VectorIndex::Tq { .. }) => "tq",
            Some(VectorIndex::IvfTq { .. }) => "ivf_tq",
            None => "flat",
        };
        let ann = segment.ann_health(Field(field_id));
        let sample = match options.sample {
            Some(count) => Some(sample_flat_vectors(flat, count).await?),
            None => None,
        };
        let probe_cost = options.probe_cost.and_then(|nprobe| {
            ann.map(|health| {
                let clusters = u64::from(health.clusters_nonempty.max(1));
                let baseline = nprobe as u64 * health.payload_bytes / clusters;
                ProbeCostReport {
                    nprobe,
                    expected_bytes: baseline,
                    expected_bytes_imbalance_adjusted: (baseline as f64 * health.imbalance) as u64,
                    expected_extents: nprobe as f64 * health.fragmentation(),
                }
            })
        });

        let aggregate = aggregates.entry(name.clone()).or_default();
        if let Some(health) = ann {
            aggregate.vectors += health.vectors;
            aggregate.payload_bytes += health.payload_bytes;
            aggregate.runs += u64::from(health.runs);
            aggregate.clusters_nonempty += u64::from(health.clusters_nonempty);
            aggregate.worst_leaf_share = aggregate
                .worst_leaf_share
                .max(health.largest_cluster_share());
        }
        if let Some(sample_report) = &sample {
            aggregate.sampled += sample_report.sampled;
            aggregate.all_zero_sampled += sample_report.all_zero;
            aggregate.all_ones_sampled += sample_report.all_ones;
        }

        dense_fields.push(DenseFieldReport {
            field: name,
            kind,
            flat_vectors: flat.num_vectors,
            flat_bytes: (flat.num_vectors * flat.vector_byte_size()) as u64,
            ann: ann.map(AnnReport::from),
            sample,
            probe_cost,
        });
    }
    dense_fields.sort_by(|a, b| a.field.cmp(&b.field));

    let residency = if options.residency {
        Some(measure_residency(&options.index, meta.id))
    } else {
        None
    };

    let mut fast_fields = Vec::new();
    for (&field_id, column) in segment.fast_fields() {
        fast_fields.push(FastFieldColumnReport {
            field: field_name(field_id),
            column_type: format!("{:?}", column.column_type),
            docs: column.num_docs,
            multi: column.multi,
            disk_bytes: column.disk_bytes(),
        });
    }
    fast_fields.sort_by(|a, b| a.field.cmp(&b.field));

    // Store shape from the block index alone (no decompression): bytes/doc is
    // the retrieval cost driver, block sizes show whether the configured block
    // budget is actually being filled.
    let store_blocks = segment.store_raw_blocks();
    let store_docs: u64 = store_blocks
        .iter()
        .map(|block| u64::from(block.num_docs))
        .sum();
    let store_compressed: u64 = store_blocks
        .iter()
        .map(|block| u64::from(block.length))
        .sum();
    let store = StoreReport {
        bytes: store_bytes,
        blocks: store_blocks.len(),
        avg_docs_per_block: ratio(store_docs, store_blocks.len() as u64),
        avg_block_bytes: ratio(store_compressed, store_blocks.len() as u64),
        bytes_per_doc: ratio(store_compressed, store_docs),
    };

    Ok(SegmentReport {
        id: format!("{:032x}", meta.id),
        docs: meta.num_docs,
        files,
        text_fields,
        term_dict,
        sparse_fields,
        dense_fields,
        fast_fields,
        store,
        term_scan: None,
        residency,
    })
}

fn ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

/// Read `count` evenly spaced vectors and look for degenerate content.
async fn sample_flat_vectors(
    flat: &hermes_core::segment::LazyFlatVectorData,
    count: usize,
) -> Result<SampleReport> {
    use hermes_core::dsl::DenseVectorQuantization;

    let total = flat.num_vectors;
    let take = count.min(total);
    let byte_size = flat.vector_byte_size();
    let is_binary = matches!(flat.quantization, DenseVectorQuantization::Binary);

    let mut all_zero = 0usize;
    let mut all_ones = 0usize;
    let mut bits_set = 0u64;
    let mut non_finite = 0usize;
    let mut raw = vec![0u8; byte_size];
    let mut floats = vec![0f32; flat.dim];
    let mut state = 0x9e37_79b9_7f4a_7c15u64 ^ total as u64;
    for _position in 0..take {
        // Deterministic splitmix64 positions rather than an even stride: flat
        // order is a BP permutation of ingestion order, and a fixed stride
        // can alias with that structure. (The first version of this scan
        // missed 100/401 zero vectors exactly that way.)
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut mixed = state;
        mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        mixed ^= mixed >> 31;
        let index = (mixed % total.max(1) as u64) as usize;
        flat.read_vector_raw_into(index, &mut raw).await?;
        if raw.iter().all(|&byte| byte == 0) {
            all_zero += 1;
        }
        // Saturated codes only mean something for binary quantization; a
        // float row of 0xff bytes is NaN garbage the non-finite check owns.
        if is_binary && raw.iter().all(|&byte| byte == 0xff) {
            all_ones += 1;
        }
        if is_binary {
            bits_set += raw
                .iter()
                .map(|byte| u64::from(byte.count_ones() as u8))
                .sum::<u64>();
        } else {
            hermes_core::segment::dequantize_raw(&raw, flat.quantization, flat.dim, &mut floats)?;
            if floats.iter().any(|value| !value.is_finite()) {
                non_finite += 1;
            }
        }
    }
    Ok(SampleReport {
        sampled: take,
        all_zero,
        all_ones,
        mean_bit_fraction: is_binary.then(|| {
            if take == 0 {
                0.0
            } else {
                bits_set as f64 / (take as f64 * byte_size as f64 * 8.0)
            }
        }),
        non_finite: (!is_binary).then_some(non_finite),
    })
}

/// Page-cache residency per segment file via `mincore(2)`.
///
/// Answers "is this index actually in RAM" — the difference between a 15 ms
/// and a 1 s dense query on the same hardware — without external tooling.
#[cfg(unix)]
fn measure_residency(
    index_path: &std::path::Path,
    segment_id: u128,
) -> BTreeMap<String, Residency> {
    use std::os::unix::io::AsRawFd;

    let mut out = BTreeMap::new();
    let files = hermes_core::segment::SegmentFiles::new(segment_id);
    for (kind, path) in [
        ("terms", &files.term_dict),
        ("postings", &files.postings),
        ("positions", &files.positions),
        ("store", &files.store),
        ("sparse", &files.sparse),
        ("vectors", &files.vectors),
        ("fast", &files.fast),
    ] {
        let full = index_path.join(path);
        let Ok(file) = std::fs::File::open(&full) else {
            continue;
        };
        let Ok(meta) = file.metadata() else { continue };
        let len = meta.len() as usize;
        if len == 0 {
            continue;
        }
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        let pages = len.div_ceil(page);
        let mapped = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if mapped == libc::MAP_FAILED {
            continue;
        }
        let mut residency_vec = vec![0u8; pages];
        let resident_pages = unsafe {
            #[cfg(target_os = "linux")]
            let vec_ptr = residency_vec.as_mut_ptr();
            #[cfg(not(target_os = "linux"))]
            let vec_ptr = residency_vec.as_mut_ptr() as *mut libc::c_char;
            if libc::mincore(mapped, len, vec_ptr) == 0 {
                residency_vec.iter().filter(|&&page| page & 1 == 1).count()
            } else {
                0
            }
        };
        unsafe { libc::munmap(mapped, len) };
        out.insert(
            kind.to_string(),
            Residency {
                resident_bytes: (resident_pages * page).min(len) as u64,
                file_bytes: len as u64,
            },
        );
    }
    out
}

#[cfg(not(unix))]
fn measure_residency(_: &std::path::Path, _: u128) -> BTreeMap<String, Residency> {
    eprintln!("--residency requires mincore(2); unsupported on this platform");
    BTreeMap::new()
}

/// One-pass term-dictionary scan: doc-frequency distribution, inline-term
/// ratio, postings/positions footprint, and the top-N terms.
///
/// O(total terms) — an explicit opt-in. Stopword bloat and pathological
/// tokenization both show up here.
async fn scan_term_dict(
    segment: &SegmentReader,
    schema: &hermes_core::Schema,
    top: usize,
) -> Result<TermScanReport> {
    use hermes_core::structures::TermInfo;

    // Dictionary keys are `field_id (4 bytes LE) ++ term bytes`.
    let decode_key = |key: &[u8]| -> (String, String) {
        if key.len() < 4 {
            return (String::new(), String::from_utf8_lossy(key).into_owned());
        }
        let field_id = u32::from_le_bytes([key[0], key[1], key[2], key[3]]);
        let field = schema
            .get_field_name(Field(field_id))
            .unwrap_or("?")
            .to_string();
        (field, String::from_utf8_lossy(&key[4..]).into_owned())
    };

    let mut doc_freqs: Vec<u32> = Vec::new();
    let mut inline_terms = 0u64;
    let mut postings_bytes = 0u64;
    let mut positions_bytes = 0u64;
    let mut heap: std::collections::BinaryHeap<std::cmp::Reverse<(u32, u64, String, String)>> =
        std::collections::BinaryHeap::with_capacity(top + 1);
    let mut posting_sizes: Vec<u64> = Vec::new();

    let mut iter = segment.term_dict_iter();
    while let Some((key, info)) = iter.next().await? {
        let (doc_freq, posting_len) = match &info {
            TermInfo::Inline { doc_freq, .. } => {
                inline_terms += 1;
                (u32::from(*doc_freq), 0u64)
            }
            TermInfo::External {
                doc_freq,
                posting_len,
                position_len,
                ..
            } => {
                postings_bytes += posting_len;
                positions_bytes += position_len;
                posting_sizes.push(*posting_len);
                (*doc_freq, *posting_len)
            }
        };
        doc_freqs.push(doc_freq);
        let (field, term) = decode_key(&key);
        heap.push(std::cmp::Reverse((doc_freq, posting_len, field, term)));
        if heap.len() > top {
            heap.pop();
        }
    }

    doc_freqs.sort_unstable();
    let percentile = |fraction: f64| -> u32 {
        if doc_freqs.is_empty() {
            0
        } else {
            doc_freqs[((doc_freqs.len() - 1) as f64 * fraction) as usize]
        }
    };
    posting_sizes.sort_unstable();
    let hot = posting_sizes.len().div_ceil(100);
    let top_1pct_bytes: u64 = posting_sizes.iter().rev().take(hot).sum();

    let mut top_terms: Vec<_> = heap
        .into_iter()
        .map(
            |std::cmp::Reverse((doc_freq, posting_bytes, field, term))| TopTerm {
                field,
                term,
                doc_freq,
                posting_bytes,
            },
        )
        .collect();
    top_terms.sort_by_key(|term| std::cmp::Reverse(term.doc_freq));

    Ok(TermScanReport {
        terms: doc_freqs.len() as u64,
        inline_terms,
        p50_doc_freq: percentile(0.50),
        p99_doc_freq: percentile(0.99),
        max_doc_freq: doc_freqs.last().copied().unwrap_or(0),
        top_1pct_postings_share: if postings_bytes == 0 {
            0.0
        } else {
            top_1pct_bytes as f64 / postings_bytes as f64
        },
        postings_bytes,
        positions_bytes,
        top_terms,
    })
}

fn print_human(report: &Report) {
    println!("index: {}", report.index);
    println!("segments: {}", report.segments.len());
    for segment in &report.segments {
        println!("\nsegment {} ({} docs)", segment.id, segment.docs);
        println!(
            "  term dict: {} terms, {} blocks, bloom {} B, dict {} B",
            segment.term_dict.terms,
            segment.term_dict.blocks,
            segment.term_dict.bloom_bytes,
            segment.term_dict.dictionary_bytes,
        );
        for text in &segment.text_fields {
            println!(
                "  text {:32} docs={} avg_tokens={:.1}",
                text.field, text.docs_with_field, text.avg_tokens_per_doc
            );
        }
        for sparse in &segment.sparse_fields {
            print!(
                "  sparse {:30} [{}] vectors={} postings={} ({:.1}/vec)",
                sparse.field,
                sparse.format,
                sparse.vectors,
                sparse.postings,
                sparse.postings_per_vector,
            );
            if let (Some(blocks), Some(padding)) = (sparse.blocks, sparse.padding_ratio) {
                print!(" blocks={blocks} padding={:.1}%", 100.0 * padding);
            }
            println!();
            if let Some(dims) = &sparse.dims_detail {
                println!(
                    "        dims: {}/{} nonzero, postings/dim p50={} p99={} max={}, \
                     top-1% share {:.1}%, impact saturation {:.2}%",
                    dims.nonzero_dims,
                    dims.declared_dims,
                    dims.p50_postings_per_dim,
                    dims.p99_postings_per_dim,
                    dims.max_postings_per_dim,
                    100.0 * dims.top_1pct_share,
                    100.0 * dims.saturation_rate,
                );
                for (dim, count) in &dims.top_dims {
                    println!("          dim {dim:<8} postings={count}");
                }
            }
        }
        for dense in &segment.dense_fields {
            print!(
                "  dense {:31} [{}] flat={} ({} B)",
                dense.field, dense.kind, dense.flat_vectors, dense.flat_bytes
            );
            if let Some(ann) = &dense.ann {
                print!(
                    " | ann: clusters={}/{} frag={:.2} imbalance={:.2} worst_leaf={:.2}%",
                    ann.clusters_nonempty,
                    ann.clusters_total,
                    ann.fragmentation,
                    ann.imbalance,
                    100.0 * ann.largest_leaf_share,
                );
            }
            println!();
            if let Some(sample) = &dense.sample {
                print!(
                    "        sample: {} vectors, {} all-zero, {} all-ones",
                    sample.sampled, sample.all_zero, sample.all_ones
                );
                if let Some(fraction) = sample.mean_bit_fraction {
                    print!(", mean bit fraction {fraction:.3}");
                }
                if let Some(non_finite) = sample.non_finite {
                    print!(", {non_finite} non-finite");
                }
                println!();
            }
            if let Some(cost) = &dense.probe_cost {
                println!(
                    "        probe@{}: {} B balanced, {} B expected, {:.0} extents",
                    cost.nprobe,
                    cost.expected_bytes,
                    cost.expected_bytes_imbalance_adjusted,
                    cost.expected_extents,
                );
            }
        }
        for column in &segment.fast_fields {
            println!(
                "  fast {:32} [{}]{} docs={} bytes={}",
                column.field,
                column.column_type,
                if column.multi { " multi" } else { "" },
                column.docs,
                column.disk_bytes,
            );
        }
        println!(
            "  store: {} B in {} blocks ({:.1} docs/block, {:.0} B/block, {:.1} B/doc)",
            segment.store.bytes,
            segment.store.blocks,
            segment.store.avg_docs_per_block,
            segment.store.avg_block_bytes,
            segment.store.bytes_per_doc,
        );
        if let Some(scan) = &segment.term_scan {
            println!(
                "  terms: {} total ({} inline), doc_freq p50={} p99={} max={}, \
                 postings={} B positions={} B, top-1% share {:.1}%",
                scan.terms,
                scan.inline_terms,
                scan.p50_doc_freq,
                scan.p99_doc_freq,
                scan.max_doc_freq,
                scan.postings_bytes,
                scan.positions_bytes,
                100.0 * scan.top_1pct_postings_share,
            );
            for term in &scan.top_terms {
                println!(
                    "    {}:{:38} doc_freq={:<10} postings={} B",
                    term.field, term.term, term.doc_freq, term.posting_bytes
                );
            }
        }
        if let Some(residency) = &segment.residency {
            for (kind, res) in residency {
                println!(
                    "  residency {:10} {:5.1}% of {} B",
                    kind,
                    100.0 * res.resident_bytes as f64 / res.file_bytes.max(1) as f64,
                    res.file_bytes,
                );
            }
        }
    }
    if !report.fields.is_empty() {
        println!("\nper-field aggregate (dense):");
        for (field, aggregate) in &report.fields {
            print!(
                "  {:32} vectors={} payload={} B frag={:.2} worst_leaf={:.2}%",
                field,
                aggregate.vectors,
                aggregate.payload_bytes,
                ratio(aggregate.runs, aggregate.clusters_nonempty),
                100.0 * aggregate.worst_leaf_share,
            );
            if aggregate.sampled > 0 {
                print!(
                    " sampled={} all_zero={} ({:.1}%) all_ones={} ({:.1}%)",
                    aggregate.sampled,
                    aggregate.all_zero_sampled,
                    100.0 * aggregate.all_zero_sampled as f64 / aggregate.sampled as f64,
                    aggregate.all_ones_sampled,
                    100.0 * aggregate.all_ones_sampled as f64 / aggregate.sampled as f64,
                );
            }
            println!();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hermes_core::{Document, IndexWriter, SchemaBuilder};

    /// End-to-end reproduction of the production incident shape: a binary
    /// field where 25% of vectors are all-zero. The cheap tier must show the
    /// resulting leaf skew and `--sample`/`--probe-cost` must quantify it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn diagnose_detects_zero_vector_collapse() {
        let dim_bits = 64;
        let byte_len = dim_bits / 8;
        let tmp = tempfile::tempdir().unwrap();

        let mut sb = SchemaBuilder::default();
        sb.set_index_name("diag_test");
        let title = sb.add_text_field("title", true, true);
        let cfg = hermes_core::BinaryDenseVectorConfig::new(dim_bits).with_ivf(Some(8), 4);
        let emb = sb.add_binary_dense_vector_field_with_config("emb", true, true, cfg);
        let views = sb.add_u64_field("views", true, true);
        sb.set_fast(views, true);
        let sparse = sb.add_sparse_vector_field_with_config(
            "sparse_emb",
            true,
            false,
            hermes_core::structures::SparseVectorConfig::splade_bmp(),
        );
        let schema = sb.build();

        let dir = MmapDirectory::new(tmp.path());
        let mut writer = IndexWriter::create(dir, schema, IndexConfig::default())
            .await
            .unwrap();
        let mut state = 0x9e3779b97f4a7c15u64;
        for i in 0..400u32 {
            let mut doc = Document::new();
            doc.add_text(title, format!("document {i} about hemoglobin"));
            let code: Vec<u8> = if i % 4 == 0 {
                vec![0u8; byte_len]
            } else if i % 8 == 1 {
                // The saturated face: signbit-packed NaN from the producer.
                vec![0xffu8; byte_len]
            } else {
                (0..byte_len)
                    .map(|_| {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        state as u8
                    })
                    .collect()
            };
            doc.add_binary_dense_vector(emb, code);
            doc.add_u64(views, u64::from(i));
            // Zipf-ish sparse vector: dim 0 in every doc (a "stopword"
            // dimension), a few mid dims, one unique dim, with a large
            // weight to exercise impact saturation.
            doc.add_sparse_vector(sparse, vec![(0, 9.5), (1 + (i % 7), 1.0), (100 + i, 0.4)]);
            writer.add_document(doc).unwrap();
        }
        writer.commit().await.unwrap();
        writer.build_vector_index().await.unwrap();
        // A second commit publishes segments rebuilt against the trained
        // quantizer generation.
        let mut trigger = Document::new();
        trigger.add_text(title, "trigger");
        trigger.add_binary_dense_vector(emb, vec![0xffu8; byte_len]);
        writer.add_document(trigger).unwrap();
        writer.commit().await.unwrap();
        writer.force_merge().await.unwrap();
        drop(writer);

        let options = DiagnoseOptions {
            index: tmp.path().to_path_buf(),
            json: false,
            sample: Some(200),
            probe_cost: Some(4),
            residency: false,
            terms: Some(5),
            sparse_stats: true,
        };
        let report = build_report(&options).await.unwrap();

        assert_eq!(report.index, "diag_test");
        assert!(!report.segments.is_empty());
        let dense: Vec<_> = report
            .segments
            .iter()
            .flat_map(|segment| segment.dense_fields.iter())
            .filter(|field| field.field == "emb")
            .collect();
        assert!(!dense.is_empty(), "emb field must appear in the report");

        let with_ann: Vec<_> = dense.iter().filter(|f| f.ann.is_some()).collect();
        assert!(
            !with_ann.is_empty(),
            "binary IVF payload expected after build_vector_index"
        );
        for field in &with_ann {
            let ann = field.ann.as_ref().unwrap();
            // 100 of ~400 vectors share the zero leaf.
            assert!(
                ann.largest_leaf_share > 0.15,
                "zero-vector collapse must show as leaf skew, got {}",
                ann.largest_leaf_share
            );
            assert!(ann.imbalance > 1.0);
            let cost = field.probe_cost.as_ref().expect("probe cost requested");
            assert_eq!(cost.nprobe, 4);
            assert!(cost.expected_bytes > 0);
            assert!(cost.expected_extents >= 4.0);
        }

        let sampled: usize = dense
            .iter()
            .filter_map(|f| f.sample.as_ref())
            .map(|s| s.sampled)
            .sum();
        let zeros: usize = dense
            .iter()
            .filter_map(|f| f.sample.as_ref())
            .map(|s| s.all_zero)
            .sum();
        assert!(sampled > 0);
        let zero_rate = zeros as f64 / sampled as f64;
        assert!(
            (0.15..=0.35).contains(&zero_rate),
            "sample must recover the ~25% zero rate, got {zero_rate:.2}"
        );
        let ones: usize = dense
            .iter()
            .filter_map(|f| f.sample.as_ref())
            .map(|s| s.all_ones)
            .sum();
        let ones_rate = ones as f64 / sampled as f64;
        assert!(
            (0.06..=0.25).contains(&ones_rate),
            "sample must recover the ~12.5% all-ones rate, got {ones_rate:.2}"
        );

        // Cheap tier covers the other structures too.
        let segment = &report.segments[0];
        assert!(segment.term_dict.terms > 0);
        assert!(
            segment.text_fields.iter().any(|t| t.field == "title"),
            "text stats missing"
        );
        assert!(segment.files.contains_key("vectors"));

        // Store shape from the block index.
        assert!(segment.store.bytes > 0);
        assert!(segment.store.blocks > 0);
        assert!(segment.store.bytes_per_doc > 0.0);

        // Fast-field column report.
        let views_column = segment
            .fast_fields
            .iter()
            .find(|column| column.field == "views")
            .expect("views fast column");
        assert_eq!(views_column.column_type, "U64");
        assert!(views_column.disk_bytes > 0);

        // Term-dict scan: 400 docs share "hemoglobin", so max doc_freq is the
        // corpus size and the top terms include it.
        let scan = segment.term_scan.as_ref().expect("term scan requested");
        assert!(scan.terms > 0);
        assert!(scan.max_doc_freq >= 400);
        assert!(
            scan.top_terms.iter().any(|term| term.term == "hemoglobin"),
            "top terms: {:?}",
            scan.top_terms.iter().map(|t| &t.term).collect::<Vec<_>>()
        );
        assert!(scan.p50_doc_freq <= scan.p99_doc_freq);
        assert!(scan.inline_terms > 0, "unique doc-number terms inline");

        // Sparse dim scan: dim 0 is in every vector (the hot dimension) and
        // its 9.5 weight saturates u8 quantization.
        let sparse_report = segment
            .sparse_fields
            .iter()
            .find(|field| field.field == "sparse_emb")
            .expect("sparse field in report");
        assert_eq!(sparse_report.format, "bmp");
        assert!(sparse_report.vectors > 0);
        let dims = sparse_report
            .dims_detail
            .as_ref()
            .expect("dim stats requested");
        assert!(dims.nonzero_dims > 100, "unique dims per doc");
        assert_eq!(
            dims.top_dims.first().map(|&(dim, _)| dim),
            Some(0),
            "dim 0 must be the hottest"
        );
        assert!(dims.max_postings_per_dim >= 400);
        assert!(
            dims.saturated_impacts > 0,
            "9.5 weight must clip the u8 impact range"
        );
    }
}
