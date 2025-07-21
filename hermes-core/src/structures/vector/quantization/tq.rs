//! TurboQuant (TQ): training-free dense-vector codec.
//!
//! Design: `docs/turboquant-quantization.md`. Per padded coordinate the codec
//! stores a 3-bit scalar code (analytic Lloyd-Max levels for the unit-sphere
//! coordinate density) plus a 1-bit QJL sign of the rotated stage-1 residual;
//! a per-vector f32 `gamma = ‖residual‖₂` makes the inner-product estimator
//! unbiased. Everything is derived from `(dim, codec constants)` — no trained
//! artifacts and no cross-segment generations.
//!
//! References: Zandieh, Daliri et al., "TurboQuant: Online Vector
//! Quantization with Near-optimal Distortion Rate" (arXiv 2504.19874);
//! layout and LUT16 scoring follow the FastScan pattern (Faiss,
//! mayflower/pg_turboquant, both MIT).

/// Bumping this refuses to mix payloads across incompatible codec revisions.
/// v2: padding-free 3-round rotation (sub-FWHT + signs + permutation per
/// round) replaced the single-round power-of-two-padded FWHT — 768-dim codes
/// shrank 33% and the codebook density now uses the true dimension.
pub const TQ_CODEC_VERSION: u32 = 2;
/// Bits per padded coordinate: 3-bit stage-1 code + 1-bit QJL sign.
pub const TQ_BITS: u32 = 4;
/// Vectors per scoring block; one lane per vector.
pub const TQ_BLOCK_LANES: usize = 16;
/// Smallest supported padded dimension. Below this the coordinate density
/// exponent `(P-3)/2` degenerates and LUT rows would not fill a SIMD lane.
pub const TQ_MIN_PADDED_DIM: usize = 8;

const TQ_STAGE1_LEVELS: usize = 8;
const TQ_STAGE1_SEED: u64 = 0x7154_5354_4147_4531; // "qTSTAGE1"
const TQ_QJL_SEED: u64 = 0x7154_514a_4c53_4b31; // "qTQJLSK1"
const TQ_LLOYD_GRID: usize = 8192;
const TQ_LLOYD_MAX_ITERATIONS: usize = 64;
const TQ_LLOYD_TOLERANCE: f64 = 1e-9;
/// i16 lane accumulators are widened to i32 at least every this many
/// dimensions: 128 * 127 = 16256 stays far from i16 saturation.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
const TQ_ACCUMULATE_CHUNK_DIMS: usize = 128;

/// Code-layout dimension: the input dimension rounded up to an even count
/// (two 4-bit coordinates per byte), floored at [`TQ_MIN_PADDED_DIM`]. Since
/// codec v2 the rotation is padding-free, so this tracks the true dimension
/// instead of the next power of two. Cheap; usable for header validation
/// without building a codec.
#[inline]
pub fn tq_padded_dim(dim: usize) -> usize {
    dim.next_multiple_of(2).max(TQ_MIN_PADDED_DIM)
}

/// Fingerprint every payload built for `dim` must carry (no codebook build).
#[inline]
pub fn tq_expected_fingerprint(dim: usize) -> u64 {
    tq_fingerprint(dim, tq_padded_dim(dim))
}

/// Process-wide codec cache. A codec is a pure function of the dimension and
/// costs a Lloyd solve to build; segment opens and merges share one instance
/// per dimension instead of re-deriving it.
pub fn tq_shared_codec(dim: usize) -> std::sync::Arc<TqCodec> {
    static CODECS: std::sync::OnceLock<
        std::sync::Mutex<rustc_hash::FxHashMap<usize, std::sync::Arc<TqCodec>>>,
    > = std::sync::OnceLock::new();
    let cache = CODECS.get_or_init(Default::default);
    let mut guard = cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    std::sync::Arc::clone(
        guard
            .entry(dim)
            .or_insert_with(|| std::sync::Arc::new(TqCodec::new(dim))),
    )
}

/// Bytes of one scoring block: 16 f32 gammas + 16 packed nibble rows.
#[inline]
pub const fn tq_block_bytes(code_size: usize) -> usize {
    TQ_BLOCK_LANES * (size_of::<f32>() + code_size)
}

/// Total codes-column bytes for `count` vectors (final block zero-padded).
#[inline]
pub const fn tq_codes_column_len(count: usize, code_size: usize) -> usize {
    count.div_ceil(TQ_BLOCK_LANES) * tq_block_bytes(code_size)
}

/// Overflow-checked [`tq_codes_column_len`] for untrusted header values.
#[inline]
pub fn tq_codes_column_len_checked(count: usize, code_size: usize) -> Option<usize> {
    count
        .div_ceil(TQ_BLOCK_LANES)
        .checked_mul(tq_block_bytes(code_size))
}

/// Bytes of one IVF-TQ scoring block: 16 f32 residual scales + 16 f32 gammas
/// + 16 packed nibble rows.
#[inline]
pub const fn tq_ivf_block_bytes(code_size: usize) -> usize {
    TQ_BLOCK_LANES * (2 * size_of::<f32>() + code_size)
}

/// Overflow-checked IVF-TQ codes-column length for untrusted header values.
#[inline]
pub fn tq_ivf_codes_column_len_checked(count: usize, code_size: usize) -> Option<usize> {
    count
        .div_ceil(TQ_BLOCK_LANES)
        .checked_mul(tq_ivf_block_bytes(code_size))
}

#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Number of sign/sub-FWHT/permutation rounds in the padding-free rotation.
/// One round mixes the largest power-of-two prefix; the permutations carry
/// every coordinate through that prefix across rounds, so three rounds give
/// near-uniform mixing for any dimension (pinned by the estimator tests).
const TQ_ROTATION_ROUNDS: usize = 3;

/// Seeded structured rotation, padding-free: per round, sign flips → a
/// normalized FWHT over the largest power-of-two prefix (identity on the
/// remainder) → a full random permutation. Every factor is orthonormal on
/// `R^padded_dim`, so the composition preserves norms and inner products
/// exactly; inputs shorter than `padded_dim` (odd dims round up by one) are
/// zero-extended, which embeds them isometrically.
#[derive(Debug, Clone)]
pub struct TqRotation {
    input_dim: usize,
    padded_dim: usize,
    /// Largest power of two ≤ `padded_dim`: the per-round FWHT span.
    fwht_len: usize,
    /// +1.0 / -1.0 per coordinate, one strip per round.
    signs: Vec<f32>,
    /// `output[i] = mixed[perm[i]]`, one strip per round.
    perms: Vec<u32>,
}

impl TqRotation {
    pub fn new(input_dim: usize, seed: u64) -> Self {
        let padded_dim = tq_padded_dim(input_dim);
        let fwht_len = if padded_dim.is_power_of_two() {
            padded_dim
        } else {
            padded_dim.next_power_of_two() / 2
        };
        let mut state = seed;
        let mut signs = Vec::with_capacity(TQ_ROTATION_ROUNDS * padded_dim);
        let mut perms = Vec::with_capacity(TQ_ROTATION_ROUNDS * padded_dim);
        for _ in 0..TQ_ROTATION_ROUNDS {
            signs.extend((0..padded_dim).map(|_| {
                if splitmix64(&mut state) & 1 == 1 {
                    1.0f32
                } else {
                    -1.0f32
                }
            }));
            let round_base = perms.len();
            perms.extend(0..padded_dim as u32);
            for i in (1..padded_dim).rev() {
                let j = (splitmix64(&mut state) % (i as u64 + 1)) as usize;
                perms.swap(round_base + i, round_base + j);
            }
        }
        Self {
            input_dim,
            padded_dim,
            fwht_len,
            signs,
            perms,
        }
    }

    #[inline]
    pub fn padded_dim(&self) -> usize {
        self.padded_dim
    }

    /// Rotate `input` (length `input_dim`, or `padded_dim` for already-padded
    /// residuals) into `output` (length `padded_dim`). `scratch` is reused
    /// across calls to keep encoding allocation-free.
    pub fn apply(&self, input: &[f32], scratch: &mut Vec<f32>, output: &mut [f32]) {
        debug_assert!(input.len() == self.input_dim || input.len() == self.padded_dim);
        debug_assert_eq!(output.len(), self.padded_dim);
        let padded_dim = self.padded_dim;
        scratch.clear();
        scratch.resize(padded_dim, 0.0);
        // Round 0 reads straight from the (implicitly zero-extended) input;
        // later rounds ping-pong between `output` and `scratch`.
        let signs = &self.signs[..padded_dim];
        for (slot, (sign, index)) in scratch.iter_mut().zip(signs.iter().zip(0..)) {
            *slot = input.get(index).copied().unwrap_or(0.0) * sign;
        }
        fwht_normalized(&mut scratch[..self.fwht_len]);
        let perm = &self.perms[..padded_dim];
        for (slot, &source) in output.iter_mut().zip(perm) {
            *slot = scratch[source as usize];
        }
        for round in 1..TQ_ROTATION_ROUNDS {
            let signs = &self.signs[round * padded_dim..(round + 1) * padded_dim];
            for (slot, (&value, sign)) in scratch.iter_mut().zip(output.iter().zip(signs)) {
                *slot = value * sign;
            }
            fwht_normalized(&mut scratch[..self.fwht_len]);
            let perm = &self.perms[round * padded_dim..(round + 1) * padded_dim];
            for (slot, &source) in output.iter_mut().zip(perm) {
                *slot = scratch[source as usize];
            }
        }
    }
}

/// In-place normalized fast Walsh-Hadamard transform (`len` a power of two).
fn fwht_normalized(values: &mut [f32]) {
    let len = values.len();
    debug_assert!(len.is_power_of_two());
    let mut step = 1;
    while step < len {
        let mut base = 0;
        while base < len {
            let (left_half, right_half) = values[base..base + step * 2].split_at_mut(step);
            for (left, right) in left_half.iter_mut().zip(right_half.iter_mut()) {
                let sum = *left + *right;
                let difference = *left - *right;
                *left = sum;
                *right = difference;
            }
            base += step * 2;
        }
        step *= 2;
    }
    let scale = 1.0 / (len as f32).sqrt();
    for value in values.iter_mut() {
        *value *= scale;
    }
}

/// Analytic 3-bit Lloyd-Max codebook for the marginal density of one
/// coordinate of a uniform unit vector in `R^padded_dim`:
/// `f(t) ∝ (1 - t²)^((padded_dim - 3) / 2)` on `[-1, 1]`.
#[derive(Debug, Clone)]
pub struct TqCodebook {
    levels: [f32; TQ_STAGE1_LEVELS],
    /// Decision boundaries between adjacent levels (midpoints).
    boundaries: [f32; TQ_STAGE1_LEVELS - 1],
}

impl TqCodebook {
    pub fn analytic(padded_dim: usize) -> Self {
        assert!(
            padded_dim >= TQ_MIN_PADDED_DIM,
            "TQ codebook requires padded_dim >= {TQ_MIN_PADDED_DIM}, got {padded_dim}"
        );
        let exponent = (padded_dim as f64 - 3.0) / 2.0;
        let cell = 2.0 / TQ_LLOYD_GRID as f64;
        // Grid midpoints and their density weights over [-1, 1].
        // Heap-allocated: two [f64; 8192] frames (~128 KiB) would risk stack
        // overflow on constrained runtimes (WASM readers, worker threads).
        let mut weights = vec![0.0f64; TQ_LLOYD_GRID];
        let mut positions = vec![0.0f64; TQ_LLOYD_GRID];
        for index in 0..TQ_LLOYD_GRID {
            let t = -1.0 + (index as f64 + 0.5) * cell;
            positions[index] = t;
            let log_density = exponent * (1.0 - t * t).max(f64::MIN_POSITIVE).ln();
            weights[index] = log_density.exp();
        }

        // Initialize boundaries at equal-mass quantiles.
        let total_mass: f64 = weights.iter().sum();
        let mut levels = [0.0f64; TQ_STAGE1_LEVELS];
        let mut boundaries = [0.0f64; TQ_STAGE1_LEVELS - 1];
        let mut accumulated = 0.0f64;
        let mut next_boundary = 0usize;
        for index in 0..TQ_LLOYD_GRID {
            accumulated += weights[index];
            while next_boundary < TQ_STAGE1_LEVELS - 1
                && accumulated
                    >= total_mass * (next_boundary as f64 + 1.0) / TQ_STAGE1_LEVELS as f64
            {
                boundaries[next_boundary] = positions[index];
                next_boundary += 1;
            }
        }

        // Lloyd-Max: centroids are density-weighted means of their cell,
        // boundaries are midpoints of adjacent centroids.
        for _ in 0..TQ_LLOYD_MAX_ITERATIONS {
            let mut mass = [0.0f64; TQ_STAGE1_LEVELS];
            let mut moment = [0.0f64; TQ_STAGE1_LEVELS];
            let mut bucket = 0usize;
            for index in 0..TQ_LLOYD_GRID {
                let t = positions[index];
                while bucket < TQ_STAGE1_LEVELS - 1 && t > boundaries[bucket] {
                    bucket += 1;
                }
                mass[bucket] += weights[index];
                moment[bucket] += weights[index] * t;
            }
            let mut shift = 0.0f64;
            for level in 0..TQ_STAGE1_LEVELS {
                if mass[level] > 0.0 {
                    let updated = moment[level] / mass[level];
                    shift = shift.max((updated - levels[level]).abs());
                    levels[level] = updated;
                }
            }
            for boundary in 0..TQ_STAGE1_LEVELS - 1 {
                boundaries[boundary] = 0.5 * (levels[boundary] + levels[boundary + 1]);
            }
            if shift < TQ_LLOYD_TOLERANCE {
                break;
            }
        }

        // The density is even, so the optimal codebook is exactly symmetric;
        // grid discretization leaves ~1e-4 asymmetry. Symmetrize so the
        // central decision boundary is exactly zero.
        for index in 0..TQ_STAGE1_LEVELS / 2 {
            let magnitude = 0.5 * (levels[TQ_STAGE1_LEVELS - 1 - index] - levels[index]);
            levels[index] = -magnitude;
            levels[TQ_STAGE1_LEVELS - 1 - index] = magnitude;
        }
        for boundary in 0..TQ_STAGE1_LEVELS - 1 {
            boundaries[boundary] = 0.5 * (levels[boundary] + levels[boundary + 1]);
        }

        Self {
            levels: levels.map(|level| level as f32),
            boundaries: boundaries.map(|boundary| boundary as f32),
        }
    }

    /// 3-bit code of the nearest level.
    #[inline]
    pub fn encode_coordinate(&self, value: f32) -> u8 {
        let mut code = 0u8;
        for &boundary in &self.boundaries {
            code += u8::from(value > boundary);
        }
        code
    }
}

/// Complete TQ codec for one field dimension. Cheap to build (sub-millisecond)
/// and immutable; share via `Arc` per open segment.
#[derive(Debug, Clone)]
pub struct TqCodec {
    dim: usize,
    padded_dim: usize,
    stage1_rotation: TqRotation,
    qjl_rotation: TqRotation,
    codebook: TqCodebook,
    /// `sqrt(π/2) / sqrt(padded_dim)`: QJL correction for an orthonormal sketch.
    qjl_scale: f32,
    fingerprint: u64,
}

impl TqCodec {
    pub fn new(dim: usize) -> Self {
        assert!(dim > 0, "TQ codec requires a non-zero dimension");
        let stage1_rotation = TqRotation::new(dim, TQ_STAGE1_SEED);
        let padded_dim = stage1_rotation.padded_dim();
        let qjl_rotation = TqRotation::new(padded_dim, TQ_QJL_SEED);
        debug_assert_eq!(qjl_rotation.padded_dim(), padded_dim);
        let codebook = TqCodebook::analytic(padded_dim);
        let qjl_scale = (std::f64::consts::PI / 2.0).sqrt() as f32 / (padded_dim as f32).sqrt();
        let fingerprint = tq_fingerprint(dim, padded_dim);
        Self {
            dim,
            padded_dim,
            stage1_rotation,
            qjl_rotation,
            codebook,
            qjl_scale,
            fingerprint,
        }
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    #[inline]
    pub fn padded_dim(&self) -> usize {
        self.padded_dim
    }

    /// Logical bytes per vector (two 4-bit coordinates per byte).
    #[inline]
    pub fn code_size(&self) -> usize {
        self.padded_dim / 2
    }

    /// Deterministic compatibility fingerprint carried as `quantizer_version`.
    #[inline]
    pub fn fingerprint(&self) -> u64 {
        self.fingerprint
    }

    /// Heap footprint: two rotations (signs f32 + perm u32 per padded coord)
    /// plus the fixed-size codebook.
    pub fn estimated_memory_bytes(&self) -> usize {
        2 * self.padded_dim * (size_of::<f32>() + size_of::<u32>()) + size_of::<TqCodebook>()
    }

    /// Encode one vector into `nibbles` (one 0..=15 value per padded
    /// coordinate) and return `gamma`. The vector is normalized internally;
    /// zero vectors encode as all-zero nibbles with `gamma = 0`.
    pub fn encode_into(
        &self,
        vector: &[f32],
        nibbles: &mut [u8],
        scratch: &mut TqEncodeScratch,
    ) -> f32 {
        self.encode_residual_into(vector, nibbles, scratch).1
    }

    /// Encode one (possibly non-unit) vector as `scale · unit_direction` and
    /// return `(scale = ‖vector‖₂, gamma)`. IVF leaves store centroid
    /// residuals, whose norms carry ranking information; `scale` restores it
    /// at score time. Zero vectors encode as all-zero nibbles with
    /// `scale = gamma = 0`.
    pub fn encode_residual_into(
        &self,
        vector: &[f32],
        nibbles: &mut [u8],
        scratch: &mut TqEncodeScratch,
    ) -> (f32, f32) {
        assert_eq!(vector.len(), self.dim, "TQ encode dimension mismatch");
        assert_eq!(nibbles.len(), self.padded_dim, "TQ nibble buffer mismatch");
        let norm = crate::structures::simd::dot_product_f32(vector, vector, vector.len()).sqrt();
        if !norm.is_finite() || norm <= 0.0 {
            nibbles.fill(0);
            return (0.0, 0.0);
        }
        scratch.normalized.clear();
        scratch
            .normalized
            .extend(vector.iter().map(|value| value / norm));

        scratch.rotated.resize(self.padded_dim, 0.0);
        let (normalized, rotated, fwht) =
            (&scratch.normalized, &mut scratch.rotated, &mut scratch.fwht);
        self.stage1_rotation.apply(normalized, fwht, rotated);

        // Stage-1 codes and residual (in stage-1 rotated space).
        scratch.residual.resize(self.padded_dim, 0.0);
        let mut residual_norm_sq = 0.0f32;
        for ((&value, nibble), residual_slot) in scratch
            .rotated
            .iter()
            .zip(nibbles.iter_mut())
            .zip(scratch.residual.iter_mut())
        {
            let code = self.codebook.encode_coordinate(value);
            *nibble = code << 1;
            let residual = value - self.codebook.levels[code as usize];
            *residual_slot = residual;
            residual_norm_sq += residual * residual;
        }

        // QJL sign bits of the rotated residual.
        scratch.rotated_residual.resize(self.padded_dim, 0.0);
        let (residual, rotated_residual, fwht) = (
            &scratch.residual,
            &mut scratch.rotated_residual,
            &mut scratch.fwht,
        );
        self.qjl_rotation.apply(residual, fwht, rotated_residual);
        for (nibble, &rotated) in nibbles.iter_mut().zip(scratch.rotated_residual.iter()) {
            *nibble |= u8::from(rotated >= 0.0);
        }
        (norm, residual_norm_sq.sqrt())
    }
}

/// Reusable per-thread encode buffers (hot-path allocation hygiene).
#[derive(Debug, Default)]
pub struct TqEncodeScratch {
    normalized: Vec<f32>,
    rotated: Vec<f32>,
    residual: Vec<f32>,
    rotated_residual: Vec<f32>,
    fwht: Vec<f32>,
}

fn tq_fingerprint(dim: usize, padded_dim: usize) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64; // FNV-1a offset basis
    let mut mix = |bytes: &[u8]| {
        for &byte in bytes {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    mix(b"hermes-tq");
    mix(&TQ_CODEC_VERSION.to_le_bytes());
    mix(&TQ_BITS.to_le_bytes());
    mix(&(dim as u64).to_le_bytes());
    mix(&(padded_dim as u64).to_le_bytes());
    mix(&TQ_STAGE1_SEED.to_le_bytes());
    mix(&TQ_QJL_SEED.to_le_bytes());
    if hash == 0 { 1 } else { hash }
}

// ---------------------------------------------------------------------------
// Query plan and block scoring
// ---------------------------------------------------------------------------

/// Per-query LUTs: `padded_dim × 16` i8 tables (globally-scaled
/// quantizations) for the block kernels. The intermediate f32 tables are
/// dropped after quantization — they are not read on the search path.
pub struct TqQueryPlan {
    padded_dim: usize,
    fingerprint: u64,
    base_lut_i8: Vec<i8>,
    qjl_lut_i8: Vec<i8>,
    base_dequant: f32,
    qjl_dequant: f32,
    /// Full-precision tables, retained for the reference estimator in tests.
    #[cfg(test)]
    reference_luts: (Vec<f32>, Vec<f32>),
}

impl std::fmt::Debug for TqQueryPlan {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TqQueryPlan")
            .field("padded_dim", &self.padded_dim)
            .field("fingerprint", &self.fingerprint)
            .finish()
    }
}

impl TqQueryPlan {
    pub fn build(codec: &TqCodec, query: &[f32]) -> Self {
        assert_eq!(query.len(), codec.dim, "TQ query dimension mismatch");
        let padded_dim = codec.padded_dim;
        let norm = crate::structures::simd::dot_product_f32(query, query, query.len()).sqrt();
        let inverse_norm = if norm.is_finite() && norm > 0.0 {
            1.0 / norm
        } else {
            0.0
        };
        let normalized: Vec<f32> = query.iter().map(|value| value * inverse_norm).collect();
        let mut fwht = Vec::with_capacity(padded_dim);
        let mut rotated = vec![0.0f32; padded_dim];
        codec
            .stage1_rotation
            .apply(&normalized, &mut fwht, &mut rotated);
        let mut qjl_rotated = vec![0.0f32; padded_dim];
        codec
            .qjl_rotation
            .apply(&rotated, &mut fwht, &mut qjl_rotated);

        let mut base_lut = vec![0.0f32; padded_dim * 16];
        let mut qjl_lut = vec![0.0f32; padded_dim * 16];
        for dim in 0..padded_dim {
            for nibble in 0..16 {
                let level = codec.codebook.levels[nibble >> 1];
                let sign = if nibble & 1 == 1 { 1.0 } else { -1.0 };
                base_lut[dim * 16 + nibble] = rotated[dim] * level;
                qjl_lut[dim * 16 + nibble] = sign * codec.qjl_scale * qjl_rotated[dim];
            }
        }
        let (base_lut_i8, base_dequant) = quantize_lut(&base_lut);
        let (qjl_lut_i8, qjl_dequant) = quantize_lut(&qjl_lut);
        Self {
            padded_dim,
            fingerprint: codec.fingerprint,
            base_lut_i8,
            qjl_lut_i8,
            base_dequant,
            qjl_dequant,
            #[cfg(test)]
            reference_luts: (base_lut, qjl_lut),
        }
    }

    #[inline]
    pub fn padded_dim(&self) -> usize {
        self.padded_dim
    }

    #[inline]
    pub fn fingerprint(&self) -> u64 {
        self.fingerprint
    }

    /// Reference f32 estimator over one unpacked nibble row (test oracle for
    /// the quantized block path).
    #[cfg(test)]
    pub(crate) fn estimate_row(&self, nibbles: &[u8], gamma: f32) -> f32 {
        debug_assert_eq!(nibbles.len(), self.padded_dim);
        let (base_lut, qjl_lut) = &self.reference_luts;
        let mut base = 0.0f32;
        let mut qjl = 0.0f32;
        for (dim, &nibble) in nibbles.iter().enumerate() {
            base += base_lut[dim * 16 + nibble as usize];
            qjl += qjl_lut[dim * 16 + nibble as usize];
        }
        base + gamma * qjl
    }
}

fn quantize_lut(values: &[f32]) -> (Vec<i8>, f32) {
    let max_abs = values
        .iter()
        .fold(0.0f32, |acc, &value| acc.max(value.abs()));
    if !max_abs.is_finite() || max_abs <= 0.0 {
        return (vec![0i8; values.len()], 0.0);
    }
    let quantize_scale = 127.0 / max_abs;
    let quantized = values
        .iter()
        .map(|&value| (value * quantize_scale).round().clamp(-127.0, 127.0) as i8)
        .collect();
    (quantized, max_abs / 127.0)
}

/// Score one block (16 lanes) into `scores`. `block` is
/// `[16 × f32 gamma][padded_dim × 8 packed nibbles]`; lanes past the run's
/// vector count hold zero padding and must be ignored by the caller.
pub fn tq_score_block(plan: &TqQueryPlan, block: &[u8], scores: &mut [f32; TQ_BLOCK_LANES]) {
    debug_assert_eq!(block.len(), tq_block_bytes(plan.padded_dim / 2));
    let (gamma_bytes, nibble_bytes) = block.split_at(TQ_BLOCK_LANES * size_of::<f32>());
    let mut base = [0i32; TQ_BLOCK_LANES];
    let mut qjl = [0i32; TQ_BLOCK_LANES];
    lut16::accumulate_block(
        &plan.base_lut_i8,
        &plan.qjl_lut_i8,
        nibble_bytes,
        plan.padded_dim,
        &mut base,
        &mut qjl,
    );
    for lane in 0..TQ_BLOCK_LANES {
        let gamma = f32::from_le_bytes(
            gamma_bytes[lane * 4..lane * 4 + 4]
                .try_into()
                .expect("gamma slice is 4 bytes"),
        );
        scores[lane] =
            base[lane] as f32 * plan.base_dequant + gamma * qjl[lane] as f32 * plan.qjl_dequant;
    }
}

/// Pack up to 16 nibble rows (+ gammas) into one block. Missing lanes are
/// zero-filled. `rows` are `padded_dim`-length 0..=15 values.
pub fn tq_pack_block(rows: &[&[u8]], gammas: &[f32], padded_dim: usize, output: &mut Vec<u8>) {
    assert!(rows.len() <= TQ_BLOCK_LANES && rows.len() == gammas.len());
    for lane in 0..TQ_BLOCK_LANES {
        let gamma = gammas.get(lane).copied().unwrap_or(0.0);
        output.extend_from_slice(&gamma.to_le_bytes());
    }
    pack_nibble_rows(rows, padded_dim, output);
}

/// Pack an IVF-TQ block: per-lane residual scales, gammas, then nibbles.
pub fn tq_pack_ivf_block(
    rows: &[&[u8]],
    scales: &[f32],
    gammas: &[f32],
    padded_dim: usize,
    output: &mut Vec<u8>,
) {
    assert!(rows.len() <= TQ_BLOCK_LANES && rows.len() == gammas.len());
    assert_eq!(scales.len(), gammas.len());
    for lane in 0..TQ_BLOCK_LANES {
        let scale = scales.get(lane).copied().unwrap_or(0.0);
        output.extend_from_slice(&scale.to_le_bytes());
    }
    for lane in 0..TQ_BLOCK_LANES {
        let gamma = gammas.get(lane).copied().unwrap_or(0.0);
        output.extend_from_slice(&gamma.to_le_bytes());
    }
    pack_nibble_rows(rows, padded_dim, output);
}

fn pack_nibble_rows(rows: &[&[u8]], padded_dim: usize, output: &mut Vec<u8>) {
    for dim in 0..padded_dim {
        for byte_index in 0..TQ_BLOCK_LANES / 2 {
            let low = rows.get(byte_index).map_or(0, |row| row[dim] & 0x0F);
            let high = rows
                .get(byte_index + TQ_BLOCK_LANES / 2)
                .map_or(0, |row| row[dim] & 0x0F);
            output.push(low | (high << 4));
        }
    }
}

/// Score one IVF-TQ block: `score[lane] = cluster_dot + scale · (base +
/// gamma · qjl)`, where `cluster_dot = ⟨normalized query, centroid⟩` is the
/// probed cluster's shared contribution and `scale = ‖residual‖`.
pub fn tq_score_ivf_block(
    plan: &TqQueryPlan,
    block: &[u8],
    cluster_dot: f32,
    scores: &mut [f32; TQ_BLOCK_LANES],
) {
    debug_assert_eq!(block.len(), tq_ivf_block_bytes(plan.padded_dim() / 2));
    let lane_f32 = TQ_BLOCK_LANES * size_of::<f32>();
    let (scale_bytes, rest) = block.split_at(lane_f32);
    let (gamma_bytes, nibble_bytes) = rest.split_at(lane_f32);
    let mut base = [0i32; TQ_BLOCK_LANES];
    let mut qjl = [0i32; TQ_BLOCK_LANES];
    lut16::accumulate_block(
        &plan.base_lut_i8,
        &plan.qjl_lut_i8,
        nibble_bytes,
        plan.padded_dim,
        &mut base,
        &mut qjl,
    );
    for lane in 0..TQ_BLOCK_LANES {
        let scale = f32::from_le_bytes(
            scale_bytes[lane * 4..lane * 4 + 4]
                .try_into()
                .expect("scale slice is 4 bytes"),
        );
        let gamma = f32::from_le_bytes(
            gamma_bytes[lane * 4..lane * 4 + 4]
                .try_into()
                .expect("gamma slice is 4 bytes"),
        );
        scores[lane] = cluster_dot
            + scale
                * (base[lane] as f32 * plan.base_dequant
                    + gamma * qjl[lane] as f32 * plan.qjl_dequant);
    }
}

mod lut16 {
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    use super::TQ_ACCUMULATE_CHUNK_DIMS;
    use super::TQ_BLOCK_LANES;

    /// Accumulate both LUT sums for 16 lanes over all dimensions.
    /// `nibble_bytes` is dimension-major: 8 bytes per dimension, byte `j`
    /// holding lane `j` (low nibble) and lane `j + 8` (high nibble).
    pub(super) fn accumulate_block(
        base_lut: &[i8],
        qjl_lut: &[i8],
        nibble_bytes: &[u8],
        padded_dim: usize,
        base: &mut [i32; TQ_BLOCK_LANES],
        qjl: &mut [i32; TQ_BLOCK_LANES],
    ) {
        #[cfg(target_arch = "aarch64")]
        {
            // NEON is baseline on aarch64.
            unsafe { accumulate_block_neon(base_lut, qjl_lut, nibble_bytes, padded_dim, base, qjl) }
            return;
        }
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                unsafe {
                    accumulate_block_avx2(base_lut, qjl_lut, nibble_bytes, padded_dim, base, qjl)
                }
                return;
            }
            if std::arch::is_x86_feature_detected!("ssse3") {
                unsafe {
                    accumulate_block_ssse3(base_lut, qjl_lut, nibble_bytes, padded_dim, base, qjl)
                }
                return;
            }
        }
        #[allow(unreachable_code)]
        accumulate_block_scalar(base_lut, qjl_lut, nibble_bytes, padded_dim, base, qjl);
    }

    /// Scalar fallback mirroring the SIMD integer arithmetic exactly
    /// (i8 lookups, i32 sums), so all paths agree bit-for-bit.
    pub(super) fn accumulate_block_scalar(
        base_lut: &[i8],
        qjl_lut: &[i8],
        nibble_bytes: &[u8],
        padded_dim: usize,
        base: &mut [i32; TQ_BLOCK_LANES],
        qjl: &mut [i32; TQ_BLOCK_LANES],
    ) {
        for dim in 0..padded_dim {
            let row = &nibble_bytes[dim * 8..dim * 8 + 8];
            let base_table = &base_lut[dim * 16..dim * 16 + 16];
            let qjl_table = &qjl_lut[dim * 16..dim * 16 + 16];
            for (lane, &byte) in row.iter().enumerate() {
                let low = (byte & 0x0F) as usize;
                let high = (byte >> 4) as usize;
                base[lane] += i32::from(base_table[low]);
                base[lane + 8] += i32::from(base_table[high]);
                qjl[lane] += i32::from(qjl_table[low]);
                qjl[lane + 8] += i32::from(qjl_table[high]);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn accumulate_block_neon(
        base_lut: &[i8],
        qjl_lut: &[i8],
        nibble_bytes: &[u8],
        padded_dim: usize,
        base: &mut [i32; TQ_BLOCK_LANES],
        qjl: &mut [i32; TQ_BLOCK_LANES],
    ) {
        use std::arch::aarch64::*;
        unsafe {
            let mask = vdup_n_u8(0x0F);
            let mut base_lo_i32 = [vdupq_n_s32(0); 4];
            let mut qjl_lo_i32 = [vdupq_n_s32(0); 4];
            let mut dim = 0;
            while dim < padded_dim {
                let chunk_end = (dim + TQ_ACCUMULATE_CHUNK_DIMS).min(padded_dim);
                let mut base_acc = [vdupq_n_s16(0); 2];
                let mut qjl_acc = [vdupq_n_s16(0); 2];
                while dim < chunk_end {
                    let row = vld1_u8(nibble_bytes.as_ptr().add(dim * 8));
                    let low = vand_u8(row, mask);
                    let high = vshr_n_u8::<4>(row);
                    let lanes = vcombine_u8(low, high);
                    let base_table = vld1q_s8(base_lut.as_ptr().add(dim * 16));
                    let qjl_table = vld1q_s8(qjl_lut.as_ptr().add(dim * 16));
                    let base_values =
                        vreinterpretq_s8_u8(vqtbl1q_u8(vreinterpretq_u8_s8(base_table), lanes));
                    let qjl_values =
                        vreinterpretq_s8_u8(vqtbl1q_u8(vreinterpretq_u8_s8(qjl_table), lanes));
                    base_acc[0] = vaddw_s8(base_acc[0], vget_low_s8(base_values));
                    base_acc[1] = vaddw_s8(base_acc[1], vget_high_s8(base_values));
                    qjl_acc[0] = vaddw_s8(qjl_acc[0], vget_low_s8(qjl_values));
                    qjl_acc[1] = vaddw_s8(qjl_acc[1], vget_high_s8(qjl_values));
                    dim += 1;
                }
                for half in 0..2 {
                    base_lo_i32[half * 2] =
                        vaddw_s16(base_lo_i32[half * 2], vget_low_s16(base_acc[half]));
                    base_lo_i32[half * 2 + 1] =
                        vaddw_s16(base_lo_i32[half * 2 + 1], vget_high_s16(base_acc[half]));
                    qjl_lo_i32[half * 2] =
                        vaddw_s16(qjl_lo_i32[half * 2], vget_low_s16(qjl_acc[half]));
                    qjl_lo_i32[half * 2 + 1] =
                        vaddw_s16(qjl_lo_i32[half * 2 + 1], vget_high_s16(qjl_acc[half]));
                }
            }
            for quarter in 0..4 {
                vst1q_s32(base.as_mut_ptr().add(quarter * 4), base_lo_i32[quarter]);
                vst1q_s32(qjl.as_mut_ptr().add(quarter * 4), qjl_lo_i32[quarter]);
            }
        }
    }

    /// AVX2: two dimensions per iteration. Adjacent dims' packed rows are
    /// contiguous (8 bytes each) and so are their 16-entry LUTs, so one 16-byte
    /// row load + one 32-byte LUT load + a 256-bit `vpshufb` covers both.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn accumulate_block_avx2(
        base_lut: &[i8],
        qjl_lut: &[i8],
        nibble_bytes: &[u8],
        padded_dim: usize,
        base: &mut [i32; TQ_BLOCK_LANES],
        qjl: &mut [i32; TQ_BLOCK_LANES],
    ) {
        use std::arch::x86_64::*;
        unsafe {
            let mask = _mm_set1_epi8(0x0F);
            let zero256 = _mm256_setzero_si256();
            let mut base_i32 = [zero256; 2];
            let mut qjl_i32 = [zero256; 2];
            let mut dim = 0;
            while dim < padded_dim {
                let chunk_end = (dim + TQ_ACCUMULATE_CHUNK_DIMS).min(padded_dim);
                let mut base_acc = zero256;
                let mut qjl_acc = zero256;
                while dim + 2 <= chunk_end {
                    // Bytes [dim*8, dim*8+16): rows for `dim` and `dim + 1`.
                    let rows = _mm_loadu_si128(nibble_bytes.as_ptr().add(dim * 8).cast());
                    let low = _mm_and_si128(rows, mask);
                    let high = _mm_and_si128(_mm_srli_epi16(rows, 4), mask);
                    // Lanes 0..16 of each dim: [low.q0 | high.q0], [low.q1 | high.q1].
                    let lanes_first = _mm_unpacklo_epi64(low, high);
                    let lanes_second = _mm_unpackhi_epi64(low, high);
                    let lanes = _mm256_inserti128_si256(
                        _mm256_castsi128_si256(lanes_first),
                        lanes_second,
                        1,
                    );
                    let base_tables = _mm256_loadu_si256(base_lut.as_ptr().add(dim * 16).cast());
                    let qjl_tables = _mm256_loadu_si256(qjl_lut.as_ptr().add(dim * 16).cast());
                    let base_values = _mm256_shuffle_epi8(base_tables, lanes);
                    let qjl_values = _mm256_shuffle_epi8(qjl_tables, lanes);
                    base_acc = _mm256_add_epi16(
                        base_acc,
                        _mm256_cvtepi8_epi16(_mm256_castsi256_si128(base_values)),
                    );
                    base_acc = _mm256_add_epi16(
                        base_acc,
                        _mm256_cvtepi8_epi16(_mm256_extracti128_si256(base_values, 1)),
                    );
                    qjl_acc = _mm256_add_epi16(
                        qjl_acc,
                        _mm256_cvtepi8_epi16(_mm256_castsi256_si128(qjl_values)),
                    );
                    qjl_acc = _mm256_add_epi16(
                        qjl_acc,
                        _mm256_cvtepi8_epi16(_mm256_extracti128_si256(qjl_values, 1)),
                    );
                    dim += 2;
                }
                // Odd remainder dim within the chunk.
                while dim < chunk_end {
                    let row = _mm_loadl_epi64(nibble_bytes.as_ptr().add(dim * 8).cast());
                    let low = _mm_and_si128(row, mask);
                    let high = _mm_and_si128(_mm_srli_epi16(row, 4), mask);
                    let lanes = _mm_unpacklo_epi64(low, high);
                    let base_table = _mm_loadu_si128(base_lut.as_ptr().add(dim * 16).cast());
                    let qjl_table = _mm_loadu_si128(qjl_lut.as_ptr().add(dim * 16).cast());
                    base_acc = _mm256_add_epi16(
                        base_acc,
                        _mm256_cvtepi8_epi16(_mm_shuffle_epi8(base_table, lanes)),
                    );
                    qjl_acc = _mm256_add_epi16(
                        qjl_acc,
                        _mm256_cvtepi8_epi16(_mm_shuffle_epi8(qjl_table, lanes)),
                    );
                    dim += 1;
                }
                for (accumulators, chunk) in [(&mut base_i32, base_acc), (&mut qjl_i32, qjl_acc)] {
                    accumulators[0] = _mm256_add_epi32(
                        accumulators[0],
                        _mm256_cvtepi16_epi32(_mm256_castsi256_si128(chunk)),
                    );
                    accumulators[1] = _mm256_add_epi32(
                        accumulators[1],
                        _mm256_cvtepi16_epi32(_mm256_extracti128_si256(chunk, 1)),
                    );
                }
            }
            _mm256_storeu_si256(base.as_mut_ptr().cast(), base_i32[0]);
            _mm256_storeu_si256(base.as_mut_ptr().add(8).cast(), base_i32[1]);
            _mm256_storeu_si256(qjl.as_mut_ptr().cast(), qjl_i32[0]);
            _mm256_storeu_si256(qjl.as_mut_ptr().add(8).cast(), qjl_i32[1]);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "ssse3")]
    unsafe fn accumulate_block_ssse3(
        base_lut: &[i8],
        qjl_lut: &[i8],
        nibble_bytes: &[u8],
        padded_dim: usize,
        base: &mut [i32; TQ_BLOCK_LANES],
        qjl: &mut [i32; TQ_BLOCK_LANES],
    ) {
        use std::arch::x86_64::*;
        unsafe {
            let mask = _mm_set1_epi8(0x0F);
            let zero = _mm_setzero_si128();
            let mut base_i32 = [zero; 4];
            let mut qjl_i32 = [zero; 4];
            let mut dim = 0;
            while dim < padded_dim {
                let chunk_end = (dim + TQ_ACCUMULATE_CHUNK_DIMS).min(padded_dim);
                let mut base_acc = [zero; 2];
                let mut qjl_acc = [zero; 2];
                while dim < chunk_end {
                    let row = _mm_loadl_epi64(nibble_bytes.as_ptr().add(dim * 8).cast());
                    let low = _mm_and_si128(row, mask);
                    let high = _mm_and_si128(_mm_srli_epi16(row, 4), mask);
                    let lanes = _mm_unpacklo_epi64(low, high);
                    let base_table = _mm_loadu_si128(base_lut.as_ptr().add(dim * 16).cast());
                    let qjl_table = _mm_loadu_si128(qjl_lut.as_ptr().add(dim * 16).cast());
                    let base_values = _mm_shuffle_epi8(base_table, lanes);
                    let qjl_values = _mm_shuffle_epi8(qjl_table, lanes);
                    // Sign-extend i8 → i16 without SSE4.1: compare-based sign mask.
                    let base_sign = _mm_cmpgt_epi8(zero, base_values);
                    let qjl_sign = _mm_cmpgt_epi8(zero, qjl_values);
                    base_acc[0] =
                        _mm_add_epi16(base_acc[0], _mm_unpacklo_epi8(base_values, base_sign));
                    base_acc[1] =
                        _mm_add_epi16(base_acc[1], _mm_unpackhi_epi8(base_values, base_sign));
                    qjl_acc[0] = _mm_add_epi16(qjl_acc[0], _mm_unpacklo_epi8(qjl_values, qjl_sign));
                    qjl_acc[1] = _mm_add_epi16(qjl_acc[1], _mm_unpackhi_epi8(qjl_values, qjl_sign));
                    dim += 1;
                }
                for half in 0..2 {
                    let base_sign = _mm_cmpgt_epi16(zero, base_acc[half]);
                    let qjl_sign = _mm_cmpgt_epi16(zero, qjl_acc[half]);
                    base_i32[half * 2] = _mm_add_epi32(
                        base_i32[half * 2],
                        _mm_unpacklo_epi16(base_acc[half], base_sign),
                    );
                    base_i32[half * 2 + 1] = _mm_add_epi32(
                        base_i32[half * 2 + 1],
                        _mm_unpackhi_epi16(base_acc[half], base_sign),
                    );
                    qjl_i32[half * 2] = _mm_add_epi32(
                        qjl_i32[half * 2],
                        _mm_unpacklo_epi16(qjl_acc[half], qjl_sign),
                    );
                    qjl_i32[half * 2 + 1] = _mm_add_epi32(
                        qjl_i32[half * 2 + 1],
                        _mm_unpackhi_epi16(qjl_acc[half], qjl_sign),
                    );
                }
            }
            for quarter in 0..4 {
                _mm_storeu_si128(base.as_mut_ptr().add(quarter * 4).cast(), base_i32[quarter]);
                _mm_storeu_si128(qjl.as_mut_ptr().add(quarter * 4).cast(), qjl_i32[quarter]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Segment build support
// ---------------------------------------------------------------------------

/// Streaming builder for one segment's TQ payload: doc/ordinal columns plus a
/// block-packed codes column ready for `ann_disk` serialization.
#[cfg(feature = "native")]
pub struct TqFlatBuilder {
    codec: std::sync::Arc<TqCodec>,
    pub doc_ids: Vec<u32>,
    pub ordinals: Vec<u16>,
    pub codes: Vec<u8>,
    pending_rows: Vec<Vec<u8>>,
    pending_gammas: Vec<f32>,
}

#[cfg(feature = "native")]
impl TqFlatBuilder {
    pub fn new(codec: std::sync::Arc<TqCodec>) -> Self {
        Self {
            codec,
            doc_ids: Vec::new(),
            ordinals: Vec::new(),
            codes: Vec::new(),
            pending_rows: Vec::with_capacity(TQ_BLOCK_LANES),
            pending_gammas: Vec::with_capacity(TQ_BLOCK_LANES),
        }
    }

    #[inline]
    pub fn codec(&self) -> &TqCodec {
        &self.codec
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.doc_ids.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.doc_ids.is_empty()
    }

    /// Encode one contiguous `(labels, vectors)` batch in parallel while
    /// preserving input order (lane order must match the doc-ID column).
    pub fn add_batch(
        &mut self,
        labels: &[(u32, u16)],
        vectors: &[f32],
    ) -> Result<(), &'static str> {
        use rayon::prelude::*;

        let dim = self.codec.dim();
        let vector_count = labels.len();
        let expected = vector_count
            .checked_mul(dim)
            .ok_or("TQ input size overflow")?;
        if vectors.len() != expected {
            return Err("TQ vector and label matrices are inconsistent");
        }
        let padded_dim = self.codec.padded_dim();
        let codec = std::sync::Arc::clone(&self.codec);
        // One contiguous nibble matrix instead of a Vec per vector: each
        // Rayon task writes its disjoint row range (allocation hygiene on
        // the ingest/merge path).
        let mut rows = vec![0u8; vector_count * padded_dim];
        let mut gammas = vec![0.0f32; vector_count];
        vectors
            .par_chunks_exact(dim)
            .zip(rows.par_chunks_exact_mut(padded_dim))
            .zip(gammas.par_iter_mut())
            .for_each_init(
                TqEncodeScratch::default,
                |scratch, ((vector, row), gamma)| {
                    *gamma = codec.encode_into(vector, row, scratch);
                },
            );

        // Top up a carried-over partial block, pack full blocks straight from
        // the contiguous matrix (no per-row copies), and carry the tail.
        let mut index = 0;
        while index < vector_count && !self.pending_rows.is_empty() {
            let (doc_id, ordinal) = labels[index];
            self.doc_ids.push(doc_id);
            self.ordinals.push(ordinal);
            self.pending_rows
                .push(rows[index * padded_dim..(index + 1) * padded_dim].to_vec());
            self.pending_gammas.push(gammas[index]);
            if self.pending_rows.len() == TQ_BLOCK_LANES {
                self.flush_block();
            }
            index += 1;
        }
        while vector_count - index >= TQ_BLOCK_LANES {
            let row_refs: Vec<&[u8]> = (0..TQ_BLOCK_LANES)
                .map(|lane| {
                    let row = index + lane;
                    &rows[row * padded_dim..(row + 1) * padded_dim]
                })
                .collect();
            tq_pack_block(
                &row_refs,
                &gammas[index..index + TQ_BLOCK_LANES],
                padded_dim,
                &mut self.codes,
            );
            for &(doc_id, ordinal) in &labels[index..index + TQ_BLOCK_LANES] {
                self.doc_ids.push(doc_id);
                self.ordinals.push(ordinal);
            }
            index += TQ_BLOCK_LANES;
        }
        for row in index..vector_count {
            let (doc_id, ordinal) = labels[row];
            self.doc_ids.push(doc_id);
            self.ordinals.push(ordinal);
            self.pending_rows
                .push(rows[row * padded_dim..(row + 1) * padded_dim].to_vec());
            self.pending_gammas.push(gammas[row]);
        }
        Ok(())
    }

    fn flush_block(&mut self) {
        let rows: Vec<&[u8]> = self.pending_rows.iter().map(Vec::as_slice).collect();
        tq_pack_block(
            &rows,
            &self.pending_gammas,
            self.codec.padded_dim(),
            &mut self.codes,
        );
        self.pending_rows.clear();
        self.pending_gammas.clear();
    }

    /// Flush the trailing partial block (zero-padded lanes).
    pub fn finish(&mut self) {
        if !self.pending_rows.is_empty() {
            self.flush_block();
        }
        debug_assert_eq!(
            self.codes.len(),
            tq_codes_column_len(self.doc_ids.len(), self.codec.code_size())
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seeded_unit_vector(dim: usize, seed: u64) -> Vec<f32> {
        // Box-Muller from splitmix64 for an isotropic direction.
        let mut state = seed;
        let mut values: Vec<f32> = (0..dim)
            .map(|_| {
                let a = (splitmix64(&mut state) >> 11) as f64 / (1u64 << 53) as f64;
                let b = (splitmix64(&mut state) >> 11) as f64 / (1u64 << 53) as f64;
                let gaussian = (-2.0 * (1.0 - a).max(f64::MIN_POSITIVE).ln()).sqrt()
                    * (2.0 * std::f64::consts::PI * b).cos();
                gaussian as f32
            })
            .collect();
        let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt();
        values.iter_mut().for_each(|v| *v /= norm);
        values
    }

    #[test]
    fn rotation_is_orthonormal_and_deterministic() {
        let rotation = TqRotation::new(100, 42);
        assert_eq!(rotation.padded_dim(), 100);
        let mut fwht_scratch = Vec::new();
        let mut probe = vec![0.0f32; 100];
        // Odd dims round up by one zero coordinate.
        assert_eq!(TqRotation::new(99, 42).padded_dim(), 100);
        TqRotation::new(99, 42).apply(&vec![1.0; 99], &mut fwht_scratch, &mut probe);
        let input = seeded_unit_vector(100, 7);
        let mut fwht = Vec::new();
        let mut output = vec![0.0f32; 100];
        rotation.apply(&input, &mut fwht, &mut output);
        let norm: f32 = output.iter().map(|v| v * v).sum();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "rotation must preserve norm, got {norm}"
        );

        let mut second = vec![0.0f32; 100];
        TqRotation::new(100, 42).apply(&input, &mut fwht, &mut second);
        assert_eq!(output, second, "rotation must be deterministic");

        // Distinct inputs keep their inner product (isometry).
        let other = seeded_unit_vector(100, 8);
        let mut other_rotated = vec![0.0f32; 100];
        rotation.apply(&other, &mut fwht, &mut other_rotated);
        let dot_before: f32 = input.iter().zip(&other).map(|(a, b)| a * b).sum();
        let dot_after: f32 = output.iter().zip(&other_rotated).map(|(a, b)| a * b).sum();
        assert!(
            (dot_before - dot_after).abs() < 1e-4,
            "rotation must preserve inner products: {dot_before} vs {dot_after}"
        );
    }

    #[test]
    fn analytic_codebook_is_symmetric_and_monotonic() {
        for padded_dim in [8, 128, 1024] {
            let codebook = TqCodebook::analytic(padded_dim);
            let levels = &codebook.levels;
            for pair in levels.windows(2) {
                assert!(pair[0] < pair[1], "levels must be strictly increasing");
            }
            for index in 0..TQ_STAGE1_LEVELS / 2 {
                assert_eq!(
                    levels[index],
                    -levels[TQ_STAGE1_LEVELS - 1 - index],
                    "levels must be exactly symmetric for P={padded_dim}: {levels:?}"
                );
            }
            assert!(levels[TQ_STAGE1_LEVELS - 1] < 1.0);
            // Coordinates concentrate near ±1/sqrt(P); the top level must be
            // on that scale, not at the interval edge.
            let scale = 1.0 / (padded_dim as f32).sqrt();
            assert!(
                levels[TQ_STAGE1_LEVELS - 1] < 6.0 * scale,
                "top level {} is implausibly large for P={padded_dim}",
                levels[TQ_STAGE1_LEVELS - 1]
            );
        }
    }

    #[test]
    fn encode_coordinate_matches_nearest_level() {
        let codebook = TqCodebook::analytic(256);
        for step in -1000i32..=1000 {
            let value = step as f32 / 1000.0;
            let code = codebook.encode_coordinate(value) as usize;
            let nearest = codebook
                .levels
                .iter()
                .enumerate()
                .min_by(|a, b| (a.1 - value).abs().total_cmp(&(b.1 - value).abs()))
                .unwrap()
                .0;
            assert_eq!(
                code, nearest,
                "value {value} coded {code}, nearest {nearest}"
            );
        }
    }

    #[test]
    fn estimator_is_unbiased_and_tight() {
        let dim = 96;
        let codec = TqCodec::new(dim);
        let mut scratch = TqEncodeScratch::default();
        let mut nibbles = vec![0u8; codec.padded_dim()];

        let pairs = 512;
        let mut signed_error_sum = 0.0f64;
        let mut squared_error_sum = 0.0f64;
        let mut stage1_signed_error_sum = 0.0f64;
        for pair in 0..pairs {
            let vector = seeded_unit_vector(dim, 1000 + pair);
            let query = seeded_unit_vector(dim, 900_000 + pair);
            let gamma = codec.encode_into(&vector, &mut nibbles, &mut scratch);
            let plan = TqQueryPlan::build(&codec, &query);
            let estimate = plan.estimate_row(&nibbles, gamma);
            let stage1_only = plan.estimate_row(&nibbles, 0.0);
            let truth: f32 = vector.iter().zip(&query).map(|(a, b)| a * b).sum();
            signed_error_sum += f64::from(estimate - truth);
            squared_error_sum += f64::from(estimate - truth).powi(2);
            stage1_signed_error_sum += f64::from(stage1_only - truth);
        }
        let mean_error = signed_error_sum / pairs as f64;
        let rmse = (squared_error_sum / pairs as f64).sqrt();
        let stage1_mean_error = stage1_signed_error_sum / pairs as f64;
        assert!(
            mean_error.abs() < 3e-3,
            "QJL-corrected estimator must be unbiased: mean error {mean_error}"
        );
        assert!(rmse < 0.05, "estimator RMSE too large: {rmse}");
        assert!(
            mean_error.abs() <= stage1_mean_error.abs() + 1e-4,
            "QJL correction must not increase bias: {mean_error} vs stage-1 {stage1_mean_error}"
        );
    }

    #[test]
    fn block_scoring_matches_row_estimates() {
        let dim = 100;
        let codec = TqCodec::new(dim);
        let mut scratch = TqEncodeScratch::default();
        let query = seeded_unit_vector(dim, 3);
        let plan = TqQueryPlan::build(&codec, &query);

        let lanes = 13; // deliberately partial block
        let mut rows = Vec::new();
        let mut gammas = Vec::new();
        for lane in 0..lanes {
            let vector = seeded_unit_vector(dim, 50 + lane as u64);
            let mut nibbles = vec![0u8; codec.padded_dim()];
            let gamma = codec.encode_into(&vector, &mut nibbles, &mut scratch);
            rows.push(nibbles);
            gammas.push(gamma);
        }
        let row_refs: Vec<&[u8]> = rows.iter().map(Vec::as_slice).collect();
        let mut block = Vec::new();
        tq_pack_block(&row_refs, &gammas, codec.padded_dim(), &mut block);
        assert_eq!(block.len(), tq_block_bytes(codec.code_size()));

        let mut scores = [0.0f32; TQ_BLOCK_LANES];
        tq_score_block(&plan, &block, &mut scores);
        for lane in 0..lanes {
            let expected = plan.estimate_row(&rows[lane], gammas[lane]);
            // The block path uses i8 LUTs; agreement is approximate.
            assert!(
                (scores[lane] - expected).abs() < 0.02,
                "lane {lane}: block score {} vs row estimate {expected}",
                scores[lane]
            );
        }
    }

    #[test]
    fn simd_accumulation_matches_scalar_reference_exactly() {
        let padded_dim = 192; // not a multiple of the widen chunk
        let mut state = 99u64;
        let mut base_lut = vec![0i8; padded_dim * 16];
        let mut qjl_lut = vec![0i8; padded_dim * 16];
        for value in base_lut.iter_mut().chain(qjl_lut.iter_mut()) {
            *value = (splitmix64(&mut state) as i32 % 255 - 127) as i8;
        }
        let mut nibble_bytes = vec![0u8; padded_dim * 8];
        for byte in nibble_bytes.iter_mut() {
            *byte = splitmix64(&mut state) as u8;
        }

        let mut base_reference = [0i32; TQ_BLOCK_LANES];
        let mut qjl_reference = [0i32; TQ_BLOCK_LANES];
        lut16::accumulate_block_scalar(
            &base_lut,
            &qjl_lut,
            &nibble_bytes,
            padded_dim,
            &mut base_reference,
            &mut qjl_reference,
        );
        let mut base_dispatch = [0i32; TQ_BLOCK_LANES];
        let mut qjl_dispatch = [0i32; TQ_BLOCK_LANES];
        lut16::accumulate_block(
            &base_lut,
            &qjl_lut,
            &nibble_bytes,
            padded_dim,
            &mut base_dispatch,
            &mut qjl_dispatch,
        );
        assert_eq!(
            base_reference, base_dispatch,
            "base sums must match exactly"
        );
        assert_eq!(qjl_reference, qjl_dispatch, "qjl sums must match exactly");
    }

    #[test]
    fn fingerprint_pins_codec_constants() {
        let codec = TqCodec::new(768);
        assert_eq!(codec.padded_dim(), 768, "codec v2 is padding-free");
        assert_eq!(codec.code_size(), 384);
        assert_ne!(codec.fingerprint(), 0);
        assert_eq!(codec.fingerprint(), TqCodec::new(768).fingerprint());
        assert_ne!(codec.fingerprint(), TqCodec::new(769).fingerprint());
        // Golden value: the fingerprint is persisted as quantizer_version in
        // every TQ segment. Any change to the hash, seeds, or codec constants
        // MUST bump TQ_CODEC_VERSION — never silently re-derive.
        assert_eq!(
            codec.fingerprint(),
            GOLDEN_FINGERPRINT_768,
            "TQ fingerprint for dim 768 changed; existing segments would be \
             rejected. Bump TQ_CODEC_VERSION deliberately instead."
        );
    }

    const GOLDEN_FINGERPRINT_768: u64 = 7026088428300072418;

    #[cfg(feature = "native")]
    #[test]
    fn builder_packs_blocks_in_input_order() {
        let dim = 32;
        let codec = std::sync::Arc::new(TqCodec::new(dim));
        let mut builder = TqFlatBuilder::new(std::sync::Arc::clone(&codec));
        let count = 37; // two full blocks + partial
        let labels: Vec<(u32, u16)> = (0..count).map(|i| (i as u32, (i % 3) as u16)).collect();
        let mut vectors = Vec::new();
        for i in 0..count {
            vectors.extend(seeded_unit_vector(dim, 7_000 + i as u64));
        }
        builder.add_batch(&labels, &vectors).unwrap();
        builder.finish();
        assert_eq!(builder.doc_ids.len(), count);
        assert_eq!(
            builder.codes.len(),
            tq_codes_column_len(count, codec.code_size())
        );

        // Every lane must score identically to encoding the row directly.
        let query = seeded_unit_vector(dim, 1);
        let plan = TqQueryPlan::build(&codec, &query);
        let block_bytes = tq_block_bytes(codec.code_size());
        let mut scratch = TqEncodeScratch::default();
        let mut scores = [0.0f32; TQ_BLOCK_LANES];
        for index in 0..count {
            let block = &builder.codes[(index / TQ_BLOCK_LANES) * block_bytes..][..block_bytes];
            tq_score_block(&plan, block, &mut scores);
            let mut nibbles = vec![0u8; codec.padded_dim()];
            let gamma = codec.encode_into(
                &vectors[index * dim..(index + 1) * dim],
                &mut nibbles,
                &mut scratch,
            );
            let expected = plan.estimate_row(&nibbles, gamma);
            assert!(
                (scores[index % TQ_BLOCK_LANES] - expected).abs() < 0.02,
                "vector {index} scored {} expected {expected}",
                scores[index % TQ_BLOCK_LANES]
            );
        }
    }
}
