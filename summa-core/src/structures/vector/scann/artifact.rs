use super::{MAX_SCANN_TREE_LEVELS, ScannConfig, ScannEncoding, ScannFormatError, ScannResult};
use std::io::Write;
use std::ops::Range;

const MAGIC: &[u8; 8] = b"HSCNGLOB";
pub const SCANN_GLOBAL_ARTIFACT_VERSION: u16 = 1;
const FINGERPRINT_OFFSET: usize = 12;

/// One routing centroid level. Float centroids are stored as per-dimension
/// fixed-point bytes with `minimums` and `steps`; binary centroids are their
/// exact packed bytes and leave both quantization vectors empty.
#[derive(Debug, Clone, PartialEq)]
pub struct ScannRoutingLevel {
    pub centroid_count: u32,
    pub centroid_codes: Vec<u8>,
    pub minimums: Vec<f32>,
    pub steps: Vec<f32>,
    /// CSR offsets from this level into the next. Empty only on the leaf level.
    pub child_offsets: Vec<u32>,
}

/// Global residual AH codebook. Binary Hamming ScaNN intentionally has no AH
/// codebook because the segment leaves retain exact packed vectors.
#[derive(Debug, Clone, PartialEq)]
pub struct ScannAhCodebook {
    pub dimensions_per_block: u16,
    pub centers_per_block: u16,
    /// Block-major, center-major, coordinate-major f32 values.
    pub centers: Vec<f32>,
}

/// A single trained routing/codebook generation shared by every segment.
#[derive(Debug, Clone, PartialEq)]
pub struct ScannTrainedArtifact {
    pub generation: u64,
    pub artifact_id: u64,
    pub trained_vectors: u64,
    pub config: ScannConfig,
    pub levels: Vec<ScannRoutingLevel>,
    pub ah_codebook: Option<ScannAhCodebook>,
}

#[derive(Debug, Clone)]
struct ScannRoutingLevelRange {
    centroid_count: u32,
    centroid_codes: Range<usize>,
    minimums: Range<usize>,
    steps: Range<usize>,
    child_offsets: Range<usize>,
}

#[derive(Debug, Clone)]
struct ScannAhCodebookRange {
    dimensions_per_block: u16,
    centers_per_block: u16,
    centers: Range<usize>,
}

/// Borrowed, zero-copy view of a persisted global ScaNN generation.
///
/// The routing centroid plane can be many gigabytes at billion-vector scale.
/// Parsing retains checked ranges into the caller's mmap-backed slice and
/// allocates only the at-most-three small level descriptors.
#[derive(Debug, Clone)]
pub struct ScannTrainedArtifactView<'a> {
    bytes: &'a [u8],
    pub generation: u64,
    pub artifact_id: u64,
    pub trained_vectors: u64,
    pub config: ScannConfig,
    levels: Vec<ScannRoutingLevelRange>,
    ah_codebook: Option<ScannAhCodebookRange>,
}

#[derive(Debug, Clone, Copy)]
pub struct ScannRoutingLevelRef<'a> {
    pub centroid_count: u32,
    pub centroid_codes: &'a [u8],
    minimums_le: &'a [u8],
    steps_le: &'a [u8],
    child_offsets_le: &'a [u8],
}

impl<'a> ScannRoutingLevelRef<'a> {
    pub fn minimums(&self) -> impl ExactSizeIterator<Item = f32> + 'a {
        self.minimums_le
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
    }

    pub fn steps(&self) -> impl ExactSizeIterator<Item = f32> + 'a {
        self.steps_le
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
    }

    pub fn child_offsets(&self) -> impl ExactSizeIterator<Item = u32> + 'a {
        self.child_offsets_le
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ScannAhCodebookRef<'a> {
    pub dimensions_per_block: u16,
    pub centers_per_block: u16,
    centers_le: &'a [u8],
}

impl<'a> ScannAhCodebookRef<'a> {
    pub fn centers(&self) -> impl ExactSizeIterator<Item = f32> + 'a {
        self.centers_le
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
    }
}

impl<'a> ScannTrainedArtifactView<'a> {
    pub fn parse(bytes: &'a [u8]) -> ScannResult<Self> {
        let mut input = Input::new(bytes);
        if input.take(8)? != MAGIC {
            return Err(ScannFormatError::new("invalid ScaNN global artifact magic"));
        }
        let version = input.u16()?;
        if version != SCANN_GLOBAL_ARTIFACT_VERSION {
            return Err(ScannFormatError::new(format!(
                "unsupported ScaNN global artifact version {version}; reader supports {SCANN_GLOBAL_ARTIFACT_VERSION}"
            )));
        }
        if input.u16()? != 0 {
            return Err(ScannFormatError::new(
                "ScaNN global artifact reserved field is non-zero",
            ));
        }
        let artifact_id = input.u64()?;
        let generation = input.u64()?;
        let trained_vectors = input.u64()?;
        let dimension = input.u32()?;
        let tree_levels = input.u8()?;
        let encoding_tag = input.u8()?;
        let dimensions_per_block = input.u16()?;
        let bits_per_code = input.u8()?;
        if input.take(3)? != [0, 0, 0] {
            return Err(ScannFormatError::new(
                "ScaNN global artifact reserved bytes are non-zero",
            ));
        }
        let config = ScannConfig {
            dimension,
            tree_levels,
            num_leaves: input.u32()?,
            encoding: ScannEncoding::from_parts(encoding_tag, dimensions_per_block, bits_per_code)?,
        };
        config.validate()?;
        let required = config.effective_training_threshold()?;
        if generation == 0 || artifact_id == 0 || trained_vectors < required {
            return Err(ScannFormatError::new(format!(
                "invalid ScaNN generation metadata: generation={generation}, fingerprint={artifact_id}, trained={trained_vectors}, required={required}"
            )));
        }
        let level_count = input.u8()?;
        if input.take(7)? != [0; 7]
            || level_count != config.tree_levels
            || level_count > MAX_SCANN_TREE_LEVELS
        {
            return Err(ScannFormatError::new(
                "ScaNN routing level count does not match configuration",
            ));
        }
        let centroid_width = match config.encoding {
            ScannEncoding::AsymmetricHash { .. } => config.dimension as usize,
            ScannEncoding::BinaryHamming => config.dimension as usize / 8,
        };
        let mut levels = Vec::with_capacity(usize::from(level_count));
        for level_index in 0..level_count {
            let centroid_count = input.u32()?;
            let code_len = input.usize()?;
            let minimum_count = input.usize()?;
            let step_count = input.usize()?;
            let child_count = input.usize()?;
            let expected_codes = (centroid_count as usize)
                .checked_mul(centroid_width)
                .ok_or_else(|| ScannFormatError::new("ScaNN centroid matrix size overflows"))?;
            if centroid_count == 0 || code_len != expected_codes {
                return Err(ScannFormatError::new(format!(
                    "invalid ScaNN centroid matrix at level {level_index}"
                )));
            }
            match config.encoding {
                ScannEncoding::AsymmetricHash { .. }
                    if minimum_count != config.dimension as usize
                        || step_count != config.dimension as usize =>
                {
                    return Err(ScannFormatError::new(format!(
                        "invalid ScaNN fixed-point parameters at level {level_index}"
                    )));
                }
                ScannEncoding::BinaryHamming if minimum_count != 0 || step_count != 0 => {
                    return Err(ScannFormatError::new(
                        "binary ScaNN centroids must not carry float quantization parameters",
                    ));
                }
                _ => {}
            }
            let centroid_codes = input.take_range(code_len)?;
            let minimums = input.take_range(checked_word_bytes(minimum_count)?)?;
            let steps = input.take_range(checked_word_bytes(step_count)?)?;
            let child_offsets = input.take_range(checked_word_bytes(child_count)?)?;
            if bytes[minimums.clone()]
                .chunks_exact(4)
                .chain(bytes[steps.clone()].chunks_exact(4))
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .any(|value| !value.is_finite())
            {
                return Err(ScannFormatError::new(format!(
                    "non-finite ScaNN fixed-point parameter at level {level_index}"
                )));
            }
            levels.push(ScannRoutingLevelRange {
                centroid_count,
                centroid_codes,
                minimums,
                steps,
                child_offsets,
            });
        }
        validate_borrowed_levels(&config, &levels, bytes)?;
        let ah_codebook = match input.u8()? {
            0 => None,
            1 => {
                let dimensions_per_block = input.u16()?;
                let centers_per_block = input.u16()?;
                let center_count = input.usize()?;
                let centers = input.take_range(checked_word_bytes(center_count)?)?;
                if bytes[centers.clone()]
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                    .any(|value| !value.is_finite())
                {
                    return Err(ScannFormatError::new(
                        "ScaNN AH codebook contains non-finite values",
                    ));
                }
                Some(ScannAhCodebookRange {
                    dimensions_per_block,
                    centers_per_block,
                    centers,
                })
            }
            _ => {
                return Err(ScannFormatError::new(
                    "invalid ScaNN AH codebook presence tag",
                ));
            }
        };
        if !input.is_empty() {
            return Err(ScannFormatError::new(
                "ScaNN global artifact has trailing bytes",
            ));
        }
        validate_borrowed_codebook(&config, ah_codebook.as_ref(), bytes)?;
        if fingerprint(bytes) != artifact_id {
            return Err(ScannFormatError::new(
                "ScaNN global artifact fingerprint mismatch",
            ));
        }
        Ok(Self {
            bytes,
            generation,
            artifact_id,
            trained_vectors,
            config,
            levels,
            ah_codebook,
        })
    }

    pub fn bytes(&self) -> &'a [u8] {
        self.bytes
    }

    pub fn level_count(&self) -> usize {
        self.levels.len()
    }

    pub fn level(&self, index: usize) -> Option<ScannRoutingLevelRef<'a>> {
        let level = self.levels.get(index)?;
        Some(ScannRoutingLevelRef {
            centroid_count: level.centroid_count,
            centroid_codes: &self.bytes[level.centroid_codes.clone()],
            minimums_le: &self.bytes[level.minimums.clone()],
            steps_le: &self.bytes[level.steps.clone()],
            child_offsets_le: &self.bytes[level.child_offsets.clone()],
        })
    }

    /// Checked byte range of one quantized centroid plane. Executable mmap
    /// models retain this descriptor rather than decoding the plane to f32.
    pub fn level_centroid_codes_range(&self, index: usize) -> Option<Range<usize>> {
        self.levels
            .get(index)
            .map(|level| level.centroid_codes.clone())
    }

    pub fn ah_codebook(&self) -> Option<ScannAhCodebookRef<'a>> {
        self.ah_codebook
            .as_ref()
            .map(|codebook| ScannAhCodebookRef {
                dimensions_per_block: codebook.dimensions_per_block,
                centers_per_block: codebook.centers_per_block,
                centers_le: &self.bytes[codebook.centers.clone()],
            })
    }
}

fn validate_borrowed_levels(
    config: &ScannConfig,
    levels: &[ScannRoutingLevelRange],
    bytes: &[u8],
) -> ScannResult<()> {
    for (level_index, level) in levels.iter().enumerate() {
        let is_leaf = level_index + 1 == levels.len();
        if is_leaf {
            if !level.child_offsets.is_empty() || level.centroid_count != config.num_leaves {
                return Err(ScannFormatError::new(
                    "ScaNN leaf level does not match configured leaves",
                ));
            }
            continue;
        }
        let offsets = bytes[level.child_offsets.clone()]
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()));
        let mut previous = None;
        let mut count = 0usize;
        let mut last = 0u32;
        for offset in offsets {
            if previous.is_some_and(|value| value > offset) {
                return Err(ScannFormatError::new(format!(
                    "invalid ScaNN child directory at level {level_index}"
                )));
            }
            previous = Some(offset);
            last = offset;
            count += 1;
        }
        if count != level.centroid_count as usize + 1
            || previous.is_none()
            || bytes[level.child_offsets.clone()].get(..4) != Some(&0u32.to_le_bytes())
            || last != levels[level_index + 1].centroid_count
        {
            return Err(ScannFormatError::new(format!(
                "invalid ScaNN child directory at level {level_index}"
            )));
        }
    }
    Ok(())
}

fn validate_borrowed_codebook(
    config: &ScannConfig,
    codebook: Option<&ScannAhCodebookRange>,
    bytes: &[u8],
) -> ScannResult<()> {
    match (config.encoding, codebook) {
        (
            ScannEncoding::AsymmetricHash {
                dimensions_per_block,
                bits_per_code,
            },
            Some(codebook),
        ) => {
            let centers_per_block = 1usize << bits_per_code;
            let blocks = (config.dimension as usize).div_ceil(usize::from(dimensions_per_block));
            let expected_values = blocks
                .checked_mul(centers_per_block)
                .and_then(|count| count.checked_mul(usize::from(dimensions_per_block)))
                .ok_or_else(|| ScannFormatError::new("ScaNN AH codebook size overflows"))?;
            if codebook.dimensions_per_block != dimensions_per_block
                || usize::from(codebook.centers_per_block) != centers_per_block
                || bytes[codebook.centers.clone()].len() != checked_word_bytes(expected_values)?
            {
                return Err(ScannFormatError::new("invalid ScaNN AH codebook shape"));
            }
        }
        (ScannEncoding::BinaryHamming, None) => {}
        (ScannEncoding::AsymmetricHash { .. }, None) => {
            return Err(ScannFormatError::new(
                "float ScaNN artifact is missing its AH codebook",
            ));
        }
        (ScannEncoding::BinaryHamming, Some(_)) => {
            return Err(ScannFormatError::new(
                "binary ScaNN must keep exact codes and has no AH codebook",
            ));
        }
    }
    Ok(())
}

fn checked_word_bytes(count: usize) -> ScannResult<usize> {
    count
        .checked_mul(4)
        .ok_or_else(|| ScannFormatError::new("ScaNN word byte length overflows"))
}

impl ScannTrainedArtifact {
    pub fn new(
        generation: u64,
        trained_vectors: u64,
        config: ScannConfig,
        levels: Vec<ScannRoutingLevel>,
        ah_codebook: Option<ScannAhCodebook>,
    ) -> ScannResult<Self> {
        let mut artifact = Self {
            generation,
            artifact_id: 0,
            trained_vectors,
            config,
            levels,
            ah_codebook,
        };
        artifact.validate_shape()?;
        artifact.artifact_id = artifact.compute_fingerprint()?;
        Ok(artifact)
    }

    pub fn to_bytes(&self) -> ScannResult<Vec<u8>> {
        self.validate()?;
        self.encode(self.artifact_id)
    }

    /// Stream the artifact without materializing a second full-size byte Vec.
    pub fn write_to(&self, writer: &mut impl Write) -> ScannResult<u64> {
        self.validate()?;
        let mut written = 0u64;
        self.for_each_encoded_chunk(self.artifact_id, |chunk| {
            writer.write_all(chunk).map_err(|error| {
                ScannFormatError::new(format!("failed to write ScaNN artifact: {error}"))
            })?;
            written = written
                .checked_add(chunk.len() as u64)
                .ok_or_else(|| ScannFormatError::new("ScaNN artifact size exceeds u64"))?;
            Ok(())
        })?;
        Ok(written)
    }

    pub fn from_bytes(bytes: &[u8]) -> ScannResult<Self> {
        let mut input = Input::new(bytes);
        if input.take(8)? != MAGIC {
            return Err(ScannFormatError::new("invalid ScaNN global artifact magic"));
        }
        let version = input.u16()?;
        if version != SCANN_GLOBAL_ARTIFACT_VERSION {
            return Err(ScannFormatError::new(format!(
                "unsupported ScaNN global artifact version {version}; reader supports {SCANN_GLOBAL_ARTIFACT_VERSION}"
            )));
        }
        if input.u16()? != 0 {
            return Err(ScannFormatError::new(
                "ScaNN global artifact reserved field is non-zero",
            ));
        }
        let artifact_id = input.u64()?;
        let generation = input.u64()?;
        let trained_vectors = input.u64()?;
        let dimension = input.u32()?;
        let tree_levels = input.u8()?;
        let encoding_tag = input.u8()?;
        let dimensions_per_block = input.u16()?;
        let bits_per_code = input.u8()?;
        let reserved = input.take(3)?;
        if reserved != [0, 0, 0] {
            return Err(ScannFormatError::new(
                "ScaNN global artifact reserved bytes are non-zero",
            ));
        }
        let config = ScannConfig {
            dimension,
            tree_levels,
            num_leaves: input.u32()?,
            encoding: ScannEncoding::from_parts(encoding_tag, dimensions_per_block, bits_per_code)?,
        };
        let level_count = input.u8()?;
        if input.take(7)? != [0; 7] {
            return Err(ScannFormatError::new(
                "ScaNN global artifact level padding is non-zero",
            ));
        }
        let mut levels = Vec::with_capacity(usize::from(level_count));
        for _ in 0..level_count {
            let centroid_count = input.u32()?;
            let code_len = input.usize()?;
            let minimum_count = input.usize()?;
            let step_count = input.usize()?;
            let child_count = input.usize()?;
            let centroid_codes = input.take(code_len)?.to_vec();
            let minimums = input.f32_vec(minimum_count)?;
            let steps = input.f32_vec(step_count)?;
            let child_offsets = input.u32_vec(child_count)?;
            levels.push(ScannRoutingLevel {
                centroid_count,
                centroid_codes,
                minimums,
                steps,
                child_offsets,
            });
        }
        let ah_codebook = match input.u8()? {
            0 => None,
            1 => {
                let dimensions_per_block = input.u16()?;
                let centers_per_block = input.u16()?;
                let center_count = input.usize()?;
                Some(ScannAhCodebook {
                    dimensions_per_block,
                    centers_per_block,
                    centers: input.f32_vec(center_count)?,
                })
            }
            _ => {
                return Err(ScannFormatError::new(
                    "invalid ScaNN AH codebook presence tag",
                ));
            }
        };
        if !input.is_empty() {
            return Err(ScannFormatError::new(
                "ScaNN global artifact has trailing bytes",
            ));
        }
        let artifact = Self {
            generation,
            artifact_id,
            trained_vectors,
            config,
            levels,
            ah_codebook,
        };
        artifact.validate()?;
        Ok(artifact)
    }

    pub fn validate(&self) -> ScannResult<()> {
        self.validate_shape()?;
        if self.artifact_id == 0 {
            return Err(ScannFormatError::new(
                "ScaNN global artifact fingerprint must be non-zero",
            ));
        }
        let expected = self.compute_fingerprint()?;
        if self.artifact_id != expected {
            return Err(ScannFormatError::new(
                "ScaNN global artifact fingerprint mismatch",
            ));
        }
        Ok(())
    }

    fn validate_shape(&self) -> ScannResult<()> {
        self.config.validate()?;
        if self.generation == 0 {
            return Err(ScannFormatError::new(
                "ScaNN global artifact generation must be non-zero",
            ));
        }
        let required = self.config.effective_training_threshold()?;
        if self.trained_vectors < required {
            return Err(ScannFormatError::new(format!(
                "ScaNN artifact was trained on {} vectors, below required threshold {required}",
                self.trained_vectors
            )));
        }
        if self.levels.len() != usize::from(self.config.tree_levels)
            || self.levels.len() > usize::from(MAX_SCANN_TREE_LEVELS)
        {
            return Err(ScannFormatError::new(
                "ScaNN routing level count does not match configuration",
            ));
        }
        let centroid_width = match self.config.encoding {
            ScannEncoding::AsymmetricHash { .. } => self.config.dimension as usize,
            ScannEncoding::BinaryHamming => self.config.dimension as usize / 8,
        };
        for (level_index, level) in self.levels.iter().enumerate() {
            let expected_centroid_bytes = (level.centroid_count as usize)
                .checked_mul(centroid_width)
                .ok_or_else(|| ScannFormatError::new("ScaNN centroid matrix size overflows"))?;
            if level.centroid_count == 0 || level.centroid_codes.len() != expected_centroid_bytes {
                return Err(ScannFormatError::new(format!(
                    "invalid ScaNN centroid matrix at level {level_index}"
                )));
            }
            match self.config.encoding {
                ScannEncoding::AsymmetricHash { .. }
                    if level.minimums.len() != self.config.dimension as usize
                        || level.steps.len() != self.config.dimension as usize
                        || level
                            .minimums
                            .iter()
                            .chain(&level.steps)
                            .any(|value| !value.is_finite()) =>
                {
                    return Err(ScannFormatError::new(format!(
                        "invalid ScaNN fixed-point parameters at level {level_index}"
                    )));
                }
                ScannEncoding::BinaryHamming
                    if !level.minimums.is_empty() || !level.steps.is_empty() =>
                {
                    return Err(ScannFormatError::new(
                        "binary ScaNN centroids must not carry float quantization parameters",
                    ));
                }
                _ => {}
            }
            let is_leaf = level_index + 1 == self.levels.len();
            if is_leaf {
                if !level.child_offsets.is_empty() || level.centroid_count != self.config.num_leaves
                {
                    return Err(ScannFormatError::new(
                        "ScaNN leaf level does not match configured leaves",
                    ));
                }
            } else {
                let next_count = self.levels[level_index + 1].centroid_count;
                if level.child_offsets.len() != level.centroid_count as usize + 1
                    || level.child_offsets.first() != Some(&0)
                    || level.child_offsets.last() != Some(&next_count)
                    || level.child_offsets.windows(2).any(|pair| pair[0] > pair[1])
                {
                    return Err(ScannFormatError::new(format!(
                        "invalid ScaNN child directory at level {level_index}"
                    )));
                }
            }
        }
        match (self.config.encoding, &self.ah_codebook) {
            (
                ScannEncoding::AsymmetricHash {
                    dimensions_per_block,
                    bits_per_code,
                },
                Some(codebook),
            ) => {
                let expected_centers = 1usize << bits_per_code;
                let blocks =
                    (self.config.dimension as usize).div_ceil(usize::from(dimensions_per_block));
                let expected_values = blocks
                    .checked_mul(expected_centers)
                    .and_then(|count| count.checked_mul(usize::from(dimensions_per_block)))
                    .ok_or_else(|| ScannFormatError::new("ScaNN AH codebook size overflows"))?;
                if codebook.dimensions_per_block != dimensions_per_block
                    || usize::from(codebook.centers_per_block) != expected_centers
                    || codebook.centers.len() != expected_values
                    || codebook.centers.iter().any(|value| !value.is_finite())
                {
                    return Err(ScannFormatError::new("invalid ScaNN AH codebook shape"));
                }
            }
            (ScannEncoding::BinaryHamming, None) => {}
            (ScannEncoding::AsymmetricHash { .. }, None) => {
                return Err(ScannFormatError::new(
                    "float ScaNN artifact is missing its AH codebook",
                ));
            }
            (ScannEncoding::BinaryHamming, Some(_)) => {
                return Err(ScannFormatError::new(
                    "binary ScaNN must keep exact codes and has no AH codebook",
                ));
            }
        }
        Ok(())
    }

    fn encode(&self, stored_fingerprint: u64) -> ScannResult<Vec<u8>> {
        let mut output = Vec::new();
        self.for_each_encoded_chunk(stored_fingerprint, |chunk| {
            output.extend_from_slice(chunk);
            Ok(())
        })?;
        Ok(output)
    }

    fn compute_fingerprint(&self) -> ScannResult<u64> {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        self.for_each_encoded_chunk(0, |chunk| {
            for &byte in chunk {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
            Ok(())
        })?;
        Ok(hash.max(1))
    }

    fn for_each_encoded_chunk(
        &self,
        stored_fingerprint: u64,
        mut output: impl FnMut(&[u8]) -> ScannResult<()>,
    ) -> ScannResult<()> {
        output(MAGIC)?;
        output(&SCANN_GLOBAL_ARTIFACT_VERSION.to_le_bytes())?;
        output(&0u16.to_le_bytes())?;
        output(&stored_fingerprint.to_le_bytes())?;
        output(&self.generation.to_le_bytes())?;
        output(&self.trained_vectors.to_le_bytes())?;
        output(&self.config.dimension.to_le_bytes())?;
        output(&[self.config.tree_levels])?;
        output(&[self.config.encoding.tag()])?;
        let (dimensions_per_block, bits_per_code) = self.config.encoding.parameters();
        output(&dimensions_per_block.to_le_bytes())?;
        output(&[bits_per_code])?;
        output(&[0; 3])?;
        output(&self.config.num_leaves.to_le_bytes())?;
        output(&[u8::try_from(self.levels.len())
            .map_err(|_| ScannFormatError::new("too many ScaNN routing levels"))?])?;
        output(&[0; 7])?;
        for level in &self.levels {
            output(&level.centroid_count.to_le_bytes())?;
            output(&encoded_len(level.centroid_codes.len())?)?;
            output(&encoded_len(level.minimums.len())?)?;
            output(&encoded_len(level.steps.len())?)?;
            output(&encoded_len(level.child_offsets.len())?)?;
            output(&level.centroid_codes)?;
            for value in &level.minimums {
                output(&value.to_bits().to_le_bytes())?;
            }
            for value in &level.steps {
                output(&value.to_bits().to_le_bytes())?;
            }
            for &value in &level.child_offsets {
                output(&value.to_le_bytes())?;
            }
        }
        match &self.ah_codebook {
            None => output(&[0])?,
            Some(codebook) => {
                output(&[1])?;
                output(&codebook.dimensions_per_block.to_le_bytes())?;
                output(&codebook.centers_per_block.to_le_bytes())?;
                output(&encoded_len(codebook.centers.len())?)?;
                for value in &codebook.centers {
                    output(&value.to_bits().to_le_bytes())?;
                }
            }
        }
        Ok(())
    }
}

fn fingerprint(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for (index, &byte) in bytes.iter().enumerate() {
        let byte = if (FINGERPRINT_OFFSET..FINGERPRINT_OFFSET + 8).contains(&index) {
            0
        } else {
            byte
        };
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash.max(1)
}

fn encoded_len(value: usize) -> ScannResult<[u8; 8]> {
    Ok(u64::try_from(value)
        .map_err(|_| ScannFormatError::new("ScaNN artifact length exceeds u64"))?
        .to_le_bytes())
}

struct Input<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Input<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, len: usize) -> ScannResult<&'a [u8]> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or_else(|| ScannFormatError::new("ScaNN artifact offset overflows"))?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or_else(|| ScannFormatError::new("truncated ScaNN global artifact"))?;
        self.offset = end;
        Ok(value)
    }

    fn take_range(&mut self, len: usize) -> ScannResult<Range<usize>> {
        let start = self.offset;
        self.take(len)?;
        Ok(start..self.offset)
    }

    fn u16(&mut self) -> ScannResult<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn u8(&mut self) -> ScannResult<u8> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> ScannResult<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> ScannResult<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn usize(&mut self) -> ScannResult<usize> {
        usize::try_from(self.u64()?)
            .map_err(|_| ScannFormatError::new("ScaNN artifact length exceeds usize"))
    }

    fn f32_vec(&mut self, count: usize) -> ScannResult<Vec<f32>> {
        let byte_len = count
            .checked_mul(4)
            .ok_or_else(|| ScannFormatError::new("ScaNN f32 vector size overflows"))?;
        let bytes = self.take(byte_len)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect())
    }

    fn u32_vec(&mut self, count: usize) -> ScannResult<Vec<u32>> {
        let byte_len = count
            .checked_mul(4)
            .ok_or_else(|| ScannFormatError::new("ScaNN u32 vector size overflows"))?;
        let bytes = self.take(byte_len)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect())
    }

    fn is_empty(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binary_artifact() -> ScannTrainedArtifact {
        let config = ScannConfig {
            dimension: 16,
            tree_levels: 2,
            num_leaves: 4,
            encoding: ScannEncoding::BinaryHamming,
        };
        ScannTrainedArtifact::new(
            7,
            100_000,
            config,
            vec![
                ScannRoutingLevel {
                    centroid_count: 2,
                    centroid_codes: vec![0, 0, 0xff, 0xff],
                    minimums: Vec::new(),
                    steps: Vec::new(),
                    child_offsets: vec![0, 2, 4],
                },
                ScannRoutingLevel {
                    centroid_count: 4,
                    centroid_codes: vec![0, 0, 1, 1, 0xfe, 0xfe, 0xff, 0xff],
                    minimums: Vec::new(),
                    steps: Vec::new(),
                    child_offsets: Vec::new(),
                },
            ],
            None,
        )
        .unwrap()
    }

    #[test]
    fn global_artifact_round_trip_preserves_generation_and_fingerprint() {
        let artifact = binary_artifact();
        let bytes = artifact.to_bytes().unwrap();
        let decoded = ScannTrainedArtifact::from_bytes(&bytes).unwrap();
        assert_eq!(decoded, artifact);
        assert_ne!(artifact.artifact_id, 0);
    }

    #[test]
    fn streaming_artifact_writer_matches_in_memory_encoding() {
        let artifact = binary_artifact();
        let expected = artifact.to_bytes().unwrap();
        let mut streamed = Vec::new();
        let written = artifact.write_to(&mut streamed).unwrap();
        assert_eq!(written as usize, expected.len());
        assert_eq!(streamed, expected);
    }

    #[test]
    fn global_artifact_view_borrows_the_original_centroid_plane() {
        let artifact = binary_artifact();
        let bytes = artifact.to_bytes().unwrap();
        let view = ScannTrainedArtifactView::parse(&bytes).unwrap();
        let level = view.level(0).unwrap();
        let original = bytes.as_ptr_range();
        assert_eq!(view.bytes().as_ptr(), bytes.as_ptr());
        assert!(level.centroid_codes.as_ptr() >= original.start);
        assert!(level.centroid_codes.as_ptr() < original.end);
        assert_eq!(level.child_offsets().collect::<Vec<_>>(), vec![0, 2, 4]);
        assert!(view.ah_codebook().is_none());
    }

    #[test]
    fn global_artifact_view_rejects_truncation_and_corruption() {
        let artifact = binary_artifact();
        let bytes = artifact.to_bytes().unwrap();
        assert!(ScannTrainedArtifactView::parse(&bytes[..bytes.len() - 1]).is_err());

        let mut corrupt = bytes;
        let centroid_offset = {
            let view = ScannTrainedArtifactView::parse(&corrupt).unwrap();
            view.level(0).unwrap().centroid_codes.as_ptr() as usize - corrupt.as_ptr() as usize
        };
        corrupt[centroid_offset] ^= 1;
        let error = ScannTrainedArtifactView::parse(&corrupt).unwrap_err();
        assert!(error.to_string().contains("fingerprint mismatch"));
    }

    #[test]
    fn global_artifact_rejects_a_future_format_version() {
        let artifact = binary_artifact();
        let mut bytes = artifact.to_bytes().unwrap();
        bytes[8..10].copy_from_slice(&(SCANN_GLOBAL_ARTIFACT_VERSION + 1).to_le_bytes());
        let error = ScannTrainedArtifact::from_bytes(&bytes).unwrap_err();
        assert!(error.to_string().contains("unsupported"));
    }

    #[test]
    fn global_artifact_rejects_content_changed_after_fingerprinting() {
        let artifact = binary_artifact();
        let mut bytes = artifact.to_bytes().unwrap();
        let centroid_offset = {
            let view = ScannTrainedArtifactView::parse(&bytes).unwrap();
            view.level(0).unwrap().centroid_codes.as_ptr() as usize - bytes.as_ptr() as usize
        };
        bytes[centroid_offset] ^= 1;
        let error = ScannTrainedArtifact::from_bytes(&bytes).unwrap_err();
        assert!(error.to_string().contains("fingerprint mismatch"));
    }
}
