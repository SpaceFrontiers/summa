use super::{ScannConfig, ScannEncoding, ScannFormatError, ScannResult, ScannTrainedArtifact};

const MAGIC: &[u8; 8] = b"HSCNSEGM";
pub const SCANN_SEGMENT_PAYLOAD_VERSION: u16 = 1;

/// One immutable physical leaf extent. Document IDs are little-endian u32
/// values relative to `doc_base`; ordinals are little-endian u16 values.
/// `codes` remain opaque to merging and are moved without re-encoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScannLeafRun {
    pub leaf_id: u32,
    pub doc_base: u32,
    pub row_count: u32,
    pub doc_ids_le: Vec<u8>,
    pub ordinals_le: Vec<u8>,
    pub codes: Vec<u8>,
}

impl ScannLeafRun {
    pub fn from_rows(
        leaf_id: u32,
        doc_base: u32,
        doc_ids: &[u32],
        ordinals: &[u16],
        codes: Vec<u8>,
        encoding: ScannEncoding,
        dimension: u32,
    ) -> ScannResult<Self> {
        if doc_ids.len() != ordinals.len() {
            return Err(ScannFormatError::new(
                "ScaNN leaf document and ordinal columns have different lengths",
            ));
        }
        let row_count = u32::try_from(doc_ids.len())
            .map_err(|_| ScannFormatError::new("ScaNN leaf row count exceeds u32"))?;
        let mut doc_ids_le = Vec::with_capacity(doc_ids.len().saturating_mul(4));
        for &doc_id in doc_ids {
            doc_ids_le.extend_from_slice(&doc_id.to_le_bytes());
        }
        let mut ordinals_le = Vec::with_capacity(ordinals.len().saturating_mul(2));
        for &ordinal in ordinals {
            ordinals_le.extend_from_slice(&ordinal.to_le_bytes());
        }
        let run = Self {
            leaf_id,
            doc_base,
            row_count,
            doc_ids_le,
            ordinals_le,
            codes,
        };
        run.validate(encoding, dimension, u32::MAX)?;
        Ok(run)
    }

    fn validate(
        &self,
        encoding: ScannEncoding,
        dimension: u32,
        segment_docs: u32,
    ) -> ScannResult<()> {
        let rows = self.row_count as usize;
        if self.doc_ids_le.len() != rows.saturating_mul(4)
            || self.ordinals_le.len() != rows.saturating_mul(2)
            || self.codes.len() != encoding.leaf_code_bytes(dimension, rows)?
        {
            return Err(ScannFormatError::new(
                "ScaNN leaf run columns do not match its row count",
            ));
        }
        for chunk in self.doc_ids_le.chunks_exact(4) {
            let local = u32::from_le_bytes(chunk.try_into().unwrap());
            let effective = self
                .doc_base
                .checked_add(local)
                .ok_or_else(|| ScannFormatError::new("ScaNN document ID overflows u32"))?;
            if effective >= segment_docs {
                return Err(ScannFormatError::new(
                    "ScaNN leaf run document ID is outside its segment",
                ));
            }
        }
        Ok(())
    }
}

/// Segment-local leaf runs tied to one global trained artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScannSegmentPayload {
    pub artifact_id: u64,
    pub generation: u64,
    pub dimension: u32,
    pub encoding: ScannEncoding,
    pub num_leaves: u32,
    pub doc_count: u32,
    runs: Vec<ScannLeafRun>,
}

impl ScannSegmentPayload {
    pub fn new(
        artifact: &ScannTrainedArtifact,
        doc_count: u32,
        runs: Vec<ScannLeafRun>,
    ) -> ScannResult<Self> {
        Self::from_generation(
            &artifact.config,
            artifact.generation,
            artifact.artifact_id,
            doc_count,
            runs,
        )
    }

    /// Construct from the already-validated generation header without
    /// decoding the global centroid plane into an owned artifact.
    pub fn from_generation(
        config: &ScannConfig,
        generation: u64,
        artifact_id: u64,
        doc_count: u32,
        mut runs: Vec<ScannLeafRun>,
    ) -> ScannResult<Self> {
        runs.sort_by_key(|run| run.leaf_id);
        let payload = Self {
            artifact_id,
            generation,
            dimension: config.dimension,
            encoding: config.encoding,
            num_leaves: config.num_leaves,
            doc_count,
            runs,
        };
        payload.validate()?;
        Ok(payload)
    }

    pub fn runs(&self) -> &[ScannLeafRun] {
        &self.runs
    }

    pub fn validate_against(&self, artifact: &ScannTrainedArtifact) -> ScannResult<()> {
        artifact.validate()?;
        if self.artifact_id != artifact.artifact_id
            || self.generation != artifact.generation
            || self.dimension != artifact.config.dimension
            || self.encoding != artifact.config.encoding
            || self.num_leaves != artifact.config.num_leaves
        {
            return Err(ScannFormatError::new(
                "ScaNN segment payload does not match the global trained generation",
            ));
        }
        self.validate()
    }

    /// Merge contiguous source segments by rebasing leaf-run document bases.
    /// Code, ordinal, and relative-document buffers are moved as-is; no vector
    /// is decoded, reassigned, or re-encoded and no codebook is retrained.
    pub fn merge_contiguous(segments: impl IntoIterator<Item = Self>) -> ScannResult<Self> {
        let mut segments = segments.into_iter();
        let mut merged = segments
            .next()
            .ok_or_else(|| ScannFormatError::new("cannot merge zero ScaNN segments"))?;
        merged.validate()?;
        let mut next_doc_base = merged.doc_count;
        for mut source in segments {
            source.validate()?;
            merged.ensure_compatible(&source)?;
            for run in &mut source.runs {
                run.doc_base = run
                    .doc_base
                    .checked_add(next_doc_base)
                    .ok_or_else(|| ScannFormatError::new("merged ScaNN doc base exceeds u32"))?;
            }
            next_doc_base = next_doc_base
                .checked_add(source.doc_count)
                .ok_or_else(|| ScannFormatError::new("merged ScaNN document count exceeds u32"))?;
            merged.runs.append(&mut source.runs);
        }
        merged.doc_count = next_doc_base;
        // Stable leaf ordering keeps source segment order within a leaf while
        // moving only run descriptors (the large column allocations stay put).
        merged.runs.sort_by_key(|run| run.leaf_id);
        merged.validate()?;
        Ok(merged)
    }

    pub fn to_bytes(&self) -> ScannResult<Vec<u8>> {
        self.validate()?;
        let mut output = Vec::new();
        output.extend_from_slice(MAGIC);
        push_u16(&mut output, SCANN_SEGMENT_PAYLOAD_VERSION);
        push_u16(&mut output, 0);
        push_u64(&mut output, self.artifact_id);
        push_u64(&mut output, self.generation);
        push_u32(&mut output, self.dimension);
        output.push(self.encoding.tag());
        let (dimensions_per_block, bits_per_code) = self.encoding.parameters();
        output.push(bits_per_code);
        push_u16(&mut output, dimensions_per_block);
        push_u32(&mut output, self.num_leaves);
        push_u32(&mut output, self.doc_count);
        push_u32(
            &mut output,
            u32::try_from(self.runs.len())
                .map_err(|_| ScannFormatError::new("ScaNN run count exceeds u32"))?,
        );
        for run in &self.runs {
            push_u32(&mut output, run.leaf_id);
            push_u32(&mut output, run.doc_base);
            push_u32(&mut output, run.row_count);
            push_u64(&mut output, run.doc_ids_le.len() as u64);
            push_u64(&mut output, run.ordinals_le.len() as u64);
            push_u64(&mut output, run.codes.len() as u64);
        }
        for run in &self.runs {
            output.extend_from_slice(&run.doc_ids_le);
            output.extend_from_slice(&run.ordinals_le);
            output.extend_from_slice(&run.codes);
        }
        Ok(output)
    }

    pub fn from_bytes(bytes: &[u8]) -> ScannResult<Self> {
        let mut input = Input::new(bytes);
        if input.take(8)? != MAGIC {
            return Err(ScannFormatError::new("invalid ScaNN segment payload magic"));
        }
        let version = input.u16()?;
        if version != SCANN_SEGMENT_PAYLOAD_VERSION {
            return Err(ScannFormatError::new(format!(
                "unsupported ScaNN segment payload version {version}; reader supports {SCANN_SEGMENT_PAYLOAD_VERSION}"
            )));
        }
        if input.u16()? != 0 {
            return Err(ScannFormatError::new(
                "ScaNN segment reserved field is non-zero",
            ));
        }
        let artifact_id = input.u64()?;
        let generation = input.u64()?;
        let dimension = input.u32()?;
        let encoding_tag = input.u8()?;
        let bits_per_code = input.u8()?;
        let dimensions_per_block = input.u16()?;
        let encoding =
            ScannEncoding::from_parts(encoding_tag, dimensions_per_block, bits_per_code)?;
        let num_leaves = input.u32()?;
        let doc_count = input.u32()?;
        let run_count = input.u32()? as usize;
        if run_count > input.remaining() / 36 {
            return Err(ScannFormatError::new(
                "ScaNN segment run directory is truncated",
            ));
        }
        let mut directory = Vec::with_capacity(run_count);
        for _ in 0..run_count {
            directory.push((
                input.u32()?,
                input.u32()?,
                input.u32()?,
                input.usize()?,
                input.usize()?,
                input.usize()?,
            ));
        }
        let mut runs = Vec::with_capacity(run_count);
        for (leaf_id, doc_base, row_count, docs_len, ordinals_len, codes_len) in directory {
            runs.push(ScannLeafRun {
                leaf_id,
                doc_base,
                row_count,
                doc_ids_le: input.take(docs_len)?.to_vec(),
                ordinals_le: input.take(ordinals_len)?.to_vec(),
                codes: input.take(codes_len)?.to_vec(),
            });
        }
        if !input.is_empty() {
            return Err(ScannFormatError::new(
                "ScaNN segment payload has trailing bytes",
            ));
        }
        let payload = Self {
            artifact_id,
            generation,
            dimension,
            encoding,
            num_leaves,
            doc_count,
            runs,
        };
        payload.validate()?;
        Ok(payload)
    }

    fn ensure_compatible(&self, other: &Self) -> ScannResult<()> {
        if self.artifact_id != other.artifact_id
            || self.generation != other.generation
            || self.dimension != other.dimension
            || self.encoding != other.encoding
            || self.num_leaves != other.num_leaves
        {
            return Err(ScannFormatError::new(
                "cannot merge ScaNN segments from different trained generations",
            ));
        }
        Ok(())
    }

    fn validate(&self) -> ScannResult<()> {
        if self.artifact_id == 0 || self.generation == 0 || self.num_leaves == 0 {
            return Err(ScannFormatError::new(
                "invalid ScaNN segment compatibility metadata",
            ));
        }
        self.encoding.row_code_bytes(self.dimension)?;
        let mut previous_leaf = None;
        for run in &self.runs {
            if run.leaf_id >= self.num_leaves
                || previous_leaf.is_some_and(|leaf| leaf > run.leaf_id)
            {
                return Err(ScannFormatError::new(
                    "ScaNN leaf runs are out of range or unsorted",
                ));
            }
            previous_leaf = Some(run.leaf_id);
            run.validate(self.encoding, self.dimension, self.doc_count)?;
        }
        Ok(())
    }
}

fn push_u16(output: &mut Vec<u8>, value: u16) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn push_u32(output: &mut Vec<u8>, value: u32) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(output: &mut Vec<u8>, value: u64) {
    output.extend_from_slice(&value.to_le_bytes());
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
            .ok_or_else(|| ScannFormatError::new("ScaNN segment offset overflows"))?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or_else(|| ScannFormatError::new("truncated ScaNN segment payload"))?;
        self.offset = end;
        Ok(value)
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
            .map_err(|_| ScannFormatError::new("ScaNN segment length exceeds usize"))
    }

    fn is_empty(&self) -> bool {
        self.offset == self.bytes.len()
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.offset
    }
}

#[cfg(test)]
mod tests {
    use super::super::{ScannConfig, ScannRoutingLevel, ScannTrainedArtifact};
    use super::*;

    fn artifact(generation: u64) -> ScannTrainedArtifact {
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

    fn segment(artifact: &ScannTrainedArtifact, doc: u32, leaf: u32) -> ScannSegmentPayload {
        let run = ScannLeafRun::from_rows(
            leaf,
            0,
            &[doc],
            &[0],
            vec![doc as u8, 0],
            ScannEncoding::BinaryHamming,
            16,
        )
        .unwrap();
        ScannSegmentPayload::new(artifact, doc + 1, vec![run]).unwrap()
    }

    #[test]
    fn compatible_segment_merge_moves_code_buffers_and_rebases_only_run_metadata() {
        let artifact = artifact(9);
        let left = segment(&artifact, 0, 1);
        let right = segment(&artifact, 0, 1);
        let left_codes = left.runs()[0].codes.as_ptr();
        let right_codes = right.runs()[0].codes.as_ptr();
        let merged = ScannSegmentPayload::merge_contiguous([left, right]).unwrap();

        assert_eq!(merged.doc_count, 2);
        assert_eq!(merged.runs().len(), 2);
        assert_eq!(merged.runs()[0].doc_base, 0);
        assert_eq!(merged.runs()[1].doc_base, 1);
        assert_eq!(merged.runs()[0].codes.as_ptr(), left_codes);
        assert_eq!(merged.runs()[1].codes.as_ptr(), right_codes);
    }

    #[test]
    fn segment_merge_refuses_different_global_generations() {
        let first_artifact = artifact(10);
        let second_artifact = artifact(11);
        let first = segment(&first_artifact, 0, 0);
        let second = segment(&second_artifact, 0, 0);
        let error = ScannSegmentPayload::merge_contiguous([first, second]).unwrap_err();
        assert!(error.to_string().contains("different trained generations"));
    }

    #[test]
    fn segment_payload_round_trip_preserves_binary_leaf_runs() {
        let artifact = artifact(12);
        let segment = segment(&artifact, 0, 1);
        let bytes = segment.to_bytes().unwrap();
        let decoded = ScannSegmentPayload::from_bytes(&bytes).unwrap();
        assert_eq!(decoded, segment);
        decoded.validate_against(&artifact).unwrap();
    }

    #[test]
    fn segment_payload_rejects_a_future_version() {
        let artifact = artifact(13);
        let segment = segment(&artifact, 0, 0);
        let mut bytes = segment.to_bytes().unwrap();
        bytes[8..10].copy_from_slice(&(SCANN_SEGMENT_PAYLOAD_VERSION + 1).to_le_bytes());
        let error = ScannSegmentPayload::from_bytes(&bytes).unwrap_err();
        assert!(error.to_string().contains("unsupported"));
    }
}
