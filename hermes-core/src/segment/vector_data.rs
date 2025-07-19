//! Vector index data structures shared between builder and reader

use std::io;
use std::mem::size_of;

use crate::directories::{FileHandle, OwnedBytes};
use crate::dsl::DenseVectorQuantization;
use crate::segment::format::{DOC_ID_ENTRY_SIZE, FLAT_BINARY_HEADER_SIZE, FLAT_BINARY_MAGIC};
use crate::structures::simd::{batch_f32_to_f16, batch_f32_to_u8, f16_to_f32, u8_to_f32};

/// Dequantize raw bytes to f32 based on storage quantization.
///
/// `raw` is the quantized byte slice, `out` receives the f32 values.
/// `num_floats` is the number of f32 values to produce (= num_vectors × dim).
/// Data-first file layout guarantees alignment for f32/f16 access.
#[inline]
pub fn dequantize_raw(
    raw: &[u8],
    quant: DenseVectorQuantization,
    num_floats: usize,
    out: &mut [f32],
) -> io::Result<()> {
    if out.len() < num_floats {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "dequantization output is too short: need {num_floats} floats, got {}",
                out.len()
            ),
        ));
    }

    let element_size = match quant {
        DenseVectorQuantization::F32 => size_of::<f32>(),
        DenseVectorQuantization::F16 => size_of::<u16>(),
        DenseVectorQuantization::UInt8 => size_of::<u8>(),
        DenseVectorQuantization::Binary => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "binary vectors cannot be dequantized to f32",
            ));
        }
    };
    let expected_bytes = num_floats.checked_mul(element_size).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "dequantization byte length overflows usize",
        )
    })?;
    if raw.len() != expected_bytes {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "dequantization input length mismatch: need {expected_bytes} bytes, got {}",
                raw.len()
            ),
        ));
    }

    match quant {
        DenseVectorQuantization::F32 => {
            if expected_bytes > 0
                && !(raw.as_ptr() as usize).is_multiple_of(std::mem::align_of::<f32>())
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "f32 vector data is not 4-byte aligned",
                ));
            }
            out[..num_floats].copy_from_slice(unsafe {
                // Safety: the exact byte length and f32 alignment were checked above.
                std::slice::from_raw_parts(raw.as_ptr() as *const f32, num_floats)
            });
        }
        DenseVectorQuantization::F16 => {
            if expected_bytes > 0
                && !(raw.as_ptr() as usize).is_multiple_of(std::mem::align_of::<u16>())
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "f16 vector data is not 2-byte aligned",
                ));
            }
            let f16_slice = unsafe {
                // Safety: the exact byte length and u16 alignment were checked above.
                std::slice::from_raw_parts(raw.as_ptr() as *const u16, num_floats)
            };
            for (i, &h) in f16_slice.iter().enumerate() {
                out[i] = f16_to_f32(h);
            }
        }
        DenseVectorQuantization::UInt8 => {
            for (i, &b) in raw.iter().enumerate() {
                out[i] = u8_to_f32(b);
            }
        }
        DenseVectorQuantization::Binary => unreachable!("validated above"),
    }
    Ok(())
}

/// Flat vector binary format helpers for writing.
///
/// Binary format v3:
/// ```text
/// [magic(u32)][dim(u32)][num_vectors(u32)][quant_type(u8)][padding(3)]
/// [vectors: N×dim×element_size]
/// [doc_ids: N×(u32+u16)]
/// ```
///
/// `element_size` is determined by `quant_type`: f32=4, f16=2, uint8=1.
/// Reading is handled by [`LazyFlatVectorData`] which loads only doc_ids into memory
/// and accesses vector data lazily via mmap-backed range reads.
pub struct FlatVectorData;

impl FlatVectorData {
    fn validate_shape(
        dim: usize,
        num_vectors: usize,
        quant: DenseVectorQuantization,
    ) -> io::Result<usize> {
        if dim == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "flat vector dimension must be greater than zero",
            ));
        }
        if quant == DenseVectorQuantization::Binary && !dim.is_multiple_of(8) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("binary flat vector dimension must be a multiple of 8, got {dim}"),
            ));
        }
        u32::try_from(dim).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("flat vector dimension {dim} exceeds u32::MAX"),
            )
        })?;
        u32::try_from(num_vectors).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("flat vector count {num_vectors} exceeds u32::MAX"),
            )
        })?;

        match quant {
            DenseVectorQuantization::Binary => dim.checked_add(7).map(|bits| bits / 8),
            _ => dim.checked_mul(quant.element_size()),
        }
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "flat vector byte size overflows usize",
            )
        })
    }

    fn validate_doc_ids(doc_ids: &[(u32, u16)]) -> io::Result<()> {
        if let Some(pair) = doc_ids.windows(2).find(|pair| pair[0] >= pair[1]) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "flat vector doc map must be strictly sorted by (doc_id, ordinal), found {:?} before {:?}",
                    pair[0], pair[1]
                ),
            ));
        }
        Ok(())
    }

    /// Validate a dense writer input completely before any bytes are emitted.
    /// Returns the exact serialized size on success.
    pub(crate) fn validate_dense_input(
        dim: usize,
        flat_vectors: &[f32],
        doc_ids: &[(u32, u16)],
        quant: DenseVectorQuantization,
    ) -> io::Result<usize> {
        if quant == DenseVectorQuantization::Binary {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "binary quantization must use serialize_binary_from_bits_streaming",
            ));
        }
        let num_vectors = doc_ids.len();
        let expected_floats = num_vectors.checked_mul(dim).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "flat f32 vector count overflows usize",
            )
        })?;
        if flat_vectors.len() != expected_floats {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "flat vector input has {} floats, expected {num_vectors} x {dim} = {expected_floats}",
                    flat_vectors.len()
                ),
            ));
        }
        Self::validate_doc_ids(doc_ids)?;
        Self::serialized_binary_size(dim, num_vectors, quant)
    }

    /// Validate a packed-binary writer input completely before any bytes are
    /// emitted. Returns the exact serialized size on success.
    pub(crate) fn validate_binary_input(
        dim_bits: usize,
        packed_vectors: &[u8],
        doc_ids: &[(u32, u16)],
    ) -> io::Result<usize> {
        let num_vectors = doc_ids.len();
        let byte_len =
            Self::validate_shape(dim_bits, num_vectors, DenseVectorQuantization::Binary)?;
        let expected_bytes = num_vectors.checked_mul(byte_len).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "packed binary vector size overflows usize",
            )
        })?;
        if packed_vectors.len() != expected_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "packed binary input has {} bytes, expected {num_vectors} x {byte_len} = {expected_bytes}",
                    packed_vectors.len()
                ),
            ));
        }
        Self::validate_doc_ids(doc_ids)?;
        Self::serialized_binary_size(dim_bits, num_vectors, DenseVectorQuantization::Binary)
    }

    /// Write the binary header to a writer.
    pub fn write_binary_header(
        dim: usize,
        num_vectors: usize,
        quant: DenseVectorQuantization,
        writer: &mut dyn std::io::Write,
    ) -> std::io::Result<()> {
        Self::validate_shape(dim, num_vectors, quant)?;
        let dim = u32::try_from(dim).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "flat vector dimension exceeds u32",
            )
        })?;
        let num_vectors = u32::try_from(num_vectors).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidInput, "flat vector count exceeds u32")
        })?;
        writer.write_all(&FLAT_BINARY_MAGIC.to_le_bytes())?;
        writer.write_all(&dim.to_le_bytes())?;
        writer.write_all(&num_vectors.to_le_bytes())?;
        writer.write_all(&[quant.tag(), 0, 0, 0])?; // quant_type + 3 bytes padding
        Ok(())
    }

    /// Compute the serialized size without actually serializing.
    pub fn serialized_binary_size(
        dim: usize,
        num_vectors: usize,
        quant: DenseVectorQuantization,
    ) -> io::Result<usize> {
        let bytes_per_vector = Self::validate_shape(dim, num_vectors, quant)?;
        let vector_bytes = num_vectors.checked_mul(bytes_per_vector).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "flat vector payload size overflows usize",
            )
        })?;
        let doc_id_bytes = num_vectors.checked_mul(DOC_ID_ENTRY_SIZE).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "flat vector doc-map size overflows usize",
            )
        })?;
        FLAT_BINARY_HEADER_SIZE
            .checked_add(vector_bytes)
            .and_then(|size| size.checked_add(doc_id_bytes))
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "flat vector serialized size overflows usize",
                )
            })
    }

    /// Stream from flat f32 storage to a writer, quantizing on write.
    ///
    /// `flat_vectors` is contiguous storage of dim*n f32 floats.
    /// Vectors are quantized to the specified format before writing.
    pub fn serialize_binary_from_flat_streaming(
        dim: usize,
        flat_vectors: &[f32],
        doc_ids: &[(u32, u16)],
        quant: DenseVectorQuantization,
        writer: &mut dyn std::io::Write,
    ) -> std::io::Result<()> {
        Self::validate_dense_input(dim, flat_vectors, doc_ids, quant)?;
        let num_vectors = doc_ids.len();
        Self::write_binary_header(dim, num_vectors, quant, writer)?;

        match quant {
            DenseVectorQuantization::F32 => {
                let bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(
                        flat_vectors.as_ptr() as *const u8,
                        std::mem::size_of_val(flat_vectors),
                    )
                };
                writer.write_all(bytes)?;
            }
            DenseVectorQuantization::F16 => {
                let mut buf = vec![0u16; dim];
                for v in flat_vectors.chunks_exact(dim) {
                    batch_f32_to_f16(v, &mut buf);
                    let bytes: &[u8] =
                        unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, dim * 2) };
                    writer.write_all(bytes)?;
                }
            }
            DenseVectorQuantization::UInt8 => {
                let mut buf = vec![0u8; dim];
                for v in flat_vectors.chunks_exact(dim) {
                    batch_f32_to_u8(v, &mut buf);
                    writer.write_all(&buf)?;
                }
            }
            DenseVectorQuantization::Binary => unreachable!("validated above"),
        }

        for &(doc_id, ordinal) in doc_ids {
            writer.write_all(&doc_id.to_le_bytes())?;
            writer.write_all(&ordinal.to_le_bytes())?;
        }

        Ok(())
    }

    /// Stream packed binary vectors (pre-packed bytes) to a writer.
    ///
    /// `packed_vectors` is contiguous storage of num_vectors * byte_len bytes.
    /// `dim_bits` is the number of bits (dimensions).
    pub fn serialize_binary_from_bits_streaming(
        dim_bits: usize,
        packed_vectors: &[u8],
        doc_ids: &[(u32, u16)],
        writer: &mut dyn std::io::Write,
    ) -> std::io::Result<()> {
        Self::validate_binary_input(dim_bits, packed_vectors, doc_ids)?;
        let num_vectors = doc_ids.len();

        Self::write_binary_header(
            dim_bits,
            num_vectors,
            DenseVectorQuantization::Binary,
            writer,
        )?;
        writer.write_all(packed_vectors)?;

        for &(doc_id, ordinal) in doc_ids {
            writer.write_all(&doc_id.to_le_bytes())?;
            writer.write_all(&ordinal.to_le_bytes())?;
        }

        Ok(())
    }

    /// Write raw pre-quantized vector bytes to a writer (for merger streaming).
    ///
    /// `raw_bytes` is already in the target quantized format.
    pub fn write_raw_vector_bytes(
        raw_bytes: &[u8],
        writer: &mut dyn std::io::Write,
    ) -> std::io::Result<()> {
        writer.write_all(raw_bytes)
    }
}

/// Lazy flat vector data — zero-copy doc_id index, vectors via range reads.
///
/// The doc_id index is kept as `OwnedBytes` (mmap-backed, zero heap copy).
/// Vector data stays on disk and is accessed via mmap-backed range reads.
/// Element size depends on quantization: f32=4, f16=2, uint8=1 bytes/dim.
///
/// Used for:
/// - Brute-force search (batched scoring with native-precision SIMD)
/// - Reranking (read individual vectors by doc_id via binary search)
/// - doc() hydration (dequantize to f32 for stored documents)
/// - Merge streaming (chunked raw vector bytes + doc_id iteration)
#[derive(Debug, Clone)]
pub struct LazyFlatVectorData {
    /// Vector dimension
    pub dim: usize,
    /// Total number of vectors
    pub num_vectors: usize,
    /// Number of distinct document IDs represented in the flat vector map.
    num_docs_with_vectors: usize,
    /// Storage quantization type
    pub quantization: DenseVectorQuantization,
    /// Zero-copy doc_id index: packed [u32_le doc_id + u16_le ordinal] × num_vectors
    doc_ids_bytes: OwnedBytes,
    /// File handle for this field's flat data region in the .vectors file
    handle: FileHandle,
    /// Byte offset within handle where raw vector data starts (after header)
    vectors_offset: u64,
    /// Bytes per vector in storage (cached: Binary = ceil(dim/8), else dim * element_size)
    vbs: usize,
    /// Exact byte length of the raw vector region, validated when opening.
    vectors_byte_len: u64,
}

impl LazyFlatVectorData {
    /// Open from a lazy file slice pointing to the flat binary data region.
    ///
    /// Reads header (16 bytes) + doc_ids (~6 bytes/vector) into memory.
    /// Vector data stays lazy on disk.
    pub async fn open(handle: FileHandle) -> io::Result<Self> {
        Self::open_with_doc_limit(handle, None).await
    }

    /// Open flat vectors while also validating every referenced document ID.
    ///
    /// Segment readers pass their durable `num_docs` here. Keeping the public
    /// `open` entry point is useful for standalone flat payloads and tests that
    /// do not have segment metadata available.
    pub(crate) async fn open_with_doc_limit(
        handle: FileHandle,
        total_docs: Option<u32>,
    ) -> io::Result<Self> {
        let header_len = u64::try_from(FLAT_BINARY_HEADER_SIZE).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "flat vector header size does not fit in u64",
            )
        })?;
        if handle.len() < header_len {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "flat vector payload is {} bytes, shorter than its {FLAT_BINARY_HEADER_SIZE}-byte header",
                    handle.len()
                ),
            ));
        }

        // Read header: magic(4) + dim(4) + num_vectors(4) + quant_type(1) + pad(3) = 16 bytes
        let header = handle.read_bytes_range(0..header_len).await?;
        if header.len() != FLAT_BINARY_HEADER_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "flat vector header read returned {} bytes, expected {FLAT_BINARY_HEADER_SIZE}",
                    header.len()
                ),
            ));
        }
        let hdr = header.as_slice();

        let magic = u32::from_le_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]);
        if magic != FLAT_BINARY_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid FlatVectorData binary magic",
            ));
        }

        let dim = u32::from_le_bytes([hdr[4], hdr[5], hdr[6], hdr[7]]) as usize;
        let num_vectors = u32::from_le_bytes([hdr[8], hdr[9], hdr[10], hdr[11]]) as usize;
        let quantization = DenseVectorQuantization::from_tag(hdr[12]).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unknown quantization tag: {}", hdr[12]),
            )
        })?;
        if hdr[13..] != [0, 0, 0] {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "flat vector header has non-zero reserved bytes",
            ));
        }

        // Read doc_ids section as zero-copy OwnedBytes (6 bytes per vector)
        let vbs =
            FlatVectorData::validate_shape(dim, num_vectors, quantization).map_err(|error| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid flat vector shape: {error}"),
                )
            })?;
        let vectors_byte_len_usize = num_vectors.checked_mul(vbs).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "flat vector payload size overflows usize",
            )
        })?;
        let doc_ids_byte_len_usize =
            num_vectors.checked_mul(DOC_ID_ENTRY_SIZE).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "flat vector doc-map size overflows usize",
                )
            })?;
        let expected_len_usize = FLAT_BINARY_HEADER_SIZE
            .checked_add(vectors_byte_len_usize)
            .and_then(|size| size.checked_add(doc_ids_byte_len_usize))
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "flat vector serialized size overflows usize",
                )
            })?;
        let expected_len = u64::try_from(expected_len_usize).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "flat vector serialized size does not fit in u64",
            )
        })?;
        if handle.len() != expected_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "flat vector payload has {} bytes, expected exactly {expected_len}",
                    handle.len()
                ),
            ));
        }

        let vectors_byte_len = u64::try_from(vectors_byte_len_usize).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "flat vector payload size does not fit in u64",
            )
        })?;
        let doc_ids_byte_len = u64::try_from(doc_ids_byte_len_usize).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "flat vector doc-map size does not fit in u64",
            )
        })?;
        let doc_ids_start = header_len.checked_add(vectors_byte_len).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "flat vector doc-map offset overflows u64",
            )
        })?;
        let doc_ids_end = doc_ids_start.checked_add(doc_ids_byte_len).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "flat vector doc-map range overflows u64",
            )
        })?;

        let doc_ids_bytes = handle.read_bytes_range(doc_ids_start..doc_ids_end).await?;
        if doc_ids_bytes.len() != doc_ids_byte_len_usize {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "flat vector doc-map read returned {} bytes, expected {doc_ids_byte_len_usize}",
                    doc_ids_bytes.len()
                ),
            ));
        }

        let mut previous = None;
        let mut num_docs_with_vectors = 0usize;
        for entry in doc_ids_bytes.as_slice().chunks_exact(DOC_ID_ENTRY_SIZE) {
            let doc_id = u32::from_le_bytes([entry[0], entry[1], entry[2], entry[3]]);
            let ordinal = u16::from_le_bytes([entry[4], entry[5]]);
            let current = (doc_id, ordinal);
            if let Some(previous) = previous
                && previous >= current
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "flat vector doc map must be strictly sorted by (doc_id, ordinal), found {previous:?} before {current:?}"
                    ),
                ));
            }
            if let Some(limit) = total_docs
                && doc_id >= limit
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "flat vector doc map references document {doc_id}, but segment contains only {} documents",
                        limit
                    ),
                ));
            }
            if previous.is_none_or(|(previous_doc_id, _)| previous_doc_id != doc_id) {
                num_docs_with_vectors = num_docs_with_vectors.checked_add(1).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "flat vector distinct-document count overflows usize",
                    )
                })?;
            }
            previous = Some(current);
        }

        Ok(Self {
            dim,
            num_vectors,
            num_docs_with_vectors,
            quantization,
            doc_ids_bytes,
            handle,
            vectors_offset: header_len,
            vbs,
            vectors_byte_len,
        })
    }

    fn checked_vector_range(
        &self,
        start_idx: usize,
        count: usize,
    ) -> io::Result<(std::ops::Range<u64>, usize)> {
        let end_idx = start_idx.checked_add(count).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "flat vector index range overflows usize",
            )
        })?;
        if end_idx > self.num_vectors {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "flat vector range {start_idx}..{end_idx} exceeds {} vectors",
                    self.num_vectors
                ),
            ));
        }

        let relative_offset = start_idx.checked_mul(self.vbs).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "flat vector byte offset overflows usize",
            )
        })?;
        let byte_len = count.checked_mul(self.vbs).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "flat vector byte length overflows usize",
            )
        })?;
        let relative_offset = u64::try_from(relative_offset).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "flat vector byte offset does not fit in u64",
            )
        })?;
        let byte_len_u64 = u64::try_from(byte_len).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "flat vector byte length does not fit in u64",
            )
        })?;
        let start = self
            .vectors_offset
            .checked_add(relative_offset)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "flat vector byte offset overflows u64",
                )
            })?;
        let end = start.checked_add(byte_len_u64).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "flat vector byte range overflows u64",
            )
        })?;
        let vectors_end = self
            .vectors_offset
            .checked_add(self.vectors_byte_len)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "flat vector payload boundary overflows u64",
                )
            })?;
        if end > vectors_end || end > self.handle.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "flat vector byte range {start}..{end} exceeds payload boundary {vectors_end}"
                ),
            ));
        }
        Ok((start..end, byte_len))
    }

    /// Pin the doc-id map (priority 3: every rerank / top-k resolution
    /// binary-searches it).
    #[cfg(feature = "native")]
    pub(crate) fn pin_doc_ids(
        &mut self,
        mode: crate::segment::pin::PinMode,
        remaining: &mut u64,
        report: &mut crate::segment::pin::PinReport,
    ) {
        crate::segment::pin::pin_section(
            &mut self.doc_ids_bytes,
            "flat doc_ids",
            mode,
            remaining,
            report,
        );
    }

    /// Advise the kernel that vector data will be accessed at random offsets.
    ///
    /// Disables kernel readahead for the raw vector region. Rerank reads
    /// scattered ~vbs-sized records; default readahead pulls in 128KB per
    /// fault, evicting useful pages in memory-bound environments.
    /// No-op for non-mmap (RAM, HTTP) backing.
    #[cfg(feature = "native")]
    pub fn advise_random_access(&self) {
        let Some(vectors_end) = self.vectors_offset.checked_add(self.vectors_byte_len) else {
            return;
        };
        self.handle
            .madvise_range(self.vectors_offset..vectors_end, libc::MADV_RANDOM);
    }

    /// Prefetch the pages backing a sorted set of vector indexes (`MADV_WILLNEED`).
    ///
    /// Coalesces adjacent candidates into ranges so the kernel can overlap
    /// the page-ins instead of taking one synchronous major fault per vector
    /// during the rerank read loop. Indexes must be yielded in ascending order.
    /// No-op for non-mmap backing.
    #[cfg(feature = "native")]
    pub fn prefetch_vectors(&self, sorted_flat_indexes: impl IntoIterator<Item = usize>) {
        /// Gap (in bytes) below which two candidate ranges are merged into one advice call.
        const COALESCE_GAP: u64 = 64 * 1024;
        let mut ranges = sorted_flat_indexes.into_iter().filter_map(|idx| {
            self.checked_vector_range(idx, 1)
                .ok()
                .map(|(range, _)| range)
        });
        let Some(first) = ranges.next() else {
            return;
        };
        let mut run_start = first.start;
        let mut run_end = first.end;
        for range in ranges {
            if range.start <= run_end.saturating_add(COALESCE_GAP) {
                run_end = run_end.max(range.end);
            } else {
                self.handle
                    .madvise_range(run_start..run_end, libc::MADV_WILLNEED);
                run_start = range.start;
                run_end = range.end;
            }
        }
        self.handle
            .madvise_range(run_start..run_end, libc::MADV_WILLNEED);
    }

    /// Read a single vector by index, dequantized to f32.
    ///
    /// `out` must have length >= `self.dim`. Returns `Ok(())` on success.
    /// Used for ANN training and doc() hydration where f32 is needed.
    pub async fn read_vector_into(&self, idx: usize, out: &mut [f32]) -> io::Result<()> {
        if out.len() < self.dim {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "flat vector output is too short: need {} floats, got {}",
                    self.dim,
                    out.len()
                ),
            ));
        }
        let bytes = self.read_vectors_batch(idx, 1).await?;
        dequantize_raw(bytes.as_slice(), self.quantization, self.dim, out)
    }

    /// Read a single vector by index, dequantized to f32 (allocates a new Vec<f32>).
    pub async fn get_vector(&self, idx: usize) -> io::Result<Vec<f32>> {
        let mut vector = vec![0f32; self.dim];
        self.read_vector_into(idx, &mut vector).await?;
        Ok(vector)
    }

    /// Read a single vector's raw bytes (no dequantization) into a caller-provided buffer.
    ///
    /// `out` must have length >= `self.vector_byte_size()`.
    /// Used for native-precision reranking where raw quantized bytes are scored directly.
    pub async fn read_vector_raw_into(&self, idx: usize, out: &mut [u8]) -> io::Result<()> {
        self.read_vector_prefix_raw_into(idx, self.vector_byte_size(), out)
            .await
    }

    /// Read a prefix of one vector's raw bytes into a caller-provided buffer.
    ///
    /// This is used by Matryoshka scoring to avoid reading the unused tail of
    /// a vector. Unlike the old full-vector boundary, all caller-controlled
    /// sizes and offset arithmetic are checked in release builds.
    pub async fn read_vector_prefix_raw_into(
        &self,
        idx: usize,
        prefix_byte_len: usize,
        out: &mut [u8],
    ) -> io::Result<()> {
        let vbs = self.vector_byte_size();
        if prefix_byte_len > vbs {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "vector prefix is {prefix_byte_len} bytes, but a vector has only {vbs} bytes"
                ),
            ));
        }
        if out.len() < prefix_byte_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "vector prefix output is too short: need {prefix_byte_len} bytes, got {}",
                    out.len()
                ),
            ));
        }
        let (full_range, _) = self.checked_vector_range(idx, 1)?;
        if prefix_byte_len == 0 {
            return Ok(());
        }
        let prefix_byte_len_u64 = u64::try_from(prefix_byte_len).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "vector prefix length does not fit in u64",
            )
        })?;
        let byte_end = full_range
            .start
            .checked_add(prefix_byte_len_u64)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "vector byte range overflows u64",
                )
            })?;
        let bytes = self
            .handle
            .read_bytes_range(full_range.start..byte_end)
            .await?;
        if bytes.len() != prefix_byte_len {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "vector prefix read returned {} bytes, expected {prefix_byte_len}",
                    bytes.len()
                ),
            ));
        }
        out[..prefix_byte_len].copy_from_slice(bytes.as_slice());
        Ok(())
    }

    /// Read a contiguous batch of raw quantized bytes by index range.
    ///
    /// Returns raw bytes for vectors `[start_idx..start_idx+count)`.
    /// Bytes are in native quantized format — pass to `batch_cosine_scores_f16/u8`
    /// or `batch_cosine_scores` (for f32) for scoring.
    pub async fn read_vectors_batch(
        &self,
        start_idx: usize,
        count: usize,
    ) -> io::Result<OwnedBytes> {
        let (range, expected_len) = self.checked_vector_range(start_idx, count)?;
        let bytes = self.handle.read_bytes_range(range).await?;
        if bytes.len() != expected_len {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "flat vector batch read returned {} bytes, expected {expected_len}",
                    bytes.len()
                ),
            ));
        }
        Ok(bytes)
    }

    /// Synchronous read of a single vector's raw bytes.
    #[cfg(feature = "sync")]
    pub fn read_vector_raw_into_sync(&self, idx: usize, out: &mut [u8]) -> io::Result<()> {
        let vbs = self.vector_byte_size();
        if out.len() < vbs {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "flat vector output is too short: need {vbs} bytes, got {}",
                    out.len()
                ),
            ));
        }
        let bytes = self.read_vectors_batch_sync(idx, 1)?;
        out[..vbs].copy_from_slice(bytes.as_slice());
        Ok(())
    }

    /// Synchronous batch read of raw quantized bytes.
    #[cfg(feature = "sync")]
    pub fn read_vectors_batch_sync(
        &self,
        start_idx: usize,
        count: usize,
    ) -> io::Result<OwnedBytes> {
        let (range, expected_len) = self.checked_vector_range(start_idx, count)?;
        let bytes = self.handle.read_bytes_range_sync(range)?;
        if bytes.len() != expected_len {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "flat vector batch read returned {} bytes, expected {expected_len}",
                    bytes.len()
                ),
            ));
        }
        Ok(bytes)
    }

    /// Find flat index range for a given doc_id (non-allocating).
    ///
    /// Returns `(start_index, count)` — the flat vector index range for this doc_id.
    /// Use `get_doc_id(start + i)` for `i in 0..count` to read individual entries.
    /// More efficient than `flat_indexes_for_doc` as it avoids Vec allocation.
    pub fn flat_indexes_for_doc_range(&self, doc_id: u32) -> (usize, usize) {
        let n = self.num_vectors;
        let start = {
            let mut lo = 0usize;
            let mut hi = n;
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                if self.doc_id_at(mid) < doc_id {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            lo
        };
        let mut count = 0;
        let mut i = start;
        while i < n && self.doc_id_at(i) == doc_id {
            count += 1;
            i += 1;
        }
        (start, count)
    }

    /// Find flat indexes for a given doc_id via binary search on sorted doc_ids.
    ///
    /// doc_ids are sorted by (doc_id, ordinal) — segment builder adds docs
    /// sequentially. Binary search runs directly on zero-copy mmap bytes.
    ///
    /// Returns `(start_index, entries)` where start_index is the flat vector index.
    pub fn flat_indexes_for_doc(&self, doc_id: u32) -> (usize, Vec<(u32, u16)>) {
        let n = self.num_vectors;
        // Binary search: find first entry where doc_id >= target
        let start = {
            let mut lo = 0usize;
            let mut hi = n;
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                if self.doc_id_at(mid) < doc_id {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            lo
        };
        // Collect entries with matching doc_id
        let mut entries = Vec::new();
        let mut i = start;
        while i < n {
            let (did, ord) = self.get_doc_id(i);
            if did != doc_id {
                break;
            }
            entries.push((did, ord));
            i += 1;
        }
        (start, entries)
    }

    /// Read doc_id at index from raw bytes (no ordinal).
    #[inline]
    fn doc_id_at(&self, idx: usize) -> u32 {
        let off = idx * DOC_ID_ENTRY_SIZE;
        let d = &self.doc_ids_bytes[off..];
        u32::from_le_bytes([d[0], d[1], d[2], d[3]])
    }

    /// Get doc_id and ordinal at index (parsed from zero-copy mmap bytes).
    #[inline]
    pub fn get_doc_id(&self, idx: usize) -> (u32, u16) {
        let off = idx * DOC_ID_ENTRY_SIZE;
        let d = &self.doc_ids_bytes[off..];
        let doc_id = u32::from_le_bytes([d[0], d[1], d[2], d[3]]);
        let ordinal = u16::from_le_bytes([d[4], d[5]]);
        (doc_id, ordinal)
    }

    /// Bytes per vector in storage (cached).
    #[inline]
    pub fn vector_byte_size(&self) -> usize {
        self.vbs
    }

    /// Number of distinct documents that have at least one vector in this field.
    #[inline]
    pub fn num_docs_with_vectors(&self) -> usize {
        self.num_docs_with_vectors
    }

    /// Total byte length of raw vector data (for chunked merger streaming).
    pub fn vector_bytes_len(&self) -> u64 {
        self.vectors_byte_len
    }

    /// Byte offset where vector data starts (for direct handle access in merger).
    pub fn vectors_byte_offset(&self) -> u64 {
        self.vectors_offset
    }

    /// Access the underlying file handle (for chunked byte-range reads in merger).
    pub fn handle(&self) -> &FileHandle {
        &self.handle
    }

    /// Estimated memory usage — doc_ids are mmap-backed (only Arc overhead).
    pub fn estimated_memory_bytes(&self) -> usize {
        size_of::<Self>() + size_of::<OwnedBytes>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dequantize_raw_accepts_valid_storage_formats() {
        let f32_values = [1.25f32, -2.5];
        let f32_bytes = unsafe {
            // Safety: viewing an initialized f32 array as bytes is always valid.
            std::slice::from_raw_parts(
                f32_values.as_ptr().cast::<u8>(),
                std::mem::size_of_val(&f32_values),
            )
        };
        let mut out = [0.0; 2];
        dequantize_raw(f32_bytes, DenseVectorQuantization::F32, 2, &mut out).unwrap();
        assert_eq!(out, f32_values);

        let f16_values = [0x3c00u16, 0xc000u16]; // 1.0, -2.0
        let f16_bytes = unsafe {
            // Safety: viewing an initialized u16 array as bytes is always valid.
            std::slice::from_raw_parts(
                f16_values.as_ptr().cast::<u8>(),
                std::mem::size_of_val(&f16_values),
            )
        };
        dequantize_raw(f16_bytes, DenseVectorQuantization::F16, 2, &mut out).unwrap();
        assert_eq!(out, [1.0, -2.0]);

        dequantize_raw(&[0, u8::MAX], DenseVectorQuantization::UInt8, 2, &mut out).unwrap();
        assert_eq!(out, [u8_to_f32(0), u8_to_f32(u8::MAX)]);
    }

    #[test]
    fn dequantize_raw_rejects_invalid_lengths_and_binary_storage() {
        let mut out = [0.0; 2];
        assert_eq!(
            dequantize_raw(&[0; 7], DenseVectorQuantization::F32, 2, &mut out)
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidData
        );
        assert_eq!(
            dequantize_raw(&[0; 8], DenseVectorQuantization::F32, 2, &mut out[..1])
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidInput
        );
        assert_eq!(
            dequantize_raw(&[], DenseVectorQuantization::Binary, 0, &mut [])
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidInput
        );
    }

    #[test]
    fn dequantize_raw_rejects_misaligned_typed_storage() {
        let storage = [0u8; 9];
        let offset = if (storage.as_ptr() as usize).is_multiple_of(4) {
            1
        } else {
            0
        };
        let raw = &storage[offset..offset + 8];
        assert!(!(raw.as_ptr() as usize).is_multiple_of(4));

        let mut out = [0.0; 2];
        assert_eq!(
            dequantize_raw(raw, DenseVectorQuantization::F32, 2, &mut out)
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidData
        );
    }

    #[test]
    fn flat_vector_writers_reject_inconsistent_shapes_and_doc_maps() {
        let mut encoded = Vec::new();
        assert!(
            FlatVectorData::serialize_binary_from_flat_streaming(
                0,
                &[],
                &[],
                DenseVectorQuantization::F32,
                &mut encoded,
            )
            .is_err()
        );
        assert!(encoded.is_empty());

        assert!(
            FlatVectorData::serialize_binary_from_flat_streaming(
                2,
                &[1.0],
                &[(0, 0)],
                DenseVectorQuantization::F32,
                &mut encoded,
            )
            .is_err()
        );
        assert!(encoded.is_empty());

        assert!(
            FlatVectorData::serialize_binary_from_flat_streaming(
                1,
                &[1.0],
                &[(0, 0)],
                DenseVectorQuantization::Binary,
                &mut encoded,
            )
            .is_err()
        );
        assert!(encoded.is_empty());

        assert!(
            FlatVectorData::serialize_binary_from_flat_streaming(
                1,
                &[1.0, 2.0],
                &[(1, 0), (0, 0)],
                DenseVectorQuantization::F32,
                &mut encoded,
            )
            .is_err()
        );
        assert!(encoded.is_empty());

        assert!(
            FlatVectorData::serialize_binary_from_flat_streaming(
                1,
                &[1.0, 2.0],
                &[(0, 0), (0, 0)],
                DenseVectorQuantization::F32,
                &mut encoded,
            )
            .is_err()
        );
        assert!(encoded.is_empty());

        assert!(
            FlatVectorData::serialize_binary_from_bits_streaming(7, &[0], &[(0, 0)], &mut encoded,)
                .is_err()
        );
        assert!(encoded.is_empty());

        assert!(
            FlatVectorData::serialize_binary_from_bits_streaming(8, &[], &[(0, 0)], &mut encoded,)
                .is_err()
        );
        assert!(encoded.is_empty());

        let vectors = [1.0f32, 2.0, 3.0, 4.0];
        let doc_ids = [(0, 0), (1, 0)];
        FlatVectorData::serialize_binary_from_flat_streaming(
            2,
            &vectors,
            &doc_ids,
            DenseVectorQuantization::F32,
            &mut encoded,
        )
        .unwrap();
        assert_eq!(
            encoded.len(),
            FlatVectorData::serialized_binary_size(2, 2, DenseVectorQuantization::F32).unwrap()
        );
    }

    fn encoded_two_vector_payload() -> Vec<u8> {
        let mut encoded = Vec::new();
        FlatVectorData::serialize_binary_from_flat_streaming(
            2,
            &[1.0, 2.0, 3.0, 4.0],
            &[(0, 0), (1, 0)],
            DenseVectorQuantization::F32,
            &mut encoded,
        )
        .unwrap();
        encoded
    }

    #[tokio::test]
    async fn flat_vector_open_rejects_corrupt_layout_and_doc_map() {
        let valid = encoded_two_vector_payload();

        let mut multi_value = Vec::new();
        FlatVectorData::serialize_binary_from_flat_streaming(
            1,
            &[1.0, 2.0, 3.0],
            &[(0, 0), (0, 1), (2, 0)],
            DenseVectorQuantization::F32,
            &mut multi_value,
        )
        .unwrap();
        let multi_value = LazyFlatVectorData::open_with_doc_limit(
            FileHandle::from_bytes(OwnedBytes::new(multi_value)),
            Some(3),
        )
        .await
        .unwrap();
        assert_eq!(multi_value.num_docs_with_vectors(), 2);

        let mut trailing = valid.clone();
        trailing.push(0);
        assert!(
            LazyFlatVectorData::open(FileHandle::from_bytes(OwnedBytes::new(trailing)))
                .await
                .is_err()
        );

        let mut truncated = valid.clone();
        truncated.pop();
        assert!(
            LazyFlatVectorData::open(FileHandle::from_bytes(OwnedBytes::new(truncated)))
                .await
                .is_err()
        );

        let mut reserved = valid.clone();
        reserved[13] = 1;
        assert!(
            LazyFlatVectorData::open(FileHandle::from_bytes(OwnedBytes::new(reserved)))
                .await
                .is_err()
        );

        let doc_map_start = FLAT_BINARY_HEADER_SIZE + 2 * 2 * size_of::<f32>();
        let mut unsorted = valid.clone();
        let (first, second) = unsorted[doc_map_start..doc_map_start + 2 * DOC_ID_ENTRY_SIZE]
            .split_at_mut(DOC_ID_ENTRY_SIZE);
        first.swap_with_slice(second);
        assert!(
            LazyFlatVectorData::open(FileHandle::from_bytes(OwnedBytes::new(unsorted)))
                .await
                .is_err()
        );

        let mut duplicate = valid.clone();
        duplicate.copy_within(
            doc_map_start..doc_map_start + DOC_ID_ENTRY_SIZE,
            doc_map_start + DOC_ID_ENTRY_SIZE,
        );
        assert!(
            LazyFlatVectorData::open(FileHandle::from_bytes(OwnedBytes::new(duplicate)))
                .await
                .is_err()
        );

        assert!(
            LazyFlatVectorData::open_with_doc_limit(
                FileHandle::from_bytes(OwnedBytes::new(valid)),
                Some(1),
            )
            .await
            .is_err()
        );

        let mut invalid_binary = Vec::new();
        FlatVectorData::serialize_binary_from_bits_streaming(
            8,
            &[0],
            &[(0, 0)],
            &mut invalid_binary,
        )
        .unwrap();
        invalid_binary[4..8].copy_from_slice(&7u32.to_le_bytes());
        assert!(
            LazyFlatVectorData::open(FileHandle::from_bytes(OwnedBytes::new(invalid_binary)))
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn flat_vector_batch_and_dequantized_reads_are_checked() {
        let flat = LazyFlatVectorData::open(FileHandle::from_bytes(OwnedBytes::new(
            encoded_two_vector_payload(),
        )))
        .await
        .unwrap();

        assert_eq!(flat.read_vectors_batch(0, 2).await.unwrap().len(), 16);
        assert_eq!(flat.read_vectors_batch(2, 0).await.unwrap().len(), 0);
        assert!(flat.read_vectors_batch(1, 2).await.is_err());
        assert!(flat.read_vectors_batch(usize::MAX, 1).await.is_err());
        assert!(flat.read_vectors_batch(0, usize::MAX).await.is_err());

        let mut values = [0.0; 2];
        flat.read_vector_into(1, &mut values).await.unwrap();
        assert_eq!(values, [3.0, 4.0]);
        assert!(flat.read_vector_into(2, &mut values).await.is_err());
        assert!(flat.read_vector_into(0, &mut values[..1]).await.is_err());

        #[cfg(feature = "sync")]
        {
            assert_eq!(flat.read_vectors_batch_sync(0, 2).unwrap().len(), 16);
            assert!(flat.read_vectors_batch_sync(1, 2).is_err());
            assert!(flat.read_vectors_batch_sync(usize::MAX, 1).is_err());
            let mut too_short = [0; 7];
            assert!(flat.read_vector_raw_into_sync(0, &mut too_short).is_err());
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[tokio::test]
    async fn flat_vector_reads_reject_short_lazy_range_results() {
        let payload = std::sync::Arc::new(encoded_two_vector_payload());
        let payload_len = payload.len() as u64;
        let read_payload = std::sync::Arc::clone(&payload);
        let read_fn: crate::directories::RangeReadFn = std::sync::Arc::new(move |range| {
            let payload = std::sync::Arc::clone(&read_payload);
            Box::pin(async move {
                let start = usize::try_from(range.start).unwrap();
                let mut end = usize::try_from(range.end).unwrap();
                // Header and doc-map reads are exact, allowing open to finish.
                // Raw vector reads deliberately violate the range-read contract.
                if range.start == FLAT_BINARY_HEADER_SIZE as u64 {
                    end -= 1;
                }
                Ok(OwnedBytes::new(payload[start..end].to_vec()))
            })
        });
        let flat = LazyFlatVectorData::open(FileHandle::lazy(payload_len, read_fn))
            .await
            .unwrap();

        let error = flat.read_vectors_batch(0, 1).await.unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);
        let mut raw = [0; 8];
        let error = flat.read_vector_raw_into(0, &mut raw).await.unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);
    }

    #[tokio::test]
    async fn vector_prefix_reads_are_checked_and_do_not_fetch_the_tail() {
        let vectors = [1.0f32, 2.0, 3.0, 4.0];
        let doc_ids = [(0, 0), (1, 0)];
        let mut encoded = Vec::new();
        FlatVectorData::serialize_binary_from_flat_streaming(
            2,
            &vectors,
            &doc_ids,
            DenseVectorQuantization::F32,
            &mut encoded,
        )
        .unwrap();
        let flat = LazyFlatVectorData::open(FileHandle::from_bytes(OwnedBytes::new(encoded)))
            .await
            .unwrap();

        let mut prefix = [0xa5; 8];
        flat.read_vector_prefix_raw_into(1, 4, &mut prefix)
            .await
            .unwrap();
        assert_eq!(&prefix[..4], &3.0f32.to_ne_bytes());
        assert_eq!(&prefix[4..], &[0xa5; 4]);

        let mut full = [0; 8];
        flat.read_vector_raw_into(1, &mut full).await.unwrap();
        assert_eq!(&full[..4], &3.0f32.to_ne_bytes());
        assert_eq!(&full[4..], &4.0f32.to_ne_bytes());

        assert!(
            flat.read_vector_prefix_raw_into(2, 4, &mut prefix)
                .await
                .is_err()
        );
        assert!(
            flat.read_vector_prefix_raw_into(0, 9, &mut prefix)
                .await
                .is_err()
        );
        assert!(
            flat.read_vector_prefix_raw_into(0, 4, &mut prefix[..3])
                .await
                .is_err()
        );
    }
}
