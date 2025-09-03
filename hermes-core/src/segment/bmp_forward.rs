//! Quantized sparse values addressed by logical document/ordinal, inside BMP.
//! The directory is compact and the payload stays evictable. Neither contains
//! physical BMP IDs, so physical reorder never rewrites forward values.

use super::logical_address::LogicalUnit;
use crate::directories::OwnedBytes;
use crate::{Error, Result};

const ROW_BYTES: usize = 16;
pub(super) const TRAILER_BYTES: usize = 16;
const STORAGE_DISABLED: u32 = 1;

#[derive(Clone)]
pub(crate) struct BmpForward {
    payload: OwnedBytes,
    rows: OwnedBytes,
    dims: u32,
}

fn corrupt(message: &str) -> Error {
    Error::Corruption(format!("BMP forward values: {message}"))
}

impl BmpForward {
    pub(crate) fn parse_optional(
        bytes: OwnedBytes,
        count: u32,
        docs: u32,
        dims: u32,
    ) -> Result<Option<Self>> {
        let trailer = bytes
            .len()
            .checked_sub(TRAILER_BYTES)
            .ok_or_else(|| corrupt("missing storage trailer"))?;
        let flags = u32::from_le_bytes(bytes[trailer + 4..trailer + 8].try_into().unwrap());
        if flags == STORAGE_DISABLED {
            if bytes.len() != TRAILER_BYTES || bytes[..4] != [0; 4] || bytes[8..] != [0; 8] {
                return Err(corrupt("disabled storage has a payload or vector count"));
            }
            return Ok(None);
        }
        Self::parse(bytes, count, docs, dims).map(Some)
    }

    pub(crate) fn parse(bytes: OwnedBytes, count: u32, docs: u32, dims: u32) -> Result<Self> {
        let trailer = bytes
            .len()
            .checked_sub(TRAILER_BYTES)
            .ok_or_else(|| corrupt("missing trailer"))?;
        let data = bytes.as_slice();
        let declared = u32::from_le_bytes(data[trailer..trailer + 4].try_into().unwrap());
        let payload_len =
            usize::try_from(u64::from_le_bytes(data[trailer + 8..].try_into().unwrap()))
                .map_err(|_| corrupt("payload exceeds address space"))?;
        let flags = u32::from_le_bytes(data[trailer + 4..trailer + 8].try_into().unwrap());
        if declared != count
            || flags != 0
            || (count as usize)
                .checked_mul(ROW_BYTES)
                .and_then(|n| n.checked_add(payload_len))
                != Some(trailer)
            || !payload_len.is_multiple_of(5)
        {
            return Err(corrupt("invalid directory length or count"));
        }
        let result = Self {
            payload: bytes.slice(0..payload_len),
            rows: bytes.slice(payload_len..trailer),
            dims,
        };
        let mut previous_key = None;
        let mut previous_offset = 0;
        for i in 0..count {
            let key = result.key(i);
            let offset = result.offset(i);
            if key.doc >= docs
                || previous_key.is_some_and(|p| p >= key)
                || result.rows.as_slice()[i as usize * ROW_BYTES + 6..i as usize * ROW_BYTES + 8]
                    != [0; 2]
                || (i == 0 && offset != 0)
                || (i > 0 && offset <= previous_offset)
                || offset >= payload_len as u64
                || !offset.is_multiple_of(5)
            {
                return Err(corrupt("invalid logical key or vector offset"));
            }
            previous_key = Some(key);
            previous_offset = offset;
        }
        if count == 0 && payload_len != 0 {
            return Err(corrupt("empty directory has a payload"));
        }
        #[cfg(feature = "native")]
        result.advise(libc::MADV_RANDOM);
        Ok(result)
    }

    pub(crate) fn len(&self) -> u32 {
        (self.rows.len() / ROW_BYTES) as u32
    }

    pub(crate) fn key(&self, index: u32) -> LogicalUnit {
        let row = &self.rows.as_slice()[index as usize * ROW_BYTES..][..ROW_BYTES];
        LogicalUnit {
            doc: u32::from_le_bytes(row[..4].try_into().unwrap()),
            ordinal: u16::from_le_bytes(row[4..6].try_into().unwrap()),
        }
    }

    fn offset(&self, index: u32) -> u64 {
        if index == self.len() {
            return self.payload.len() as u64;
        }
        let start = index as usize * ROW_BYTES + 8;
        u64::from_le_bytes(self.rows.as_slice()[start..start + 8].try_into().unwrap())
    }

    fn lower_bound(&self, target: LogicalUnit) -> u32 {
        let mut low = 0;
        let mut high = self.len();
        while low < high {
            let mid = low + (high - low) / 2;
            if self.key(mid) < target {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        low
    }

    pub(crate) fn find(&self, target: LogicalUnit) -> Option<u32> {
        let index = self.lower_bound(target);
        (index < self.len() && self.key(index) == target).then_some(index)
    }

    pub(crate) fn for_document(&self, doc: u32) -> impl Iterator<Item = (u16, u32)> + '_ {
        let start = self.lower_bound(LogicalUnit { doc, ordinal: 0 });
        (start..self.len())
            .map(|i| (self.key(i), i))
            .take_while(move |(key, _)| key.doc == doc)
            .map(|(key, i)| (key.ordinal, i))
    }

    /// Validate only the selected vector's payload, before returning its view.
    pub(crate) fn vector(&self, index: u32) -> Result<ForwardVector<'_>> {
        if index >= self.len() {
            return Err(corrupt("vector index out of bounds"));
        }
        let start = self.offset(index) as usize;
        let end = self.offset(index + 1) as usize;
        let vector = ForwardVector(&self.payload.as_slice()[start..end]);
        let mut previous = None;
        for (dimension, impact) in vector.iter() {
            if dimension >= self.dims || impact == 0 || previous.is_some_and(|p| p > dimension) {
                return Err(corrupt("invalid dimension order or impact"));
            }
            previous = Some(dimension);
        }
        Ok(vector)
    }

    pub(crate) fn encoded_bytes(&self) -> usize {
        self.payload.len() + self.rows.len() + TRAILER_BYTES
    }
    #[cfg(feature = "native")]
    pub(crate) fn payload_bytes(&self) -> usize {
        self.payload.len()
    }

    #[cfg(any(feature = "native", feature = "wasm", test))]
    pub(crate) fn validate_payload(&self) -> Result<ValidatedForward<'_>> {
        for i in 0..self.len() {
            self.vector(i)?;
        }
        Ok(ValidatedForward(self))
    }

    #[cfg(feature = "native")]
    pub(crate) fn advise(&self, advice: i32) {
        self.rows.madvise(advice);
        self.payload.madvise(advice);
    }
}

#[cfg(any(feature = "native", feature = "wasm", test))]
pub(crate) fn write_disabled(writer: &mut dyn std::io::Write) -> std::io::Result<u64> {
    use byteorder::{LittleEndian, WriteBytesExt};
    writer.write_u32::<LittleEndian>(0)?;
    writer.write_u32::<LittleEndian>(STORAGE_DISABLED)?;
    writer.write_u64::<LittleEndian>(0)?;
    Ok(TRAILER_BYTES as u64)
}

/// A background scan validates once before BP's repeated infallible passes.
#[cfg(any(feature = "native", feature = "wasm", test))]
pub(crate) struct ValidatedForward<'a>(&'a BmpForward);
#[cfg(any(feature = "native", feature = "wasm", test))]
impl ValidatedForward<'_> {
    pub(crate) fn vector(&self, index: u32) -> ForwardVector<'_> {
        ForwardVector(
            &self.0.payload.as_slice()
                [self.0.offset(index) as usize..self.0.offset(index + 1) as usize],
        )
    }
}

#[derive(Clone, Copy)]
pub(crate) struct ForwardVector<'a>(&'a [u8]);
impl ForwardVector<'_> {
    pub(crate) fn iter(&self) -> impl Iterator<Item = (u32, u8)> + '_ {
        self.0
            .chunks_exact(5)
            .map(|entry| (u32::from_le_bytes(entry[..4].try_into().unwrap()), entry[4]))
    }
    #[cfg(feature = "native")]
    pub(crate) fn len(&self) -> usize {
        self.0.len() / 5
    }
}

#[cfg(any(feature = "native", feature = "wasm", test))]
pub(crate) fn write_directory(
    writer: &mut dyn std::io::Write,
    rows: impl IntoIterator<Item = std::io::Result<(LogicalUnit, u64)>>,
    count: u32,
    payload_bytes: u64,
) -> std::io::Result<u64> {
    use byteorder::{LittleEndian, WriteBytesExt};
    let mut written = 0u32;
    let mut previous = None;
    let mut previous_offset = 0;
    for row in rows {
        let (key, offset) = row?;
        if previous.is_some_and(|p| p >= key)
            || (written == 0 && offset != 0)
            || (written > 0 && offset <= previous_offset)
            || offset >= payload_bytes
            || !offset.is_multiple_of(5)
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid BMP forward directory",
            ));
        }
        writer.write_u32::<LittleEndian>(key.doc)?;
        writer.write_u16::<LittleEndian>(key.ordinal)?;
        writer.write_u16::<LittleEndian>(0)?;
        writer.write_u64::<LittleEndian>(offset)?;
        written = written
            .checked_add(1)
            .ok_or_else(|| std::io::Error::other("BMP forward count overflow"))?;
        previous = Some(key);
        previous_offset = offset;
    }
    if written != count {
        return Err(std::io::Error::other("BMP forward count mismatch"));
    }
    writer.write_u32::<LittleEndian>(count)?;
    writer.write_u32::<LittleEndian>(0)?;
    writer.write_u64::<LittleEndian>(payload_bytes)?;
    Ok(u64::from(count) * ROW_BYTES as u64 + TRAILER_BYTES as u64)
}

#[cfg(feature = "native")]
mod rewrite;
#[cfg(feature = "native")]
pub(crate) use rewrite::{validate_copy_sources, write_forward_sources};

#[cfg(all(test, feature = "native"))]
mod tests;
