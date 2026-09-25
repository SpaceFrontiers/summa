//! Immutable little-endian L1 metadata, including unaligned mmap views.

use crate::directories::OwnedBytes;

#[derive(Debug, Clone)]
pub(super) struct GroupWords(OwnedBytes);

impl GroupWords {
    pub(super) fn borrowed(bytes: OwnedBytes) -> Self {
        // Footer parsing proves the extent; no scan or realignment is needed.
        debug_assert!(bytes.len().is_multiple_of(4));
        Self(bytes)
    }

    #[inline]
    pub(super) fn words(&self) -> &[[u8; 4]] {
        self.0.as_slice().as_chunks::<4>().0
    }

    #[inline]
    pub(super) fn get(&self, index: usize) -> Option<u32> {
        self.words()
            .get(index)
            .map(|word| u32::from_le_bytes(*word))
    }

    pub(super) fn bytes(&self) -> &[u8] {
        self.0.as_slice()
    }

    pub(super) fn len(&self) -> usize {
        self.0.len() / 4
    }

    pub(super) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl From<Vec<u32>> for GroupWords {
    fn from(words: Vec<u32>) -> Self {
        let mut bytes = Vec::with_capacity(words.len() * 4);
        for word in words {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        Self(OwnedBytes::new(bytes))
    }
}
