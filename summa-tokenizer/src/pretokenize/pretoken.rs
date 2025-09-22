//! Once we have a document, we can pretokenize it (potentially in parallel)

use std::ops::Deref;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct Pretoken<'a>(pub &'a [u8]);

impl AsRef<[u8]> for Pretoken<'_> {
    fn as_ref(&self) -> &[u8] {
        self.0
    }
}

impl<'a> Deref for Pretoken<'a> {
    type Target = &'a [u8];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
