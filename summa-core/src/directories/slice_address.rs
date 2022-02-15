use std::borrow::Borrow;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use lru::KeyRef;

#[derive(Hash, Clone, Eq, PartialEq)]
pub struct SliceAddress {
    pub path: PathBuf,
    pub generation: u32,
    pub index: u64,
}

#[derive(Clone, Eq, Hash, PartialEq)]
pub(crate) struct SliceAddressRef<'a> {
    pub path: &'a Path,
    pub generation: u32,
    pub index: u64,
}

pub(crate) trait SliceAddressKey {
    fn key(&self) -> SliceAddressRef;
}

impl SliceAddressKey for SliceAddress {
    fn key(&self) -> SliceAddressRef {
        SliceAddressRef {
            path: self.path.as_path(),
            generation: self.generation,
            index: self.index,
        }
    }
}

impl<'a> SliceAddressKey for SliceAddressRef<'a> {
    fn key(&self) -> SliceAddressRef {
        self.clone()
    }
}

impl<'a> Borrow<dyn SliceAddressKey + 'a> for KeyRef<SliceAddress> {
    fn borrow(&self) -> &(dyn SliceAddressKey + 'a) {
        let slice_address: &SliceAddress = self.borrow();
        slice_address
    }
}
impl<'a> PartialEq for (dyn SliceAddressKey + 'a) {
    fn eq(&self, other: &Self) -> bool {
        self.key().eq(&other.key())
    }
}

impl<'a> Eq for (dyn SliceAddressKey + 'a) {}

impl<'a> Hash for (dyn SliceAddressKey + 'a) {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.key().hash(state)
    }
}
