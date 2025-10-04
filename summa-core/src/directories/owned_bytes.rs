//! Immutable byte ownership and checked zero-copy views.
use std::{ops::Range, sync::Arc};

/// Backing store for OwnedBytes — supports both heap Vec and mmap.
#[derive(Clone)]
enum SharedBytes {
    Vec(Arc<Vec<u8>>),
    #[cfg(feature = "native")]
    Mmap(Arc<memmap2::Mmap>),
    Local(Arc<SharedBytes>),
    #[cfg(feature = "native")]
    Recycled(Arc<super::payload::buffers::RecycledBytes>),
}

impl SharedBytes {
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        match self {
            SharedBytes::Vec(v) => v.as_slice(),
            #[cfg(feature = "native")]
            SharedBytes::Recycled(v) => v.as_slice(),
            #[cfg(feature = "native")]
            SharedBytes::Mmap(m) => m.as_ref(),
            SharedBytes::Local(owner) => owner.as_bytes(),
        }
    }
}

impl std::fmt::Debug for SharedBytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SharedBytes::Vec(v) => write!(f, "Vec(len={})", v.len()),
            #[cfg(feature = "native")]
            SharedBytes::Recycled(v) => write!(f, "Recycled(len={})", v.as_slice().len()),
            #[cfg(feature = "native")]
            SharedBytes::Mmap(m) => write!(f, "Mmap(len={})", m.len()),
            SharedBytes::Local(owner) => owner.fmt(f),
        }
    }
}

/// Owned bytes with cheap cloning (Arc-backed)
///
/// Supports two backing stores:
/// - `Vec<u8>` for owned data (RamDirectory, FsDirectory, decompressed blocks)
/// - `Mmap` for zero-copy memory-mapped files (MmapDirectory, native only)
#[derive(Clone)]
pub struct OwnedBytes {
    data: SharedBytes,
    /// Validated subview into `data`. Its allocation is immutable and stable
    /// for the lifetime of the retained Arc; see `docs/owned-byte-views.md`.
    view: std::ptr::NonNull<[u8]>,
}

// SAFETY: the view points into immutable storage owned by `data`. Both Arc
// owners are Send + Sync, keep their allocation stable, and expose no mutable
// access through this type. Every clone retains that same backing allocation.
unsafe impl Send for OwnedBytes {}
// SAFETY: shared access only yields immutable slices tied to the owner's borrow.
unsafe impl Sync for OwnedBytes {}

impl std::fmt::Debug for OwnedBytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OwnedBytes")
            .field("data", &self.data)
            .field("len", &self.len())
            .finish()
    }
}

impl OwnedBytes {
    pub(super) fn with_local_owner(mut self) -> Self {
        if !matches!(self.data, SharedBytes::Local(_)) {
            self.data = SharedBytes::Local(Arc::new(self.data));
        }
        self
    }

    /// Validate a view while its stable backing owner is available. Moving the
    /// Arc handle below does not move the Vec buffer or memory mapping.
    fn with_range(data: SharedBytes, range: Range<usize>) -> Self {
        let view = std::ptr::NonNull::from(&data.as_bytes()[range]);
        Self { data, view }
    }

    pub fn new(data: Vec<u8>) -> Self {
        let len = data.len();
        Self::with_range(SharedBytes::Vec(Arc::new(data)), 0..len)
    }

    #[cfg(feature = "native")]
    pub(super) fn from_recycled(
        data: Arc<super::payload::buffers::RecycledBytes>,
        len: usize,
    ) -> Self {
        Self::with_range(SharedBytes::Recycled(data), 0..len)
    }

    pub fn empty() -> Self {
        Self::new(Vec::new())
    }

    /// Create from a pre-existing Arc<Vec<u8>> with a checked sub-range.
    /// Used by RamDirectory and CachingDirectory to share data without copying.
    pub(crate) fn from_arc_vec(data: Arc<Vec<u8>>, range: Range<usize>) -> Self {
        Self::with_range(SharedBytes::Vec(data), range)
    }

    /// Create from a memory-mapped file (zero-copy).
    #[cfg(feature = "native")]
    pub(crate) fn from_mmap(mmap: Arc<memmap2::Mmap>) -> Self {
        let len = mmap.len();
        Self::with_range(SharedBytes::Mmap(mmap), 0..len)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.view.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Create a checked subview bounded by this view, retaining the same owner.
    pub fn slice(&self, range: Range<usize>) -> Self {
        let view = std::ptr::NonNull::from(&self.as_slice()[range]);
        Self {
            data: self.data.clone(),
            view,
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: constructors and slice validate this view against immutable
        // Arc-owned storage. The owner outlives the returned borrow of self;
        // neither moving a handle nor cloning it can move its backing bytes.
        unsafe { self.view.as_ref() }
    }

    /// Returns `true` if the backing store is a memory-mapped file.
    ///
    /// Used to guard `madvise` calls: `MADV_DONTNEED` on heap memory
    /// zeroes pages on Linux and corrupts allocator metadata.
    #[cfg(feature = "native")]
    #[inline]
    pub fn is_mmap(&self) -> bool {
        match &self.data {
            SharedBytes::Mmap(_) => true,
            SharedBytes::Local(owner) => matches!(owner.as_ref(), SharedBytes::Mmap(_)),
            SharedBytes::Vec(_) | SharedBytes::Recycled(_) => false,
        }
    }

    /// Advise the kernel about the access pattern for these bytes.
    ///
    /// No-op unless the backing store is mmap (heap memory must never be
    /// madvised: `MADV_DONTNEED` on heap zeroes pages and corrupts allocator
    /// metadata) or the range is empty.
    #[cfg(feature = "native")]
    pub fn madvise(&self, advice: libc::c_int) {
        self.madvise_range(0..self.len(), advice);
    }

    /// Pin these bytes in physical memory (`mlock`). mmap-backed only —
    /// heap memory is not evictable by the page cache. Returns whether the
    /// lock succeeded; failure (e.g. RLIMIT_MEMLOCK) is not fatal.
    /// Locks are released automatically when the mapping is unmapped.
    #[cfg(feature = "native")]
    pub fn mlock(&self) -> bool {
        if !self.is_mmap() {
            return false;
        }
        let slice = self.as_slice();
        if slice.is_empty() {
            return true;
        }
        let ptr = slice.as_ptr();
        let len = slice.len();
        let page_size = 4096usize;
        let aligned_ptr = (ptr as usize) & !(page_size - 1);
        let aligned_len = len + (ptr as usize - aligned_ptr);
        unsafe { libc::mlock(aligned_ptr as *const libc::c_void, aligned_len) == 0 }
    }

    /// Advise the kernel about the access pattern for a sub-range.
    ///
    /// The range is relative to these bytes. Same mmap-only guard as
    /// [`Self::madvise`]. The pointer is aligned down to a page boundary
    /// as required by `madvise`.
    #[cfg(feature = "native")]
    pub fn madvise_range(&self, range: Range<usize>, advice: libc::c_int) {
        if !self.is_mmap() {
            return;
        }
        let slice = &self.as_slice()[range];
        if slice.is_empty() {
            return;
        }
        let ptr = slice.as_ptr();
        let len = slice.len();
        let page_size = 4096usize;
        let aligned_ptr = (ptr as usize) & !(page_size - 1);
        let aligned_len = len + (ptr as usize - aligned_ptr);
        unsafe {
            libc::madvise(aligned_ptr as *mut libc::c_void, aligned_len, advice);
        }
    }

    /// Whether the leading pages of a sub-range are in the page cache.
    ///
    /// Samples at most [`Self::RESIDENCY_SAMPLE_PAGES`] pages from the start
    /// of the range with one `mincore` call. Heap bytes and empty ranges are
    /// resident. A failed call reports `false`, so callers that use this to
    /// skip prefetch keep prefetching.
    #[cfg(feature = "native")]
    pub fn range_resident(&self, range: Range<usize>) -> bool {
        if !self.is_mmap() {
            return true;
        }
        let slice = &self.as_slice()[range];
        if slice.is_empty() {
            return true;
        }
        let page_size = system_page_size();
        let ptr = slice.as_ptr() as usize;
        let aligned_ptr = ptr & !(page_size - 1);
        let aligned_len =
            (slice.len() + (ptr - aligned_ptr)).min(Self::RESIDENCY_SAMPLE_PAGES * page_size);
        let mut pages = [0u8; Self::RESIDENCY_SAMPLE_PAGES];
        // SAFETY: the range lies inside this live mapping and `pages` holds
        // one byte for each of the at most RESIDENCY_SAMPLE_PAGES queried pages.
        let result = unsafe {
            libc::mincore(
                aligned_ptr as *mut libc::c_void,
                aligned_len,
                pages.as_mut_ptr().cast(),
            )
        };
        result == 0
            && pages[..aligned_len.div_ceil(page_size)]
                .iter()
                .all(|&page| page & 1 != 0)
    }

    /// Upper bound on pages sampled by [`Self::range_resident`].
    #[cfg(feature = "native")]
    pub const RESIDENCY_SAMPLE_PAGES: usize = 64;

    pub fn to_vec(&self) -> Vec<u8> {
        self.as_slice().to_vec()
    }
}

impl AsRef<[u8]> for OwnedBytes {
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl std::ops::Deref for OwnedBytes {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

/// The OS page size (`mincore` requires page-aligned addresses; Apple
/// silicon uses 16 KiB pages).
#[cfg(feature = "native")]
fn system_page_size() -> usize {
    static PAGE_SIZE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *PAGE_SIZE.get_or_init(|| {
        // SAFETY: sysconf has no preconditions.
        usize::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) })
            .ok()
            .filter(|size| size.is_power_of_two())
            .unwrap_or(4096)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn owned_byte_views_retain_heap_storage_across_moves_clones_and_empty_slices() {
        let backing = Arc::new((0u8..64).collect::<Vec<_>>());
        let weak = Arc::downgrade(&backing);
        let bytes = OwnedBytes::from_arc_vec(backing.clone(), 3..61);
        let nested = bytes.slice(1..57).slice(2..53);
        let expected = (6u8..57).collect::<Vec<_>>();
        let pointer = nested.as_slice().as_ptr();
        let empty = nested.slice(nested.len()..nested.len());
        assert!(empty.is_empty());
        assert!(OwnedBytes::empty().slice(0..0).as_slice().is_empty());
        let cloned = nested.clone();
        drop(backing);
        drop(bytes);
        drop(nested);
        assert_eq!(cloned.as_slice().as_ptr(), pointer);
        assert_eq!(cloned.as_slice(), expected);
        drop(cloned);
        assert!(
            weak.upgrade().is_some(),
            "empty views also retain their owner"
        );
        drop(empty);
        assert!(weak.upgrade().is_none());
        assert_eq!(
            std::mem::size_of::<OwnedBytes>(),
            std::mem::size_of::<super::SharedBytes>() + 2 * std::mem::size_of::<usize>()
        );
    }

    #[cfg(feature = "native")]
    #[test]
    fn owned_byte_views_keep_heap_and_mmap_owners_alive_across_threads() {
        fn check(bytes: OwnedBytes, mapped: bool) {
            assert_eq!(bytes.is_mmap(), mapped);
            let survivor = bytes.slice(3..61).slice(1..56);
            let copied = survivor.clone();
            drop(bytes);
            let thread = std::thread::spawn(move || {
                assert_eq!(survivor.as_slice(), &(4u8..59).collect::<Vec<_>>());
                assert_eq!(survivor.is_mmap(), mapped);
                survivor.slice(2..9)
            });
            assert_eq!(copied.as_slice(), &(4u8..59).collect::<Vec<_>>());
            drop(copied);
            let final_view = thread.join().unwrap();
            assert_eq!(final_view.as_slice(), &[6, 7, 8, 9, 10, 11, 12]);
            assert_eq!(final_view.is_mmap(), mapped);
        }
        check(OwnedBytes::new((0u8..64).collect()), false);
        let mut mapping = memmap2::MmapMut::map_anon(64).unwrap();
        mapping.copy_from_slice(&(0u8..64).collect::<Vec<_>>());
        let mapping = Arc::new(mapping.make_read_only().unwrap());
        let weak = Arc::downgrade(&mapping);
        check(OwnedBytes::from_mmap(mapping.clone()), true);
        check(
            OwnedBytes::from_mmap(mapping.clone()).with_local_owner(),
            true,
        );
        check(
            OwnedBytes::new((0u8..64).collect()).with_local_owner(),
            false,
        );
        assert_eq!(Arc::strong_count(&mapping), 1);
        drop(mapping);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn owned_byte_subslices_reject_access_outside_the_parent_view() {
        let bytes = OwnedBytes::new(vec![1, 2, 3, 4, 5]);
        let parent = bytes.slice(1..3);
        assert_eq!(parent.slice(0..2).as_slice(), &[2, 3]);
        assert!(std::panic::catch_unwind(|| parent.slice(0..3).to_vec()).is_err());
        assert!(
            std::panic::catch_unwind(|| parent.slice(Range { start: 2, end: 1 }).to_vec()).is_err()
        );
        assert!(
            std::panic::catch_unwind(|| parent.slice(usize::MAX..usize::MAX).to_vec()).is_err()
        );
    }

    #[tokio::test]
    async fn test_owned_bytes() {
        let bytes = OwnedBytes::new(vec![1, 2, 3, 4, 5]);

        assert_eq!(bytes.len(), 5);
        assert_eq!(bytes.as_slice(), &[1, 2, 3, 4, 5]);

        let sliced = bytes.slice(1..4);
        assert_eq!(sliced.as_slice(), &[2, 3, 4]);

        // Original unchanged
        assert_eq!(bytes.as_slice(), &[1, 2, 3, 4, 5]);
    }
}
