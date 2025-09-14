//! Bounded structural-validation reuse tied to one segment's immutable text files.
use std::hash::{Hash, Hasher};
use std::io;
use std::ops::Range;

use parking_lot::RwLock;
use rustc_hash::FxHasher;

use super::super::positions_v2::{PositionStream, PositionStreamLayout, TermPositions};
use super::{BlockPostingList, Footer};
use crate::directories::{FileHandle, OwnedBytes};

#[derive(Clone, Copy)]
enum ValidationProof {
    Postings(Footer),
    Positions(PositionStreamLayout),
}

#[derive(Clone, Copy)]
struct ValidatedRange {
    start: u64,
    end: u64,
    proof: ValidationProof,
}

/// One budget for both text files. Proofs retain only structural metadata,
/// never decoded payloads. Lazy callbacks cannot reuse these proofs.
pub(crate) struct PostingListReader {
    file: FileHandle,
    positions: Option<FileHandle>,
    validated: RwLock<Box<[Option<ValidatedRange>]>>,
    slots: usize,
    content_error: std::sync::Arc<std::sync::OnceLock<usize>>,
    #[cfg(test)]
    validations: std::sync::atomic::AtomicUsize,
}

impl PostingListReader {
    pub(crate) fn validate_budget(bytes: usize) -> io::Result<()> {
        if bytes > 64 * 1024 * 1024 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "posting_validation_cache_bytes must be at most 64 MiB per segment",
            ));
        }
        Ok(())
    }

    pub(crate) fn new(
        file: FileHandle,
        positions: Option<FileHandle>,
        budget_bytes: usize,
    ) -> io::Result<Self> {
        Self::validate_budget(budget_bytes)?;
        let slots = if file.is_sync() || positions.as_ref().is_some_and(FileHandle::is_sync) {
            budget_bytes / std::mem::size_of::<Option<ValidatedRange>>()
        } else {
            0
        };
        if budget_bytes != 0 {
            if !file.is_sync() {
                log::warn!(
                    "posting validation cache disabled for a lazy file handle; every range read will be validated"
                );
            }
            if positions.as_ref().is_some_and(|file| !file.is_sync()) {
                log::warn!(
                    "position validation cache disabled for a lazy file handle; every range read will be validated"
                );
            }
        }
        Ok(Self {
            file,
            positions,
            slots,
            content_error: Default::default(),
            validated: RwLock::new(vec![None; slots].into_boxed_slice()),
            #[cfg(test)]
            validations: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    pub(crate) fn check_integrity(&self) -> io::Result<()> {
        match self.content_error.get() {
            Some(block) => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("posting payload corruption detected in block {block}"),
            )),
            None => Ok(()),
        }
    }

    pub(crate) fn integrity_heap_bytes(&self) -> usize {
        std::mem::size_of::<std::sync::OnceLock<usize>>() + 2 * std::mem::size_of::<usize>()
    }

    pub(crate) fn file(&self) -> &FileHandle {
        &self.file
    }

    pub(crate) fn positions_file(&self) -> Option<&FileHandle> {
        self.positions.as_ref()
    }

    pub(crate) fn heap_bytes(&self) -> usize {
        self.slots * std::mem::size_of::<Option<ValidatedRange>>()
    }

    pub(crate) async fn read(&self, range: Range<u64>) -> io::Result<BlockPostingList> {
        Self::check_range(&self.file, &range)?;
        let bytes = self.file.read_bytes_range(range.clone()).await?;
        self.decode(range, bytes)
    }

    #[cfg(feature = "sync")]
    pub(crate) fn read_sync(&self, range: Range<u64>) -> io::Result<BlockPostingList> {
        Self::check_range(&self.file, &range)?;
        let bytes = self.file.read_bytes_range_sync(range.clone())?;
        self.decode(range, bytes)
    }

    pub(crate) async fn read_positions(&self, range: Range<u64>) -> io::Result<TermPositions> {
        let file = self.position_file()?;
        Self::check_range(file, &range)?;
        let bytes = file.read_bytes_range(range.clone()).await?;
        self.decode_positions(range, bytes, file.is_sync())
    }

    #[cfg(feature = "sync")]
    pub(crate) fn read_positions_sync(&self, range: Range<u64>) -> io::Result<TermPositions> {
        let file = self.position_file()?;
        Self::check_range(file, &range)?;
        let bytes = file.read_bytes_range_sync(range.clone())?;
        self.decode_positions(range, bytes, file.is_sync())
    }

    fn position_file(&self) -> io::Result<&FileHandle> {
        self.positions
            .as_ref()
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "missing position file"))
    }

    fn check_range(file: &FileHandle, range: &Range<u64>) -> io::Result<()> {
        if range.start > range.end || range.end > file.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "text posting range out of bounds",
            ));
        }
        Ok(())
    }

    fn check_length(range: &Range<u64>, bytes: &OwnedBytes) -> io::Result<()> {
        if bytes.len() as u64 != range.end - range.start {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "short text posting range read",
            ));
        }
        Ok(())
    }

    // A bounded set avoids rescanning large immutable streams when two keys
    // share a direct-mapped slot. The configured entry budget is unchanged.
    fn set_range(&self, hash: usize) -> Range<usize> {
        let start = (hash % self.slots.div_ceil(4)) * 4;
        start..(start + 4).min(self.slots)
    }

    fn same_key(entry: &ValidatedRange, range: &Range<u64>, positions: bool) -> bool {
        entry.start == range.start
            && entry.end == range.end
            && matches!(entry.proof, ValidationProof::Positions(_)) == positions
    }

    fn cached(
        &self,
        range: &Range<u64>,
        positions: bool,
        immutable: bool,
    ) -> (Option<usize>, Option<ValidationProof>) {
        if self.slots == 0 || !immutable {
            return (None, None);
        }
        let mut hash = FxHasher::default();
        range.hash(&mut hash);
        if positions {
            true.hash(&mut hash);
        }
        let hash = hash.finish() as usize;
        let proof = self.validated.read()[self.set_range(hash)]
            .iter()
            .flatten()
            .find(|entry| Self::same_key(entry, range, positions))
            .map(|entry| entry.proof);
        (Some(hash), proof)
    }

    fn publish(&self, hash: Option<usize>, range: Range<u64>, proof: ValidationProof) {
        if let Some(hash) = hash {
            let mut entries = self.validated.write();
            let set = &mut entries[self.set_range(hash)];
            let positions = matches!(proof, ValidationProof::Positions(_));
            let slot = set
                .iter()
                .position(|entry| {
                    entry.is_some_and(|entry| Self::same_key(&entry, &range, positions))
                })
                .or_else(|| set.iter().position(Option::is_none))
                .unwrap_or_else(|| hash.rotate_right(16) % set.len());
            set[slot] = Some(ValidatedRange {
                start: range.start,
                end: range.end,
                proof,
            });
        }
    }

    fn decode(&self, range: Range<u64>, bytes: OwnedBytes) -> io::Result<BlockPostingList> {
        Self::check_length(&range, &bytes)?;
        let (slot, proof) = self.cached(&range, false, self.file.is_sync());
        if let Some(ValidationProof::Postings(footer)) = proof {
            crate::observe::search_work!(postings_proof_hits += 1);
            return Ok(self.attach_validated_bytes(bytes, footer));
        }
        #[cfg(test)]
        self.validations
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let footer = BlockPostingList::validate_bytes(&bytes)?;
        crate::observe::search_work!(
            postings_admitted += 1,
            posting_blocks_admitted += footer.l0_count
        );
        self.publish(slot, range, ValidationProof::Postings(footer));
        Ok(self.attach_validated_bytes(bytes, footer))
    }

    fn attach_validated_bytes(&self, bytes: OwnedBytes, footer: Footer) -> BlockPostingList {
        let mut list = BlockPostingList::from_validated_bytes(bytes, footer);
        list.content_error = Some(std::sync::Arc::clone(&self.content_error));
        list
    }

    fn decode_positions(
        &self,
        range: Range<u64>,
        bytes: OwnedBytes,
        immutable: bool,
    ) -> io::Result<TermPositions> {
        Self::check_length(&range, &bytes)?;
        let (slot, proof) = self.cached(&range, true, immutable);
        if let Some(ValidationProof::Positions(layout)) = proof {
            crate::observe::search_work!(positions_proof_hits += 1);
            return Ok(TermPositions::Stream(PositionStream::from_validated_bytes(
                bytes, layout,
            )));
        }
        #[cfg(test)]
        self.validations
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if !PositionStream::is_stream(bytes.as_slice()) {
            return TermPositions::open(bytes);
        }
        let layout = PositionStream::validate_bytes(bytes.as_slice())?;
        crate::observe::search_work!(positions_admitted += 1);
        self.publish(slot, range, ValidationProof::Positions(layout));
        Ok(TermPositions::Stream(PositionStream::from_validated_bytes(
            bytes, layout,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structures::{PostingCodec, PostingList};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn new_reader(file: FileHandle, bytes: usize) -> io::Result<PostingListReader> {
        PostingListReader::new(file, None, bytes)
    }

    fn fixture(codec: PostingCodec) -> Vec<u8> {
        let mut postings = PostingList::new();
        for doc in 0..1291 {
            postings.push(doc * 7, doc % 13 + 1);
        }
        let list = BlockPostingList::from_posting_list_with_ratio_bounds(
            &postings,
            true,
            Some(&|doc| doc % 100 + 1),
            codec,
        )
        .unwrap();
        let mut bytes = Vec::new();
        list.serialize(&mut bytes).unwrap();
        bytes
    }

    fn validations(reader: &PostingListReader) -> usize {
        reader.validations.load(Ordering::Relaxed)
    }

    fn assert_bytes(list: BlockPostingList, expected: &[u8]) {
        let mut bytes = Vec::new();
        list.serialize(&mut bytes).unwrap();
        assert_eq!(bytes, expected);
    }

    fn position_fixture() -> Vec<u8> {
        let mut bytes = Vec::new();
        let mut encoder = crate::structures::PositionStreamEncoder::new(&mut bytes);
        encoder.push_doc(&mut [1, 5, 9]).unwrap();
        encoder.finish().unwrap();
        bytes
    }

    #[tokio::test]
    async fn equal_cross_file_ranges_never_alias_admission_proofs_or_exceed_one_budget() {
        let mut list = PostingList::new();
        list.push(0, 64);
        let mut postings = Vec::new();
        BlockPostingList::from_posting_list(&list)
            .unwrap()
            .serialize(&mut postings)
            .unwrap();
        // A single 8-bit position block is count + 32 bytes. Build an equally
        // sized stream so only the file/format identity distinguishes the keys.
        let count = postings.len() - 32;
        assert!((2..=128).contains(&count));
        let expected: Vec<u32> = (0..count as u32).collect();
        let mut positions = Vec::new();
        let mut encoder = crate::structures::PositionStreamEncoder::new(&mut positions);
        encoder.push_doc(&mut expected.clone()).unwrap();
        encoder.finish().unwrap();
        assert_eq!(postings.len(), positions.len());
        let len = positions.len() as u64;
        let budget = std::mem::size_of::<Option<ValidatedRange>>();
        let reader = PostingListReader::new(
            FileHandle::from_bytes(OwnedBytes::new(postings.clone())),
            Some(FileHandle::from_bytes(OwnedBytes::new(positions))),
            budget,
        )
        .unwrap();
        assert_eq!(reader.heap_bytes(), budget);
        for _ in 0..3 {
            assert_bytes(reader.read(0..len).await.unwrap(), &postings);
            let positions = reader.read_positions(0..len).await.unwrap();
            assert_eq!(
                positions.positions(0, 0, count as u32),
                Some(expected.clone())
            );
        }
        assert_eq!(validations(&reader), 6);
        for _ in 0..3 {
            assert_eq!(
                reader
                    .read_positions(0..len)
                    .await
                    .unwrap()
                    .positions(0, 0, count as u32),
                Some(expected.clone())
            );
            #[cfg(feature = "sync")]
            assert_eq!(
                reader
                    .read_positions_sync(0..len)
                    .unwrap()
                    .positions(0, 0, count as u32),
                Some(expected.clone())
            );
        }
        assert_eq!(validations(&reader), 6);
    }

    #[tokio::test]
    async fn position_admission_without_cache_revalidates_and_retains_no_entries() {
        for budget in [0, std::mem::size_of::<Option<ValidatedRange>>() - 1] {
            let bytes = position_fixture();
            let len = bytes.len() as u64;
            let reader = PostingListReader::new(
                FileHandle::empty(),
                Some(FileHandle::from_bytes(OwnedBytes::new(bytes))),
                budget,
            )
            .unwrap();
            reader.read_positions(0..len).await.unwrap();
            reader.read_positions(0..len).await.unwrap();
            assert_eq!(validations(&reader), 2);
            assert_eq!(reader.heap_bytes(), 0);
        }
    }

    #[tokio::test]
    async fn lazy_postings_stay_uncached_when_inline_positions_enable_shared_cache() {
        let bytes = fixture(PostingCodec::Rounded);
        let len = bytes.len() as u64;
        let positions = position_fixture();
        let position_len = positions.len() as u64;
        let calls = Arc::new(AtomicUsize::new(0));
        let reader = PostingListReader::new(
            FileHandle::lazy(
                len,
                Arc::new({
                    let calls = calls.clone();
                    move |_| {
                        let mut data = bytes.clone();
                        if calls.fetch_add(1, Ordering::Relaxed) != 0 {
                            data[6] = 0xe1;
                        }
                        Box::pin(async move { Ok(OwnedBytes::new(data)) })
                    }
                }),
            ),
            Some(FileHandle::from_bytes(OwnedBytes::new(positions))),
            4096,
        )
        .unwrap();
        assert!(reader.heap_bytes() > 0 && reader.heap_bytes() <= 4096);
        reader.read_positions(0..position_len).await.unwrap();
        reader.read_positions(0..position_len).await.unwrap();
        assert_eq!(validations(&reader), 1);
        reader.read(0..len).await.unwrap();
        assert!(
            reader
                .read(0..len)
                .await
                .unwrap_err()
                .to_string()
                .contains("exceeds 32 bits")
        );
        assert_eq!(validations(&reader), 3);
        assert_eq!(calls.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn lazy_positions_validate_actual_reads_even_when_postings_can_cache() {
        let bytes = Arc::new(position_fixture());
        let len = bytes.len() as u64;
        let calls = Arc::new(AtomicUsize::new(0));
        let reader = PostingListReader::new(
            FileHandle::from_bytes(OwnedBytes::new(fixture(PostingCodec::Rounded))),
            Some(FileHandle::lazy(
                len,
                Arc::new({
                    let calls = calls.clone();
                    let bytes = bytes.clone();
                    move |_| {
                        let call = calls.fetch_add(1, Ordering::Relaxed);
                        let mut data = bytes.as_ref().clone();
                        Box::pin(async move {
                            match call {
                                1 => data[2] = 7,
                                2 => {
                                    data.pop();
                                }
                                3 => return Err(io::Error::other("injected position I/O error")),
                                4 => futures::future::pending().await,
                                _ => (),
                            }
                            Ok(OwnedBytes::new(data))
                        })
                    }
                }),
            )),
            4096,
        )
        .unwrap();
        assert!(reader.heap_bytes() > 0 && reader.heap_bytes() <= 4096);
        reader.read_positions(0..len).await.unwrap();
        assert!(
            reader
                .read_positions(0..len)
                .await
                .unwrap_err()
                .to_string()
                .contains("invalid position block")
        );
        assert_eq!(
            reader.read_positions(0..len).await.unwrap_err().kind(),
            io::ErrorKind::UnexpectedEof
        );
        assert!(
            reader
                .read_positions(0..len)
                .await
                .unwrap_err()
                .to_string()
                .contains("injected position I/O")
        );
        let mut cancelled = Box::pin(reader.read_positions(0..len));
        assert!(futures::poll!(cancelled.as_mut()).is_pending());
        drop(cancelled);
        assert_eq!(
            reader
                .read_positions(0..len)
                .await
                .unwrap()
                .positions(0, 0, 3),
            Some(vec![1, 5, 9])
        );
        assert_eq!(validations(&reader), 3);
        assert!(reader.validated.read().iter().all(Option::is_none));
        assert!(reader.read_positions(len..len + 1).await.is_err());
        assert!(
            reader
                .read_positions(Range { start: 2, end: 1 })
                .await
                .is_err()
        );
        assert_eq!(calls.load(Ordering::Relaxed), 6);
    }

    #[tokio::test]
    async fn immutable_repeated_reads_share_validation_and_preserve_all_codec_bytes() {
        for codec in [
            PostingCodec::Rounded,
            PostingCodec::Packed,
            PostingCodec::Pfor,
            PostingCodec::Simd4x,
        ] {
            let bytes = fixture(codec);
            let reader =
                new_reader(FileHandle::from_bytes(OwnedBytes::new(bytes.clone())), 4096).unwrap();
            for _ in 0..3 {
                assert_bytes(reader.read(0..bytes.len() as u64).await.unwrap(), &bytes);
                #[cfg(feature = "sync")]
                assert_bytes(reader.read_sync(0..bytes.len() as u64).unwrap(), &bytes);
            }
            assert_eq!(validations(&reader), 1);
            assert!(reader.heap_bytes() <= 4096);
            assert_eq!(
                reader
                    .decode(0..bytes.len() as u64, OwnedBytes::new(Vec::new()))
                    .unwrap_err()
                    .kind(),
                io::ErrorKind::UnexpectedEof
            );
            assert_eq!(validations(&reader), 1);
        }
    }

    #[tokio::test]
    async fn colliding_immutable_ranges_reuse_proofs_within_the_existing_budget() {
        let bytes = fixture(PostingCodec::Rounded);
        let len = bytes.len() as u64;
        let budget = 4 * std::mem::size_of::<Option<ValidatedRange>>();
        let reader = new_reader(
            FileHandle::from_bytes(OwnedBytes::new(bytes.repeat(8))),
            budget,
        )
        .unwrap();
        // Select two ranges that compete for one original direct-mapped slot.
        let mut first = [None; 4];
        let mut pair = None;
        for i in 0..8u64 {
            let range = i * len..(i + 1) * len;
            let mut hash = FxHasher::default();
            range.hash(&mut hash);
            let slot = hash.finish() as usize % first.len();
            if let Some(old) = first[slot] {
                pair = Some((old, i));
                break;
            }
            first[slot] = Some(i);
        }
        let (a, b) = pair.unwrap();
        for _ in 0..3 {
            for i in [a, b] {
                assert_bytes(reader.read(i * len..(i + 1) * len).await.unwrap(), &bytes);
                #[cfg(feature = "sync")]
                assert_bytes(reader.read_sync(i * len..(i + 1) * len).unwrap(), &bytes);
            }
        }
        assert_eq!(
            validations(&reader),
            2,
            "both immutable proofs fit in the existing budget"
        );
        assert_eq!(reader.heap_bytes(), budget);
    }

    #[tokio::test]
    async fn collisions_revalidate_exact_ranges_without_reusing_another_footer() {
        let first = fixture(PostingCodec::Rounded);
        let second = fixture(PostingCodec::Pfor);
        let split = first.len() as u64;
        let end = split + second.len() as u64;
        let data = [first.as_slice(), second.as_slice()].concat();
        let reader = new_reader(
            FileHandle::from_bytes(OwnedBytes::new(data)),
            std::mem::size_of::<Option<ValidatedRange>>(),
        )
        .unwrap();
        for _ in 0..3 {
            assert_bytes(reader.read(0..split).await.unwrap(), &first);
            assert_bytes(reader.read(split..end).await.unwrap(), &second);
        }
        assert_eq!(validations(&reader), 6);
        assert_eq!(reader.slots, 1);
    }

    #[tokio::test]
    async fn disabled_and_sub_entry_budgets_revalidate_without_allocation() {
        for budget in [0, std::mem::size_of::<Option<ValidatedRange>>() - 1] {
            let bytes = fixture(PostingCodec::Rounded);
            let len = bytes.len() as u64;
            let reader =
                new_reader(FileHandle::from_bytes(OwnedBytes::new(bytes)), budget).unwrap();
            reader.read(0..len).await.unwrap();
            reader.read(0..len).await.unwrap();
            assert_eq!(validations(&reader), 2);
            assert_eq!(reader.heap_bytes(), 0);
        }
        assert!(new_reader(FileHandle::empty(), 64 * 1024 * 1024 + 1).is_err());
    }

    #[tokio::test]
    async fn replacement_owner_rejects_corruption_at_previously_validated_offsets() {
        let bytes = fixture(PostingCodec::Rounded);
        let len = bytes.len() as u64;
        let old = new_reader(FileHandle::from_bytes(OwnedBytes::new(bytes.clone())), 4096).unwrap();
        old.read(0..len).await.unwrap();
        let mut corrupt = bytes.clone();
        corrupt[6] = 0xe1;
        let new = new_reader(FileHandle::from_bytes(OwnedBytes::new(corrupt)), 4096).unwrap();
        for _ in 0..2 {
            assert!(
                new.read(0..len)
                    .await
                    .unwrap_err()
                    .to_string()
                    .contains("exceeds 32 bits")
            );
            assert_bytes(old.read(0..len).await.unwrap(), &bytes);
        }
        assert_eq!(validations(&old), 1);
        assert_eq!(validations(&new), 2);
        assert!(new.validated.read().iter().all(Option::is_none));
    }

    #[tokio::test]
    async fn lazy_reads_never_reuse_validation_or_hide_io_errors_or_cancellation() {
        let bytes = Arc::new(fixture(PostingCodec::Rounded));
        let len = bytes.len() as u64;
        let calls = Arc::new(AtomicUsize::new(0));
        let reader = new_reader(
            FileHandle::lazy(
                len,
                Arc::new({
                    let calls = Arc::clone(&calls);
                    let bytes = Arc::clone(&bytes);
                    move |_| {
                        let call = calls.fetch_add(1, Ordering::Relaxed);
                        let mut data = bytes.as_ref().clone();
                        Box::pin(async move {
                            match call {
                                1 => return Err(io::Error::other("injected I/O error")),
                                2 => data[6] = 0xe1,
                                3 => {
                                    data.pop();
                                }
                                4 => futures::future::pending().await,
                                _ => (),
                            }
                            Ok(OwnedBytes::new(data))
                        })
                    }
                }),
            ),
            4096,
        )
        .unwrap();
        assert_eq!(reader.heap_bytes(), 0);
        assert_bytes(reader.read(0..len).await.unwrap(), &bytes);
        assert!(
            reader
                .read(0..len)
                .await
                .unwrap_err()
                .to_string()
                .contains("injected I/O")
        );
        assert!(
            reader
                .read(0..len)
                .await
                .unwrap_err()
                .to_string()
                .contains("exceeds 32 bits")
        );
        assert_eq!(
            reader.read(0..len).await.unwrap_err().kind(),
            io::ErrorKind::UnexpectedEof
        );
        let mut cancelled = Box::pin(reader.read(0..len));
        assert!(futures::poll!(cancelled.as_mut()).is_pending());
        drop(cancelled);
        assert_bytes(reader.read(0..len).await.unwrap(), &bytes);
        assert_eq!(calls.load(Ordering::Relaxed), 6);
        assert_eq!(validations(&reader), 3);
        assert!(reader.read(len..len + 1).await.is_err());
        assert!(reader.read(Range { start: 2, end: 1 }).await.is_err());
        assert_eq!(
            calls.load(Ordering::Relaxed),
            6,
            "invalid ranges must fail before I/O"
        );
    }
}
