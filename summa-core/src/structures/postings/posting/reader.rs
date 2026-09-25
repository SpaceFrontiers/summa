//! Borrowed query views over writer-produced text files.
use std::io;
use std::ops::Range;

use super::super::positions_v2::{PositionStream, TermPositions};
use super::{BlockPostingList, Footer};
use crate::directories::{FileHandle, OwnedBytes};

/// Query-local handles share one segment-wide write-once error state.
#[derive(Clone, Debug, Default)]
pub(super) struct PostingIntegrity {
    first_error: std::sync::Arc<std::sync::OnceLock<usize>>,
}

impl PostingIntegrity {
    pub(super) fn record(&self, block: usize) {
        let _ = self.first_error.set(block);
    }
}

/// One validated posting envelope; optional list views are built only on use.
#[derive(Debug)]
pub(crate) struct DeferredPosting {
    bytes: OwnedBytes,
    footer: Footer,
    content_error: std::sync::Arc<PostingIntegrity>,
}

impl DeferredPosting {
    pub(crate) fn doc_count(&self) -> u32 {
        self.footer.doc_count
    }

    pub(crate) fn first_doc(&self) -> Option<crate::DocId> {
        (self.footer.l0_count != 0)
            .then(|| super::read_l0(&self.bytes[self.footer.l0_start()..self.footer.l0_end()], 0).0)
    }

    pub(crate) fn into_list(self) -> BlockPostingList {
        let mut list = BlockPostingList::from_layout(self.bytes, self.footer);
        list.verify_content = false;
        list.content_error = Some(self.content_error);
        list
    }

    #[cfg(test)]
    pub(crate) fn from_list(list: &BlockPostingList) -> Self {
        let mut bytes = Vec::new();
        list.serialize(&mut bytes).unwrap();
        let bytes = OwnedBytes::new(bytes);
        let footer = Footer::parse(&bytes).unwrap();
        Self {
            bytes,
            footer,
            content_error: Default::default(),
        }
    }
}

#[derive(Clone)]
pub(crate) struct PostingListReader {
    file: FileHandle,
    positions: Option<FileHandle>,
    content_error: std::sync::Arc<PostingIntegrity>,
}

impl PostingListReader {
    pub(crate) fn for_expansion(&self) -> Self {
        Self {
            file: self.file.with_local_owner(),
            positions: self.positions.clone(),
            content_error: std::sync::Arc::new((*self.content_error).clone()),
        }
    }

    pub(crate) fn new(file: FileHandle, positions: Option<FileHandle>) -> Self {
        Self {
            file,
            positions,
            content_error: Default::default(),
        }
    }

    pub(crate) fn check_integrity(&self) -> io::Result<()> {
        match self.content_error.first_error.get() {
            Some(block) => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("posting payload corruption detected in block {block}"),
            )),
            None => Ok(()),
        }
    }

    pub(crate) fn integrity_heap_bytes(&self) -> usize {
        std::mem::size_of::<std::sync::OnceLock<usize>>()
            + std::mem::size_of::<PostingIntegrity>()
            + 4 * std::mem::size_of::<usize>()
    }

    pub(crate) fn file(&self) -> &FileHandle {
        &self.file
    }

    pub(crate) fn positions_file(&self) -> Option<&FileHandle> {
        self.positions.as_ref()
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

    pub(crate) async fn read_deferred(&self, range: Range<u64>) -> io::Result<DeferredPosting> {
        Self::check_range(&self.file, &range)?;
        let bytes = self.file.read_bytes_range(range.clone()).await?;
        self.decode_deferred(range, bytes)
    }

    #[cfg(feature = "sync")]
    pub(crate) fn read_deferred_sync(&self, range: Range<u64>) -> io::Result<DeferredPosting> {
        Self::check_range(&self.file, &range)?;
        let bytes = self.file.read_bytes_range_sync(range.clone())?;
        self.decode_deferred(range, bytes)
    }

    pub(crate) async fn read_positions(&self, range: Range<u64>) -> io::Result<TermPositions> {
        let file = self.position_file()?;
        Self::check_range(file, &range)?;
        let bytes = file.read_bytes_range(range.clone()).await?;
        self.decode_positions(range, bytes)
    }

    #[cfg(feature = "sync")]
    pub(crate) fn read_positions_sync(&self, range: Range<u64>) -> io::Result<TermPositions> {
        let file = self.position_file()?;
        Self::check_range(file, &range)?;
        let bytes = file.read_bytes_range_sync(range.clone())?;
        self.decode_positions(range, bytes)
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

    fn decode(&self, range: Range<u64>, bytes: OwnedBytes) -> io::Result<BlockPostingList> {
        self.decode_deferred(range, bytes)
            .map(DeferredPosting::into_list)
    }

    fn decode_deferred(&self, range: Range<u64>, bytes: OwnedBytes) -> io::Result<DeferredPosting> {
        Self::check_length(&range, &bytes)?;
        let footer = Footer::parse(&bytes)?;
        crate::observe::search_work!(postings_opened += 1);
        Ok(DeferredPosting {
            bytes,
            footer,
            content_error: std::sync::Arc::clone(&self.content_error),
        })
    }

    fn decode_positions(&self, range: Range<u64>, bytes: OwnedBytes) -> io::Result<TermPositions> {
        Self::check_length(&range, &bytes)?;
        let stream = PositionStream::open_for_query(bytes)?;
        crate::observe::search_work!(positions_opened += 1);
        Ok(TermPositions(stream))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structures::{PostingCodec, PostingList};

    #[tokio::test]
    async fn expansion_decode_failures_remain_visible_to_all_segment_readers() {
        let mut bytes = fixture(PostingCodec::Rounded);
        bytes[..2].copy_from_slice(&129_u16.to_le_bytes());
        let len = bytes.len() as u64;
        let reader = PostingListReader::new(FileHandle::from_bytes(OwnedBytes::new(bytes)), None);
        let sibling = reader.for_expansion();
        let expansion = reader.for_expansion();
        let list = expansion.read(0..len).await.unwrap();
        drop(expansion);
        assert!(list.decode_block_doc_ids_only(0, &mut Vec::new()).is_none());
        assert!(reader.check_integrity().is_err());
        assert!(sibling.check_integrity().is_err());
        sibling.content_error.record(99);
        assert_eq!(reader.content_error.first_error.get(), Some(&0));
    }

    #[tokio::test]
    async fn deferred_views_preserve_bytes_and_outlive_their_reader_for_every_codec() {
        for codec in [
            PostingCodec::Rounded,
            PostingCodec::Packed,
            PostingCodec::Pfor,
            PostingCodec::Simd4x,
        ] {
            let bytes = fixture(codec);
            let reader = PostingListReader::new(
                FileHandle::from_bytes(OwnedBytes::new(bytes.clone())),
                None,
            );
            let deferred = reader.read_deferred(0..bytes.len() as u64).await.unwrap();
            assert_eq!(deferred.doc_count(), 1291);
            assert_eq!(deferred.first_doc(), Some(0));
            #[cfg(feature = "sync")]
            {
                let sync = reader.read_deferred_sync(0..bytes.len() as u64).unwrap();
                let mut encoded = Vec::new();
                sync.into_list().serialize(&mut encoded).unwrap();
                assert_eq!(encoded, bytes);
            }
            drop(reader);
            let list = deferred.into_list();
            let mut encoded = Vec::new();
            list.serialize(&mut encoded).unwrap();
            assert_eq!(encoded, bytes);
            let strict = BlockPostingList::deserialize(&bytes).unwrap();
            for block in 0..strict.num_blocks() {
                let (mut expected, mut actual) = (Vec::new(), Vec::new());
                assert!(
                    strict
                        .decode_block_doc_ids_only(block, &mut expected)
                        .is_some()
                );
                assert!(list.decode_block_doc_ids_only(block, &mut actual).is_some());
                assert_eq!(actual, expected);
            }
        }
    }

    #[tokio::test]
    async fn deferred_open_rejects_bad_envelopes_and_retains_the_segment_error_observer() {
        let mut bytes = fixture(PostingCodec::Rounded);
        bytes[..2].copy_from_slice(&129_u16.to_le_bytes());
        let len = bytes.len() as u64;
        let reader = PostingListReader::new(FileHandle::from_bytes(OwnedBytes::new(bytes)), None);
        let expansion = reader.for_expansion();
        let deferred = expansion.read_deferred(0..len).await.unwrap();
        drop(expansion);
        assert!(reader.check_integrity().is_ok());
        assert!(
            deferred
                .into_list()
                .decode_block_doc_ids_only(0, &mut Vec::new())
                .is_none()
        );
        assert!(reader.check_integrity().is_err());
        assert!(reader.read_deferred(0..len + 1).await.is_err());
        assert!(reader.read_deferred(0..len - 1).await.is_err());
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

    #[tokio::test]
    async fn trusted_query_views_preserve_bytes_and_decoded_blocks_for_every_codec() {
        for codec in [
            PostingCodec::Rounded,
            PostingCodec::Packed,
            PostingCodec::Pfor,
            PostingCodec::Simd4x,
        ] {
            let bytes = fixture(codec);
            let strict = BlockPostingList::deserialize(&bytes).unwrap();
            let reader = PostingListReader::new(
                FileHandle::from_bytes(OwnedBytes::new(bytes.clone())),
                None,
            );
            let list = reader.read(0..bytes.len() as u64).await.unwrap();
            assert!(!list.verify_content);
            let mut encoded = Vec::new();
            list.serialize(&mut encoded).unwrap();
            assert_eq!(encoded, bytes);
            let (mut expected_docs, mut expected_tfs, mut docs, mut tfs) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for block in 0..strict.num_blocks() {
                assert!(strict.decode_block_into(block, &mut expected_docs, &mut expected_tfs));
                assert!(list.decode_block_into(block, &mut docs, &mut tfs));
                assert_eq!((&docs, &tfs), (&expected_docs, &expected_tfs));
            }
            #[cfg(feature = "sync")]
            {
                let sync = reader.read_sync(0..bytes.len() as u64).unwrap();
                let mut encoded = Vec::new();
                sync.serialize(&mut encoded).unwrap();
                assert_eq!(encoded, bytes);
            }
        }
    }

    #[tokio::test]
    async fn query_open_does_not_scan_block_headers_but_explicit_deserialization_does() {
        let mut bytes = fixture(PostingCodec::Rounded);
        bytes[6] = 0xe1; // Invalid codec width, outside the footer envelope.
        assert!(BlockPostingList::deserialize(&bytes).is_err());
        let len = bytes.len() as u64;
        let reader = PostingListReader::new(FileHandle::from_bytes(OwnedBytes::new(bytes)), None);
        let list = reader.read(0..len).await.unwrap();
        assert!(list.decode_block_doc_ids_only(0, &mut Vec::new()).is_none());
        assert!(reader.check_integrity().is_err());
        assert!(reader.read(0..len + 1).await.is_err());
    }

    #[tokio::test]
    async fn trusted_block_decode_bounds_output_before_entering_fixed_size_kernels() {
        for count in [0u16, 129, u16::MAX] {
            let mut bytes = fixture(PostingCodec::Simd4x);
            bytes[..2].copy_from_slice(&count.to_le_bytes());
            let len = bytes.len() as u64;
            let reader =
                PostingListReader::new(FileHandle::from_bytes(OwnedBytes::new(bytes)), None);
            let list = reader.read(0..len).await.unwrap();
            let mut docs = Vec::new();
            assert!(list.decode_block_doc_ids_only(0, &mut docs).is_none());
            assert_eq!(docs.capacity(), 0);
            assert!(reader.check_integrity().is_err());
        }
    }

    #[tokio::test]
    async fn query_decode_trusts_document_order_while_explicit_deserialization_checks_it() {
        let mut bytes = fixture(PostingCodec::Rounded);
        assert_eq!(bytes[6], 8);
        bytes[8] = 0; // A duplicate ID is a writer invariant, not a query check.
        let strict = BlockPostingList::deserialize(&bytes).unwrap();
        assert!(
            strict
                .decode_block_doc_ids_only(0, &mut Vec::new())
                .is_none()
        );
        let len = bytes.len() as u64;
        let reader = PostingListReader::new(FileHandle::from_bytes(OwnedBytes::new(bytes)), None);
        let list = reader.read(0..len).await.unwrap();
        let mut docs = Vec::new();
        assert!(list.decode_block_doc_ids_only(0, &mut docs).is_some());
        assert_eq!(docs[0], docs[1]);
        reader.check_integrity().unwrap();
    }

    #[tokio::test]
    async fn position_query_open_skips_payload_scan_and_preserves_healthy_positions() {
        let mut bytes = Vec::new();
        let mut encoder = crate::structures::PositionStreamEncoder::new(&mut bytes);
        encoder.push_doc(&mut [1, 5, 9]).unwrap();
        encoder.finish().unwrap();
        let len = bytes.len() as u64;
        let reader = PostingListReader::new(
            FileHandle::empty(),
            Some(FileHandle::from_bytes(OwnedBytes::new(bytes.clone()))),
        );
        assert_eq!(
            reader.read_positions(0..len).await.unwrap().positions(0, 3),
            Some(vec![1, 5, 9])
        );
        bytes[2] = 7;
        assert!(PositionStream::open(OwnedBytes::new(bytes.clone())).is_err());
        let reader = PostingListReader::new(
            FileHandle::empty(),
            Some(FileHandle::from_bytes(OwnedBytes::new(bytes))),
        );
        reader.read_positions(0..len).await.unwrap();
    }
}
