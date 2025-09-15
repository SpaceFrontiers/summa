//! Borrowed query views over writer-produced text files.
use std::io;
use std::ops::Range;

use super::super::positions_v2::{PositionStream, TermPositions};
use super::{BlockPostingList, Footer};
use crate::directories::{FileHandle, OwnedBytes};

pub(crate) struct PostingListReader {
    file: FileHandle,
    positions: Option<FileHandle>,
    content_error: std::sync::Arc<std::sync::OnceLock<usize>>,
}

impl PostingListReader {
    pub(crate) fn new(file: FileHandle, positions: Option<FileHandle>) -> Self {
        Self {
            file,
            positions,
            content_error: Default::default(),
        }
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
        Self::check_length(&range, &bytes)?;
        let footer = Footer::parse(&bytes)?;
        let mut list = BlockPostingList::from_layout(bytes, footer);
        list.verify_content = false;
        list.content_error = Some(std::sync::Arc::clone(&self.content_error));
        crate::observe::search_work!(postings_opened += 1);
        Ok(list)
    }

    fn decode_positions(&self, range: Range<u64>, bytes: OwnedBytes) -> io::Result<TermPositions> {
        Self::check_length(&range, &bytes)?;
        if !PositionStream::is_stream(bytes.as_slice()) {
            return TermPositions::open(bytes);
        }
        let stream = PositionStream::open_for_query(bytes)?;
        crate::observe::search_work!(positions_opened += 1);
        Ok(TermPositions::Stream(stream))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structures::{PostingCodec, PostingList};

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
            reader
                .read_positions(0..len)
                .await
                .unwrap()
                .positions(0, 0, 3),
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
