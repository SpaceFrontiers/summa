//! Bounded encoded lookahead owned by one MaxScore cursor.
use super::{OwnedBytes, SparseBlock, SparseIndex, checked_sparse_block_range};

const MAX_BLOCKS: usize = 8;
const MAX_BYTES: u64 = 16 * 1024;

#[derive(Default)]
pub(crate) struct SparseReadWindow {
    bytes: Option<OwnedBytes>,
    start: u64,
}

#[cfg(all(test, feature = "native"))]
mod tests {
    use super::*;
    use crate::directories::FileHandle;
    use crate::structures::{SparseSkipEntry, WeightQuantization};
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    };

    #[tokio::test]
    async fn cursor_window_caps_bytes_before_decoding_and_rejects_out_of_file_metadata() {
        let bytes = OwnedBytes::new(vec![0; 40 * 1024]);
        let mut skips = Vec::new();
        for block in 0..10 {
            SparseSkipEntry::new(block, block, u64::from(block) * 4096, 4096, 1.0)
                .write_to_vec(&mut skips);
        }
        let mut index = SparseIndex::new(
            FileHandle::from_bytes(bytes.clone()),
            super::super::DimensionTable::new(),
            OwnedBytes::new(skips),
            10,
            10,
        );
        index.set_payload_handle(FileHandle::from_bytes(bytes));
        let mut window = SparseReadWindow::default();
        // Invalid block contents must fail decoding; the admitted byte window
        // is nevertheless bounded to four 4 KiB blocks, not eight.
        assert!(
            index
                .load_cursor_block(0, 0, 0, 10, &mut window)
                .await
                .is_err()
        );
        assert_eq!(window.bytes.as_ref().unwrap().len(), MAX_BYTES as usize);
        assert!(
            index
                .load_cursor_block(0, u64::MAX, 0, 10, &mut window)
                .await
                .is_err()
        );
    }

    fn encoded_blocks(count: u32) -> (SparseIndex, OwnedBytes, Vec<std::ops::Range<usize>>) {
        let mut bytes = Vec::new();
        let mut skips = Vec::new();
        let mut ranges = Vec::new();
        for block in 0..count {
            let start = bytes.len();
            SparseBlock::from_postings(&[(block, 2, 0.5)], WeightQuantization::Float32)
                .unwrap()
                .write(&mut bytes)
                .unwrap();
            ranges.push(start..bytes.len());
            SparseSkipEntry::new(
                block,
                block,
                start as u64,
                (bytes.len() - start) as u32,
                0.5,
            )
            .write_to_vec(&mut skips);
        }
        let bytes = OwnedBytes::new(bytes);
        let mut dims = super::super::DimensionTable::new();
        dims.push(1, 0, 0, count, count, 0.5);
        let index = SparseIndex::new(
            FileHandle::from_bytes(bytes.clone()),
            dims,
            OwnedBytes::new(skips),
            count,
            count,
        );
        (index, bytes, ranges)
    }

    #[tokio::test]
    async fn cursor_window_preserves_bytes_and_recovers_after_failed_or_cancelled_refill() {
        let (mut index, bytes, ranges) = encoded_blocks(20);
        let mode = Arc::new(AtomicUsize::new(0));
        let reads = Arc::new(Mutex::new(Vec::new()));
        index.set_payload_handle(FileHandle::lazy(
            bytes.len() as u64,
            Arc::new({
                let bytes = bytes.clone();
                let mode = mode.clone();
                let reads = reads.clone();
                move |range| {
                    reads.lock().unwrap().push(range.clone());
                    let bytes = bytes.clone();
                    let mode = mode.load(Ordering::Relaxed);
                    Box::pin(async move {
                        match mode {
                            1 => Err(std::io::Error::other("injected read failure")),
                            2 => std::future::pending().await,
                            3 => Ok(OwnedBytes::empty()),
                            _ => Ok(bytes.slice(range.start as usize..range.end as usize)),
                        }
                    })
                }
            }),
        ));
        let mut window = SparseReadWindow::default();
        for (block, range) in ranges.iter().enumerate().take(8) {
            let loaded = index
                .load_cursor_block(0, 0, block, 17, &mut window)
                .await
                .unwrap()
                .unwrap();
            let mut encoded = Vec::new();
            loaded.write(&mut encoded).unwrap();
            assert_eq!(encoded.as_slice(), &bytes.as_slice()[range.clone()]);
        }
        assert_eq!(*reads.lock().unwrap(), vec![0..ranges[7].end as u64]);
        for failure in [1, 3] {
            mode.store(failure, Ordering::Relaxed);
            assert!(
                index
                    .load_cursor_block(0, 0, 8, 17, &mut window)
                    .await
                    .is_err()
            );
            assert!(window.bytes.is_none());
        }
        mode.store(2, Ordering::Relaxed);
        {
            let future = index.load_cursor_block(0, 0, 8, 17, &mut window);
            futures::pin_mut!(future);
            assert!(futures::poll!(future).is_pending());
        }
        assert!(window.bytes.is_none());
        mode.store(0, Ordering::Relaxed);
        // The window must stop at this dimension's end, before the next list.
        index
            .load_cursor_block(0, 0, 16, 17, &mut window)
            .await
            .unwrap();
        assert_eq!(
            reads.lock().unwrap().last().unwrap(),
            &(ranges[16].start as u64..ranges[16].end as u64)
        );
        index
            .load_cursor_block(0, 0, 0, 17, &mut window)
            .await
            .unwrap();
        let count = reads.lock().unwrap().len();
        assert!(
            index
                .load_cursor_block(usize::MAX, 0, 0, 17, &mut window)
                .await
                .is_err()
        );
        assert_eq!(reads.lock().unwrap().len(), count);
        assert!(
            reads
                .lock()
                .unwrap()
                .iter()
                .all(|r| r.end - r.start <= MAX_BYTES)
        );
    }

    #[tokio::test]
    async fn maxscore_refill_failure_is_not_partial_success_and_cancelled_queries_release_reads() {
        use crate::query::MaxScoreExecutor;
        let (mut index, bytes, _) = encoded_blocks(20);
        let mode = Arc::new(AtomicUsize::new(0));
        let pending = Arc::new(AtomicUsize::new(0));
        struct PendingRead(Arc<AtomicUsize>);
        impl Drop for PendingRead {
            fn drop(&mut self) {
                self.0.fetch_sub(1, Ordering::Relaxed);
            }
        }
        index.set_payload_handle(FileHandle::lazy(
            bytes.len() as u64,
            Arc::new({
                let mode = mode.clone();
                let pending = pending.clone();
                move |range| {
                    let bytes = bytes.clone();
                    let mode = mode.load(Ordering::Relaxed);
                    let pending = pending.clone();
                    Box::pin(async move {
                        // First eight blocks complete; suspend/fail only a refill
                        // after scoring has already accumulated partial results.
                        if range.start != 0 {
                            match mode {
                                1 => return Err(std::io::Error::other("injected refill failure")),
                                2 => {
                                    pending.fetch_add(1, Ordering::Relaxed);
                                    let _read = PendingRead(pending);
                                    std::future::pending::<()>().await;
                                }
                                _ => {}
                            }
                        }
                        Ok(bytes.slice(range.start as usize..range.end as usize))
                    })
                }
            }),
        ));
        let make = || MaxScoreExecutor::sparse(&index, vec![(1, 1.0)], 20, 1.0);
        let expected = make().execute_sync().unwrap();
        mode.store(1, Ordering::Relaxed);
        assert!(make().execute().await.is_err());
        mode.store(2, Ordering::Relaxed);
        {
            let query = make().execute();
            futures::pin_mut!(query);
            assert!(futures::poll!(query).is_pending());
            assert_eq!(pending.load(Ordering::Relaxed), 1);
        }
        assert_eq!(pending.load(Ordering::Relaxed), 0);
        mode.store(0, Ordering::Relaxed);
        let actual = make().execute().await.unwrap();
        let identities = |rows: Vec<crate::query::ScoredDoc>| {
            rows.into_iter()
                .map(|row| (row.doc_id, row.score.to_bits(), row.ordinal))
                .collect::<Vec<_>>()
        };
        assert_eq!(identities(actual), identities(expected));
    }
}

impl SparseIndex {
    pub(crate) async fn load_cursor_block(
        &self,
        skip_start: usize,
        base: u64,
        block: usize,
        count: usize,
        window: &mut SparseReadWindow,
    ) -> crate::Result<Option<SparseBlock>> {
        if self.payload_handle.is_none() {
            return self.load_block_direct(skip_start, base, block).await;
        }
        if block >= count {
            return Ok(None);
        }
        let end = skip_start
            .checked_add(count)
            .filter(|&end| end <= self.skip_entry_count())
            .ok_or_else(|| {
                crate::Error::Corruption("sparse cursor skip range exceeds metadata".into())
            })?;
        let entry = skip_start + block;
        let requested =
            checked_sparse_block_range(base, self.read_skip_entry(entry), self.handle.len())?;
        if requested.end - requested.start > MAX_BYTES {
            window.bytes = None;
            return self.load_block_direct(skip_start, base, block).await;
        }
        let present = window.bytes.as_ref().is_some_and(|bytes| {
            requested.start >= window.start && requested.end - window.start <= bytes.len() as u64
        });
        if !present {
            let mut read_end = requested.end;
            for next in entry + 1..entry.saturating_add(MAX_BLOCKS).min(end) {
                let range = checked_sparse_block_range(
                    base,
                    self.read_skip_entry(next),
                    self.handle.len(),
                )?;
                if range.start != read_end || range.end - requested.start > MAX_BYTES {
                    break;
                }
                read_end = range.end;
            }
            // Drop the old window before awaiting. Cancellation/failure leaves
            // it empty; only a fully completed read is installed.
            window.bytes = None;
            let bytes = self
                .payload_handle()
                .read_bytes_range(requested.start..read_end)
                .await?;
            window.start = requested.start;
            window.bytes = Some(bytes);
        }
        let start = (requested.start - window.start) as usize;
        let end = (requested.end - window.start) as usize;
        SparseBlock::from_owned_bytes(window.bytes.as_ref().unwrap().slice(start..end))
            .map(Some)
            .map_err(|error| {
                crate::Error::Corruption(format!("sparse cursor block {block}: {error}"))
            })
    }
}
