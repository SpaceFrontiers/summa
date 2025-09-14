use super::*;

#[test]
fn default_dictionary_writer_preserves_frozen_v5_bytes() {
    let mut writer = SSTableWriter::<_, TermInfo>::new(Vec::new());
    for i in 0..5000u64 {
        let key = format!("word{i:08}");
        writer
            .insert(
                key.as_bytes(),
                &TermInfo::external(i * 1024, 768, (i % 301 + 4) as u32),
            )
            .unwrap();
    }
    assert_eq!(
        writer.finish().unwrap(),
        include_bytes!("testdata/default_v5.bin")
    );
}

#[test]
fn oversized_dictionary_key_is_rejected_before_writing_unreadable_data() {
    let mut writer = SSTableWriter::<_, TermInfo>::new(Vec::new());
    let key = vec![b'a'; MAX_SSTABLE_BLOCK_BYTES + 1];
    assert!(writer.insert(&key, &TermInfo::external(0, 1, 4)).is_err());
    assert_eq!(writer.num_entries, 0);
    assert_eq!(writer.current_offset, 0);
    assert!(writer.block_buffer.is_empty());
}

#[test]
fn oversized_dictionary_value_cannot_publish_a_block_the_reader_rejects() {
    let mut writer = SSTableWriter::<_, Vec<u8>>::new(Vec::new());
    let value = vec![0; MAX_SSTABLE_BLOCK_BYTES];
    assert!(writer.insert(b"key", &value).is_err());
    assert_eq!(writer.current_offset, 0);
    assert!(writer.finish().is_err());
}

fn small_block_table() -> Vec<u8> {
    let mut writer = SSTableWriter::<_, u64>::with_config(
        Vec::new(),
        SSTableWriterConfig {
            block_size: SSTableBlockSize::try_from(512).unwrap(),
            ..Default::default()
        },
    );
    for i in 0..1000u64 {
        writer.insert(format!("key{i:08}").as_bytes(), &i).unwrap();
    }
    writer.finish().unwrap()
}

#[tokio::test]
async fn bulk_dictionary_prefetch_never_expands_the_configured_cache() {
    let reader = AsyncSSTableReader::<u64>::open(
        FileHandle::from_bytes(OwnedBytes::new(small_block_table())),
        2,
    )
    .await
    .unwrap();
    assert!(reader.stats().num_blocks > 2);
    reader.prefetch_leading_blocks().await.unwrap();
    assert!(reader.cached_blocks() <= 2);
}

#[tokio::test]
async fn zero_dictionary_cache_budget_prevents_bulk_payload_reads() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let bytes = OwnedBytes::new(small_block_table());
    let reads = Arc::new(AtomicUsize::new(0));
    let file = FileHandle::lazy(
        bytes.len() as u64,
        Arc::new({
            let reads = Arc::clone(&reads);
            move |range| {
                reads.fetch_add(1, Ordering::Relaxed);
                let data = bytes.slice(range.start as usize..range.end as usize);
                Box::pin(async move { Ok(data) })
            }
        }),
    );
    let reader = AsyncSSTableReader::<u64>::open_with_cache_budget(file, 100, Some(0))
        .await
        .unwrap();
    reads.store(0, Ordering::Relaxed);
    reader.prefetch_leading_blocks().await.unwrap();
    assert_eq!(reads.load(Ordering::Relaxed), 0);
    assert_eq!(reader.cached_bytes(), 0);
}

#[test]
fn dictionary_cache_counts_bytes_through_eviction_duplicates_and_oversized_bypass() {
    let mut cache = BlockCache::new(2, Some(1000));
    cache.insert(1, Arc::from(vec![1; 600]));
    cache.insert(2, Arc::from(vec![2; 300]));
    assert_eq!(cache.retained_bytes, 900);
    cache.insert(3, Arc::from(vec![3; 800]));
    assert!(cache.peek(1).is_none() && cache.peek(2).is_none());
    let held_by_reader = cache.peek(3).unwrap();
    cache.insert(3, Arc::from(vec![0; 100]));
    assert_eq!(cache.retained_bytes, 800);
    assert_eq!(cache.peek(3).unwrap().len(), 800);
    cache.insert(4, Arc::from(vec![4; 1100]));
    assert!(cache.peek(4).is_none());
    assert_eq!(cache.retained_bytes, 800);
    cache.insert(5, Arc::from(vec![5; 500]));
    assert_eq!(cache.retained_bytes, 500);
    assert_eq!(held_by_reader.as_ref(), vec![3; 800]);
    cache.insert(6, Arc::from(vec![6; 1000]));
    assert_eq!(cache.retained_bytes, 1000);
    assert_eq!(cache.blocks.len(), 1);
    for budget in [None, Some(0), Some(1), Some(10)] {
        let mut cache = BlockCache::new(0, budget);
        cache.insert(0, Arc::from(vec![0; 10]));
        assert_eq!(cache.retained_bytes, 0);
        assert!(cache.blocks.is_empty());
    }
}

#[tokio::test]
async fn dictionary_targets_preserve_every_value_and_bounded_prefix_results() {
    let mut previous_blocks = usize::MAX;
    for target in [512, 1024, 4096, 16384, 1024 * 1024] {
        let mut writer = SSTableWriter::<_, u64>::with_config(
            Vec::new(),
            SSTableWriterConfig {
                block_size: SSTableBlockSize::try_from(target).unwrap(),
                ..Default::default()
            },
        );
        for i in 0..1000u64 {
            writer.insert(format!("key{i:08}").as_bytes(), &i).unwrap();
        }
        let reader = AsyncSSTableReader::<u64>::open_with_cache_budget(
            FileHandle::from_bytes(OwnedBytes::new(writer.finish().unwrap())),
            8,
            Some(4096),
        )
        .await
        .unwrap();
        assert!(reader.stats().num_blocks <= previous_blocks);
        previous_blocks = reader.stats().num_blocks;
        let mut iter = reader.iter();
        for i in 0..1000u64 {
            let (key, value) = iter.next().await.unwrap().unwrap();
            assert_eq!(key, format!("key{i:08}").as_bytes());
            assert_eq!(value, i);
            assert!(reader.cached_bytes() <= 4096);
        }
        assert!(iter.next().await.unwrap().is_none());
        for i in [0, 1, 16, 127, 128, 511, 999] {
            let key = format!("key{i:08}");
            assert_eq!(reader.get(key.as_bytes()).await.unwrap(), Some(i));
            #[cfg(feature = "sync")]
            assert_eq!(reader.get_sync(key.as_bytes()).unwrap(), Some(i));
        }
        let (entries, truncated) = reader.prefix_scan_limited(b"key00000", 17).await.unwrap();
        assert!(truncated);
        assert_eq!(
            entries.iter().map(|(_, value)| *value).collect::<Vec<_>>(),
            (0..17).collect::<Vec<_>>()
        );
        #[cfg(feature = "sync")]
        assert_eq!(
            reader.prefix_scan_limited_sync(b"key00000", 17).unwrap(),
            (entries, truncated)
        );
        assert!(reader.cached_bytes() <= 4096 && reader.cached_blocks() <= 8);
    }
    for invalid in [0, 511, 1024 * 1024 + 1, usize::MAX] {
        assert!(SSTableBlockSize::try_from(invalid).is_err());
    }
}

#[tokio::test]
async fn dictionary_entry_larger_than_target_remains_readable_without_cache_retention() {
    let mut writer = SSTableWriter::<_, Vec<u8>>::with_config(
        Vec::new(),
        SSTableWriterConfig {
            block_size: SSTableBlockSize::try_from(512).unwrap(),
            ..Default::default()
        },
    );
    let value = vec![7; 4096];
    writer.insert(b"large", &value).unwrap();
    let reader = AsyncSSTableReader::<Vec<u8>>::open_with_cache_budget(
        FileHandle::from_bytes(OwnedBytes::new(writer.finish().unwrap())),
        4,
        Some(512),
    )
    .await
    .unwrap();
    assert_eq!(reader.get(b"large").await.unwrap(), Some(value));
    assert_eq!(reader.cached_bytes(), 0);
    assert_eq!(reader.cached_blocks(), 0);
}

#[tokio::test]
async fn lazy_concurrent_dictionary_misses_and_prefix_scans_respect_byte_caps() {
    let bytes = OwnedBytes::new(small_block_table());
    let file = FileHandle::lazy(
        bytes.len() as u64,
        Arc::new(move |range| {
            let data = bytes.slice(range.start as usize..range.end as usize);
            Box::pin(async move {
                tokio::task::yield_now().await;
                Ok(data)
            })
        }),
    );
    let reader = AsyncSSTableReader::<u64>::open_with_cache_budget(file, 100, Some(2048))
        .await
        .unwrap();
    let work: Vec<_> = (0..100u64)
        .map(|i| {
            let reader = &reader;
            async move {
                assert_eq!(
                    reader.get(format!("key{i:08}").as_bytes()).await.unwrap(),
                    Some(i)
                );
            }
        })
        .collect();
    futures::future::join_all(work).await;
    assert!(reader.cached_bytes() <= 2048);
    let (entries, truncated) = reader.prefix_scan_limited(b"key", 17).await.unwrap();
    assert!(truncated && entries.len() == 17);
    reader.prefetch_leading_blocks().await.unwrap();
    assert!(reader.cached_bytes() <= 2048);
    reader.preload_all_blocks().await.unwrap();
    assert!(reader.cached_bytes() <= 2048 && reader.cached_blocks() <= 100);
    assert_eq!(reader.get(b"key00000999").await.unwrap(), Some(999));
}

#[tokio::test]
async fn cancelled_bulk_dictionary_prefetch_releases_io_without_publishing_cache_entries() {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    struct Dropped(Arc<AtomicUsize>);
    impl Drop for Dropped {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }
    let bytes = OwnedBytes::new(small_block_table());
    let blocked = Arc::new(AtomicBool::new(false));
    let dropped = Arc::new(AtomicUsize::new(0));
    let file = FileHandle::lazy(
        bytes.len() as u64,
        Arc::new({
            let blocked = Arc::clone(&blocked);
            let dropped = Arc::clone(&dropped);
            move |range| {
                let blocked = blocked.load(Ordering::Relaxed);
                let dropped = Arc::clone(&dropped);
                let data = bytes.slice(range.start as usize..range.end as usize);
                Box::pin(async move {
                    if blocked {
                        let _guard = Dropped(dropped);
                        futures::future::pending::<()>().await;
                    }
                    Ok(data)
                })
            }
        }),
    );
    let reader = AsyncSSTableReader::<u64>::open_with_cache_budget(file, 2, Some(1024))
        .await
        .unwrap();
    blocked.store(true, Ordering::Relaxed);
    let mut pending = Box::pin(reader.prefetch_leading_blocks());
    assert!(futures::poll!(pending.as_mut()).is_pending());
    drop(pending);
    assert_eq!(dropped.load(Ordering::Relaxed), 1);
    assert_eq!(reader.cached_bytes(), 0);
    blocked.store(false, Ordering::Relaxed);
    assert_eq!(reader.get(b"key00000000").await.unwrap(), Some(0));
    assert!(reader.cached_bytes() <= 1024);
}

#[derive(Clone)]
struct FailingValue;
impl SSTableValue for FailingValue {
    fn serialize<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_all(&[42])?;
        Err(io::Error::other("injected value serializer failure"))
    }
    fn deserialize<R: Read>(_: &mut R) -> io::Result<Self> {
        unreachable!()
    }
}

#[test]
fn failed_dictionary_value_serialization_cannot_publish_a_partial_entry() {
    let mut writer = SSTableWriter::new(Vec::new());
    assert!(writer.insert(b"key", &FailingValue).is_err());
    assert!(writer.finish().is_err());
}

#[derive(Clone)]
struct StreamingOversizedValue;
impl SSTableValue for StreamingOversizedValue {
    fn serialize<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        // Stream fixed scratch: the writer must stop growth before its limit.
        let chunk = [0u8; 1024 * 1024];
        for _ in 0..66 {
            writer.write_all(&chunk)?;
        }
        Ok(())
    }
    fn deserialize<R: Read>(_: &mut R) -> io::Result<Self> {
        unreachable!()
    }
}

#[test]
fn streaming_dictionary_values_cannot_grow_writer_scratch_beyond_the_reader_limit() {
    let mut writer = SSTableWriter::new(Vec::new());
    assert!(writer.insert(b"key", &StreamingOversizedValue).is_err());
    assert!(writer.block_buffer.len() <= MAX_SSTABLE_BLOCK_BYTES);
    assert!(writer.insert(b"another", &StreamingOversizedValue).is_err());
    assert!(writer.finish().is_err());
}

#[derive(Clone)]
struct PanickingValue;
impl SSTableValue for PanickingValue {
    fn serialize<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_all(&[42])?;
        panic!("injected value serializer panic");
    }
    fn deserialize<R: Read>(_: &mut R) -> io::Result<Self> {
        unreachable!()
    }
}

#[test]
fn panicking_dictionary_value_cannot_be_published_after_unwinding() {
    let mut writer = SSTableWriter::new(Vec::new());
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            writer.insert(b"key", &PanickingValue).unwrap();
        }))
        .is_err()
    );
    assert!(writer.finish().is_err());
}

#[test]
fn failed_dictionary_output_write_poisoning_prevents_later_publication() {
    struct FailingOutput(Vec<u8>);
    impl Write for FailingOutput {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if self.0.is_empty() {
                self.0.push(bytes[0]);
                Ok(1)
            } else {
                Err(io::Error::other("injected partial output write"))
            }
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let mut writer = SSTableWriter::with_config(
        FailingOutput(Vec::new()),
        SSTableWriterConfig {
            block_size: SSTableBlockSize::try_from(512).unwrap(),
            ..Default::default()
        },
    );
    assert!(writer.insert(b"key", &vec![0u8; 1024]).is_err());
    assert!(writer.insert(b"next", &vec![1u8; 1024]).is_err());
    assert!(writer.finish().is_err());
}
