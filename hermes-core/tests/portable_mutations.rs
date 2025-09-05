//! Exercise the portable writer on a native runtime with cancellable/failing I/O.
#![cfg(all(feature = "wasm", not(feature = "native")))]
use async_trait::async_trait;
use hermes_core::directories::{
    Directory, DirectoryWriter, FileHandle, OwnedBytes, RamDirectory, StreamingWriter,
};
use hermes_core::{Document, IndexConfig, SchemaBuilder, WasmIndexWriter};
use std::{
    io,
    ops::Range,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicU8, Ordering},
    },
};

#[derive(Clone, Default)]
struct FaultDirectory {
    ram: RamDirectory,
    mode: Arc<AtomicU8>, // 1 fail build; 2 cancel build; 3 fail rename; 4 cancel after rename
}
#[async_trait]
impl Directory for FaultDirectory {
    async fn exists(&self, p: &Path) -> io::Result<bool> {
        self.ram.exists(p).await
    }
    async fn file_size(&self, p: &Path) -> io::Result<u64> {
        self.ram.file_size(p).await
    }
    async fn open_read(&self, p: &Path) -> io::Result<FileHandle> {
        self.ram.open_read(p).await
    }
    async fn read_range(&self, p: &Path, r: Range<u64>) -> io::Result<OwnedBytes> {
        self.ram.read_range(p, r).await
    }
    async fn list_files(&self, p: &Path) -> io::Result<Vec<PathBuf>> {
        self.ram.list_files(p).await
    }
    async fn open_lazy(&self, p: &Path) -> io::Result<FileHandle> {
        self.ram.open_lazy(p).await
    }
}
#[async_trait]
impl DirectoryWriter for FaultDirectory {
    async fn write(&self, p: &Path, d: &[u8]) -> io::Result<()> {
        self.ram.write(p, d).await
    }
    async fn delete(&self, p: &Path) -> io::Result<()> {
        self.ram.delete(p).await
    }
    async fn rename(&self, a: &Path, b: &Path) -> io::Result<()> {
        if b == Path::new("metadata.json") && self.mode.load(Ordering::SeqCst) == 3 {
            return Err(io::Error::other("injected metadata rename failure"));
        }
        self.ram.rename(a, b).await?;
        if b == Path::new("metadata.json") && self.mode.load(Ordering::SeqCst) == 4 {
            std::future::pending::<()>().await;
        }
        Ok(())
    }
    async fn sync(&self) -> io::Result<()> {
        Ok(())
    }
    async fn streaming_writer(&self, p: &Path) -> io::Result<Box<dyn StreamingWriter>> {
        if p.to_string_lossy().starts_with("seg_") {
            match self.mode.load(Ordering::SeqCst) {
                1 => return Err(io::Error::other("injected build failure")),
                2 => std::future::pending::<()>().await,
                _ => {}
            }
        }
        self.ram.streaming_writer(p).await
    }
}

#[tokio::test]
async fn failed_or_cancelled_portable_build_never_publishes_a_replacement_deletion() {
    for mode in [1, 2] {
        let dir = FaultDirectory::default();
        let mut schema = SchemaBuilder::default();
        let id = schema.add_text_field("id", true, true);
        schema.set_primary_key(id);
        let mut writer =
            WasmIndexWriter::create(dir.clone(), schema.build(), IndexConfig::default())
                .await
                .unwrap();
        let mut doc = Document::new();
        doc.add_text(id, "a");
        writer.add_document(doc.clone()).await.unwrap();
        writer.commit().await.unwrap();
        writer.upsert_document(doc.clone()).await.unwrap();
        dir.mode.store(mode, Ordering::SeqCst);
        let result =
            tokio::time::timeout(std::time::Duration::from_millis(10), writer.commit()).await;
        if mode == 1 {
            assert!(result.unwrap().is_err());
        } else {
            assert!(result.is_err());
        }
        dir.mode.store(0, Ordering::SeqCst);
        assert!(
            writer
                .commit()
                .await
                .unwrap_err()
                .to_string()
                .contains("abort")
        );
        assert_eq!(
            writer
                .metadata()
                .segment_metas
                .values()
                .map(|info| info.num_live_docs())
                .sum::<u32>(),
            1
        );
        writer.abort().await.unwrap();
        assert!(!writer.commit().await.unwrap());
        assert!(writer.add_document(doc.clone()).await.is_err());
        writer.upsert_document(doc).await.unwrap();
        writer.commit().await.unwrap();
        assert_eq!(
            writer
                .metadata()
                .segment_metas
                .values()
                .map(|info| info.num_live_docs())
                .sum::<u32>(),
            1
        );
        let reopened = WasmIndexWriter::open(dir, IndexConfig::default())
            .await
            .unwrap();
        assert_eq!(
            reopened
                .metadata()
                .segment_metas
                .values()
                .map(|info| info.num_live_docs())
                .sum::<u32>(),
            1
        );
    }
}

#[tokio::test]
async fn portable_publication_retry_reconciles_both_sides_of_the_metadata_rename() {
    for mode in [3, 4] {
        let dir = FaultDirectory::default();
        let mut schema = SchemaBuilder::default();
        let id = schema.add_text_field("id", true, true);
        schema.set_primary_key(id);
        let mut writer =
            WasmIndexWriter::create(dir.clone(), schema.build(), IndexConfig::default())
                .await
                .unwrap();
        let mut doc = Document::new();
        doc.add_text(id, "a");
        writer.add_document(doc.clone()).await.unwrap();
        writer.commit().await.unwrap();
        writer.upsert_document(doc.clone()).await.unwrap();
        dir.mode.store(mode, Ordering::SeqCst);
        let result =
            tokio::time::timeout(std::time::Duration::from_millis(10), writer.commit()).await;
        if mode == 3 {
            assert!(result.unwrap().is_err());
        } else {
            assert!(result.is_err());
        }
        dir.mode.store(0, Ordering::SeqCst);
        assert!(writer.commit().await.unwrap());
        assert!(!writer.commit().await.unwrap());
        assert_eq!(
            writer
                .metadata()
                .segment_metas
                .values()
                .map(|info| info.num_live_docs())
                .sum::<u32>(),
            1
        );
        assert!(writer.add_document(doc.clone()).await.is_err());
        let mut reopened = WasmIndexWriter::open(dir.clone(), IndexConfig::default())
            .await
            .unwrap();
        assert!(reopened.add_document(doc).await.is_err());
        assert_eq!(
            reopened
                .metadata()
                .segment_metas
                .values()
                .map(|info| info.num_live_docs())
                .sum::<u32>(),
            1
        );
        let files = dir.list_files(Path::new("")).await.unwrap();
        assert_eq!(
            files
                .iter()
                .filter(|p| p.extension().is_some_and(|e| e == "del"))
                .count(),
            1
        );
    }
}
