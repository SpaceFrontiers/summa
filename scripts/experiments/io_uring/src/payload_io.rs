//! Thin fixture adapter for the filesystem-demand control. All pool/ring I/O
//! uses the core directory-owned service; no diagnostic lifecycle protocol.
use std::{
    io,
    ops::Range,
    path::{Path, PathBuf},
    sync::Arc,
};
use summa_core::directories::{
    Directory, DirectoryWriter, FileHandle, FsDirectory, MmapDirectory, OwnedBytes, StreamingWriter,
};
pub use summa_core::directories::{PayloadReadBackend as Backend, PayloadReadService as Service};
#[derive(Clone)]
pub struct PayloadDirectory {
    mapped: MmapDirectory,
    fs: FsDirectory,
    positional: bool,
}
impl PayloadDirectory {
    pub fn new(root: &Path, service: Option<Arc<Service>>, positional: bool) -> Self {
        Self {
            mapped: match service {
                Some(service) => MmapDirectory::new(root).with_payload_reads(service),
                None => MmapDirectory::new(root),
            },
            fs: FsDirectory::new(root),
            positional,
        }
    }
    #[allow(dead_code)] // Only the sparse probe opts into this role.
    pub fn with_sparse_payload_reads(mut self, service: Arc<Service>) -> Self {
        self.mapped = self.mapped.with_sparse_payload_reads(service);
        self
    }
}
#[async_trait::async_trait]
impl Directory for PayloadDirectory {
    async fn exists(&self, p: &Path) -> io::Result<bool> {
        self.mapped.exists(p).await
    }
    async fn file_size(&self, p: &Path) -> io::Result<u64> {
        self.mapped.file_size(p).await
    }
    async fn open_read(&self, p: &Path) -> io::Result<FileHandle> {
        self.mapped.open_read(p).await
    }
    async fn read_range(&self, p: &Path, r: Range<u64>) -> io::Result<OwnedBytes> {
        self.mapped.read_range(p, r).await
    }
    async fn list_files(&self, p: &Path) -> io::Result<Vec<PathBuf>> {
        self.mapped.list_files(p).await
    }
    async fn open_lazy(&self, p: &Path) -> io::Result<FileHandle> {
        self.mapped.open_lazy(p).await
    }
    async fn open_payload(&self, p: &Path) -> io::Result<FileHandle> {
        if self.positional {
            self.fs.open_lazy(p).await
        } else {
            self.mapped.open_payload(p).await
        }
    }
    async fn open_sparse_payload(&self, p: &Path) -> io::Result<Option<FileHandle>> {
        self.mapped.open_sparse_payload(p).await
    }
    fn set_index_label(&self, label: &str) {
        self.mapped.set_index_label(label);
        self.fs.set_index_label(label);
    }
    fn local_path(&self, p: &Path) -> Option<PathBuf> {
        self.mapped.local_path(p)
    }
}
#[async_trait::async_trait]
impl DirectoryWriter for PayloadDirectory {
    async fn write(&self, p: &Path, b: &[u8]) -> io::Result<()> {
        self.mapped.write(p, b).await
    }
    async fn delete(&self, p: &Path) -> io::Result<()> {
        self.mapped.delete(p).await
    }
    async fn rename(&self, a: &Path, b: &Path) -> io::Result<()> {
        self.mapped.rename(a, b).await
    }
    async fn link(&self, a: &Path, b: &Path) -> io::Result<()> {
        self.mapped.link(a, b).await
    }
    async fn sync(&self) -> io::Result<()> {
        self.mapped.sync().await
    }
    async fn streaming_writer(&self, p: &Path) -> io::Result<Box<dyn StreamingWriter>> {
        self.mapped.streaming_writer(p).await
    }
    async fn streaming_writer_cold(&self, p: &Path) -> io::Result<Box<dyn StreamingWriter>> {
        self.mapped.streaming_writer_cold(p).await
    }
    async fn streaming_writer_cold_with_capacity(
        &self,
        p: &Path,
        capacity: usize,
    ) -> io::Result<Box<dyn StreamingWriter>> {
        self.mapped
            .streaming_writer_cold_with_capacity(p, capacity)
            .await
    }
}
