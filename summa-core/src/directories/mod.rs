mod chunk_generator;
mod chunked_caching_directory;
mod debug_proxy_directory;
mod external_requests;
mod hot_cache_directory;
mod memory_sized_cache;
mod network_directory;
mod requests_composer;
mod slice_address;
mod stored_item;

pub use chunked_caching_directory::ChunkedCachingDirectory;
pub use debug_proxy_directory::DebugProxyDirectory;
pub use external_requests::{
    DefaultExternalRequestGenerator, ExternalRequest, ExternalRequestGenerator, ExternalRequestGeneratorClone, ExternalResponse, Header,
};
pub use hot_cache_directory::{write_hotcache, HotDirectory};
pub use memory_sized_cache::MemorySizedCache;
pub use network_directory::NetworkDirectory;

struct Noop {}

impl std::io::Write for Noop {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl tantivy::directory::TerminatingWrite for Noop {
    fn terminate_ref(&mut self, _: tantivy::directory::AntiCallToken) -> std::io::Result<()> {
        Ok(())
    }
}

#[macro_export]
macro_rules! read_only_directory {
    () => {
        fn atomic_write(&self, _: &Path, _: &[u8]) -> std::io::Result<()> {
            Ok(())
        }

        fn delete(&self, _: &Path) -> Result<(), tantivy::directory::error::DeleteError> {
            Ok(())
        }

        fn open_write(&self, _: &Path) -> Result<tantivy::directory::WritePtr, tantivy::directory::error::OpenWriteError> {
            Ok(std::io::BufWriter::new(Box::new(Noop {})))
        }

        fn sync_directory(&self) -> std::io::Result<()> {
            Ok(())
        }

        fn watch(&self, _: tantivy::directory::WatchCallback) -> tantivy::Result<tantivy::directory::WatchHandle> {
            Ok(tantivy::directory::WatchHandle::empty())
        }

        fn acquire_lock(&self, _lock: &tantivy::directory::Lock) -> Result<tantivy::directory::DirectoryLock, tantivy::directory::error::LockError> {
            Ok(tantivy::directory::DirectoryLock::from(Box::new(|| {})))
        }
    };
}
pub use read_only_directory;
