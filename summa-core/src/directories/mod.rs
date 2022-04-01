mod byte_range_cache;
mod caching_directory;
mod debug_proxy_directory;
mod external_requests;
mod hot_cache_directory;
mod network_directory;

pub use caching_directory::{CachingDirectory, FileStat, FileStats};
pub use debug_proxy_directory::DebugProxyDirectory;
pub use external_requests::{
    DefaultExternalRequestGenerator, ExternalRequest, ExternalRequestGenerator, ExternalRequestGeneratorClone, ExternalResponse, Header, RequestError,
};
pub use hot_cache_directory::{create_hotcache, deserialize_cbor, HotDirectory, StaticDirectoryCache};
pub use network_directory::{NetworkDirectory, NetworkFile};

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
macro_rules! write_proxy_directory {
    () => {
        fn atomic_write(&self, path: &Path, data: &[u8]) -> std::io::Result<()> {
            self.underlying.atomic_write(path, data)
        }

        fn delete(&self, path: &Path) -> Result<(), tantivy::directory::error::DeleteError> {
            self.underlying.delete(path)
        }

        async fn delete_async(&self, path: &Path) -> Result<(), tantivy::directory::error::DeleteError> {
            self.underlying.delete_async(path).await
        }

        fn open_write(&self, path: &Path) -> Result<tantivy::directory::WritePtr, tantivy::directory::error::OpenWriteError> {
            self.underlying.open_write(path)
        }

        fn sync_directory(&self) -> std::io::Result<()> {
            self.underlying.sync_directory()
        }

        fn watch(&self, watch_callback: tantivy::directory::WatchCallback) -> tantivy::Result<tantivy::directory::WatchHandle> {
            self.underlying.watch(watch_callback)
        }

        fn acquire_lock(&self, lock: &tantivy::directory::Lock) -> Result<tantivy::directory::DirectoryLock, tantivy::directory::error::LockError> {
            self.underlying.acquire_lock(lock)
        }
    };
}
pub use write_proxy_directory;
