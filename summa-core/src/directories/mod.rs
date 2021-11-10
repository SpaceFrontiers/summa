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

#[macro_export]
macro_rules! read_only_directory {
    () => {
        fn atomic_write(&self, _path: &Path, _data: &[u8]) -> std::io::Result<()> {
            unimplemented!("read-only")
        }

        fn delete(&self, _path: &Path) -> Result<(), tantivy::directory::error::DeleteError> {
            unimplemented!("read-only")
        }

        fn open_write(&self, _path: &Path) -> Result<tantivy::directory::WritePtr, tantivy::directory::error::OpenWriteError> {
            unimplemented!("read-only")
        }

        fn sync_directory(&self) -> std::io::Result<()> {
            unimplemented!("read-only")
        }

        fn watch(&self, _watch_callback: tantivy::directory::WatchCallback) -> tantivy::Result<tantivy::directory::WatchHandle> {
            Ok(tantivy::directory::WatchHandle::empty())
        }

        fn acquire_lock(&self, _lock: &tantivy::directory::Lock) -> Result<tantivy::directory::DirectoryLock, tantivy::directory::error::LockError> {
            Ok(tantivy::directory::DirectoryLock::from(Box::new(|| {})))
        }
    };
}
pub use read_only_directory;
