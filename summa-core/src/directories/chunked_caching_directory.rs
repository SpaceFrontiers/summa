use crate::directories::chunk_generator::{Chunk, ChunkGenerator};
use crate::directories::requests_composer::{Request, RequestsComposer};
use crate::directories::MemorySizedCache;
use crate::metrics::CacheMetrics;
use std::fmt;
use std::fmt::{Debug, Formatter};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tantivy::directory::error::OpenReadError;
use tantivy::directory::{FileHandle, OwnedBytes};
use tantivy::error::AsyncIoError;
use tantivy::{Directory, HasLen};

#[derive(Clone)]
pub struct ChunkedCachingDirectory {
    chunk_size: usize,
    cache: Option<Arc<MemorySizedCache>>,
    underlying: Arc<dyn Directory>,
}

impl ChunkedCachingDirectory {
    pub fn new(underlying: Arc<dyn Directory>, chunk_size: usize) -> ChunkedCachingDirectory {
        ChunkedCachingDirectory {
            chunk_size,
            cache: None,
            underlying,
        }
    }
    pub fn new_with_capacity_in_bytes(
        underlying: Arc<dyn Directory>,
        chunk_size: usize,
        capacity_in_bytes: usize,
        cache_metrics: CacheMetrics,
    ) -> ChunkedCachingDirectory {
        ChunkedCachingDirectory {
            chunk_size,
            cache: Some(Arc::new(MemorySizedCache::with_capacity_in_bytes(capacity_in_bytes, cache_metrics))),
            underlying,
        }
    }
    pub fn new_unbounded(underlying: Arc<dyn Directory>, chunk_size: usize, cache_metrics: CacheMetrics) -> ChunkedCachingDirectory {
        ChunkedCachingDirectory {
            chunk_size,
            cache: Some(Arc::new(MemorySizedCache::with_infinite_capacity(cache_metrics))),
            underlying,
        }
    }
}

impl Debug for ChunkedCachingDirectory {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "ChunkedCachingDirectory({:?})", self.underlying)
    }
}

impl Directory for ChunkedCachingDirectory {
    fn get_file_handle(&self, path: &Path) -> Result<Arc<dyn FileHandle>, OpenReadError> {
        let underlying_filehandle = self.underlying.get_file_handle(path)?;
        Ok(Arc::new(ChunkedCachingFileHandle {
            path: path.to_path_buf(),
            cache: self.cache.clone(),
            chunk_size: self.chunk_size,
            underlying_filehandle,
        }))
    }

    fn exists(&self, path: &Path) -> Result<bool, OpenReadError> {
        self.underlying.exists(path)
    }

    fn atomic_read(&self, path: &Path) -> Result<Vec<u8>, OpenReadError> {
        let file_handle = self.get_file_handle(path)?;
        let len = file_handle.len();
        let owned_bytes = file_handle
            .read_bytes(0..len)
            .map_err(|io_error| OpenReadError::wrap_io_error(io_error, path.to_path_buf()))?;
        Ok(owned_bytes.as_slice().to_vec())
    }

    super::read_only_directory!();
}

struct ChunkedCachingFileHandle {
    path: PathBuf,
    cache: Option<Arc<MemorySizedCache>>,
    chunk_size: usize,
    underlying_filehandle: Arc<dyn FileHandle>,
}

impl Debug for ChunkedCachingFileHandle {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "ChunkedCachingFileHandle(path={:?}, underlying={:?})",
            &self.path,
            self.underlying_filehandle.as_ref()
        )
    }
}

impl ChunkedCachingFileHandle {
    fn try_fill_from_cache(&self, range: Range<usize>, response_buffer: &mut [u8]) -> Vec<Chunk> {
        let chunks = ChunkGenerator::new(range, self.len(), self.chunk_size);
        match &self.cache {
            None => chunks.collect(),
            Some(cache) => {
                let mut missing_chunks = vec![];
                for chunk in chunks {
                    match cache.get_slice(&self.path, chunk.index) {
                        Some(item) => response_buffer[chunk.target_ix..][..chunk.len()].clone_from_slice(&item.slice(chunk.data_bounds())),
                        None => missing_chunks.push(chunk),
                    };
                }
                missing_chunks
            }
        }
    }

    fn adopt_response(&self, total_response: &mut [u8], response: OwnedBytes, original_request: &Request) {
        for chunk in original_request.chunks() {
            let item = response.slice(chunk.shifted_chunk_range(original_request.bounds().start));
            total_response[chunk.target_ix..][..chunk.len()].clone_from_slice(&item.slice(chunk.data_bounds()));
            if let Some(cache) = &self.cache {
                cache.put_slice(self.path.to_path_buf(), chunk.index, item);
            }
        }
    }
}

#[async_trait]
impl FileHandle for ChunkedCachingFileHandle {
    fn read_bytes(&self, range: Range<usize>) -> std::io::Result<OwnedBytes> {
        let mut response_buffer = vec![0; range.end - range.start];
        let missing_chunks = self.try_fill_from_cache(range, &mut response_buffer);

        for missing_chunks_request in RequestsComposer::for_chunks(missing_chunks).requests() {
            let missing_chunks_response = self.underlying_filehandle.read_bytes(missing_chunks_request.bounds())?;
            self.adopt_response(&mut response_buffer, missing_chunks_response, &missing_chunks_request)
        }
        Ok(OwnedBytes::new(response_buffer))
    }

    async fn read_bytes_async(&self, range: Range<usize>) -> Result<OwnedBytes, AsyncIoError> {
        let mut response_buffer = vec![0; range.end - range.start];
        let missing_chunks = self.try_fill_from_cache(range, &mut response_buffer);

        for missing_chunks_request in RequestsComposer::for_chunks(missing_chunks).requests() {
            let missing_chunks_response = self.underlying_filehandle.read_bytes_async(missing_chunks_request.bounds()).await?;
            self.adopt_response(&mut response_buffer, missing_chunks_response, &missing_chunks_request)
        }

        Ok(OwnedBytes::new(response_buffer))
    }
}

impl HasLen for ChunkedCachingFileHandle {
    fn len(&self) -> usize {
        self.underlying_filehandle.len()
    }
}
