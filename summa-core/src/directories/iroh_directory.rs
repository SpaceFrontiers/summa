use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;
use std::{ops::Range, path::Path, sync::Arc, usize};

use iroh_metrics::resolver::OutMetrics;
use iroh_resolver::resolver::{OutPrettyReader, Resolver};
use iroh_unixfs::content_loader::ContentLoader;
use tantivy::directory::error::{DeleteError, LockError, OpenWriteError};
use tantivy::directory::{AntiCallToken, DirectoryLock, Lock, TerminatingWrite, WatchCallback, WatchHandle, WritePtr};
use tantivy::{
    directory::{error::OpenReadError, FileHandle, OwnedBytes},
    Directory, HasLen,
};
use tokio::io::{AsyncReadExt, AsyncSeekExt};

use crate::components::Executor;
use crate::errors::SummaResult;

pub struct IrohTerminatingWrite {
    inner: WritePtr,
}

impl Write for IrohTerminatingWrite {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.inner.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

impl TerminatingWrite for IrohTerminatingWrite {
    fn terminate_ref(&mut self, act: AntiCallToken) -> std::io::Result<()> {
        self.inner.terminate_ref(act)
    }
}

/// Allow to implement searching over Iroh
///
/// `IrohDirectory` translates `read_bytes` calls into Iroh requests to content
#[derive(Clone)]
pub struct IrohDirectory<D: Directory + Clone, T: ContentLoader + Unpin + 'static> {
    underlying: D,
    resolver: Resolver<T>,
    cache: HashMap<PathBuf, iroh_resolver::resolver::Out>,
    executor: Executor,
}

impl<D: Directory + Clone, T: ContentLoader + Unpin + 'static> Debug for IrohDirectory<D, T> {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "IrohDirectory")
    }
}

impl<D: Directory + Clone, T: ContentLoader + Unpin + 'static> IrohDirectory<D, T> {
    pub async fn new(underlying: D, loader: T, cid: &str, executor: Executor) -> SummaResult<IrohDirectory<D, T>> {
        let resolver = Resolver::new(loader);
        let root_path = resolver.resolve(iroh_resolver::Path::from_parts("ipfs", cid, "")?).await?;
        let mut cache = HashMap::new();
        for (file_name, cid) in root_path.named_links()?.into_iter() {
            let file_name = PathBuf::from(file_name.expect("file should have name"));
            let resolved_path = resolver.resolve(iroh_resolver::Path::from_cid(cid)).await?;
            cache.insert(file_name, resolved_path);
        }
        Ok(IrohDirectory {
            underlying,
            resolver,
            cache,
            executor,
        })
    }

    fn get_iroh_file_handle(&self, path: &Path) -> Result<IrohFile<T>, OpenReadError> {
        self.cache
            .get(path)
            .map(|file| IrohFile::new(file.clone(), self.resolver.clone(), self.executor.clone()))
            .ok_or_else(|| OpenReadError::FileDoesNotExist(path.to_path_buf()))
    }
}

#[async_trait]
impl<D: Directory + Clone, T: ContentLoader + Unpin + 'static> Directory for IrohDirectory<D, T> {
    fn get_file_handle(&self, path: &Path) -> Result<Arc<dyn FileHandle>, OpenReadError> {
        Ok(if let Ok(file_handle) = self.underlying.get_file_handle(path) {
            file_handle
        } else {
            Arc::new(self.get_iroh_file_handle(path)?)
        })
    }

    fn delete(&self, path: &Path) -> Result<(), DeleteError> {
        if let Ok(exists) = self.underlying.exists(path) {
            if exists {
                self.underlying.delete(path)?;
            }
        }
        Ok(())
    }

    async fn delete_async(&self, path: &Path) -> Result<(), DeleteError> {
        if let Ok(exists) = self.underlying.exists(path) {
            if exists {
                self.underlying.delete_async(path).await?;
            }
        }
        Ok(())
    }

    fn exists(&self, path: &Path) -> Result<bool, OpenReadError> {
        Ok(self.underlying.exists(path)? || self.cache.contains_key(path))
    }

    fn open_write(&self, path: &Path) -> Result<WritePtr, OpenWriteError> {
        self.underlying.open_write(path)
    }

    fn atomic_read(&self, path: &Path) -> Result<Vec<u8>, OpenReadError> {
        match self.underlying.atomic_read(path) {
            Ok(r) => Ok(r),
            Err(_) => {
                let file_handle = self.get_iroh_file_handle(path)?;
                Ok(file_handle.read_bytes(0..file_handle.len()).expect("cannot read").to_vec())
            }
        }
    }

    async fn atomic_read_async(&self, path: &Path) -> Result<Vec<u8>, OpenReadError> {
        match self.underlying.atomic_read_async(path).await {
            Ok(r) => Ok(r),
            Err(_) => {
                let file_handle = self.get_iroh_file_handle(path)?;
                Ok(file_handle.read_bytes_async(0..file_handle.len()).await.expect("cannot read").to_vec())
            }
        }
    }

    fn atomic_write(&self, path: &Path, data: &[u8]) -> std::io::Result<()> {
        self.underlying.atomic_write(path, data)
    }

    fn sync_directory(&self) -> std::io::Result<()> {
        self.underlying.sync_directory()
    }

    fn acquire_lock(&self, lock: &Lock) -> Result<DirectoryLock, LockError> {
        self.underlying.acquire_lock(lock)
    }

    fn watch(&self, watch_callback: WatchCallback) -> tantivy::Result<WatchHandle> {
        self.underlying.watch(watch_callback)
    }
}

/// `IrohDirectory` creates `IrohFile` for translating `read_bytes` calls into Iroh requests to content
#[derive(Debug, Clone)]
struct IrohFile<T: ContentLoader + Unpin + 'static> {
    out: iroh_resolver::resolver::Out,
    resolver: Resolver<T>,
    executor: Executor,
}

impl<T: ContentLoader + Unpin + 'static> IrohFile<T> {
    pub fn new(out: iroh_resolver::resolver::Out, resolver: Resolver<T>, executor: Executor) -> IrohFile<T> {
        IrohFile { out, resolver, executor }
    }

    fn pretty_reader(&self, end: Option<usize>) -> OutPrettyReader<T> {
        self.out
            .clone()
            .pretty(self.resolver.clone(), OutMetrics { start: Instant::now() }, end)
            .expect("cannot create pretty reader")
    }
}

#[async_trait]
impl<T: ContentLoader + Unpin + 'static> FileHandle for IrohFile<T> {
    fn read_bytes(&self, byte_range: Range<usize>) -> std::io::Result<OwnedBytes> {
        let file = self.clone();
        self.executor.spawn_blocking(async move { file.read_bytes_async(byte_range).await })
    }

    async fn read_bytes_async(&self, byte_range: Range<usize>) -> std::io::Result<OwnedBytes> {
        let mut reader = self.pretty_reader(Some(byte_range.end));
        reader
            .seek(tokio::io::SeekFrom::Start(byte_range.start as u64))
            .await
            .expect("iroh seek failed");
        let mut buffer = Vec::new();
        reader.read_to_end(&mut buffer).await?;
        Ok(OwnedBytes::new(buffer))
    }
}

impl<T: ContentLoader + Unpin + 'static> HasLen for IrohFile<T> {
    fn len(&self) -> usize {
        self.out.metadata().size.expect("size must be set") as usize
    }
}
