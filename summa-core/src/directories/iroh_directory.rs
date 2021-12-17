use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::path::PathBuf;
use std::time::Instant;
use std::{ops::Range, path::Path, sync::Arc, usize};

use iroh_metrics::resolver::OutMetrics;
use iroh_resolver::resolver::{OutPrettyReader, Resolver};
use iroh_unixfs::content_loader::ContentLoader;
use tantivy::directory::error::{DeleteError, OpenWriteError};
use tantivy::directory::{WatchCallback, WatchHandle, WritePtr};
use tantivy::{
    directory::{error::OpenReadError, FileHandle, OwnedBytes},
    AsyncIoResult, Directory, HasLen,
};
use tokio::io::{AsyncReadExt, AsyncSeekExt};
use tokio::runtime::Handle;
use tracing::info;

use crate::errors::SummaResult;

#[derive(Clone)]
pub struct IrohDirectory<D: Directory + Clone, T: ContentLoader + Unpin + 'static> {
    underlying: D,
    resolver: Resolver<T>,
    files: HashMap<PathBuf, iroh_resolver::resolver::Out>,
    cid: String,
}

impl<D: Directory + Clone, T: ContentLoader + Unpin + 'static> Debug for IrohDirectory<D, T> {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "IrohDirectory({:?})", &self.cid)
    }
}
impl<D: Directory + Clone, T: ContentLoader + Unpin + 'static> IrohDirectory<D, T> {
    pub async fn new(underlying: D, loader: T, cid: &str) -> SummaResult<IrohDirectory<D, T>> {
        let resolver = Resolver::new(loader);
        let root_path = resolver.resolve(iroh_resolver::Path::from_parts("ipfs", cid, "")?).await?;
        let mut files = HashMap::new();
        for (file_name, cid) in root_path.named_links()?.into_iter() {
            let file_name = PathBuf::from(file_name.expect("file without name"));
            let resolved_path = resolver.resolve(iroh_resolver::Path::from_cid(cid)).await?;
            files.insert(file_name, resolved_path);
        }
        Ok(IrohDirectory {
            underlying,
            resolver,
            files,
            cid: cid.to_string(),
        })
    }

    fn get_iroh_file_handle(&self, file_name: &Path) -> Result<IrohFile<T>, OpenReadError> {
        self.files
            .get(file_name)
            .map(|file| IrohFile::new(file_name, file.clone(), self.resolver.clone()))
            .ok_or_else(|| OpenReadError::FileDoesNotExist(file_name.to_path_buf()))
    }
}

#[async_trait]
impl<D: Directory + Clone, T: ContentLoader + Unpin + 'static> Directory for IrohDirectory<D, T> {
    fn get_file_handle(&self, file_name: &Path) -> Result<Arc<dyn FileHandle>, OpenReadError> {
        if self.underlying.exists(file_name)? {
            Ok(self.underlying.get_file_handle(file_name)?)
        } else {
            Ok(Arc::new(self.get_iroh_file_handle(file_name)?))
        }
    }

    fn delete(&self, path: &Path) -> Result<(), DeleteError> {
        if let Ok(exists) = self.underlying.exists(path) {
            if exists {
                self.underlying.delete(path)?;
            }
        }
        Ok(())
    }

    fn exists(&self, path: &Path) -> Result<bool, OpenReadError> {
        Ok(self.underlying.exists(path)? || self.files.contains_key(path))
    }

    fn open_write(&self, path: &Path) -> Result<WritePtr, OpenWriteError> {
        self.underlying.open_write(path)
    }

    fn atomic_read(&self, path: &Path) -> Result<Vec<u8>, OpenReadError> {
        if self.underlying.exists(path)? {
            self.underlying.atomic_read(path)
        } else {
            let file_handle = self.get_iroh_file_handle(path)?;
            Ok(file_handle.read_bytes(0..file_handle.len()).expect("cannot read").to_vec())
        }
    }

    async fn atomic_read_async(&self, path: &Path) -> AsyncIoResult<Vec<u8>> {
        if self.underlying.exists(path)? {
            self.underlying.atomic_read_async(path).await
        } else {
            let file_handle = self.get_iroh_file_handle(path)?;
            Ok(file_handle.read_bytes_async(0..file_handle.len()).await.expect("cannot read").to_vec())
        }
    }

    fn atomic_write(&self, path: &Path, data: &[u8]) -> std::io::Result<()> {
        self.underlying.atomic_write(path, data)
    }

    fn sync_directory(&self) -> std::io::Result<()> {
        self.underlying.sync_directory()
    }

    fn watch(&self, watch_callback: WatchCallback) -> tantivy::Result<WatchHandle> {
        self.underlying.watch(watch_callback)
    }
}

#[derive(Debug)]
struct IrohFile<T: ContentLoader + Unpin + 'static> {
    file_name: PathBuf,
    out: iroh_resolver::resolver::Out,
    resolver: Resolver<T>,
}

impl<T: ContentLoader + Unpin + 'static> IrohFile<T> {
    pub fn new(file_name: &Path, out: iroh_resolver::resolver::Out, resolver: Resolver<T>) -> IrohFile<T> {
        IrohFile {
            file_name: file_name.to_path_buf(),
            out,
            resolver,
        }
    }

    fn pretty_reader(&self, end: Option<usize>) -> OutPrettyReader<T> {
        self.out
            .clone()
            .pretty(self.resolver.clone(), OutMetrics { start: Instant::now() }, end)
            .expect("cannot create pretty reader")
    }

    fn do_read_bytes(reader: OutPrettyReader<T>, byte_range: Option<Range<usize>>) -> std::io::Result<OwnedBytes> {
        let result = Handle::current().block_on(IrohFile::do_read_bytes_async(reader, byte_range.map(|r| r.start)));
        Ok(result.expect("failed block_on"))
    }

    async fn do_read_bytes_async(mut reader: OutPrettyReader<T>, start: Option<usize>) -> AsyncIoResult<OwnedBytes> {
        if let Some(start) = start {
            reader.seek(tokio::io::SeekFrom::Start(start as u64)).await.expect("iroh seek failed");
        }
        let mut buffer = Vec::new();
        reader.read_to_end(&mut buffer).await?;
        Ok(OwnedBytes::new(buffer))
    }
}

#[async_trait]
impl<T: ContentLoader + Unpin + 'static> FileHandle for IrohFile<T> {
    fn read_bytes(&self, byte_range: Range<usize>) -> std::io::Result<OwnedBytes> {
        info!(action = "read_bytes");
        let reader = self.pretty_reader(Some(byte_range.end));
        IrohFile::do_read_bytes(reader, Some(byte_range))
    }

    async fn read_bytes_async(&self, byte_range: Range<usize>) -> AsyncIoResult<OwnedBytes> {
        info!(action = "read_bytes_async", file = ?self.file_name, range = ?byte_range);
        let reader = self.pretty_reader(Some(byte_range.end));
        let r = IrohFile::do_read_bytes_async(reader, Some(byte_range.start)).await;
        info!(action = "read_bytes_async_done", file = ?self.file_name, range = ?byte_range);
        r
    }
}

impl<T: ContentLoader + Unpin + 'static> HasLen for IrohFile<T> {
    fn len(&self) -> usize {
        self.out.metadata().size.expect("size must be set") as usize
    }
}
