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
    Directory, HasLen,
};
use tokio::io::{AsyncReadExt, AsyncSeekExt};
use tracing::trace;

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
            trace!(action = "resolved_iroh_file", file_name = ?file_name, cid = ?cid);
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
            .map(|file| IrohFile::new(file.clone(), self.resolver.clone()))
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

#[derive(Debug, Clone)]
struct IrohFile<T: ContentLoader + Unpin + 'static> {
    out: iroh_resolver::resolver::Out,
    resolver: Resolver<T>,
}

impl<T: ContentLoader + Unpin + 'static> IrohFile<T> {
    pub fn new(out: iroh_resolver::resolver::Out, resolver: Resolver<T>) -> IrohFile<T> {
        IrohFile { out, resolver }
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
        let (s, mut r) = tokio::sync::mpsc::unbounded_channel();
        let file = self.clone();
        tokio::spawn(async move { s.send(file.read_bytes_async(byte_range).await).expect("cannot send to channel") });
        r.blocking_recv().expect("cannot block on channel")
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
