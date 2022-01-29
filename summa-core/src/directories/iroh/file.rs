use std::fmt::{Debug, Formatter};
use std::io;
use std::ops::Range;
use std::path::{Path, PathBuf};

use bytes::Bytes;
use cid::Cid;
use instant::Instant;
use iroh_metrics::resolver::OutMetrics;
use iroh_resolver::resolver::{OutPrettyReader, Resolver};
use iroh_unixfs::content_loader::FullLoader;
use tantivy::directory::OwnedBytes;
use tokio::io::{AsyncReadExt, AsyncSeekExt};

use crate::components::Driver;
use crate::errors::SummaResult;

#[derive(Clone, Debug)]
pub struct IrohFileDescriptor {
    pub cid: Cid,
    pub path: PathBuf,
    pub data: Bytes,
    pub links: Vec<Cid>,
    pub size: u64,
}

impl IrohFileDescriptor {
    pub fn new(cid: Cid, path: impl AsRef<Path>, size: u64) -> Self {
        IrohFileDescriptor {
            cid,
            path: path.as_ref().to_path_buf(),
            data: Bytes::new(),
            links: vec![],
            size,
        }
    }
}

/// `IrohDirectory` creates `IrohFile` for translating `read_bytes` calls into Iroh requests to content
#[derive(Clone)]
pub struct IrohFile {
    iroh_fd: IrohFileDescriptor,
    resolver: Resolver<FullLoader>,
    driver: Driver,
}

impl Debug for IrohFile {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IrohFile").field("iroh_fd", &self.iroh_fd).finish()
    }
}

impl IrohFile {
    pub fn new(iroh_fd: &IrohFileDescriptor, resolver: &Resolver<FullLoader>, driver: &Driver) -> IrohFile {
        IrohFile {
            iroh_fd: iroh_fd.clone(),
            resolver: resolver.clone(),
            driver: driver.clone(),
        }
    }

    async fn pretty_reader(&self, end: Option<usize>) -> SummaResult<OutPrettyReader<FullLoader>> {
        let resolved = self.resolver.resolve(iroh_resolver::Path::from_cid(self.iroh_fd.cid)).await?;
        Ok(resolved.pretty(self.resolver.clone(), OutMetrics { start: Instant::now() }, end)?)
    }

    pub async fn read_pretty_bytes_async(&self, byte_range: Range<usize>) -> io::Result<OwnedBytes> {
        let mut reader = self.pretty_reader(Some(byte_range.end)).await?;
        reader.seek(io::SeekFrom::Start(byte_range.start as u64)).await.expect("iroh seek failed");
        let mut buffer = Vec::new();
        reader.read_to_end(&mut buffer).await?;
        Ok(OwnedBytes::new(buffer))
    }

    pub fn read_pretty_bytes(&self, byte_range: Range<usize>) -> io::Result<OwnedBytes> {
        let file = self.clone();
        // ToDo: consider direct sync reading
        self.driver.block_on(async move { file.read_pretty_bytes_async(byte_range).await })
    }

    pub fn size(&self) -> u64 {
        self.iroh_fd.size
    }
}
