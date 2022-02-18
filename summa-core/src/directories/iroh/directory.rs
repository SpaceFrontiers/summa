use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;

use cid::multihash::MultihashDigest;
use cid::Cid;
use iroh_resolver::resolver::Resolver;
use iroh_unixfs::builder::encode_unixfs_pb;
use iroh_unixfs::codecs::Codec;
use iroh_unixfs::content_loader::FullLoader;
use iroh_unixfs::unixfs::{dag_pb, unixfs_pb, DataType};
use parking_lot::RwLock;
use prost::Message;
use tantivy::directory::error::OpenReadError;

use crate::components::Driver;
use crate::directories::iroh::file::{IrohFile, IrohFileDescriptor};
use crate::directories::iroh::writer::IrohWriter;
use crate::errors::SummaResult;

pub const DEFAULT_CHUNK_SIZE: u64 = 1024 * 1024;
pub(crate) const DEFAULT_DEGREE: u32 = 174;
pub(crate) const DEFAULT_CODE: cid::multihash::Code = cid::multihash::Code::Blake3_256;

/// `IrohDirectory` implements simple file system interface over Iroh Store and Iroh Resolver
/// `IrohDirectory` works in mixed mode by doing save operations directly to store and loading operations
/// through resolver.
///
/// It is a subject of refactoring in the future for better separation Store and Resolver.
#[derive(Clone)]
pub struct IrohDirectory {
    resolver: Resolver<FullLoader>,
    store: iroh_store::Store,
    inner: Arc<RwLock<IrohDirectoryInner>>,
    chunk_size: u64,
    driver: Driver,
}

impl Debug for IrohDirectory {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IrohDirectoryDescriptor").field("cid", &self.cid()).finish()
    }
}

impl IrohDirectory {
    pub fn new(loader: &FullLoader, store: &iroh_store::Store, chunk_size: u64, driver: Driver) -> Self {
        let resolver = Resolver::new(loader.clone());
        IrohDirectory {
            resolver,
            store: store.clone(),
            inner: Arc::new(RwLock::new(IrohDirectoryInner::new(store))),
            driver,
            chunk_size,
        }
    }

    pub async fn from_cid(loader: &FullLoader, store: &iroh_store::Store, driver: Driver, cid: &str, chunk_size: u64) -> SummaResult<Self> {
        let resolver = Resolver::new(loader.clone());
        let root_path = resolver.resolve(iroh_resolver::Path::from_parts("ipfs", cid, "")?).await?;
        let mut files = HashMap::new();
        for (file_name, cid) in root_path.named_links()?.into_iter() {
            let path = PathBuf::from(file_name.expect("file should have name"));
            let fd = IrohFileDescriptor::new(
                cid,
                &path,
                resolver
                    .resolve(iroh_resolver::Path::from_cid(cid))
                    .await?
                    .metadata()
                    .size
                    .expect("should have size"),
            );
            files.insert(path, fd);
        }
        Ok(IrohDirectory {
            resolver,
            store: store.clone(),
            inner: Arc::new(RwLock::new(IrohDirectoryInner::from_files(
                store,
                Cid::from_str(cid).expect("should be cid"),
                files,
            ))),
            chunk_size,
            driver: driver.clone(),
        })
    }

    pub fn cid(&self) -> Option<Cid> {
        self.inner.read().cid
    }

    pub fn insert(&self, path: &Path, file: IrohFileDescriptor) -> SummaResult<Option<IrohFileDescriptor>> {
        self.inner.write().insert(path, file)
    }

    pub fn delete(&self, path: impl AsRef<Path>) -> SummaResult<Option<IrohFileDescriptor>> {
        self.inner.write().delete(path.as_ref())
    }

    pub fn get_file(&self, path: impl AsRef<Path>) -> Result<IrohFile, OpenReadError> {
        Ok(IrohFile::new(
            self.inner
                .read()
                .files
                .get(path.as_ref())
                .ok_or_else(|| OpenReadError::FileDoesNotExist(path.as_ref().to_path_buf()))?,
            &self.resolver,
            &self.driver,
        ))
    }

    pub fn get_writer(&self, path: impl AsRef<Path>) -> IrohWriter {
        IrohWriter::new(&self.store, self.clone(), path, self.chunk_size)
    }

    pub fn exists(&self, path: impl AsRef<Path>) -> bool {
        self.inner.read().exists(path)
    }
}

struct IrohDirectoryInner {
    store: iroh_store::Store,
    cid: Option<Cid>,
    files: HashMap<PathBuf, IrohFileDescriptor>,
}

impl IrohDirectoryInner {
    pub fn new(store: &iroh_store::Store) -> Self {
        IrohDirectoryInner {
            store: store.clone(),
            cid: None,
            files: HashMap::new(),
        }
    }

    pub fn from_files(store: &iroh_store::Store, cid: Cid, files: HashMap<PathBuf, IrohFileDescriptor>) -> Self {
        IrohDirectoryInner {
            store: store.clone(),
            cid: Some(cid),
            files,
        }
    }

    pub fn insert(&mut self, path: &Path, file: IrohFileDescriptor) -> SummaResult<Option<IrohFileDescriptor>> {
        let old_file = self.files.insert(path.to_path_buf(), file);
        self.update_root_directory()?;
        Ok(old_file)
    }

    pub fn update_root_directory(&mut self) -> SummaResult<()> {
        let mut links = vec![];
        let mut cids = vec![];
        for (path, fd) in &self.files {
            cids.push(fd.cid);
            links.push(dag_pb::PbLink {
                hash: Some(fd.cid.to_bytes()),
                name: Some(path.to_string_lossy().to_string()),
                tsize: Some(fd.size),
            });
        }
        // directory itself comes last
        let unixfs_pb_data = unixfs_pb::Data {
            r#type: DataType::Directory as i32,
            ..Default::default()
        };
        let unixfs_pb_node = encode_unixfs_pb(&unixfs_pb_data, links)?;
        let encoded_unixfs_pb_node = unixfs_pb_node.encode_to_vec();
        let cid = Cid::new_v1(Codec::DagPb as _, DEFAULT_CODE.digest(&encoded_unixfs_pb_node));
        self.store.put(cid, encoded_unixfs_pb_node, cids)?;
        self.cid = Some(cid);
        Ok(())
    }

    pub fn delete(&mut self, path: impl AsRef<Path>) -> SummaResult<Option<IrohFileDescriptor>> {
        let old_file = self.files.remove(path.as_ref());
        self.update_root_directory()?;
        Ok(old_file)
    }

    pub fn exists(&self, path: impl AsRef<Path>) -> bool {
        self.files.contains_key(path.as_ref())
    }
}
