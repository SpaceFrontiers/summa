use std::any::Any;
use std::io::{BufWriter, Write};
use std::{io, ops::Range, path::Path, sync::Arc, usize};

use tantivy::directory::error::{DeleteError, LockError, OpenWriteError};
use tantivy::directory::{DirectoryLock, Lock, TerminatingWrite, WatchCallback, WatchHandle, WritePtr};
use tantivy::{
    directory::{error::OpenReadError, FileHandle, OwnedBytes},
    Directory, HasLen,
};

use crate::directories::iroh::directory::IrohDirectory;
use crate::directories::iroh::file::IrohFile;

#[async_trait]
impl Directory for IrohDirectory {
    fn get_file_handle(&self, path: &Path) -> Result<Arc<dyn FileHandle>, OpenReadError> {
        Ok(Arc::new(self.get_file(path)?))
    }

    fn delete(&self, path: &Path) -> Result<(), DeleteError> {
        match self.delete(path) {
            Ok(None) => Err(DeleteError::FileDoesNotExist(path.to_path_buf())),
            Ok(_) => Ok(()),
            Err(e) => Err(DeleteError::IoError {
                io_error: Arc::new(io::Error::new(io::ErrorKind::Other, e)),
                filepath: path.to_path_buf(),
            }),
        }
    }

    fn exists(&self, path: &Path) -> Result<bool, OpenReadError> {
        Ok(self.exists(path))
    }

    fn open_write(&self, path: &Path) -> Result<WritePtr, OpenWriteError> {
        Ok(BufWriter::new(Box::new(self.get_writer(path))))
    }

    fn atomic_read(&self, path: &Path) -> Result<Vec<u8>, OpenReadError> {
        let file_handle = self.get_file_handle(path)?;
        Ok(file_handle.read_bytes(0..file_handle.len()).expect("cannot read").to_vec())
    }

    async fn atomic_read_async(&self, path: &Path) -> Result<Vec<u8>, OpenReadError> {
        // ToDo: Use async
        self.atomic_read(path)
    }

    fn atomic_write(&self, path: &Path, data: &[u8]) -> io::Result<()> {
        let mut writer = self.get_writer(path);
        writer.write_all(data)?;
        writer.terminate()
    }

    fn sync_directory(&self) -> io::Result<()> {
        Ok(())
    }

    fn acquire_lock(&self, _lock: &Lock) -> Result<DirectoryLock, LockError> {
        Ok(tantivy::directory::DirectoryLock::from(Box::new(|| {})))
    }

    fn watch(&self, _: WatchCallback) -> tantivy::Result<WatchHandle> {
        Ok(WatchHandle::empty())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[async_trait]
impl FileHandle for IrohFile {
    fn read_bytes(&self, byte_range: Range<usize>) -> io::Result<OwnedBytes> {
        self.read_pretty_bytes(byte_range)
    }

    async fn read_bytes_async(&self, byte_range: Range<usize>) -> io::Result<OwnedBytes> {
        self.read_pretty_bytes_async(byte_range).await
    }
}

impl HasLen for IrohFile {
    fn len(&self) -> usize {
        self.size() as usize
    }
}
