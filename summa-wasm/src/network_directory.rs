use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::{io, io::BufWriter, ops::Range, path::Path, sync::Arc, usize};
use tantivy::{
    directory::{
        error::{DeleteError, OpenReadError, OpenWriteError},
        FileHandle, OwnedBytes, WatchCallback, WatchHandle, WritePtr,
    },
    Directory, HasLen,
};

use crate::request_generator::RequestGenerator;
use summa_directory::noop_writer::Noop;

#[derive(Clone)]
pub struct NetworkDirectory {
    request_generator: RequestGenerator,
    file_sizes: HashMap<String, usize>,
}

impl Debug for NetworkDirectory {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        "NetworkDirectory".fmt(f)
    }
}

impl NetworkDirectory {
    pub fn new(request_generator: RequestGenerator, file_sizes: HashMap<String, usize>) -> NetworkDirectory {
        NetworkDirectory { request_generator, file_sizes }
    }
}

impl Directory for NetworkDirectory {
    fn get_file_handle(&self, file_name: &Path) -> Result<Arc<dyn FileHandle>, OpenReadError> {
        let file_name = file_name.to_string_lossy().to_string();
        Ok(Arc::new(NetworkFile::new(
            file_name.clone(),
            self.file_sizes[&file_name],
            self.request_generator.clone(),
        )?))
    }

    fn delete(&self, path: &Path) -> Result<(), DeleteError> {
        Ok(())
    }

    fn exists(&self, _path: &Path) -> Result<bool, OpenReadError> {
        todo!()
    }

    fn open_write(&self, _path: &Path) -> Result<WritePtr, OpenWriteError> {
        Ok(BufWriter::new(Box::new(Noop {})))
    }

    fn atomic_read(&self, path: &Path) -> Result<Vec<u8>, OpenReadError> {
        let file_handle = self.get_file_handle(path)?;
        Ok(file_handle
            .read_bytes(0..file_handle.len())
            .map_err(|e| OpenReadError::wrap_io_error(e, path.to_path_buf()))?
            .to_vec())
    }

    fn atomic_write(&self, _path: &Path, _data: &[u8]) -> io::Result<()> {
        todo!()
    }

    fn sync_directory(&self) -> io::Result<()> {
        todo!()
    }

    fn watch(&self, _watch_callback: WatchCallback) -> tantivy::Result<WatchHandle> {
        Ok(WatchHandle::empty())
    }
}

#[derive(Clone, Debug)]
struct NetworkFile {
    file_name: String,
    file_size: usize,
    request_generator: RequestGenerator,
}

impl NetworkFile {
    pub fn new(file_name: String, file_size: usize, request_generator: RequestGenerator) -> Result<NetworkFile, OpenReadError> {
        Ok(NetworkFile {
            file_name,
            file_size,
            request_generator,
        })
    }
}

impl FileHandle for NetworkFile {
    fn read_bytes(&self, byte_range: Range<usize>) -> io::Result<OwnedBytes> {
        let response = self.request_generator.generate(&self.file_name, byte_range).send();
        Ok(OwnedBytes::new(response.to_vec()))
    }
}

impl HasLen for NetworkFile {
    fn len(&self) -> usize {
        self.file_size
    }
}
