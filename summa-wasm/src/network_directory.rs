use async_channel::bounded;
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::{io, ops::Range, path::Path, sync::Arc, usize};
use tantivy::{
    directory::{error::OpenReadError, FileHandle, OwnedBytes},
    AsyncIoResult, Directory, HasLen,
};
use wasm_bindgen::UnwrapThrowExt;
use wasm_bindgen_futures::spawn_local;

use crate::requests::RequestGenerator;

#[derive(Clone)]
pub struct NetworkDirectory {
    files: HashMap<String, usize>,
    request_generator: RequestGenerator,
}

impl Debug for NetworkDirectory {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        "NetworkDirectory".fmt(f)
    }
}

impl NetworkDirectory {
    pub fn new(files: HashMap<String, usize>, request_generator: RequestGenerator) -> NetworkDirectory {
        NetworkDirectory { files, request_generator }
    }
}

impl Directory for NetworkDirectory {
    fn get_file_handle(&self, file_name: &Path) -> Result<Arc<dyn FileHandle>, OpenReadError> {
        let file_name_str = file_name.to_string_lossy();
        let file_size = self
            .files
            .get(file_name_str.as_ref())
            .ok_or_else(|| OpenReadError::FileDoesNotExist(file_name.to_path_buf()))?;
        Ok(Arc::new(NetworkFile::new(
            file_name_str.to_string(),
            *file_size,
            self.request_generator.clone(),
        )?))
    }

    fn exists(&self, path: &Path) -> Result<bool, OpenReadError> {
        Ok(self.files.contains_key(path.to_string_lossy().as_ref()))
    }

    fn atomic_read(&self, path: &Path) -> Result<Vec<u8>, OpenReadError> {
        let file_handle = self.get_file_handle(path)?;
        Ok(file_handle
            .read_bytes(0..file_handle.len())
            .map_err(|e| OpenReadError::wrap_io_error(e, path.to_path_buf()))?
            .to_vec())
    }

    summa_core::directories::read_only_directory!();
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

#[async_trait]
impl FileHandle for NetworkFile {
    fn read_bytes(&self, byte_range: Range<usize>) -> io::Result<OwnedBytes> {
        let response = self.request_generator.generate(&self.file_name, byte_range)?.send()?;
        Ok(OwnedBytes::new(response.to_vec()))
    }

    async fn read_bytes_async(&self, byte_range: Range<usize>) -> AsyncIoResult<OwnedBytes> {
        let request = self.request_generator.generate(&self.file_name, byte_range)?;
        let (sender, receiver) = bounded(1);
        spawn_local(async move {
            let response = request.send_async().await.map(|response| response.to_vec());
            sender.send(response).await.unwrap_throw();
        });
        let response = receiver.recv().await.unwrap_throw()?;
        Ok(OwnedBytes::new(response))
    }
}

impl HasLen for NetworkFile {
    fn len(&self) -> usize {
        self.file_size
    }
}
