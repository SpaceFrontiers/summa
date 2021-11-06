use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::{io, ops::Range, path::Path, sync::Arc, usize};

use tantivy::directory::DirectoryClone;
use tantivy::{
    directory::{error::OpenReadError, FileHandle, OwnedBytes},
    AsyncIoResult, Directory, HasLen,
};

use super::ExternalRequestGenerator;
use crate::directories::ExternalRequest;

pub struct NetworkDirectory<TExternalRequest: ExternalRequest> {
    files: HashMap<String, usize>,
    external_request_generator: Box<dyn ExternalRequestGenerator<TExternalRequest>>,
}

impl<TExternalRequest: ExternalRequest> Debug for NetworkDirectory<TExternalRequest> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        "NetworkDirectory".fmt(f)
    }
}

impl<TExternalRequest: ExternalRequest> NetworkDirectory<TExternalRequest> {
    pub fn new(
        files: HashMap<String, usize>,
        external_request_generator: Box<dyn ExternalRequestGenerator<TExternalRequest>>,
    ) -> NetworkDirectory<TExternalRequest> {
        NetworkDirectory {
            files,
            external_request_generator,
        }
    }
}

impl<TExternalRequest: ExternalRequest + 'static> DirectoryClone for NetworkDirectory<TExternalRequest> {
    fn box_clone(&self) -> Box<dyn Directory> {
        Box::new(NetworkDirectory {
            files: self.files.clone(),
            external_request_generator: self.external_request_generator.box_clone(),
        })
    }
}

impl<TExternalRequest: ExternalRequest + 'static> Directory for NetworkDirectory<TExternalRequest> {
    fn get_file_handle(&self, file_name: &Path) -> Result<Arc<dyn FileHandle>, OpenReadError> {
        let file_name_str = file_name.to_string_lossy();
        let file_size = self
            .files
            .get(file_name_str.as_ref())
            .ok_or_else(|| OpenReadError::FileDoesNotExist(file_name.to_path_buf()))?;
        Ok(Arc::new(NetworkFile::new(
            file_name_str.to_string(),
            *file_size,
            self.external_request_generator.box_clone(),
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

    super::read_only_directory!();
}

#[derive(Debug)]
struct NetworkFile<TExternalRequest: ExternalRequest> {
    file_name: String,
    file_size: usize,
    request_generator: Box<dyn ExternalRequestGenerator<TExternalRequest>>,
}

impl<TExternalRequest: ExternalRequest> NetworkFile<TExternalRequest> {
    pub fn new(
        file_name: String,
        file_size: usize,
        request_generator: Box<dyn ExternalRequestGenerator<TExternalRequest>>,
    ) -> Result<NetworkFile<TExternalRequest>, OpenReadError> {
        Ok(NetworkFile {
            file_name,
            file_size,
            request_generator,
        })
    }
}

#[async_trait]
impl<TExternalRequest: ExternalRequest + Debug + 'static> FileHandle for NetworkFile<TExternalRequest> {
    fn read_bytes(&self, byte_range: Range<usize>) -> io::Result<OwnedBytes> {
        Ok(OwnedBytes::new(self.request_generator.generate(&self.file_name, byte_range)?.request()?))
    }

    async fn read_bytes_async(&self, byte_range: Range<usize>) -> AsyncIoResult<OwnedBytes> {
        let request = self.request_generator.generate(&self.file_name, byte_range)?;
        let request_fut = request.request_async();
        Ok(OwnedBytes::new(request_fut.await?))
    }
}

impl<TExternalRequest: ExternalRequest> HasLen for NetworkFile<TExternalRequest> {
    fn len(&self) -> usize {
        self.file_size
    }
}
