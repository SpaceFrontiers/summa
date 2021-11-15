use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::{io, ops::Range, path::Path, sync::Arc, usize};

use parking_lot::RwLock;
use tantivy::directory::error::{DeleteError, OpenWriteError};
use tantivy::directory::{AntiCallToken, DirectoryClone, TerminatingWrite, WatchCallback, WatchHandle, WritePtr};
use tantivy::{
    directory::{error::OpenReadError, FileHandle, OwnedBytes},
    AsyncIoResult, Directory, HasLen,
};

use super::ExternalRequestGenerator;
use crate::directories::ExternalRequest;
use crate::errors::ValidationError::InvalidHttpHeader;
use crate::errors::{SummaResult, ValidationError};

struct Noop {}

impl Write for Noop {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl TerminatingWrite for Noop {
    fn terminate_ref(&mut self, _: AntiCallToken) -> io::Result<()> {
        Ok(())
    }
}

pub struct NetworkDirectory<TExternalRequest: ExternalRequest> {
    file_lengths: Arc<RwLock<HashMap<PathBuf, u64>>>,
    external_request_generator: Box<dyn ExternalRequestGenerator<TExternalRequest>>,
}

impl<TExternalRequest: ExternalRequest> Debug for NetworkDirectory<TExternalRequest> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        "NetworkDirectory".fmt(f)
    }
}

impl<TExternalRequest: ExternalRequest> NetworkDirectory<TExternalRequest> {
    pub fn open(
        external_request_generator: Box<dyn ExternalRequestGenerator<TExternalRequest>>,
        file_lengths: Arc<RwLock<HashMap<PathBuf, u64>>>,
    ) -> NetworkDirectory<TExternalRequest> {
        NetworkDirectory {
            file_lengths,
            external_request_generator,
        }
    }
}

impl<TExternalRequest: ExternalRequest + 'static> DirectoryClone for NetworkDirectory<TExternalRequest> {
    fn box_clone(&self) -> Box<dyn Directory> {
        Box::new(NetworkDirectory {
            file_lengths: self.file_lengths.clone(),
            external_request_generator: self.external_request_generator.box_clone(),
        })
    }
}

impl<TExternalRequest: ExternalRequest + 'static> Directory for NetworkDirectory<TExternalRequest> {
    fn get_file_handle(&self, file_name: &Path) -> Result<Arc<dyn FileHandle>, OpenReadError> {
        let file_name_str = file_name.to_string_lossy();
        Ok(Arc::new(NetworkFile::new(
            file_name_str.to_string(),
            self.file_lengths.read().get(file_name).cloned(),
            self.external_request_generator.box_clone(),
        )?))
    }

    fn delete(&self, _: &Path) -> Result<(), DeleteError> {
        Ok(())
    }

    fn exists(&self, path: &Path) -> Result<bool, OpenReadError> {
        Ok(self.get_file_handle(path)?.len() > 0)
    }

    fn open_write(&self, _: &Path) -> Result<WritePtr, OpenWriteError> {
        Ok(BufWriter::new(Box::new(Noop {})))
    }

    fn atomic_read(&self, path: &Path) -> Result<Vec<u8>, OpenReadError> {
        let file_handle = self.get_file_handle(path)?;
        Ok(file_handle
            .read_bytes(0..file_handle.len())
            .map_err(|e| OpenReadError::wrap_io_error(e, path.to_path_buf()))?
            .to_vec())
    }

    fn atomic_write(&self, _: &Path, _: &[u8]) -> io::Result<()> {
        Ok(())
    }

    fn sync_directory(&self) -> io::Result<()> {
        Ok(())
    }

    fn watch(&self, _: WatchCallback) -> tantivy::Result<WatchHandle> {
        Ok(WatchHandle::empty())
    }
}

#[derive(Debug)]
struct NetworkFile<TExternalRequest: ExternalRequest> {
    file_name: String,
    file_length: Option<u64>,
    request_generator: Box<dyn ExternalRequestGenerator<TExternalRequest>>,
}

impl<TExternalRequest: ExternalRequest> NetworkFile<TExternalRequest> {
    pub fn new(
        file_name: String,
        file_length: Option<u64>,
        request_generator: Box<dyn ExternalRequestGenerator<TExternalRequest>>,
    ) -> Result<NetworkFile<TExternalRequest>, OpenReadError> {
        Ok(NetworkFile {
            file_name,
            file_length,
            request_generator,
        })
    }

    pub fn internal_length(&self) -> SummaResult<u64> {
        Ok(match self.file_length {
            Some(file_length) => file_length,
            None => {
                let external_response = self.request_generator.generate_length_request(&self.file_name)?.request()?;
                external_response
                    .headers
                    .iter()
                    .find_map(|header| {
                        if header.name == "content-length" {
                            Some(
                                header
                                    .value
                                    .parse::<u64>()
                                    .map_err(|_| InvalidHttpHeader(header.name.clone(), header.value.clone())),
                            )
                        } else {
                            None
                        }
                    })
                    .ok_or_else(|| ValidationError::MissingHeader("content_range".to_string()))??
            }
        })
    }
}

#[async_trait]
impl<TExternalRequest: ExternalRequest + Debug + 'static> FileHandle for NetworkFile<TExternalRequest> {
    fn read_bytes(&self, byte_range: Range<usize>) -> io::Result<OwnedBytes> {
        let request_response = self.request_generator.generate_range_request(&self.file_name, byte_range)?.request()?;
        Ok(OwnedBytes::new(request_response.data))
    }

    async fn read_bytes_async(&self, byte_range: Range<usize>) -> AsyncIoResult<OwnedBytes> {
        let request = self.request_generator.generate_range_request(&self.file_name, byte_range)?;
        let request_fut = request.request_async();
        Ok(OwnedBytes::new(request_fut.await?.data))
    }
}

impl<TExternalRequest: ExternalRequest> HasLen for NetworkFile<TExternalRequest> {
    fn len(&self) -> usize {
        self.internal_length().unwrap_or(0) as usize
    }
}
