use std::any::Any;
use std::fmt::{Debug, Formatter};
use std::{io, ops::Range, path::Path, sync::Arc};

use tantivy::directory::error::LockError;
use tantivy::directory::{DirectoryClone, DirectoryLock, Lock, WatchCallback, WatchHandle};
use tantivy::{
    directory::{error::OpenReadError, FileHandle, OwnedBytes},
    Directory, HasLen,
};
use tracing::{info, trace};

use super::ExternalRequestGenerator;
use crate::directories::{ExternalRequest, RequestError};
use crate::errors::ValidationError::InvalidHttpHeader;
use crate::errors::{SummaResult, ValidationError};

/// Allow to implement searching over HTTP
///
/// `NetworkDirectory` translates `read_bytes` calls into network requests.
pub struct NetworkDirectory<TExternalRequest: ExternalRequest> {
    external_request_generator: Box<dyn ExternalRequestGenerator<TExternalRequest>>,
}

impl<TExternalRequest: ExternalRequest> Debug for NetworkDirectory<TExternalRequest> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        "NetworkDirectory".fmt(f)
    }
}

impl<TExternalRequest: ExternalRequest> NetworkDirectory<TExternalRequest> {
    pub fn open(external_request_generator: Box<dyn ExternalRequestGenerator<TExternalRequest>>) -> NetworkDirectory<TExternalRequest> {
        NetworkDirectory { external_request_generator }
    }

    pub fn get_network_file_handle(&self, file_name: &Path) -> NetworkFile<TExternalRequest> {
        let file_name_str = file_name.to_string_lossy();
        NetworkFile::new(file_name_str.to_string(), self.external_request_generator.box_clone())
    }
}

impl<TExternalRequest: ExternalRequest + 'static> DirectoryClone for NetworkDirectory<TExternalRequest> {
    fn box_clone(&self) -> Box<dyn Directory> {
        Box::new(NetworkDirectory {
            external_request_generator: self.external_request_generator.box_clone(),
        })
    }
}

#[async_trait]
impl<TExternalRequest: ExternalRequest + 'static> Directory for NetworkDirectory<TExternalRequest> {
    fn get_file_handle(&self, file_name: &Path) -> Result<Arc<dyn FileHandle>, OpenReadError> {
        Ok(Arc::new(self.get_network_file_handle(file_name)))
    }

    fn exists(&self, path: &Path) -> Result<bool, OpenReadError> {
        Ok(self.get_file_handle(path)?.len() > 0)
    }

    fn atomic_read(&self, path: &Path) -> Result<Vec<u8>, OpenReadError> {
        let file_handle = self.get_network_file_handle(path);
        info!(action = "atomic_read", path = ?path);
        match file_handle.do_read_bytes(None) {
            Ok(bytes) => Ok(bytes.to_vec()),
            Err(RequestError::NotFound(p)) => Err(OpenReadError::FileDoesNotExist(p)),
            Err(RequestError::IoError(e, p)) => Err(OpenReadError::wrap_io_error(e, p)),
            Err(e) => panic!("{:?}", e),
        }
    }

    async fn atomic_read_async(&self, path: &Path) -> Result<Vec<u8>, OpenReadError> {
        let file_handle = self.get_network_file_handle(path);
        info!(action = "atomic_read_async", path = ?path);
        match file_handle.do_read_bytes_async(None).await {
            Ok(bytes) => Ok(bytes.to_vec()),
            Err(RequestError::NotFound(p)) => Err(OpenReadError::FileDoesNotExist(p)),
            Err(RequestError::IoError(e, p)) => Err(OpenReadError::wrap_io_error(e, p)),
            Err(e) => panic!("{:?}", e),
        }
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

    fn real_directory(&self) -> &dyn Directory {
        self
    }
}

/// `NetworkDirectory` creates `NetworkFile` for translating `read_bytes` calls into network requests
#[derive(Debug)]
pub struct NetworkFile<TExternalRequest: ExternalRequest> {
    file_name: String,
    request_generator: Box<dyn ExternalRequestGenerator<TExternalRequest>>,
}

impl<TExternalRequest: ExternalRequest> NetworkFile<TExternalRequest> {
    pub fn new(file_name: String, request_generator: Box<dyn ExternalRequestGenerator<TExternalRequest>>) -> NetworkFile<TExternalRequest> {
        NetworkFile { file_name, request_generator }
    }

    pub fn file_name(&self) -> &str {
        &self.file_name
    }

    pub fn url(&self) -> String {
        self.request_generator.generate_length_request(&self.file_name).url().to_string()
    }

    fn do_read_bytes(&self, byte_range: Option<Range<u64>>) -> Result<OwnedBytes, RequestError> {
        self.request_generator
            .generate_range_request(&self.file_name, byte_range)
            .request()
            .map(|r| OwnedBytes::new(r.data))
    }

    pub async fn do_read_bytes_async(&self, byte_range: Option<Range<u64>>) -> Result<OwnedBytes, RequestError> {
        let request = self.request_generator.generate_range_request(&self.file_name, byte_range);
        let url = request.url().to_string();
        trace!(action = "start_reading_file", url = ?url);
        let request_fut = request.request_async();
        trace!(action = "finish_reading_file", url = ?url);
        request_fut.await.map(|r| OwnedBytes::new(r.data))
    }

    pub fn internal_length(&self) -> SummaResult<u64> {
        let external_response = self.request_generator.generate_length_request(&self.file_name).request()?;
        Ok(external_response
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
            .ok_or_else(|| ValidationError::MissingHeader("content_range".to_string()))??)
    }
}

#[async_trait]
impl<TExternalRequest: ExternalRequest + Debug + 'static> FileHandle for NetworkFile<TExternalRequest> {
    fn read_bytes(&self, byte_range: Range<u64>) -> io::Result<OwnedBytes> {
        self.do_read_bytes(Some(byte_range))
            .map_err(|e| io::Error::new(io::ErrorKind::ConnectionRefused, e))
    }

    async fn read_bytes_async(&self, byte_range: Range<u64>) -> io::Result<OwnedBytes> {
        self.do_read_bytes_async(Some(byte_range))
            .await
            .map_err(|e| io::Error::new(io::ErrorKind::ConnectionRefused, e))
    }
}

impl<TExternalRequest: ExternalRequest> HasLen for NetworkFile<TExternalRequest> {
    fn len(&self) -> u64 {
        self.internal_length().unwrap_or_default()
    }
}
