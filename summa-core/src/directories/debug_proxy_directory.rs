use std::any::Any;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use std::{fmt, io, mem};

use async_trait::async_trait;
use tantivy::directory::error::{LockError, OpenReadError};
use tantivy::directory::{DirectoryLock, FileHandle, Lock, OwnedBytes};
use tantivy::{Directory, HasLen};
use time::OffsetDateTime;

#[derive(Clone, Default)]
struct OperationBuffer(Arc<Mutex<Vec<ReadOperation>>>);

impl fmt::Debug for OperationBuffer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "OperationBuffer")
    }
}

impl OperationBuffer {
    fn drain(&self) -> impl Iterator<Item = ReadOperation> + 'static {
        let mut guard = self.0.lock().expect("poisoned");
        let ops: Vec<ReadOperation> = mem::take(guard.as_mut());
        ops.into_iter()
    }

    fn push(&self, read_operation: ReadOperation) {
        let mut guard = self.0.lock().expect("poisoned");
        guard.push(read_operation);
    }
}

/// A ReadOperation records meta data about a read operation.
/// It is recorded by the `DebugProxyDirectory`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReadOperation {
    /// Path that was read
    pub path: PathBuf,
    /// If fetching a range of data, the start offset, else 0.
    pub offset: u64,
    /// The number of bytes fetched
    pub num_bytes: usize,
    /// The date at which the operation was performed (UTC timezone).
    pub start_date: OffsetDateTime,
    /// The elapsed time to run the read operatioon.
    pub duration: Duration,
}

struct ReadOperationBuilder {
    start_date: OffsetDateTime,
    start_instant: Instant,
    path: PathBuf,
    offset: u64,
}

impl ReadOperationBuilder {
    pub fn new(path: &Path) -> Self {
        let start_instant = Instant::now();
        let start_date = OffsetDateTime::now_utc();
        ReadOperationBuilder {
            start_date,
            start_instant,
            path: path.to_path_buf(),
            offset: 0,
        }
    }

    pub fn with_offset(self, offset: u64) -> Self {
        ReadOperationBuilder {
            start_date: self.start_date,
            start_instant: self.start_instant,
            path: self.path,
            offset,
        }
    }

    fn terminate(self, num_bytes: usize) -> ReadOperation {
        let duration = self.start_instant.elapsed();
        ReadOperation {
            path: self.path.clone(),
            offset: self.offset,
            num_bytes,
            start_date: self.start_date,
            duration,
        }
    }
}

/// The debug proxy wraps another directory and simply acts as a proxy
/// recording all of its read operations.
///
/// It has two purpose
/// - It is used when building our hotcache, to identify the file sections that
/// should be in the hotcache.
/// - It is used in the search-api to provide debugging/performance information.
#[derive(Debug)]
pub struct DebugProxyDirectory {
    underlying: Box<dyn Directory>,
    operations: OperationBuffer,
}

impl Clone for DebugProxyDirectory {
    fn clone(&self) -> Self {
        DebugProxyDirectory {
            underlying: self.underlying.box_clone(),
            operations: self.operations.clone(),
        }
    }
}

impl DebugProxyDirectory {
    /// Wraps another directory to log all of its read operations.
    pub fn wrap(directory: Box<dyn Directory>) -> Self {
        DebugProxyDirectory {
            underlying: directory,
            operations: OperationBuffer::default(),
        }
    }

    /// Returns all of the existing read operations.
    ///
    /// Calling this "drains" the existing queue of operations.
    pub fn drain_read_operations(&self) -> impl Iterator<Item = ReadOperation> + '_ {
        self.operations.drain()
    }

    /// Adds a new operation
    fn register(&self, read_op: ReadOperation) {
        self.operations.push(read_op);
    }

    /// Adds a new operation in an async fashion.
    async fn register_async(&self, read_op: ReadOperation) {
        self.operations.push(read_op);
    }
}

struct DebugProxyFileHandle {
    directory: DebugProxyDirectory,
    underlying: Arc<dyn FileHandle>,
    path: PathBuf,
}

#[async_trait]
impl FileHandle for DebugProxyFileHandle {
    fn read_bytes(&self, byte_range: Range<u64>) -> io::Result<OwnedBytes> {
        let read_operation_builder = ReadOperationBuilder::new(&self.path).with_offset(byte_range.start);
        let payload = self.underlying.read_bytes(byte_range)?;
        let read_operation = read_operation_builder.terminate(payload.len());
        self.directory.register(read_operation);
        Ok(payload)
    }

    async fn read_bytes_async(&self, byte_range: Range<u64>) -> io::Result<OwnedBytes> {
        let read_operation_builder = ReadOperationBuilder::new(&self.path).with_offset(byte_range.start);
        let payload = self.underlying.read_bytes_async(byte_range).await?;
        let read_operation = read_operation_builder.terminate(payload.len());
        self.directory.register_async(read_operation).await;
        Ok(payload)
    }
}

impl fmt::Debug for DebugProxyFileHandle {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "DebugProxyFileHandle({:?})", &self.underlying)
    }
}

impl HasLen for DebugProxyFileHandle {
    fn len(&self) -> u64 {
        self.underlying.len()
    }
}

#[async_trait]
impl Directory for DebugProxyDirectory {
    fn get_file_handle(&self, path: &Path) -> Result<Arc<dyn FileHandle>, OpenReadError> {
        let underlying = self.underlying.get_file_handle(path)?;
        Ok(Arc::new(DebugProxyFileHandle {
            underlying,
            directory: self.clone(),
            path: path.to_owned(),
        }))
    }

    fn exists(&self, path: &Path) -> Result<bool, OpenReadError> {
        self.underlying.exists(path)
    }

    fn atomic_read(&self, path: &Path) -> Result<Vec<u8>, OpenReadError> {
        let read_operation_builder = ReadOperationBuilder::new(path);
        let payload = self.underlying.atomic_read(path)?;
        let read_operation = read_operation_builder.terminate(payload.len());
        self.register(read_operation);
        Ok(payload.to_vec())
    }

    async fn atomic_read_async(&self, path: &Path) -> Result<Vec<u8>, OpenReadError> {
        let read_operation_builder = ReadOperationBuilder::new(path);
        let payload = self.underlying.atomic_read_async(path).await?;
        let read_operation = read_operation_builder.terminate(payload.len());
        self.register(read_operation);
        Ok(payload.to_vec())
    }

    fn acquire_lock(&self, _lock: &Lock) -> Result<DirectoryLock, LockError> {
        Ok(tantivy::directory::DirectoryLock::from(Box::new(|| {})))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
    fn underlying_directory(&self) -> Option<&dyn Directory> {
        Some(self.underlying.as_ref())
    }

    fn real_directory(&self) -> &dyn Directory {
        self.underlying.real_directory()
    }
}
