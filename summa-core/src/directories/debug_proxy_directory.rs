// Copyright (C) 2022 Quickwit, Inc.
//
// Quickwit is offered under the AGPL v3.0 and as commercial software.
// For commercial licensing, contact us at hello@quickwit.io.
//
// AGPL:
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as
// published by the Free Software Foundation, either version 3 of the
// License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.

use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use std::{fmt, io, mem};

use async_trait::async_trait;
use tantivy::directory::error::OpenReadError;
use tantivy::directory::{FileHandle, OwnedBytes};
use tantivy::{Directory, HasLen};
use time::OffsetDateTime;

use crate::directories::Noop;

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
    pub offset: usize,
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
    offset: usize,
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

    pub fn with_offset(self, offset: usize) -> Self {
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
pub struct DebugProxyDirectory<D: Directory> {
    underlying: Arc<D>,
    operations: OperationBuffer,
}

impl<D: Directory> Clone for DebugProxyDirectory<D> {
    fn clone(&self) -> Self {
        DebugProxyDirectory {
            underlying: self.underlying.clone(),
            operations: self.operations.clone(),
        }
    }
}

impl<D: Directory> DebugProxyDirectory<D> {
    /// Wraps another directory to log all of its read operations.
    pub fn wrap(directory: D) -> Self {
        DebugProxyDirectory {
            underlying: Arc::new(directory),
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

struct DebugProxyFileHandle<D: Directory> {
    directory: DebugProxyDirectory<D>,
    underlying: Arc<dyn FileHandle>,
    path: PathBuf,
}

#[async_trait]
impl<D: Directory> FileHandle for DebugProxyFileHandle<D> {
    fn read_bytes(&self, byte_range: Range<usize>) -> io::Result<OwnedBytes> {
        let read_operation_builder = ReadOperationBuilder::new(&self.path).with_offset(byte_range.start);
        let payload = self.underlying.read_bytes(byte_range)?;
        let read_operation = read_operation_builder.terminate(payload.len());
        self.directory.register(read_operation);
        Ok(payload)
    }

    async fn read_bytes_async(&self, byte_range: Range<usize>) -> tantivy::AsyncIoResult<OwnedBytes> {
        let read_operation_builder = ReadOperationBuilder::new(&self.path).with_offset(byte_range.start);
        let payload = self.underlying.read_bytes_async(byte_range).await?;
        let read_operation = read_operation_builder.terminate(payload.len());
        self.directory.register_async(read_operation).await;
        Ok(payload)
    }
}

impl<D: Directory> fmt::Debug for DebugProxyFileHandle<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "DebugProxyFileHandle({:?})", &self.underlying)
    }
}

impl<D: Directory> HasLen for DebugProxyFileHandle<D> {
    fn len(&self) -> usize {
        self.underlying.len()
    }
}

impl<D: Directory> Directory for DebugProxyDirectory<D> {
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

    super::read_only_directory!();
}
