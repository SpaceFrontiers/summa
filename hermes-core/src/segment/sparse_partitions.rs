//! Segment-owned sparse envelopes for immutable Seismic nomination partitions.
//!
//! The Seismic codec owns payloads; this adapter owns file writers and shares the
//! ordinary sparse TOC/footer with the root file. Publication remains with the
//! segment lifecycle.

use std::io::Write;

use super::format::{SparseFieldToc, write_sparse_toc_and_footer};
use super::seismic::{OutputLengths, PARTITIONS, SeismicWriter};
use super::{OffsetWriter, SegmentFiles};
use crate::Result;
use crate::directories::DirectoryWriter;
use crate::structures::WeightQuantization;

pub(super) const PARTITION_BUFFER_BYTES: usize = 64 * 1024;

struct PartitionWriter {
    writer: OffsetWriter,
    fields: Vec<SparseFieldToc>,
}

pub(super) struct SparsePartitionWriters {
    writers: Vec<Option<PartitionWriter>>,
}

impl SparsePartitionWriters {
    pub(super) async fn create<D: DirectoryWriter>(
        dir: &D,
        files: &SegmentFiles,
        selected: impl Fn(usize) -> bool,
    ) -> Result<Self> {
        let mut writers = Vec::with_capacity(PARTITIONS);
        for partition in 0..PARTITIONS {
            writers.push(if selected(partition) {
                Some(PartitionWriter {
                    writer: OffsetWriter::new(
                        dir.streaming_writer_cold_with_capacity(
                            &files.seismic_partition(partition),
                            PARTITION_BUFFER_BYTES,
                        )
                        .await?,
                    ),
                    fields: Vec::new(),
                })
            } else {
                None
            });
        }
        Ok(Self { writers })
    }

    /// Local buffers and field directories retained alongside codec scratch.
    #[cfg(feature = "native")]
    pub(super) fn scratch_bytes(&self) -> usize {
        self.writers.capacity() * std::mem::size_of::<Option<PartitionWriter>>()
            + self
                .writers
                .iter()
                .flatten()
                .map(|output| {
                    PARTITION_BUFFER_BYTES
                        + output.fields.capacity() * std::mem::size_of::<SparseFieldToc>()
                        + output
                            .fields
                            .iter()
                            .map(|field| {
                                field.dims.capacity()
                                    * std::mem::size_of::<super::format::SparseDimTocEntry>()
                            })
                            .sum::<usize>()
                })
                .sum::<usize>()
    }

    pub(super) fn writer(&mut self, partition: usize) -> &mut OffsetWriter {
        &mut self.writers[partition]
            .as_mut()
            .expect("partition was opened before its codec writes")
            .writer
    }

    pub(super) fn record_field(
        &mut self,
        partition: usize,
        field: u32,
        rows: u32,
        quantization: WeightQuantization,
        bytes: u64,
    ) {
        let output = self.writers[partition].as_mut().unwrap();
        output.fields.push(SparseFieldToc::seismic(
            field,
            rows,
            output.writer.offset() - bytes,
            bytes,
            quantization,
        ));
    }

    pub(super) fn record_outputs(
        &mut self,
        field: u32,
        rows: u32,
        quantization: WeightQuantization,
        lengths: &OutputLengths,
    ) {
        for (partition, &bytes) in lengths.partitions.iter().enumerate() {
            self.record_field(partition, field, rows, quantization, bytes);
        }
    }

    pub(super) fn finish(self) -> Result<usize> {
        let mut bytes = 0usize;
        for mut output in self.writers.into_iter().flatten() {
            let offset = output.writer.offset();
            write_sparse_toc_and_footer(&mut output.writer, offset, offset, &output.fields)?;
            bytes += output.writer.offset() as usize;
            output.writer.finish()?;
        }
        Ok(bytes)
    }
}

pub(super) struct SeismicFieldWriter<'a> {
    pub(super) root: &'a mut dyn Write,
    pub(super) partitions: &'a mut SparsePartitionWriters,
}

impl SeismicWriter for SeismicFieldWriter<'_> {
    fn root(&mut self) -> &mut dyn Write {
        self.root
    }

    fn partition(&mut self, partition: usize) -> &mut dyn Write {
        self.partitions.writer(partition)
    }
}
