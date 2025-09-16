use super::*;

impl SeismicIndex {
    /// Admit the format envelope and payload extents once, before search views.
    pub(crate) fn parse(bytes: OwnedBytes, total_docs: u32, total_vectors: u32) -> Result<Self> {
        Self::parse_component(bytes, total_docs, total_vectors, None)
    }

    pub(crate) fn attach_partition(
        &mut self,
        id: usize,
        bytes: OwnedBytes,
        _source_offset: u64,
    ) -> Result<()> {
        if id >= PARTITIONS || self.partitions[id].is_some() {
            return Err(corrupt("invalid or duplicate nomination partition"));
        }
        let part = Self::parse_component(bytes, u32::MAX, self.rows, Some(id))?;
        if part.dims != self.dims
            || part.quantization != self.quantization
            || part.postings != self.postings
            || part.cluster_size != self.cluster_size
            || part.summary_energy.to_bits() != self.summary_energy.to_bits()
        {
            return Err(corrupt("nomination partition settings differ from root"));
        }
        self.pending_terms = self
            .pending_terms
            .checked_add(part.pending_terms)
            .ok_or_else(|| corrupt("term debt overflow"))?;
        self.pending_bytes = self
            .pending_bytes
            .checked_add(part.pending_bytes)
            .ok_or_else(|| corrupt("fragmented term byte count overflow"))?;
        self.partitions[id] = Some(Partition {
            bytes: part.bytes,
            #[cfg(any(feature = "native", test))]
            source_offset: _source_offset,
            runs: part.runs,
            #[cfg(any(feature = "native", test))]
            pending_terms: part.pending_terms,
            #[cfg(any(feature = "native", test))]
            pending_bytes: part.pending_bytes,
        });
        Ok(())
    }

    pub(crate) fn finish_partitions(&self) -> Result<()> {
        if self.partitions.iter().any(Option::is_none) {
            return Err(corrupt("missing nomination partition"));
        }
        Ok(())
    }

    fn parse_component(
        bytes: OwnedBytes,
        total_docs: u32,
        total_vectors: u32,
        partition: Option<usize>,
    ) -> Result<Self> {
        let footer = bytes
            .len()
            .checked_sub(FOOTER)
            .ok_or_else(|| corrupt("missing footer"))?;
        let f = &bytes[footer..];
        if u32_at(f, 0) != MAGIC
            || u32_at(f, 4) != VERSION
            || u32_at(f, 36) != partition.map_or(0, |id| id as u32 + 1)
        {
            return Err(corrupt("unsupported format"));
        }
        let quantization = match u32_at(f, 12) {
            0 => WeightQuantization::Float32,
            1 => WeightQuantization::Float16,
            2 => WeightQuantization::UInt8,
            3 => WeightQuantization::UInt4,
            _ => return Err(corrupt("unknown precision")),
        };
        let dims = u32_at(f, 8);
        let postings = u32_at(f, 16);
        let cluster_size = u32_at(f, 20);
        let summary_energy = f32::from_bits(u32_at(f, 24));
        let rows = u32_at(f, 28);
        let count = u32_at(f, 32) as usize;
        if dims == 0
            || postings == 0
            || postings > 65_536
            || cluster_size == 0
            || cluster_size > postings
            || !summary_energy.is_finite()
            || !(0.0..=1.0).contains(&summary_energy)
            || summary_energy == 0.0
            || rows != total_vectors
        {
            return Err(corrupt("invalid settings or vector count"));
        }
        let dir = count
            .checked_mul(RUN_ENTRY)
            .and_then(|n| footer.checked_sub(n))
            .ok_or_else(|| corrupt("invalid run directory"))?;
        let mut single_valued = true;
        let mut forward_entries = 0u64;
        let mut term_dims = Vec::new();
        let mut runs = Vec::with_capacity(count);
        let mut expected_offset = 0usize;
        let mut expected_row = 0u32;
        let mut previous_key = None;
        for entry in bytes[dir..footer].chunks_exact(RUN_ENTRY) {
            let offset = usize::try_from(u64_at(entry, 0))
                .map_err(|_| corrupt("run offset exceeds address space"))?;
            let len = usize::try_from(u64_at(entry, 8))
                .map_err(|_| corrupt("run length exceeds address space"))?;
            let end = offset
                .checked_add(len)
                .ok_or_else(|| corrupt("run overflow"))?;
            let doc_base = u32_at(entry, 16);
            let row_base = u32_at(entry, 20);
            let run_rows = u32_at(entry, 24);
            if offset != expected_offset
                || end > dir
                || len < RUN_FOOTER
                || row_base != expected_row
                || u32_at(entry, 28) != 0
            {
                return Err(corrupt("invalid run extent"));
            }
            let payload = bytes.slice(offset..end);
            let rf = payload.len() - RUN_FOOTER;
            let row_offset = usize::try_from(u64_at(&payload, rf + 8))
                .map_err(|_| corrupt("row offset overflow"))?;
            let term_offset = usize::try_from(u64_at(&payload, rf + 16))
                .map_err(|_| corrupt("term offset overflow"))?;
            let terms = u32_at(&payload, rf + 4);
            let directory_rows = if partition.is_none() { run_rows } else { 0 };
            let row_end = (directory_rows as usize)
                .checked_mul(ROW_ENTRY)
                .and_then(|n| row_offset.checked_add(n))
                .ok_or_else(|| corrupt("row directory overflow"))?;
            if u32_at(&payload, rf) != run_rows
                || u32_at(&payload, rf + 24) != RUN_MAGIC
                || u32_at(&payload, rf + 28) != 0
                || row_end > term_offset
                || (partition.is_none() && (terms != 0 || row_end != term_offset))
                || (partition.is_some() && row_offset != 0)
                || (terms as usize)
                    .checked_mul(TERM_ENTRY)
                    .and_then(|n| term_offset.checked_add(n))
                    != Some(rf)
            {
                return Err(corrupt("invalid run directories"));
            }
            let mut next_vector = 0usize;
            for r in payload[row_offset..row_end].chunks_exact(ROW_ENTRY) {
                let doc = u32_at(r, 0)
                    .checked_add(doc_base)
                    .ok_or_else(|| corrupt("document overflow"))?;
                let key = LogicalUnit {
                    doc,
                    ordinal: u16::from_le_bytes(r[4..6].try_into().unwrap()),
                };
                let at =
                    usize::try_from(u64_at(r, 8)).map_err(|_| corrupt("vector offset overflow"))?;
                let n = u32_at(r, 20) as usize;
                forward_entries = forward_entries
                    .checked_add(n as u64)
                    .ok_or_else(|| corrupt("forward coordinate count overflow"))?;
                let len = u32_at(r, 16) as usize;
                if doc >= total_docs
                    || previous_key.is_some_and(|old| old >= key)
                    || r[6..8] != [0; 2]
                    || at != next_vector
                    || forward::vector_bytes(n, quantization) != Some(len)
                {
                    return Err(corrupt("invalid forward row"));
                }
                next_vector = at
                    .checked_add(len)
                    .filter(|&end| end <= row_offset)
                    .ok_or_else(|| corrupt("invalid forward payload extent"))?;
                single_valued &= key.ordinal == 0;
                previous_key = Some(key);
            }
            if next_vector != row_offset {
                return Err(corrupt("unowned forward bytes"));
            }
            let mut previous_dim = None;
            let mut term_extents = Vec::with_capacity(terms as usize);
            for t in payload[term_offset..rf].chunks_exact(TERM_ENTRY) {
                let dim = u32_at(t, 0);
                let clusters = u32_at(t, 4);
                let at =
                    usize::try_from(u64_at(t, 8)).map_err(|_| corrupt("term offset overflow"))?;
                let len =
                    usize::try_from(u64_at(t, 16)).map_err(|_| corrupt("term length overflow"))?;
                let end = at
                    .checked_add(len)
                    .filter(|&end| end <= term_offset)
                    .ok_or_else(|| corrupt("term extent overflow"))?;
                if dim >= dims
                    || partition.is_some_and(|id| dim as usize % PARTITIONS != id)
                    || previous_dim.is_some_and(|p| p > dim)
                    || at < row_end
                    || u32_at(t, 24) > run_rows
                    || (clusters > 0 && u32_at(t, 24) == run_rows)
                    || u32_at(t, 28) > 65_536
                    || u32_at(t, 32) > run_rows
                    || u32_at(t, 36) != 0
                {
                    return Err(corrupt("invalid term directory"));
                }
                term_dims.push((dim, len as u64, u32_at(t, 32)));
                term_extents.push((at, end, clusters));
                previous_dim = Some(dim);
            }
            term_extents.sort_unstable();
            let mut next_term = row_end;
            for &(start, end, _) in &term_extents {
                if start != next_term {
                    return Err(corrupt("overlapping or gapped term payloads"));
                }
                next_term = end;
            }
            if next_term != term_offset {
                return Err(corrupt("unowned term bytes"));
            }
            // Summary payloads are immutable writer output. Opening a run
            // does not read, prefetch or validate their contents.
            runs.push(Run {
                row_directory: payload.slice(row_offset..row_end),
                term_directory: payload.slice(term_offset..rf),
                bytes: payload,
                #[cfg(any(feature = "native", test))]
                offset: offset as u64,
                doc_base,
                row_base,
                #[cfg(any(feature = "native", test))]
                rows: run_rows,
                #[cfg(test)]
                term_offset,
                terms,
            });
            expected_offset = end;
            expected_row = expected_row
                .checked_add(run_rows)
                .ok_or_else(|| corrupt("row count overflow"))?;
        }
        if expected_offset != dir || expected_row != rows {
            return Err(corrupt("run totals disagree"));
        }
        term_dims.sort_unstable();
        for group in term_dims.chunk_by(|a, b| a.0 == b.0) {
            let frequency = group
                .iter()
                .try_fold(0u32, |sum, entry| sum.checked_add(entry.2))
                .ok_or_else(|| corrupt("term frequency overflow"))?;
            if frequency > rows {
                return Err(corrupt("term frequency exceeds forward rows"));
            }
        }
        let pending_terms = term_dims
            .chunk_by(|a, b| a.0 == b.0)
            .filter(|group| group.len() > 1)
            .count() as u32;
        let pending_bytes = term_dims
            .chunk_by(|a, b| a.0 == b.0)
            .filter(|group| group.len() > 1)
            .flat_map(|group| group.iter().map(|entry| entry.1))
            .try_fold(0u64, |sum, bytes| sum.checked_add(bytes))
            .ok_or_else(|| corrupt("fragmented term byte count overflow"))?;
        Ok(Self {
            bytes,
            source_offset: 0,
            runs,
            partitions: vec![None; PARTITIONS],
            dims,
            quantization,
            postings,
            cluster_size,
            summary_energy,
            rows,
            single_valued,
            pending_terms,
            pending_bytes,
            forward_entries,
        })
    }
}
