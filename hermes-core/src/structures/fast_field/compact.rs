//! Re-encode only a bounded mixed source range with the existing column writer.
use super::*;

impl ColumnBlock {
    pub(crate) fn compact_range(
        &self,
        column_type: FastFieldColumnType,
        multi: bool,
        range: std::ops::Range<u32>,
        keep: impl Fn(u32) -> bool,
        budget: usize,
    ) -> crate::Result<Vec<u8>> {
        use crate::Error;
        let text = column_type == FastFieldColumnType::TextOrdinal;
        if range.start > range.end || range.end > self.num_docs || range.len() > 4096 {
            return Err(Error::Corruption("invalid compaction column range".into()));
        }
        let mut output = match (text, multi) {
            (true, false) => FastFieldWriter::new_text(),
            (true, true) => FastFieldWriter::new_text_multi(),
            (false, false) => FastFieldWriter::new_numeric(column_type),
            (false, true) => FastFieldWriter::new_numeric_multi(column_type),
        };
        let mut bytes = range.len().saturating_mul(32).saturating_add(8192);
        let admit = |bytes| {
            if bytes > budget / 4 {
                Err(Error::Schema(
                    "mixed column chunk exceeds compaction scratch budget".into(),
                ))
            } else {
                Ok(())
            }
        };
        admit(bytes)?;
        let mut next = 0;
        let mut values = [0u64; 256];
        let mut offsets = [0u64; 257];
        for start in (range.start..range.end).step_by(256) {
            let count = (range.end - start).min(256) as usize;
            if multi {
                codec::auto_read_batch(
                    self.offset_data.as_slice(),
                    start as usize,
                    &mut offsets[..count + 1],
                );
            } else {
                codec::auto_read_batch(self.data.as_slice(), start as usize, &mut values[..count]);
            }
            for local in 0..count {
                if !keep(self.cumulative_docs + start + local as u32) {
                    continue;
                }
                let value_range = if multi {
                    let begin = usize::try_from(offsets[local])
                        .map_err(|_| Error::Corruption("column value offset overflow".into()))?;
                    let end = usize::try_from(offsets[local + 1])
                        .map_err(|_| Error::Corruption("column value offset overflow".into()))?;
                    if begin > end {
                        return Err(Error::Corruption("inverted column value range".into()));
                    }
                    admit(bytes.saturating_add((end - begin).saturating_mul(64)))?;
                    Some((begin, end))
                } else {
                    None
                };
                let mut append = |value: u64| -> crate::Result<()> {
                    bytes = bytes.saturating_add(64);
                    if text {
                        let value = u32::try_from(value)
                            .ok()
                            .and_then(|ordinal| self.dict.as_ref()?.get(ordinal))
                            .ok_or_else(|| {
                                Error::Corruption("invalid local text ordinal".into())
                            })?;
                        bytes = bytes.saturating_add(value.len().saturating_mul(4));
                        admit(bytes)?;
                        output.add_text(next, value);
                    } else {
                        admit(bytes)?;
                        output.add_u64(next, value);
                    }
                    Ok(())
                };
                if let Some((begin, end)) = value_range {
                    for at in (begin..end).step_by(256) {
                        let size = (end - at).min(256);
                        codec::auto_read_batch(self.value_data.as_slice(), at, &mut values[..size]);
                        for &value in &values[..size] {
                            append(value)?;
                        }
                    }
                } else if values[local] != FAST_FIELD_MISSING {
                    append(values[local])?;
                }
                next += 1;
            }
        }
        output.pad_to(next);
        let mut encoded = Vec::new();
        output.serialize(&mut encoded, 0)?;
        Ok(encoded)
    }
}
