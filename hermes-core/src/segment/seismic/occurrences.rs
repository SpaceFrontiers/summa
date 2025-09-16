//! Restart groups for cardinality-aware summary cluster-ID sets.
use crate::structures::combination;
use crate::{Error, Result};
pub(super) const GROUP: usize = 128;

pub(super) fn bit_offsets(
    ends: &[u32],
    previous: u32,
    clusters: usize,
    out: &mut [usize; GROUP + 1],
) -> usize {
    let mut previous = previous;
    out[0] = 0;
    for (i, &end) in ends.iter().enumerate() {
        let count = (end - previous) as usize;
        out[i + 1] = out[i] + usize::from(combination::width(clusters, count));
        previous = end;
    }
    out[ends.len()].div_ceil(8)
}

/// Offsets are relative to the payload after the checkpoint directory.
/// Local layout appends each group's unchanged weight bytes after its ranks.
pub(super) fn encode(
    ids: &[u64],
    ends: &[u64],
    weights: &[u8],
    clusters: usize,
    local: bool,
) -> Result<Vec<u8>> {
    let mut output = vec![0; ends.len().div_ceil(GROUP) * 4];
    let directory = output.len();
    let mut previous = 0;
    for (block, group) in ends.chunks(GROUP).enumerate() {
        let offset = u32::try_from(output.len() - directory)
            .map_err(|_| Error::Corruption("summary subset stream exceeds address space".into()))?;
        output[block * 4..block * 4 + 4].copy_from_slice(&offset.to_le_bytes());
        let start = previous;
        let mut ranks = Vec::with_capacity(1024);
        let mut bit = 0;
        for &end in group {
            let end = end as usize;
            let values = &ids[previous..end];
            combination::append(
                &mut ranks,
                &mut bit,
                combination::rank(values),
                combination::width(clusters, values.len()),
            );
            previous = end;
        }
        output.extend(ranks);
        if local {
            output.extend_from_slice(&weights[start..previous]);
        }
    }
    Ok(output)
}
