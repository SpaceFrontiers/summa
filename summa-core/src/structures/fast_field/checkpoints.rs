//! Bounded header checkpoints; payloads remain in their original byte owner.
use super::{ColumnBlock, codec};

const MAX_CHECKPOINTS: usize = 256;

struct Checkpoint {
    column_block: u32,
    codec_block: u32,
    offset: u32,
}

pub(super) struct Checkpoints(Box<[Checkpoint]>);

impl Checkpoints {
    pub(super) fn new(blocks: &[ColumnBlock], multi: bool) -> Self {
        if multi {
            return Self(Box::default());
        }
        let count = |block: &ColumnBlock| {
            let data = block.data.as_slice();
            if data.first() == Some(&(codec::CodecType::BlockwiseLinear as u8)) {
                u32::from_le_bytes(data[5..9].try_into().unwrap()) as usize
            } else {
                0
            }
        };
        let total: usize = blocks.iter().map(count).sum();
        let stride = total.div_ceil(MAX_CHECKPOINTS).max(1);
        let mut entries = Vec::with_capacity(total.div_ceil(stride));
        let mut ordinal = 0;
        // Opening has already validated every header and payload extent.
        for (column_block, block) in blocks.iter().enumerate() {
            let data = block.data.as_slice();
            let mut offset = 9;
            for codec_block in 0..count(block) {
                if ordinal % stride == 0 {
                    entries.push(Checkpoint {
                        column_block: column_block as u32,
                        codec_block: codec_block as u32,
                        offset: (offset - 1) as u32,
                    });
                }
                let packed = u32::from_le_bytes(data[offset + 25..offset + 29].try_into().unwrap());
                offset += 29 + packed as usize;
                ordinal += 1;
            }
        }
        Self(entries.into_boxed_slice())
    }

    pub(super) fn heap_bytes(&self) -> usize {
        std::mem::size_of_val(self.0.as_ref())
    }

    #[inline]
    pub(super) fn read(&self, blocks: &[ColumnBlock], block: usize, local: u32) -> u64 {
        let data = blocks[block].data.as_slice();
        if data.first() != Some(&(codec::CodecType::BlockwiseLinear as u8)) {
            return codec::auto_read(data, local as usize);
        }
        let target = (
            block as u32,
            local / codec::BLOCKWISE_LINEAR_BLOCK_SIZE as u32,
        );
        let end = self
            .0
            .partition_point(|c| (c.column_block, c.codec_block) <= target);
        let (first, offset) = end
            .checked_sub(1)
            .map(|i| &self.0[i])
            .filter(|c| c.column_block == block as u32)
            .map_or((0, 8), |c| (c.codec_block as usize, c.offset as usize));
        codec::blockwise_linear_read_from(&data[1..], local as usize, first, offset)
    }
}
