use std::ops::Range;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Chunk {
    pub index: u64,
    pub chunk_left_ix: u64,
    pub chunk_right_ix: u64,
    pub inner_left_ix: usize,
    pub inner_right_ix: usize,
    pub target_ix: u64,
}

impl Chunk {
    /// Real chunk length
    #[inline]
    pub fn len(&self) -> usize {
        self.inner_right_ix - self.inner_left_ix
    }

    /// Requested data bounds, relative to chunk start.
    #[inline]
    pub fn data_bounds(&self) -> Range<usize> {
        self.inner_left_ix..self.inner_right_ix
    }

    /// Absolute chunk bounds shifted by `shift` to the left
    #[inline]
    pub fn shifted_chunk_range(&self, shift: u64) -> Range<u64> {
        (self.chunk_left_ix - shift)..(self.chunk_right_ix - shift)
    }
}

/// Used for producing `Chunk`s between left and right border
pub(crate) struct ChunkGenerator {
    current: u64,
    range: Range<u64>,
    file_size: u64,
    chunk_size: u64,
}

impl ChunkGenerator {
    pub fn new(range: Range<u64>, file_size: u64, chunk_size: u64) -> ChunkGenerator {
        ChunkGenerator {
            current: range.start,
            range,
            file_size,
            chunk_size,
        }
    }

    /// Absolute chunk index
    #[inline]
    pub fn index(&self) -> u64 {
        self.current / self.chunk_size
    }

    /// Starting index of where this chunk should be copied to
    #[inline]
    pub fn target_ix(&self) -> u64 {
        self.current - self.range.start
    }

    /// Left index of the chunk
    #[inline]
    pub fn chunk_left_ix(&self) -> u64 {
        self.current - self.current % self.chunk_size
    }

    #[inline]
    pub fn inner_left_ix(&self) -> usize {
        (self.current % self.chunk_size) as usize
    }

    #[inline]
    pub fn chunk_right_ix(&self) -> u64 {
        std::cmp::min(self.chunk_left_ix() + self.chunk_size, self.file_size)
    }

    #[inline]
    pub fn inner_right_ix(&self) -> usize {
        ((std::cmp::min(self.chunk_right_ix(), self.range.end) - 1) % self.chunk_size + 1) as usize
    }
}

impl Iterator for ChunkGenerator {
    type Item = Chunk;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.range.end {
            None
        } else {
            let chunk = Chunk {
                index: self.index(),
                chunk_left_ix: self.chunk_left_ix(),
                chunk_right_ix: self.chunk_right_ix(),
                inner_left_ix: self.inner_left_ix(),
                inner_right_ix: self.inner_right_ix(),
                target_ix: self.target_ix(),
            };
            self.current = self.chunk_right_ix();
            Some(chunk)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunker() {
        let mut chunk_iter = ChunkGenerator::new(3..27, 27, 10);
        let chunk = chunk_iter.next().unwrap();
        assert_eq!(chunk.chunk_left_ix, 0);
        assert_eq!(chunk.chunk_right_ix, 10);
        assert_eq!(chunk.inner_left_ix, 3);
        assert_eq!(chunk.inner_right_ix, 10);
        assert_eq!(chunk.target_ix, 0);
        assert_eq!(chunk.index, 0);

        let chunk = chunk_iter.next().unwrap();
        assert_eq!(chunk.chunk_left_ix, 10);
        assert_eq!(chunk.chunk_right_ix, 20);
        assert_eq!(chunk.inner_left_ix, 0);
        assert_eq!(chunk.inner_right_ix, 10);
        assert_eq!(chunk.target_ix, 7);
        assert_eq!(chunk.index, 1);

        let chunk = chunk_iter.next().unwrap();
        assert_eq!(chunk.chunk_left_ix, 20);
        assert_eq!(chunk.chunk_right_ix, 27);
        assert_eq!(chunk.inner_left_ix, 0);
        assert_eq!(chunk.inner_right_ix, 7);
        assert_eq!(chunk.target_ix, 17);
        assert_eq!(chunk.index, 2);
    }

    #[test]
    fn test_chunker_border_cases() {
        let mut chunk_iter = ChunkGenerator::new(3..5, 7, 10);
        let chunk = chunk_iter.next().unwrap();
        assert_eq!(chunk.chunk_left_ix, 0);
        assert_eq!(chunk.chunk_right_ix, 7);
        assert_eq!(chunk.inner_left_ix, 3);
        assert_eq!(chunk.inner_right_ix, 5);
        assert_eq!(chunk.target_ix, 0);
        assert_eq!(chunk.index, 0);
        assert!(chunk_iter.next().is_none());

        let mut chunk_iter = ChunkGenerator::new(3..11, 11, 10);
        let chunk = chunk_iter.next().unwrap();
        assert_eq!(chunk.chunk_left_ix, 0);
        assert_eq!(chunk.chunk_right_ix, 10);
        assert_eq!(chunk.inner_left_ix, 3);
        assert_eq!(chunk.inner_right_ix, 10);
        assert_eq!(chunk.target_ix, 0);
        assert_eq!(chunk.index, 0);
        let chunk = chunk_iter.next().unwrap();

        assert_eq!(chunk.chunk_left_ix, 10);
        assert_eq!(chunk.chunk_right_ix, 11);
        assert_eq!(chunk.inner_left_ix, 0);
        assert_eq!(chunk.inner_right_ix, 1);
        assert_eq!(chunk.target_ix, 7);
        assert_eq!(chunk.index, 1);
        assert!(chunk_iter.next().is_none());

        let mut chunk_iter = ChunkGenerator::new(0..2, 10, 2);
        let chunk = chunk_iter.next().unwrap();
        assert_eq!(chunk.chunk_left_ix, 0);
        assert_eq!(chunk.chunk_right_ix, 2);
        assert_eq!(chunk.inner_left_ix, 0);
        assert_eq!(chunk.inner_right_ix, 2);
        assert_eq!(chunk.target_ix, 0);
        assert_eq!(chunk.index, 0);
        assert!(chunk_iter.next().is_none());
    }
}
