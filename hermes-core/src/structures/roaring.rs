//! Roaring Bitmap implementation for compressed integer sets
//!
//! Roaring bitmaps use a hybrid approach:
//! - Sparse containers: sorted arrays for low-density chunks
//! - Dense containers: bitmaps for high-density chunks
//! - Run containers: RLE for consecutive ranges
//!
//! This provides excellent compression and fast set operations.
//! Used by Apache Lucene, Spark, Druid, and many databases.

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{self, Read, Write};

/// Threshold for switching from array to bitmap container
/// If a container has more than this many elements, use bitmap
const ARRAY_TO_BITMAP_THRESHOLD: usize = 4096;

/// Container types for different density patterns
#[derive(Debug, Clone)]
enum Container {
    /// Sorted array of 16-bit values (for sparse containers)
    Array(Vec<u16>),
    /// Bitmap of 2^16 bits (for dense containers)
    Bitmap(Box<[u64; 1024]>),
    /// Run-length encoded (start, length) pairs
    Runs(Vec<(u16, u16)>),
}

impl Container {
    fn new_array() -> Self {
        Container::Array(Vec::new())
    }

    #[allow(dead_code)]
    fn new_bitmap() -> Self {
        Container::Bitmap(Box::new([0u64; 1024]))
    }

    fn cardinality(&self) -> u32 {
        match self {
            Container::Array(arr) => arr.len() as u32,
            Container::Bitmap(bm) => bm.iter().map(|w| w.count_ones()).sum(),
            Container::Runs(runs) => runs.iter().map(|(_, len)| *len as u32 + 1).sum(),
        }
    }

    fn contains(&self, val: u16) -> bool {
        match self {
            Container::Array(arr) => arr.binary_search(&val).is_ok(),
            Container::Bitmap(bm) => {
                let word_idx = (val / 64) as usize;
                let bit_idx = val % 64;
                (bm[word_idx] >> bit_idx) & 1 == 1
            }
            Container::Runs(runs) => {
                for &(start, len) in runs {
                    if val >= start && val <= start + len {
                        return true;
                    }
                    if val < start {
                        return false;
                    }
                }
                false
            }
        }
    }

    fn insert(&mut self, val: u16) -> bool {
        match self {
            Container::Array(arr) => {
                match arr.binary_search(&val) {
                    Ok(_) => false, // Already exists
                    Err(pos) => {
                        arr.insert(pos, val);
                        true
                    }
                }
            }
            Container::Bitmap(bm) => {
                let word_idx = (val / 64) as usize;
                let bit_idx = val % 64;
                let old = bm[word_idx];
                bm[word_idx] |= 1u64 << bit_idx;
                old != bm[word_idx]
            }
            Container::Runs(runs) => {
                // For simplicity, convert to array, insert, then optimize
                let mut arr: Vec<u16> = Vec::new();
                for &(start, len) in runs.iter() {
                    for i in 0..=len {
                        arr.push(start + i);
                    }
                }
                let inserted = match arr.binary_search(&val) {
                    Ok(_) => false,
                    Err(pos) => {
                        arr.insert(pos, val);
                        true
                    }
                };
                *self = Container::Array(arr);
                inserted
            }
        }
    }

    fn optimize(&mut self) {
        let card = self.cardinality() as usize;

        match self {
            Container::Array(arr) if card > ARRAY_TO_BITMAP_THRESHOLD => {
                // Convert to bitmap
                let mut bm = Box::new([0u64; 1024]);
                for &val in arr.iter() {
                    let word_idx = (val / 64) as usize;
                    let bit_idx = val % 64;
                    bm[word_idx] |= 1u64 << bit_idx;
                }
                *self = Container::Bitmap(bm);
            }
            Container::Bitmap(bm) if card <= ARRAY_TO_BITMAP_THRESHOLD => {
                // Convert to array
                let mut arr = Vec::with_capacity(card);
                for (word_idx, &word) in bm.iter().enumerate() {
                    let mut w = word;
                    while w != 0 {
                        let bit_idx = w.trailing_zeros();
                        arr.push((word_idx * 64 + bit_idx as usize) as u16);
                        w &= w - 1;
                    }
                }
                *self = Container::Array(arr);
            }
            _ => {}
        }

        // Try run-length encoding if beneficial
        self.try_run_encode();
    }

    fn try_run_encode(&mut self) {
        let arr = match self {
            Container::Array(arr) if arr.len() >= 4 => arr,
            _ => return,
        };

        let mut runs = Vec::new();
        let mut i = 0;

        while i < arr.len() {
            let start = arr[i];
            let mut end = start;

            while i + 1 < arr.len() && arr[i + 1] == end + 1 {
                end = arr[i + 1];
                i += 1;
            }

            runs.push((start, end - start));
            i += 1;
        }

        // Only use runs if it saves space
        // Array: 2 bytes per element
        // Runs: 4 bytes per run
        if runs.len() * 4 < arr.len() * 2 {
            *self = Container::Runs(runs);
        }
    }

    fn serialize<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        match self {
            Container::Array(arr) => {
                writer.write_u8(0)?; // Type tag
                writer.write_u16::<LittleEndian>(arr.len() as u16)?;
                for &val in arr {
                    writer.write_u16::<LittleEndian>(val)?;
                }
            }
            Container::Bitmap(bm) => {
                writer.write_u8(1)?; // Type tag
                for &word in bm.iter() {
                    writer.write_u64::<LittleEndian>(word)?;
                }
            }
            Container::Runs(runs) => {
                writer.write_u8(2)?; // Type tag
                writer.write_u16::<LittleEndian>(runs.len() as u16)?;
                for &(start, len) in runs {
                    writer.write_u16::<LittleEndian>(start)?;
                    writer.write_u16::<LittleEndian>(len)?;
                }
            }
        }
        Ok(())
    }

    fn deserialize<R: Read>(reader: &mut R) -> io::Result<Self> {
        let tag = reader.read_u8()?;
        match tag {
            0 => {
                let len = reader.read_u16::<LittleEndian>()? as usize;
                let mut arr = Vec::with_capacity(len);
                for _ in 0..len {
                    arr.push(reader.read_u16::<LittleEndian>()?);
                }
                Ok(Container::Array(arr))
            }
            1 => {
                let mut bm = Box::new([0u64; 1024]);
                for word in bm.iter_mut() {
                    *word = reader.read_u64::<LittleEndian>()?;
                }
                Ok(Container::Bitmap(bm))
            }
            2 => {
                let len = reader.read_u16::<LittleEndian>()? as usize;
                let mut runs = Vec::with_capacity(len);
                for _ in 0..len {
                    let start = reader.read_u16::<LittleEndian>()?;
                    let run_len = reader.read_u16::<LittleEndian>()?;
                    runs.push((start, run_len));
                }
                Ok(Container::Runs(runs))
            }
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid container type",
            )),
        }
    }

    fn size_bytes(&self) -> usize {
        match self {
            Container::Array(arr) => arr.len() * 2 + 4,
            Container::Bitmap(_) => 8 * 1024 + 1,
            Container::Runs(runs) => runs.len() * 4 + 4,
        }
    }
}

/// Roaring Bitmap for compressed integer sets
#[derive(Debug, Clone)]
pub struct RoaringBitmap {
    /// High 16 bits -> container mapping
    containers: Vec<(u16, Container)>,
}

impl RoaringBitmap {
    /// Create empty bitmap
    pub fn new() -> Self {
        Self {
            containers: Vec::new(),
        }
    }

    /// Create from sorted slice
    pub fn from_sorted_slice(values: &[u32]) -> Self {
        let mut bitmap = Self::new();

        if values.is_empty() {
            return bitmap;
        }

        let mut current_high = (values[0] >> 16) as u16;
        let mut current_container = Container::new_array();

        for &val in values {
            let high = (val >> 16) as u16;
            let low = val as u16;

            if high != current_high {
                current_container.optimize();
                bitmap.containers.push((current_high, current_container));
                current_high = high;
                current_container = Container::new_array();
            }

            current_container.insert(low);
        }

        current_container.optimize();
        bitmap.containers.push((current_high, current_container));

        bitmap
    }

    /// Insert a value
    pub fn insert(&mut self, val: u32) -> bool {
        let high = (val >> 16) as u16;
        let low = val as u16;

        // Find or create container
        match self.containers.binary_search_by_key(&high, |&(h, _)| h) {
            Ok(idx) => self.containers[idx].1.insert(low),
            Err(idx) => {
                let mut container = Container::new_array();
                container.insert(low);
                self.containers.insert(idx, (high, container));
                true
            }
        }
    }

    /// Check if value exists
    pub fn contains(&self, val: u32) -> bool {
        let high = (val >> 16) as u16;
        let low = val as u16;

        match self.containers.binary_search_by_key(&high, |&(h, _)| h) {
            Ok(idx) => self.containers[idx].1.contains(low),
            Err(_) => false,
        }
    }

    /// Number of elements
    pub fn cardinality(&self) -> u32 {
        self.containers.iter().map(|(_, c)| c.cardinality()).sum()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.containers.is_empty()
    }

    /// Optimize all containers
    pub fn optimize(&mut self) {
        for (_, container) in &mut self.containers {
            container.optimize();
        }
    }

    /// Intersection with another bitmap
    pub fn and(&self, other: &RoaringBitmap) -> RoaringBitmap {
        let mut result = RoaringBitmap::new();

        let mut i = 0;
        let mut j = 0;

        while i < self.containers.len() && j < other.containers.len() {
            let (high1, c1) = &self.containers[i];
            let (high2, c2) = &other.containers[j];

            match high1.cmp(high2) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    let intersected = Self::intersect_containers(c1, c2);
                    if intersected.cardinality() > 0 {
                        result.containers.push((*high1, intersected));
                    }
                    i += 1;
                    j += 1;
                }
            }
        }

        result
    }

    fn intersect_containers(c1: &Container, c2: &Container) -> Container {
        match (c1, c2) {
            (Container::Array(a1), Container::Array(a2)) => {
                let mut result = Vec::new();
                let mut i = 0;
                let mut j = 0;
                while i < a1.len() && j < a2.len() {
                    match a1[i].cmp(&a2[j]) {
                        std::cmp::Ordering::Less => i += 1,
                        std::cmp::Ordering::Greater => j += 1,
                        std::cmp::Ordering::Equal => {
                            result.push(a1[i]);
                            i += 1;
                            j += 1;
                        }
                    }
                }
                Container::Array(result)
            }
            (Container::Bitmap(b1), Container::Bitmap(b2)) => {
                let mut result = Box::new([0u64; 1024]);
                for i in 0..1024 {
                    result[i] = b1[i] & b2[i];
                }
                let mut c = Container::Bitmap(result);
                c.optimize();
                c
            }
            (Container::Array(arr), Container::Bitmap(bm))
            | (Container::Bitmap(bm), Container::Array(arr)) => {
                let mut result = Vec::new();
                for &val in arr {
                    let word_idx = (val / 64) as usize;
                    let bit_idx = val % 64;
                    if (bm[word_idx] >> bit_idx) & 1 == 1 {
                        result.push(val);
                    }
                }
                Container::Array(result)
            }
            _ => {
                // For runs, convert to array first
                let arr1 = Self::container_to_array(c1);
                let arr2 = Self::container_to_array(c2);
                Self::intersect_containers(&Container::Array(arr1), &Container::Array(arr2))
            }
        }
    }

    fn container_to_array(c: &Container) -> Vec<u16> {
        match c {
            Container::Array(arr) => arr.clone(),
            Container::Bitmap(bm) => {
                let mut arr = Vec::new();
                for (word_idx, &word) in bm.iter().enumerate() {
                    let mut w = word;
                    while w != 0 {
                        let bit_idx = w.trailing_zeros();
                        arr.push((word_idx * 64 + bit_idx as usize) as u16);
                        w &= w - 1;
                    }
                }
                arr
            }
            Container::Runs(runs) => {
                let mut arr = Vec::new();
                for &(start, len) in runs {
                    for i in 0..=len {
                        arr.push(start + i);
                    }
                }
                arr
            }
        }
    }

    /// Union with another bitmap
    pub fn or(&self, other: &RoaringBitmap) -> RoaringBitmap {
        let mut result = RoaringBitmap::new();

        let mut i = 0;
        let mut j = 0;

        while i < self.containers.len() || j < other.containers.len() {
            if i >= self.containers.len() {
                result.containers.push(other.containers[j].clone());
                j += 1;
            } else if j >= other.containers.len() {
                result.containers.push(self.containers[i].clone());
                i += 1;
            } else {
                let (high1, c1) = &self.containers[i];
                let (high2, c2) = &other.containers[j];

                match high1.cmp(high2) {
                    std::cmp::Ordering::Less => {
                        result.containers.push(self.containers[i].clone());
                        i += 1;
                    }
                    std::cmp::Ordering::Greater => {
                        result.containers.push(other.containers[j].clone());
                        j += 1;
                    }
                    std::cmp::Ordering::Equal => {
                        let united = Self::union_containers(c1, c2);
                        result.containers.push((*high1, united));
                        i += 1;
                        j += 1;
                    }
                }
            }
        }

        result
    }

    fn union_containers(c1: &Container, c2: &Container) -> Container {
        match (c1, c2) {
            (Container::Bitmap(b1), Container::Bitmap(b2)) => {
                let mut result = Box::new([0u64; 1024]);
                for i in 0..1024 {
                    result[i] = b1[i] | b2[i];
                }
                Container::Bitmap(result)
            }
            _ => {
                // Convert both to arrays and merge
                let arr1 = Self::container_to_array(c1);
                let arr2 = Self::container_to_array(c2);

                let mut result = Vec::with_capacity(arr1.len() + arr2.len());
                let mut i = 0;
                let mut j = 0;

                while i < arr1.len() && j < arr2.len() {
                    match arr1[i].cmp(&arr2[j]) {
                        std::cmp::Ordering::Less => {
                            result.push(arr1[i]);
                            i += 1;
                        }
                        std::cmp::Ordering::Greater => {
                            result.push(arr2[j]);
                            j += 1;
                        }
                        std::cmp::Ordering::Equal => {
                            result.push(arr1[i]);
                            i += 1;
                            j += 1;
                        }
                    }
                }

                result.extend_from_slice(&arr1[i..]);
                result.extend_from_slice(&arr2[j..]);

                let mut c = Container::Array(result);
                c.optimize();
                c
            }
        }
    }

    /// Serialize
    pub fn serialize<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_u32::<LittleEndian>(self.containers.len() as u32)?;
        for (high, container) in &self.containers {
            writer.write_u16::<LittleEndian>(*high)?;
            container.serialize(writer)?;
        }
        Ok(())
    }

    /// Deserialize
    pub fn deserialize<R: Read>(reader: &mut R) -> io::Result<Self> {
        let num_containers = reader.read_u32::<LittleEndian>()? as usize;
        let mut containers = Vec::with_capacity(num_containers);

        for _ in 0..num_containers {
            let high = reader.read_u16::<LittleEndian>()?;
            let container = Container::deserialize(reader)?;
            containers.push((high, container));
        }

        Ok(Self { containers })
    }

    /// Approximate size in bytes
    pub fn size_bytes(&self) -> usize {
        4 + self
            .containers
            .iter()
            .map(|(_, c)| 2 + c.size_bytes())
            .sum::<usize>()
    }

    /// Create an iterator
    pub fn iter(&self) -> RoaringIterator<'_> {
        RoaringIterator {
            bitmap: self,
            container_idx: 0,
            value_idx: 0,
            current_values: Vec::new(),
        }
    }
}

impl Default for RoaringBitmap {
    fn default() -> Self {
        Self::new()
    }
}

/// Iterator over Roaring Bitmap
pub struct RoaringIterator<'a> {
    bitmap: &'a RoaringBitmap,
    container_idx: usize,
    value_idx: usize,
    current_values: Vec<u16>,
}

impl<'a> Iterator for RoaringIterator<'a> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.value_idx < self.current_values.len() {
                let high = self.bitmap.containers[self.container_idx - 1].0 as u32;
                let low = self.current_values[self.value_idx] as u32;
                self.value_idx += 1;
                return Some((high << 16) | low);
            }

            if self.container_idx >= self.bitmap.containers.len() {
                return None;
            }

            let (_, container) = &self.bitmap.containers[self.container_idx];
            self.current_values = RoaringBitmap::container_to_array(container);
            self.value_idx = 0;
            self.container_idx += 1;
        }
    }
}

/// Block size for Roaring BlockMax (matches container size = 65536 doc_ids)
pub const ROARING_BLOCK_SIZE: usize = 65536;

/// Block metadata for BlockMax WAND optimization in Roaring
#[derive(Debug, Clone)]
pub struct RoaringBlockInfo {
    /// High 16 bits (container key)
    pub container_key: u16,
    /// First doc_id in this container
    pub first_doc_id: u32,
    /// Last doc_id in this container
    pub last_doc_id: u32,
    /// Maximum term frequency in this block
    pub max_tf: u32,
    /// Upper bound BM25 score for this block
    pub max_block_score: f32,
    /// Number of documents in this container
    pub num_docs: u32,
}

impl RoaringBlockInfo {
    /// Serialize block info
    pub fn serialize<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_u16::<LittleEndian>(self.container_key)?;
        writer.write_u32::<LittleEndian>(self.first_doc_id)?;
        writer.write_u32::<LittleEndian>(self.last_doc_id)?;
        writer.write_u32::<LittleEndian>(self.max_tf)?;
        writer.write_f32::<LittleEndian>(self.max_block_score)?;
        writer.write_u32::<LittleEndian>(self.num_docs)?;
        Ok(())
    }

    /// Deserialize block info
    pub fn deserialize<R: Read>(reader: &mut R) -> io::Result<Self> {
        Ok(Self {
            container_key: reader.read_u16::<LittleEndian>()?,
            first_doc_id: reader.read_u32::<LittleEndian>()?,
            last_doc_id: reader.read_u32::<LittleEndian>()?,
            max_tf: reader.read_u32::<LittleEndian>()?,
            max_block_score: reader.read_f32::<LittleEndian>()?,
            num_docs: reader.read_u32::<LittleEndian>()?,
        })
    }
}

/// Roaring bitmap with term frequencies for posting lists
#[derive(Debug, Clone)]
pub struct RoaringPostingList {
    /// Document IDs as roaring bitmap
    pub doc_ids: RoaringBitmap,
    /// Term frequencies (sparse map for non-1 frequencies)
    /// Most terms have tf=1, so we only store exceptions
    pub term_freqs: Vec<(u32, u32)>,
    /// Default term frequency (usually 1)
    pub default_tf: u32,
    /// Maximum term frequency
    pub max_tf: u32,
    /// Block metadata for BlockMax WAND (one per container)
    pub blocks: Vec<RoaringBlockInfo>,
    /// Global maximum score across all blocks
    pub max_score: f32,
}

impl RoaringPostingList {
    /// BM25 parameters for block-max score calculation
    const K1: f32 = 1.2;
    const B: f32 = 0.75;

    /// Compute BM25 upper bound score for a given max_tf and IDF
    #[inline]
    pub fn compute_bm25_upper_bound(max_tf: u32, idf: f32) -> f32 {
        let tf = max_tf as f32;
        // Conservative upper bound: assume dl=0, so length_norm = 1 - b = 0.25
        let min_length_norm = 1.0 - Self::B;
        let tf_norm = (tf * (Self::K1 + 1.0)) / (tf + Self::K1 * min_length_norm);
        idf * tf_norm
    }

    /// Create from doc_ids and term frequencies (without IDF - for backwards compatibility)
    pub fn from_postings(doc_ids: &[u32], term_freqs: &[u32]) -> Self {
        Self::from_postings_with_idf(doc_ids, term_freqs, 1.0)
    }

    /// Create from doc_ids and term frequencies with IDF for block-max scores
    pub fn from_postings_with_idf(doc_ids: &[u32], term_freqs: &[u32], idf: f32) -> Self {
        assert_eq!(doc_ids.len(), term_freqs.len());

        if doc_ids.is_empty() {
            return Self {
                doc_ids: RoaringBitmap::new(),
                term_freqs: Vec::new(),
                default_tf: 1,
                max_tf: 0,
                blocks: Vec::new(),
                max_score: 0.0,
            };
        }

        let bitmap = RoaringBitmap::from_sorted_slice(doc_ids);

        // Find most common TF (usually 1)
        let mut tf_counts = std::collections::HashMap::new();
        for &tf in term_freqs {
            *tf_counts.entry(tf).or_insert(0u32) += 1;
        }
        let default_tf = tf_counts
            .iter()
            .max_by_key(|&(_, count)| count)
            .map(|(&tf, _)| tf)
            .unwrap_or(1);

        // Store only non-default TFs
        let exceptions: Vec<(u32, u32)> = doc_ids
            .iter()
            .zip(term_freqs.iter())
            .filter(|&(_, &tf)| tf != default_tf)
            .map(|(&doc, &tf)| (doc, tf))
            .collect();

        let max_tf = *term_freqs.iter().max().unwrap_or(&1);

        // Build block metadata (one block per container)
        // Group doc_ids by container (high 16 bits)
        let mut blocks = Vec::new();
        let mut max_score = 0.0f32;
        let mut i = 0;

        while i < doc_ids.len() {
            let container_key = (doc_ids[i] >> 16) as u16;
            let block_start = i;

            // Find all docs in this container
            while i < doc_ids.len() && (doc_ids[i] >> 16) as u16 == container_key {
                i += 1;
            }

            let block_doc_ids = &doc_ids[block_start..i];
            let block_tfs = &term_freqs[block_start..i];
            let block_max_tf = *block_tfs.iter().max().unwrap_or(&1);
            let block_score = Self::compute_bm25_upper_bound(block_max_tf, idf);
            max_score = max_score.max(block_score);

            blocks.push(RoaringBlockInfo {
                container_key,
                first_doc_id: block_doc_ids[0],
                last_doc_id: *block_doc_ids.last().unwrap(),
                max_tf: block_max_tf,
                max_block_score: block_score,
                num_docs: block_doc_ids.len() as u32,
            });
        }

        Self {
            doc_ids: bitmap,
            term_freqs: exceptions,
            default_tf,
            max_tf,
            blocks,
            max_score,
        }
    }

    /// Get term frequency for a document
    pub fn get_tf(&self, doc_id: u32) -> u32 {
        // Binary search in exceptions
        match self.term_freqs.binary_search_by_key(&doc_id, |&(d, _)| d) {
            Ok(idx) => self.term_freqs[idx].1,
            Err(_) => self.default_tf,
        }
    }

    /// Number of documents
    pub fn len(&self) -> u32 {
        self.doc_ids.cardinality()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.doc_ids.is_empty()
    }

    /// Get number of blocks (containers)
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Get block index for a doc_id
    pub fn block_for_doc(&self, doc_id: u32) -> Option<usize> {
        let container_key = (doc_id >> 16) as u16;
        self.blocks
            .binary_search_by_key(&container_key, |b| b.container_key)
            .ok()
    }

    /// Serialize
    pub fn serialize<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        self.doc_ids.serialize(writer)?;
        writer.write_u32::<LittleEndian>(self.default_tf)?;
        writer.write_u32::<LittleEndian>(self.max_tf)?;
        writer.write_f32::<LittleEndian>(self.max_score)?;
        writer.write_u32::<LittleEndian>(self.term_freqs.len() as u32)?;
        for &(doc, tf) in &self.term_freqs {
            writer.write_u32::<LittleEndian>(doc)?;
            writer.write_u32::<LittleEndian>(tf)?;
        }

        // Write block metadata
        writer.write_u32::<LittleEndian>(self.blocks.len() as u32)?;
        for block in &self.blocks {
            block.serialize(writer)?;
        }

        Ok(())
    }

    /// Deserialize
    pub fn deserialize<R: Read>(reader: &mut R) -> io::Result<Self> {
        let doc_ids = RoaringBitmap::deserialize(reader)?;
        let default_tf = reader.read_u32::<LittleEndian>()?;
        let max_tf = reader.read_u32::<LittleEndian>()?;
        let max_score = reader.read_f32::<LittleEndian>()?;
        let num_exceptions = reader.read_u32::<LittleEndian>()? as usize;
        let mut term_freqs = Vec::with_capacity(num_exceptions);
        for _ in 0..num_exceptions {
            let doc = reader.read_u32::<LittleEndian>()?;
            let tf = reader.read_u32::<LittleEndian>()?;
            term_freqs.push((doc, tf));
        }

        // Read block metadata
        let num_blocks = reader.read_u32::<LittleEndian>()? as usize;
        let mut blocks = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            blocks.push(RoaringBlockInfo::deserialize(reader)?);
        }

        Ok(Self {
            doc_ids,
            term_freqs,
            default_tf,
            max_tf,
            blocks,
            max_score,
        })
    }

    /// Create iterator
    pub fn iterator(&self) -> RoaringPostingIterator<'_> {
        RoaringPostingIterator {
            list: self,
            doc_iter: self.doc_ids.iter(),
            current_doc: None,
            current_block: 0,
        }
    }
}

/// Iterator over Roaring posting list with BlockMax support
pub struct RoaringPostingIterator<'a> {
    list: &'a RoaringPostingList,
    doc_iter: RoaringIterator<'a>,
    current_doc: Option<u32>,
    current_block: usize,
}

impl<'a> RoaringPostingIterator<'a> {
    /// Current document ID
    pub fn doc(&self) -> u32 {
        self.current_doc.unwrap_or(u32::MAX)
    }

    /// Current term frequency
    pub fn term_freq(&self) -> u32 {
        match self.current_doc {
            Some(doc) => self.list.get_tf(doc),
            None => 0,
        }
    }

    /// Advance to next document
    pub fn advance(&mut self) -> u32 {
        self.current_doc = self.doc_iter.next();
        // Update current block if needed
        if let Some(doc) = self.current_doc
            && !self.list.blocks.is_empty()
        {
            let container_key = (doc >> 16) as u16;
            // Move to next block if we've passed current one
            while self.current_block < self.list.blocks.len()
                && self.list.blocks[self.current_block].container_key < container_key
            {
                self.current_block += 1;
            }
        }
        self.doc()
    }

    /// Initialize (must be called before first use)
    pub fn init(&mut self) {
        self.current_doc = self.doc_iter.next();
        self.current_block = 0;
    }

    /// Seek to first doc >= target
    pub fn seek(&mut self, target: u32) -> u32 {
        // Use block skip list for faster seeking
        if !self.list.blocks.is_empty() {
            let target_container = (target >> 16) as u16;

            // Skip blocks that are entirely before target
            while self.current_block < self.list.blocks.len()
                && self.list.blocks[self.current_block].last_doc_id < target
            {
                self.current_block += 1;
            }

            // If we've exhausted all blocks
            if self.current_block >= self.list.blocks.len() {
                self.current_doc = None;
                return u32::MAX;
            }

            // Skip docs until we reach the target block's first doc
            let block = &self.list.blocks[self.current_block];
            if block.container_key > target_container
                || (block.container_key == target_container && block.first_doc_id > self.doc())
            {
                // Need to advance iterator to this block
                while let Some(doc) = self.current_doc {
                    if doc >= block.first_doc_id {
                        break;
                    }
                    self.current_doc = self.doc_iter.next();
                }
            }
        }

        // Linear scan within block
        while let Some(doc) = self.current_doc {
            if doc >= target {
                return doc;
            }
            self.current_doc = self.doc_iter.next();
        }
        u32::MAX
    }

    /// Check if iterator is exhausted
    pub fn is_exhausted(&self) -> bool {
        self.current_doc.is_none()
    }

    /// Get current block's max score (for BlockMax WAND)
    pub fn current_block_max_score(&self) -> f32 {
        if self.current_doc.is_none() || self.list.blocks.is_empty() {
            return 0.0;
        }
        if self.current_block < self.list.blocks.len() {
            self.list.blocks[self.current_block].max_block_score
        } else {
            0.0
        }
    }

    /// Get current block's max term frequency
    pub fn current_block_max_tf(&self) -> u32 {
        if self.current_doc.is_none() || self.list.blocks.is_empty() {
            return 0;
        }
        if self.current_block < self.list.blocks.len() {
            self.list.blocks[self.current_block].max_tf
        } else {
            0
        }
    }

    /// Get max score for remaining blocks (for MaxScore optimization)
    pub fn max_remaining_score(&self) -> f32 {
        if self.current_doc.is_none() || self.list.blocks.is_empty() {
            return 0.0;
        }
        self.list.blocks[self.current_block..]
            .iter()
            .map(|b| b.max_block_score)
            .fold(0.0f32, |a, b| a.max(b))
    }

    /// Skip to next block containing doc >= target (for BlockWAND)
    /// Returns (first_doc_in_block, block_max_score) or None if exhausted
    pub fn skip_to_block_with_doc(&mut self, target: u32) -> Option<(u32, f32)> {
        if self.list.blocks.is_empty() {
            return None;
        }

        while self.current_block < self.list.blocks.len() {
            let block = &self.list.blocks[self.current_block];
            if block.last_doc_id >= target {
                // Advance iterator to this block's first doc
                while let Some(doc) = self.current_doc {
                    if doc >= block.first_doc_id {
                        break;
                    }
                    self.current_doc = self.doc_iter.next();
                }
                return Some((block.first_doc_id, block.max_block_score));
            }
            self.current_block += 1;
        }

        self.current_doc = None;
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roaring_basic() {
        let mut bitmap = RoaringBitmap::new();
        bitmap.insert(1);
        bitmap.insert(100);
        bitmap.insert(1000);
        bitmap.insert(100000);

        assert!(bitmap.contains(1));
        assert!(bitmap.contains(100));
        assert!(bitmap.contains(1000));
        assert!(bitmap.contains(100000));
        assert!(!bitmap.contains(2));
        assert!(!bitmap.contains(50000));

        assert_eq!(bitmap.cardinality(), 4);
    }

    #[test]
    fn test_roaring_from_sorted() {
        let values: Vec<u32> = (0..10000).map(|i| i * 3).collect();
        let bitmap = RoaringBitmap::from_sorted_slice(&values);

        assert_eq!(bitmap.cardinality(), 10000);

        for &val in &values {
            assert!(bitmap.contains(val), "Missing value {}", val);
        }
    }

    #[test]
    fn test_roaring_intersection() {
        let a = RoaringBitmap::from_sorted_slice(&[1, 2, 3, 100, 200, 300]);
        let b = RoaringBitmap::from_sorted_slice(&[2, 3, 4, 200, 300, 400]);

        let result = a.and(&b);

        assert_eq!(result.cardinality(), 4);
        assert!(result.contains(2));
        assert!(result.contains(3));
        assert!(result.contains(200));
        assert!(result.contains(300));
    }

    #[test]
    fn test_roaring_union() {
        let a = RoaringBitmap::from_sorted_slice(&[1, 2, 3]);
        let b = RoaringBitmap::from_sorted_slice(&[3, 4, 5]);

        let result = a.or(&b);

        assert_eq!(result.cardinality(), 5);
        for i in 1..=5 {
            assert!(result.contains(i));
        }
    }

    #[test]
    fn test_roaring_serialization() {
        let values: Vec<u32> = (0..1000).map(|i| i * 7).collect();
        let bitmap = RoaringBitmap::from_sorted_slice(&values);

        let mut buffer = Vec::new();
        bitmap.serialize(&mut buffer).unwrap();

        let restored = RoaringBitmap::deserialize(&mut &buffer[..]).unwrap();

        assert_eq!(restored.cardinality(), bitmap.cardinality());
        for &val in &values {
            assert!(restored.contains(val));
        }
    }

    #[test]
    fn test_roaring_posting_list() {
        let doc_ids: Vec<u32> = vec![1, 5, 10, 50, 100, 500, 1000];
        let term_freqs: Vec<u32> = vec![1, 1, 2, 1, 3, 1, 1];

        let list = RoaringPostingList::from_postings(&doc_ids, &term_freqs);

        assert_eq!(list.len(), 7);
        assert_eq!(list.default_tf, 1);
        assert_eq!(list.max_tf, 3);

        // Check TF lookups
        assert_eq!(list.get_tf(1), 1);
        assert_eq!(list.get_tf(10), 2);
        assert_eq!(list.get_tf(100), 3);
    }

    #[test]
    fn test_roaring_iterator() {
        let values: Vec<u32> = vec![1, 10, 100, 1000, 10000];
        let bitmap = RoaringBitmap::from_sorted_slice(&values);

        let collected: Vec<u32> = bitmap.iter().collect();
        assert_eq!(collected, values);
    }

    #[test]
    fn test_roaring_block_max() {
        // Create posting list spanning multiple containers (high 16 bits)
        // Container 0: doc_ids 0-65535
        // Container 1: doc_ids 65536-131071
        // Container 2: doc_ids 131072-196607
        let mut doc_ids = Vec::new();
        let mut term_freqs = Vec::new();

        // Container 0: 100 docs with max_tf = 2
        for i in 0..100 {
            doc_ids.push(i * 100);
            term_freqs.push(if i == 50 { 2 } else { 1 });
        }

        // Container 1: 100 docs with max_tf = 5
        for i in 0..100 {
            doc_ids.push(65536 + i * 100);
            term_freqs.push(if i == 25 { 5 } else { 1 });
        }

        // Container 2: 100 docs with max_tf = 3
        for i in 0..100 {
            doc_ids.push(131072 + i * 100);
            term_freqs.push(if i == 75 { 3 } else { 1 });
        }

        let list = RoaringPostingList::from_postings_with_idf(&doc_ids, &term_freqs, 2.0);

        // Should have 3 blocks (one per container)
        assert_eq!(list.num_blocks(), 3);
        assert_eq!(list.blocks[0].container_key, 0);
        assert_eq!(list.blocks[1].container_key, 1);
        assert_eq!(list.blocks[2].container_key, 2);

        assert_eq!(list.blocks[0].max_tf, 2);
        assert_eq!(list.blocks[1].max_tf, 5);
        assert_eq!(list.blocks[2].max_tf, 3);

        // Block 1 should have highest score (max_tf = 5)
        assert!(list.blocks[1].max_block_score > list.blocks[0].max_block_score);
        assert!(list.blocks[1].max_block_score > list.blocks[2].max_block_score);

        // Global max_score should equal block 1's score
        assert_eq!(list.max_score, list.blocks[1].max_block_score);

        // Test iterator block-max methods
        let mut iter = list.iterator();
        iter.init();
        assert_eq!(iter.current_block_max_tf(), 2); // Block 0

        // Seek to block 1
        iter.seek(65536);
        assert_eq!(iter.current_block_max_tf(), 5);

        // Seek to block 2
        iter.seek(131072);
        assert_eq!(iter.current_block_max_tf(), 3);
    }

    #[test]
    fn test_roaring_block_max_serialization() {
        let mut doc_ids = Vec::new();
        let mut term_freqs = Vec::new();

        // Two containers
        for i in 0..50 {
            doc_ids.push(i * 10);
            term_freqs.push((i % 5) + 1);
        }
        for i in 0..50 {
            doc_ids.push(65536 + i * 10);
            term_freqs.push((i % 3) + 1);
        }

        let list = RoaringPostingList::from_postings_with_idf(&doc_ids, &term_freqs, 1.5);

        let mut buffer = Vec::new();
        list.serialize(&mut buffer).unwrap();

        let restored = RoaringPostingList::deserialize(&mut &buffer[..]).unwrap();

        assert_eq!(restored.len(), list.len());
        assert_eq!(restored.max_tf, list.max_tf);
        assert_eq!(restored.max_score, list.max_score);
        assert_eq!(restored.num_blocks(), list.num_blocks());

        // Verify block metadata
        for (orig, rest) in list.blocks.iter().zip(restored.blocks.iter()) {
            assert_eq!(orig.container_key, rest.container_key);
            assert_eq!(orig.first_doc_id, rest.first_doc_id);
            assert_eq!(orig.last_doc_id, rest.last_doc_id);
            assert_eq!(orig.max_tf, rest.max_tf);
            assert_eq!(orig.max_block_score, rest.max_block_score);
        }

        // Verify iteration produces same results
        let mut iter1 = list.iterator();
        let mut iter2 = restored.iterator();
        iter1.init();
        iter2.init();

        while !iter1.is_exhausted() {
            assert_eq!(iter1.doc(), iter2.doc());
            assert_eq!(iter1.term_freq(), iter2.term_freq());
            iter1.advance();
            iter2.advance();
        }
        assert!(iter2.is_exhausted());
    }
}
