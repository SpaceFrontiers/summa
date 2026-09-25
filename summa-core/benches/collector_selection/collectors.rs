//! Experimental collectors live only in this benchmark; production stays unchanged.

use summa_core::query::{HeapEntry, ScoreCollector};

pub type Results = Vec<(u32, f32, u16)>;

pub trait Collect: Sized {
    const NAME: &'static str;
    fn new(k: usize) -> Self;
    fn threshold(&self) -> f32;
    fn add(&mut self, entry: HeapEntry);
    fn finish(self) -> Results;
    /// Retained backing allocations, excluding the collector object and output.
    fn backing_bytes(k: usize) -> usize;
}

impl Collect for ScoreCollector {
    const NAME: &'static str = "heap";
    fn new(k: usize) -> Self {
        Self::new(k)
    }
    fn threshold(&self) -> f32 {
        self.threshold()
    }
    fn add(&mut self, entry: HeapEntry) {
        self.insert_with_ordinal(entry.doc_id, entry.score, entry.ordinal);
    }
    fn finish(self) -> Results {
        self.into_sorted_results()
    }
    fn backing_bytes(k: usize) -> usize {
        // All measured k <= the production initial capacity cap (8192).
        k * size_of::<HeapEntry>()
    }
}

fn sorted(mut entries: Vec<HeapEntry>) -> Results {
    entries.sort_unstable();
    entries
        .into_iter()
        .map(|entry| (entry.doc_id, entry.score, entry.ordinal))
        .collect()
}

/// Retain the best k after every k newly admitted candidates. The cached cutoff
/// is conservative between partitions, so pruning may do additional work.
pub struct Partition {
    entries: Vec<HeapEntry>,
    k: usize,
    cutoff: Option<HeapEntry>,
}

impl Partition {
    fn partition(&mut self) {
        let (_, worst, _) = self.entries.select_nth_unstable(self.k - 1);
        self.cutoff = Some(*worst);
        self.entries.truncate(self.k);
    }
}

impl Collect for Partition {
    const NAME: &'static str = "partition";
    fn new(k: usize) -> Self {
        Self {
            entries: Vec::with_capacity(2 * k),
            k,
            cutoff: None,
        }
    }
    fn threshold(&self) -> f32 {
        self.cutoff.map_or(0.0, |entry| entry.score)
    }
    fn add(&mut self, entry: HeapEntry) {
        if self.k == 0
            || self
                .cutoff
                .is_some_and(|cutoff| entry.score < cutoff.score || entry >= cutoff)
        {
            return;
        }
        self.entries.push(entry);
        if self.entries.len() == self.k || self.entries.len() == 2 * self.k {
            self.partition();
        }
    }
    fn finish(mut self) -> Results {
        if self.entries.len() > self.k {
            self.partition();
        }
        sorted(self.entries)
    }
    fn backing_bytes(k: usize) -> usize {
        2 * k * size_of::<HeapEntry>()
    }
}

#[derive(Clone, Copy)]
struct Node {
    entry: HeapEntry,
    leaf: u32,
}

const EMPTY: Node = Node {
    entry: HeapEntry {
        doc_id: 0,
        score: 0.0,
        ordinal: 0,
    },
    leaf: u32::MAX,
};

/// A tournament whose winner is the worst retained hit. Internal nodes retain
/// the other competitor, so replacing the winner replays only its ancestor path.
/// Unlike the upstream score-only comparison, use Summa's complete tie order.
pub struct LoserTree {
    entries: Vec<HeapEntry>,
    tree: Vec<Node>,
    root: Node,
    k: usize,
}

impl LoserTree {
    fn build(&mut self) {
        for (leaf, &entry) in self.entries.iter().enumerate() {
            let mut current = Node {
                entry,
                leaf: leaf as u32,
            };
            let mut node = (self.k + leaf) >> 1;
            while node != 0 {
                let slot = &mut self.tree[node];
                if slot.leaf == u32::MAX {
                    *slot = current;
                    current.leaf = u32::MAX;
                    break;
                }
                if slot.entry > current.entry {
                    std::mem::swap(slot, &mut current);
                }
                node >>= 1;
            }
            if current.leaf != u32::MAX {
                self.root = current;
            }
        }
    }
}

impl Collect for LoserTree {
    const NAME: &'static str = "loser_tree";
    fn new(k: usize) -> Self {
        Self {
            entries: Vec::with_capacity(k),
            tree: vec![EMPTY; k],
            root: EMPTY,
            k,
        }
    }
    fn threshold(&self) -> f32 {
        if self.entries.len() == self.k && self.k != 0 {
            self.root.entry.score
        } else {
            0.0
        }
    }
    fn add(&mut self, entry: HeapEntry) {
        if self.k == 0 {
            return;
        }
        if self.entries.len() < self.k {
            self.entries.push(entry);
            if self.entries.len() == self.k {
                self.build();
            }
            return;
        }
        if entry.score < self.root.entry.score || entry >= self.root.entry {
            return;
        }
        let leaf = self.root.leaf;
        self.entries[leaf as usize] = entry;
        let mut current = Node { entry, leaf };
        let mut node = (self.k + leaf as usize) >> 1;
        while node != 0 {
            let loser = self.tree[node];
            let wins = loser.entry > current.entry;
            self.tree[node] = if wins { current } else { loser };
            current = if wins { loser } else { current };
            node >>= 1;
        }
        self.root = current;
    }
    fn finish(self) -> Results {
        sorted(self.entries)
    }
    fn backing_bytes(k: usize) -> usize {
        k * (size_of::<HeapEntry>() + size_of::<Node>())
    }
}
