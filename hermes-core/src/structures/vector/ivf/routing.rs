//! Metric-agnostic IVF routing primitives.
//!
//! Quantizers provide metric-specific centroid scores. This module owns the
//! topology-independent parts: flat/two-level policy, bounded beam sizing,
//! deterministic top selection, and the versioned probe plan shared by every
//! segment participating in one query.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::Arc;

use crate::dsl::IvfRoutingMode;
use rand::prelude::*;
use serde::{Deserialize, Serialize};

/// Automatic routing switches to a centroid index at this leaf count. Below
/// it, a SIMD-friendly flat pass is normally cheaper than another level of
/// indirection.
pub const HNSW_AUTO_THRESHOLD: usize = 4_096;

/// Extra leaf coverage requested from the parent level. A beam of four times
/// the minimum parent count avoids the recall cliff of greedy one-parent
/// hierarchical routing while keeping parent/leaf scoring sublinear.
const PARENT_BEAM_OVERSAMPLE: usize = 4;

const HNSW_M: usize = 32;
const HNSW_EF_CONSTRUCTION: usize = 200;
const HNSW_QUERY_OVERSAMPLE: usize = 4;
const HNSW_MIN_EF_SEARCH: usize = 128;

#[derive(Clone, Copy, Debug)]
struct GraphCandidate {
    node: u32,
    distance: f32,
}

impl PartialEq for GraphCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.node == other.node && self.distance.to_bits() == other.distance.to_bits()
    }
}

impl Eq for GraphCandidate {}

impl PartialOrd for GraphCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for GraphCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.node.cmp(&other.node))
    }
}

struct VisitedNodes {
    epochs: Vec<u32>,
    current: u32,
}

impl VisitedNodes {
    fn new(nodes: usize) -> Self {
        Self {
            epochs: vec![0; nodes],
            current: 0,
        }
    }

    fn reset(&mut self) {
        self.current = self.current.wrapping_add(1);
        if self.current == 0 {
            self.epochs.fill(0);
            self.current = 1;
        }
    }

    fn ensure_nodes(&mut self, nodes: usize) {
        if self.epochs.len() < nodes {
            self.epochs.resize(nodes, 0);
        }
    }

    fn insert(&mut self, node: u32) -> bool {
        let slot = &mut self.epochs[node as usize];
        if *slot == self.current {
            false
        } else {
            *slot = self.current;
            true
        }
    }
}

struct HnswQueryScratch {
    visited: VisitedNodes,
    candidates: BinaryHeap<Reverse<GraphCandidate>>,
    best: BinaryHeap<GraphCandidate>,
    ordered: Vec<GraphCandidate>,
}

impl HnswQueryScratch {
    fn new() -> Self {
        Self {
            visited: VisitedNodes::new(0),
            candidates: BinaryHeap::new(),
            best: BinaryHeap::new(),
            ordered: Vec::new(),
        }
    }
}

thread_local! {
    /// Segment construction routes millions of vectors through the same graph.
    /// Retaining scratch per worker avoids zeroing the visited bitmap and
    /// reallocating both heaps for every assignment.
    static HNSW_QUERY_SCRATCH: std::cell::RefCell<HnswQueryScratch> =
        std::cell::RefCell::new(HnswQueryScratch::new());
}

/// Compact, centroid-free HNSW topology. Node IDs are global leaf IDs, so the
/// graph shares the quantizer's existing centroid matrix rather than storing a
/// second copy of every vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswRoutingGraph {
    m: u16,
    ef_construction: u32,
    entry_point: u32,
    max_level: u8,
    node_levels: Vec<u8>,
    /// Per-node ranges into `level_offsets`; each node owns level_count + 1
    /// offsets so every adjacency is a direct pair of indexed loads.
    node_offsets: Vec<u32>,
    level_offsets: Vec<u32>,
    neighbors: Vec<u32>,
}

impl HnswRoutingGraph {
    pub fn build(node_count: usize, distance: impl Fn(u32, u32) -> f32, seed: u64) -> Self {
        assert!(node_count > 0 && node_count <= u32::MAX as usize);
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let level_multiplier = 1.0 / (HNSW_M as f64).ln();
        let node_levels: Vec<u8> = (0..node_count)
            .map(|_| {
                let uniform = rng.random::<f64>().clamp(f64::MIN_POSITIVE, 1.0);
                (-uniform.ln() * level_multiplier).floor().min(31.0) as u8
            })
            .collect();
        let mut insertion_order: Vec<u32> = (0..node_count as u32).collect();
        insertion_order.shuffle(&mut rng);
        let mut links: Vec<Vec<Vec<u32>>> = node_levels
            .iter()
            .map(|&level| vec![Vec::new(); level as usize + 1])
            .collect();
        let mut visited = VisitedNodes::new(node_count);
        let mut entry_point = insertion_order[0];
        let mut max_level = node_levels[entry_point as usize];

        for &node in insertion_order.iter().skip(1) {
            let node_level = node_levels[node as usize];
            let mut entry = entry_point;
            let node_distance = |candidate| distance(node, candidate);

            for level in ((node_level as usize + 1)..=max_level as usize).rev() {
                entry = greedy_search_level(&links, entry, level, &node_distance);
            }

            for level in (0..=usize::min(node_level as usize, max_level as usize)).rev() {
                let candidates = search_graph_layer(
                    &links,
                    entry,
                    level,
                    HNSW_EF_CONSTRUCTION,
                    &node_distance,
                    &mut visited,
                );
                if let Some(best) = candidates.first() {
                    entry = best.node;
                }
                let max_connections = if level == 0 { HNSW_M * 2 } else { HNSW_M };
                let selected =
                    select_diverse_neighbors(node, candidates, max_connections, &distance);
                links[node as usize][level] = selected.clone();
                for neighbor in selected {
                    let adjacency = &mut links[neighbor as usize][level];
                    if !adjacency.contains(&node) {
                        adjacency.push(node);
                    }
                    if adjacency.len() > max_connections {
                        let candidates = adjacency
                            .iter()
                            .copied()
                            .map(|candidate| GraphCandidate {
                                node: candidate,
                                distance: distance(neighbor, candidate),
                            })
                            .collect();
                        *adjacency = select_diverse_neighbors(
                            neighbor,
                            candidates,
                            max_connections,
                            &distance,
                        );
                    }
                }
            }

            if node_level > max_level {
                entry_point = node;
                max_level = node_level;
            }
        }

        Self::compact(
            HNSW_M,
            HNSW_EF_CONSTRUCTION,
            entry_point,
            max_level,
            node_levels,
            links,
        )
    }

    fn compact(
        m: usize,
        ef_construction: usize,
        entry_point: u32,
        max_level: u8,
        node_levels: Vec<u8>,
        links: Vec<Vec<Vec<u32>>>,
    ) -> Self {
        let mut node_offsets = Vec::with_capacity(links.len() + 1);
        let level_count: usize = links.iter().map(|levels| levels.len() + 1).sum();
        let neighbor_count: usize = links
            .iter()
            .flat_map(|levels| levels.iter())
            .map(Vec::len)
            .sum();
        let mut level_offsets = Vec::with_capacity(level_count);
        let mut neighbors = Vec::with_capacity(neighbor_count);
        for levels in links {
            node_offsets.push(level_offsets.len() as u32);
            for mut adjacency in levels {
                adjacency.sort_unstable();
                adjacency.dedup();
                level_offsets.push(neighbors.len() as u32);
                neighbors.extend(adjacency);
            }
            level_offsets.push(neighbors.len() as u32);
        }
        node_offsets.push(level_offsets.len() as u32);
        Self {
            m: m as u16,
            ef_construction: ef_construction as u32,
            entry_point,
            max_level,
            node_levels,
            node_offsets,
            level_offsets,
            neighbors,
        }
    }

    #[inline]
    pub fn neighbors(&self, node: u32, level: usize) -> &[u32] {
        if (self.node_levels[node as usize] as usize) < level {
            return &[];
        }
        let offset_index = self.node_offsets[node as usize] as usize + level;
        let start = self.level_offsets[offset_index] as usize;
        let end = self.level_offsets[offset_index + 1] as usize;
        &self.neighbors[start..end]
    }

    pub fn search(&self, query_distance: impl Fn(u32) -> f32, take: usize) -> Vec<u32> {
        let take = take.min(self.node_levels.len());
        if take == 0 {
            return Vec::new();
        }
        let mut entry = self.entry_point;
        for level in (1..=self.max_level as usize).rev() {
            entry = greedy_search_compact(self, entry, level, &query_distance);
        }
        let ef_search = take
            .saturating_mul(HNSW_QUERY_OVERSAMPLE)
            .max(HNSW_MIN_EF_SEARCH)
            .min(self.node_levels.len());
        HNSW_QUERY_SCRATCH.with(|scratch| {
            let mut scratch = scratch.borrow_mut();
            search_compact_layer_reusing(self, entry, ef_search, &query_distance, &mut scratch);
            scratch
                .ordered
                .iter()
                .take(take)
                .map(|candidate| candidate.node)
                .collect()
        })
    }

    pub fn search_one(&self, query_distance: impl Fn(u32) -> f32) -> u32 {
        let mut entry = self.entry_point;
        for level in (1..=self.max_level as usize).rev() {
            entry = greedy_search_compact(self, entry, level, &query_distance);
        }
        let ef_search = HNSW_MIN_EF_SEARCH.min(self.node_levels.len());
        HNSW_QUERY_SCRATCH.with(|scratch| {
            let mut scratch = scratch.borrow_mut();
            search_compact_layer_reusing(self, entry, ef_search, &query_distance, &mut scratch);
            scratch
                .ordered
                .first()
                .map_or(entry, |candidate| candidate.node)
        })
    }

    pub fn validate(&self, expected_nodes: usize) -> bool {
        if self.m as usize != HNSW_M
            || self.ef_construction as usize != HNSW_EF_CONSTRUCTION
            || expected_nodes == 0
            || self.node_levels.len() != expected_nodes
            || self.node_offsets.len() != expected_nodes + 1
            || self.node_offsets.first() != Some(&0)
            || self.node_offsets.last().copied() != Some(self.level_offsets.len() as u32)
            || self.node_offsets.windows(2).any(|pair| pair[0] > pair[1])
            || self
                .node_offsets
                .iter()
                .any(|&offset| offset as usize > self.level_offsets.len())
            || self.entry_point as usize >= expected_nodes
            || self.node_levels[self.entry_point as usize] != self.max_level
            || self.node_levels.iter().copied().max() != Some(self.max_level)
            || self.level_offsets.last().copied() != Some(self.neighbors.len() as u32)
            || self.level_offsets.windows(2).any(|pair| pair[0] > pair[1])
            || self
                .neighbors
                .iter()
                .any(|&node| node as usize >= expected_nodes)
        {
            return false;
        }
        for node in 0..expected_nodes {
            let start = self.node_offsets[node] as usize;
            let end = self.node_offsets[node + 1] as usize;
            if end.saturating_sub(start) != self.node_levels[node] as usize + 2 {
                return false;
            }
            for level in 0..=self.node_levels[node] as usize {
                let adjacency = self.neighbors(node as u32, level);
                let max_connections = if level == 0 { HNSW_M * 2 } else { HNSW_M };
                if adjacency.len() > max_connections
                    || adjacency.contains(&(node as u32))
                    || adjacency.windows(2).any(|pair| pair[0] >= pair[1])
                {
                    return false;
                }
            }
        }
        true
    }

    pub fn size_bytes(&self) -> usize {
        self.node_levels.len()
            + self.node_offsets.len() * size_of::<u32>()
            + self.level_offsets.len() * size_of::<u32>()
            + self.neighbors.len() * size_of::<u32>()
            + 32
    }

    /// Visit the compact, immutable arrays touched by every HNSW route.
    /// Query scratch is thread-local and intentionally excluded.
    pub(crate) fn visit_resident_regions(&self, visit: &mut dyn FnMut(&'static str, &[u8])) {
        visit("HNSW node levels", bytes_of_slice(&self.node_levels));
        visit("HNSW node offsets", bytes_of_slice(&self.node_offsets));
        visit("HNSW level offsets", bytes_of_slice(&self.level_offsets));
        visit("HNSW neighbors", bytes_of_slice(&self.neighbors));
    }
}

fn greedy_search_level(
    links: &[Vec<Vec<u32>>],
    mut current: u32,
    level: usize,
    query_distance: &impl Fn(u32) -> f32,
) -> u32 {
    let mut current_distance = query_distance(current);
    loop {
        let mut changed = false;
        for &candidate in &links[current as usize][level] {
            let distance = query_distance(candidate);
            if distance < current_distance || (distance == current_distance && candidate < current)
            {
                current = candidate;
                current_distance = distance;
                changed = true;
            }
        }
        if !changed {
            return current;
        }
    }
}

fn greedy_search_compact(
    graph: &HnswRoutingGraph,
    mut current: u32,
    level: usize,
    query_distance: &impl Fn(u32) -> f32,
) -> u32 {
    let mut current_distance = query_distance(current);
    loop {
        let mut changed = false;
        for &candidate in graph.neighbors(current, level) {
            let distance = query_distance(candidate);
            if distance < current_distance || (distance == current_distance && candidate < current)
            {
                current = candidate;
                current_distance = distance;
                changed = true;
            }
        }
        if !changed {
            return current;
        }
    }
}

fn search_graph_layer(
    links: &[Vec<Vec<u32>>],
    entry: u32,
    level: usize,
    ef: usize,
    query_distance: &impl Fn(u32) -> f32,
    visited: &mut VisitedNodes,
) -> Vec<GraphCandidate> {
    search_layer_impl(entry, ef, query_distance, visited, |node| {
        &links[node as usize][level]
    })
}

fn search_compact_layer_reusing(
    graph: &HnswRoutingGraph,
    entry: u32,
    ef: usize,
    query_distance: &impl Fn(u32) -> f32,
    scratch: &mut HnswQueryScratch,
) {
    scratch.visited.ensure_nodes(graph.node_levels.len());
    scratch.visited.reset();
    scratch.candidates.clear();
    scratch.best.clear();
    scratch.ordered.clear();
    scratch.visited.insert(entry);
    let first = GraphCandidate {
        node: entry,
        distance: query_distance(entry),
    };
    scratch.candidates.push(Reverse(first));
    scratch.best.push(first);

    while let Some(Reverse(current)) = scratch.candidates.pop() {
        if scratch.best.len() >= ef
            && scratch
                .best
                .peek()
                .is_some_and(|worst| current.distance > worst.distance)
        {
            break;
        }
        for &neighbor in graph.neighbors(current.node, 0) {
            if !scratch.visited.insert(neighbor) {
                continue;
            }
            let candidate = GraphCandidate {
                node: neighbor,
                distance: query_distance(neighbor),
            };
            if scratch.best.len() < ef
                || scratch.best.peek().is_some_and(|worst| candidate < *worst)
            {
                scratch.candidates.push(Reverse(candidate));
                scratch.best.push(candidate);
                if scratch.best.len() > ef {
                    scratch.best.pop();
                }
            }
        }
    }
    scratch.ordered.extend(scratch.best.drain());
    scratch.ordered.sort_unstable();
}

fn search_layer_impl<'a>(
    entry: u32,
    ef: usize,
    query_distance: &impl Fn(u32) -> f32,
    visited: &mut VisitedNodes,
    neighbors: impl Fn(u32) -> &'a [u32],
) -> Vec<GraphCandidate> {
    visited.reset();
    visited.insert(entry);
    let first = GraphCandidate {
        node: entry,
        distance: query_distance(entry),
    };
    let mut candidates = BinaryHeap::new();
    let mut best = BinaryHeap::new();
    candidates.push(Reverse(first));
    best.push(first);

    while let Some(Reverse(current)) = candidates.pop() {
        if best.len() >= ef
            && best
                .peek()
                .is_some_and(|worst| current.distance > worst.distance)
        {
            break;
        }
        for &neighbor in neighbors(current.node) {
            if !visited.insert(neighbor) {
                continue;
            }
            let candidate = GraphCandidate {
                node: neighbor,
                distance: query_distance(neighbor),
            };
            if best.len() < ef || best.peek().is_some_and(|worst| candidate < *worst) {
                candidates.push(Reverse(candidate));
                best.push(candidate);
                if best.len() > ef {
                    best.pop();
                }
            }
        }
    }
    best.into_sorted_vec()
}

fn select_diverse_neighbors(
    query_node: u32,
    mut candidates: Vec<GraphCandidate>,
    limit: usize,
    distance: &impl Fn(u32, u32) -> f32,
) -> Vec<u32> {
    candidates.sort_unstable();
    candidates.dedup_by_key(|candidate| candidate.node);
    let mut selected = Vec::with_capacity(limit);
    let mut deferred = Vec::new();
    for candidate in candidates {
        if candidate.node == query_node {
            continue;
        }
        if selected
            .iter()
            .all(|&neighbor| distance(candidate.node, neighbor) > candidate.distance)
        {
            selected.push(candidate.node);
            if selected.len() == limit {
                return selected;
            }
        } else {
            deferred.push(candidate.node);
        }
    }
    for candidate in deferred {
        if selected.len() == limit {
            break;
        }
        selected.push(candidate);
    }
    selected
}

/// Compact parent-to-leaf adjacency shared by float and binary quantizers.
/// Offsets avoid one heap allocation per parent and serialize as two flat
/// arrays in the single index-level quantizer artifact.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct IvfRoutingTopology {
    child_offsets: Vec<u32>,
    leaf_ids: Vec<u32>,
}

impl IvfRoutingTopology {
    pub fn from_children(children: &[Vec<u32>]) -> Self {
        let mut child_offsets = Vec::with_capacity(children.len() + 1);
        let mut leaf_ids = Vec::new();
        child_offsets.push(0);
        for child_list in children {
            leaf_ids.extend_from_slice(child_list);
            child_offsets.push(leaf_ids.len() as u32);
        }
        Self {
            child_offsets,
            leaf_ids,
        }
    }

    pub fn parent_count(&self) -> usize {
        self.child_offsets.len().saturating_sub(1)
    }

    pub fn children(&self, parent: usize) -> &[u32] {
        let start = self.child_offsets[parent] as usize;
        let end = self.child_offsets[parent + 1] as usize;
        &self.leaf_ids[start..end]
    }

    pub fn validate(&self, num_leaves: usize) -> bool {
        if self.parent_count() == 0 {
            return self.child_offsets.is_empty() && self.leaf_ids.is_empty();
        }
        self.child_offsets.first() == Some(&0)
            && self.child_offsets.last().copied() == Some(self.leaf_ids.len() as u32)
            && self.child_offsets.windows(2).all(|pair| pair[0] <= pair[1])
            && self.leaf_ids.len() == num_leaves
            && self.leaf_ids.iter().all(|&leaf| leaf < num_leaves as u32)
            && {
                let mut leaves = self.leaf_ids.clone();
                leaves.sort_unstable();
                leaves.iter().copied().eq(0..num_leaves as u32)
            }
    }

    pub(crate) fn visit_resident_regions(&self, visit: &mut dyn FnMut(&'static str, &[u8])) {
        visit(
            "two-level child offsets",
            bytes_of_slice(&self.child_offsets),
        );
        visit("two-level leaf IDs", bytes_of_slice(&self.leaf_ids));
    }
}

/// View an initialized plain-data slice as bytes for residency operations.
/// The returned slice cannot outlive the source and is never mutated.
pub(crate) fn bytes_of_slice<T>(slice: &[T]) -> &[u8] {
    let byte_len = std::mem::size_of_val(slice);
    if byte_len == 0 {
        return &[];
    }
    // SAFETY: every byte in an initialized `T` allocation may be read as u8;
    // the lifetime remains tied to `slice`, and callers receive no mutation.
    unsafe { std::slice::from_raw_parts(slice.as_ptr().cast::<u8>(), byte_len) }
}

pub fn routing_parent_count(num_leaves: usize) -> usize {
    if num_leaves <= 1 {
        return num_leaves;
    }
    ((num_leaves as f64).sqrt().ceil() as usize)
        .clamp(2, 4_096)
        .min(num_leaves)
}

/// Allocate exactly `total_clusters` child cells proportionally to populated
/// parent groups, without assigning more cells than training points.
pub fn allocate_child_clusters(group_sizes: &[usize], total_clusters: usize) -> Vec<usize> {
    let mut allocated: Vec<usize> = group_sizes
        .iter()
        .map(|&size| usize::from(size > 0))
        .collect();
    let mut remaining = total_clusters.saturating_sub(allocated.iter().sum());
    let total_points: usize = group_sizes.iter().sum();
    if remaining == 0 || total_points == 0 {
        return allocated;
    }
    for (allocation, &size) in allocated.iter_mut().zip(group_sizes) {
        let capacity = size.saturating_sub(*allocation);
        let share = remaining
            .saturating_mul(size)
            .checked_div(total_points)
            .unwrap_or(0)
            .min(capacity);
        *allocation += share;
    }
    remaining = total_clusters.saturating_sub(allocated.iter().sum());
    while remaining > 0 {
        let Some((index, _)) = group_sizes
            .iter()
            .enumerate()
            .filter(|(index, size)| allocated[*index] < **size)
            .max_by_key(|(index, size)| (**size, std::cmp::Reverse(allocated[*index])))
        else {
            break;
        };
        allocated[index] += 1;
        remaining -= 1;
    }
    allocated
}

/// A centroid selection computed once and reused by every segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IvfProbePlan {
    pub quantizer_version: u64,
    /// Hash of the query, routing mode, and requested leaf count. This keeps a
    /// reused mutable query object from accidentally reusing an older route.
    pub request_fingerprint: u64,
    pub cluster_ids: Arc<[u32]>,
}

impl IvfProbePlan {
    pub fn new(quantizer_version: u64, request_fingerprint: u64, cluster_ids: Vec<u32>) -> Self {
        Self {
            quantizer_version,
            request_fingerprint,
            cluster_ids: cluster_ids.into(),
        }
    }
}

fn fingerprint_words(
    mode: IvfRoutingMode,
    nprobe: usize,
    words: impl IntoIterator<Item = u64>,
) -> u64 {
    // FNV-1a with an extra avalanche. This is a cache key, not a persisted
    // identity or an adversarial hash table key.
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mode_tag = match mode {
        IvfRoutingMode::Auto => 0u64,
        IvfRoutingMode::Flat => 1,
        IvfRoutingMode::TwoLevel => 2,
        IvfRoutingMode::Hnsw => 3,
    };
    for word in std::iter::once(mode_tag)
        .chain(std::iter::once(nprobe as u64))
        .chain(words)
    {
        hash ^= word;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xff51_afd7_ed55_8ccd);
    hash ^ (hash >> 33)
}

pub fn float_probe_fingerprint(query: &[f32], nprobe: usize, mode: IvfRoutingMode) -> u64 {
    fingerprint_words(
        mode,
        nprobe,
        query.iter().map(|value| value.to_bits() as u64),
    )
}

pub fn binary_probe_fingerprint(query: &[u8], nprobe: usize, mode: IvfRoutingMode) -> u64 {
    fingerprint_words(mode, nprobe, query.iter().map(|&value| value as u64))
}

#[inline]
pub fn effective_routing_mode(mode: IvfRoutingMode, num_leaves: usize) -> IvfRoutingMode {
    match mode {
        IvfRoutingMode::Auto if num_leaves >= HNSW_AUTO_THRESHOLD => IvfRoutingMode::Hnsw,
        IvfRoutingMode::Auto => IvfRoutingMode::Flat,
        explicit => explicit,
    }
}

/// Number of parent cells to put in the routing beam.
pub fn parent_probe_count(nprobe: usize, num_leaves: usize, num_parents: usize) -> usize {
    if num_parents == 0 || num_leaves == 0 {
        return 0;
    }
    let leaves_per_parent = num_leaves.div_ceil(num_parents).max(1);
    nprobe
        .saturating_mul(PARENT_BEAM_OVERSAMPLE)
        .div_ceil(leaves_per_parent)
        .clamp(1, num_parents)
}

/// Deterministically select the best score indexes without fully sorting the
/// input. `HIGHER_IS_BETTER` covers Hamming similarity; `false` covers L2.
pub fn select_best<const HIGHER_IS_BETTER: bool>(scores: &[f32], take: usize) -> Vec<u32> {
    let take = take.min(scores.len());
    if take == 0 {
        return Vec::new();
    }
    let mut order: Vec<u32> = (0..scores.len() as u32).collect();
    let compare = |left: &u32, right: &u32| {
        let left_score = scores[*left as usize];
        let right_score = scores[*right as usize];
        let score_order = if HIGHER_IS_BETTER {
            right_score.total_cmp(&left_score)
        } else {
            left_score.total_cmp(&right_score)
        };
        score_order.then_with(|| left.cmp(right))
    };
    if take < order.len() {
        order.select_nth_unstable_by(take, compare);
        order.truncate(take);
    }
    order.sort_unstable_by(compare);
    order
}

/// Select leaf IDs from a scored candidate set. Candidate IDs need not be
/// contiguous, which lets both metrics share the exact same two-level beam
/// implementation.
pub fn select_best_candidates<const HIGHER_IS_BETTER: bool>(
    candidates: &mut Vec<(u32, f32)>,
    take: usize,
) -> Vec<u32> {
    let take = take.min(candidates.len());
    if take == 0 {
        return Vec::new();
    }
    let compare = |left: &(u32, f32), right: &(u32, f32)| {
        let score_order = if HIGHER_IS_BETTER {
            right.1.total_cmp(&left.1)
        } else {
            left.1.total_cmp(&right.1)
        };
        score_order.then_with(|| left.0.cmp(&right.0))
    };
    if take < candidates.len() {
        candidates.select_nth_unstable_by(take, compare);
        candidates.truncate(take);
    }
    candidates.sort_unstable_by(compare);
    candidates
        .iter()
        .map(|(cluster_id, _)| *cluster_id)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_selection_supports_both_metric_directions() {
        let scores = [0.5, 0.9, 0.1, 0.9];
        assert_eq!(select_best::<true>(&scores, 2), vec![1, 3]);
        assert_eq!(select_best::<false>(&scores, 2), vec![2, 0]);
    }

    #[test]
    fn two_level_beam_is_oversubscribed_but_bounded() {
        assert_eq!(parent_probe_count(32, 65_536, 256), 1);
        assert_eq!(parent_probe_count(256, 65_536, 256), 4);
        assert_eq!(parent_probe_count(65_536, 65_536, 256), 256);
    }

    #[test]
    fn compact_hnsw_routes_without_copying_points() {
        let points: Vec<[f32; 2]> = (0..512)
            .map(|index| {
                let angle = index as f32 * std::f32::consts::TAU / 512.0;
                [angle.cos(), angle.sin()]
            })
            .collect();
        let distance = |left: u32, right: u32| {
            let [lx, ly] = points[left as usize];
            let [rx, ry] = points[right as usize];
            (lx - rx).powi(2) + (ly - ry).powi(2)
        };
        let graph = HnswRoutingGraph::build(points.len(), distance, 42);
        assert!(graph.validate(points.len()));
        assert!(graph.size_bytes() < points.len() * 512);

        let query = [0.37f32, -0.91];
        let routed = graph.search(
            |node| {
                let [x, y] = points[node as usize];
                (x - query[0]).powi(2) + (y - query[1]).powi(2)
            },
            10,
        );
        let mut exact: Vec<u32> = (0..points.len() as u32).collect();
        exact.sort_unstable_by(|&left, &right| {
            let score = |node: u32| {
                let [x, y] = points[node as usize];
                (x - query[0]).powi(2) + (y - query[1]).powi(2)
            };
            score(left)
                .total_cmp(&score(right))
                .then_with(|| left.cmp(&right))
        });
        assert_eq!(routed, exact[..10]);

        let bytes = bincode::serde::encode_to_vec(&graph, bincode::config::standard()).unwrap();
        let (decoded, consumed): (HnswRoutingGraph, usize) =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
        assert_eq!(consumed, bytes.len());
        assert!(decoded.validate(points.len()));

        let mut corrupted = decoded;
        corrupted.node_offsets[1] = u32::MAX;
        assert!(!corrupted.validate(points.len()));
    }
}
