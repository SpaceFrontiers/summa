use anyhow::{anyhow, Result};
use bitvec::prelude::*;
use cid::Cid;
use multihash::{Code, MultihashDigest};
use prost::Message;
use sled::Tree;

#[derive(Clone, Debug)]
pub struct HamtLink {
    hash_bits: BitVec<u8, Msb0>,
    pb_link: summa_proto::proto::dag_pb::PbLink,
}

impl HamtLink {
    pub fn new(hash: &[u8], pb_link: summa_proto::proto::dag_pb::PbLink) -> Self {
        HamtLink {
            hash_bits: BitVec::<u8, Msb0>::from_slice(&hash),
            pb_link,
        }
    }
    pub fn get_bits(&self, depth: usize, width: usize) -> usize {
        let offset = (depth * width) as usize;
        let index = &self.hash_bits[offset..(offset + width)];
        to_int(index)
    }
}

#[derive(Debug)]
enum ShardOrLink {
    Shard(Shard),
    HamtLink(HamtLink),
}

#[derive(Debug)]
struct Shard {
    hasher: Code,
    size: usize,
    width: usize,
    depth: usize,
    children: Vec<Option<ShardOrLink>>,
}

#[derive(Debug)]
struct CollapsedShard {
    hasher: Code,
    map: BitVec<u8, Msb0>,
    size: usize,
    links: Vec<summa_proto::proto::dag_pb::PbLink>,
}

// Reverse engineered from Kubo, do not ask me.
pub fn murmur_hash(data: &[u8], output: &mut [u8; 8]) {
    let value = fastmurmur3::hash(data);
    output[0] = (value >> 56) as u8;
    output[1] = (value >> 48) as u8;
    output[2] = (value >> 40) as u8;
    output[3] = (value >> 32) as u8;
    output[4] = (value >> 24) as u8;
    output[5] = (value >> 16) as u8;
    output[6] = (value >> 8) as u8;
    output[7] = value as u8;
}

impl CollapsedShard {
    pub fn from_links(hasher: Code, size: usize, links: Vec<Option<summa_proto::proto::dag_pb::PbLink>>) -> Self {
        // The ordering is reversed engineered from kubo implementation.
        // The bit that corresponds to block with index N is set to
        // at position N couting from the right (least significant) side.
        // .rev() and Msb0 is choosen due to the layout of how BitVec stored.
        let map: BitVec<u8, Msb0> = links.iter().rev().map(|x| x.is_some()).collect();
        let links = links.into_iter().flatten().collect();
        CollapsedShard { map, hasher, size, links }
    }

    fn serialize_to_block(mut self) -> (Cid, Vec<u8>) {
        let unixfs_proto = summa_proto::proto::unixfs::Data {
            r#type: summa_proto::proto::unixfs::data::DataType::HamtShard.into(),
            hash_type: Some(0x22), // Murmur3,
            fanout: Some(self.size as u64),
            data: Some(self.map.clone().into_vec()),
            ..Default::default()
        };
        self.links.sort_by(|a, b| a.name.partial_cmp(&b.name).unwrap());
        let node_pb = summa_proto::proto::dag_pb::PbNode {
            data: Some(unixfs_proto.encode_to_vec()),
            links: self.links,
        };
        let block = node_pb.encode_to_vec();
        let cid = Cid::new_v1(0x70, self.hasher.digest(&block));
        return (cid, block);
    }
}

impl Shard {
    fn new(hasher: Code, size: usize, depth: usize) -> Self {
        Shard {
            hasher,
            size,
            width: size.trailing_zeros() as usize,
            depth,
            children: (0..size).map(|_x| None).collect(),
        }
    }

    fn add(&mut self, hamt_link: HamtLink) -> Result<()> {
        let index = hamt_link.get_bits(self.depth, self.width);
        match &mut self.children[index] {
            Some(e) => match e {
                ShardOrLink::Shard(shard) => shard.add(hamt_link),
                ShardOrLink::HamtLink(existing_hamt_link) => {
                    let mut new_shard = Shard::new(self.hasher, self.size, self.depth + 1);
                    new_shard.add(existing_hamt_link.clone())?;
                    new_shard.add(hamt_link)?;
                    self.children[index] = Some(ShardOrLink::Shard(new_shard));
                    Ok(())
                }
            },
            None => {
                self.children[index] = Some(ShardOrLink::HamtLink(hamt_link));
                Ok(())
            }
        }
    }

    fn collapse(self, tree: &Tree) -> (Cid, u64) {
        let mut t_size_sum = 0;
        let links = self
            .children
            .into_iter()
            .enumerate()
            .map(|(index, child)| {
                child.map(|child| match child {
                    ShardOrLink::Shard(shard) => {
                        let (cid, t_size) = shard.collapse(tree);
                        t_size_sum += t_size;
                        summa_proto::proto::dag_pb::PbLink {
                            name: Some(format!("{:02X}", index)),
                            hash: Some(cid.to_bytes()),
                            t_size: Some(t_size),
                        }
                    }
                    ShardOrLink::HamtLink(link) => {
                        t_size_sum += link.pb_link.t_size.unwrap();
                        summa_proto::proto::dag_pb::PbLink {
                            name: link.pb_link.name.map(|name| format!("{:02X}{}", index, name)),
                            hash: link.pb_link.hash,
                            t_size: link.pb_link.t_size,
                        }
                    }
                })
            })
            .collect();
        let collapsed_shard = CollapsedShard::from_links(self.hasher, self.size, links);
        let (cid, block) = collapsed_shard.serialize_to_block();
        tree.insert(cid.to_bytes(), block).unwrap();
        (cid, t_size_sum)
    }
}

#[derive(Debug)]
pub struct IpldHamt {
    root: Shard,
}

impl IpldHamt {
    pub fn new(hasher: Code, size: usize) -> Self {
        IpldHamt {
            root: Shard::new(hasher, size, 0),
        }
    }

    pub fn set_depth(&mut self, new_depth: usize) {
        self.root.depth = new_depth
    }

    pub fn add(&mut self, hamt_link: HamtLink) -> Result<()> {
        self.root.add(hamt_link)
    }

    pub fn collapse(self, tree: &Tree) -> (Cid, u64) {
        let hasher = self.root.hasher;
        let (cid, t_size) = self.root.collapse(tree);

        let root = tree.get(cid.to_bytes()).unwrap().unwrap();
        let root_block = root.to_vec();

        tree.remove(cid.to_bytes()).unwrap();
        let cid = Cid::new_v1(0x70, hasher.digest(&root_block));
        tree.insert(cid.to_bytes(), root_block).unwrap();

        (cid, t_size)
    }

    pub fn serialize_root_of_subtrees(self, tree: &Tree, subtrees: Vec<summa_proto::proto::dag_pb::PbLink>) -> Result<Cid> {
        if subtrees.len() != self.root.size {
            return Err(anyhow!("Subtree count does not match width of tree"));
        }

        let collapsed_shard = CollapsedShard::from_links(self.root.hasher, self.root.size, subtrees.into_iter().map(Some).collect());
        let (cid, block) = collapsed_shard.serialize_to_block();
        tree.insert(cid.to_bytes(), block).unwrap();
        Ok(cid)
    }
}

pub fn to_int(slice: &BitSlice<u8, Msb0>) -> usize {
    slice.iter().by_vals().fold(0, |acc, b| (acc << 1) | (b as usize))
}
