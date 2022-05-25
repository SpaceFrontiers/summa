use std::io;
use std::io::Write;
use std::path::{Path, PathBuf};

use cid::Cid;
use multihash::{Code, MultihashDigest};
use prost::Message;
use rayon::prelude::*;

use crate::car::HeadlessCar;
use crate::ipld_hamt::{murmur_hash, HamtLink, IpldHamt};

pub struct DirectoryBuilder {
    hash_db: sled::Db,
    tree_db: sled::Db,
    temporary_data_path: PathBuf,
    hasher: Code,
    bucket_size: usize,
    total_size: u64,
}

pub struct StoringItem {
    name: String,
    cid: Cid,
    filesize: u64,
}

impl StoringItem {
    pub fn new(name: String, cid: Cid, filesize: u64) -> Self {
        StoringItem { name, cid, filesize }
    }
}

impl DirectoryBuilder {
    pub fn new<P: AsRef<Path>>(temporary_data_path: P, hasher: Code) -> Self {
        let hash_db = sled::Config::new()
            .path(temporary_data_path.as_ref().join("hash.db"))
            .flush_every_ms(Some(5000))
            .mode(sled::Mode::HighThroughput)
            .open()
            .unwrap();
        let tree_db = sled::Config::new()
            .path(temporary_data_path.as_ref().join("tree.db"))
            .flush_every_ms(Some(1000))
            .mode(sled::Mode::HighThroughput)
            .open()
            .unwrap();
        DirectoryBuilder {
            hash_db,
            tree_db,
            temporary_data_path: temporary_data_path.as_ref().to_path_buf(),
            hasher,
            bucket_size: 256,
            total_size: 0u64,
        }
    }

    pub fn add_items<I>(&mut self, items: I)
    where
        I: Iterator<Item = StoringItem> + Send,
    {
        let collected_size: u64 = items
            .par_bridge()
            .map(|storing_item| {
                let mut hashed_name: [u8; 8] = [0; 8];
                let name_bytes = storing_item.name.as_bytes();

                // Murmur hash is the only possible hash for now (2023-05-27) in IPLD HAMT
                murmur_hash(name_bytes, &mut hashed_name);
                let cid_bytes = storing_item.cid.to_bytes();

                let name_cid_filesize = bincode::serialize(&(name_bytes, &cid_bytes, storing_item.filesize)).unwrap();
                self.hash_db.insert(hashed_name, name_cid_filesize).unwrap();
                (name_bytes.len() + cid_bytes.len()) as u64
            })
            .sum();
        self.total_size += collected_size;
    }

    fn build_subtrees(&self) -> Vec<summa_proto::proto::dag_pb::PbLink> {
        (0..self.bucket_size)
            .collect::<Vec<_>>()
            .par_iter()
            .map(|bucket| {
                let hash_db = self.hash_db.clone();
                let mut tree = IpldHamt::new(self.hasher, self.bucket_size.into());
                tree.set_depth(1);

                for hash_keycid in hash_db.scan_prefix(&[*bucket as u8]) {
                    let hash_keycid = hash_keycid.unwrap();
                    let hash = hash_keycid.0;
                    let keycid = hash_keycid.1;

                    let (key, cid, t_size): (&[u8], &[u8], u64) = bincode::deserialize(&keycid).unwrap();
                    let key_str = unsafe { String::from_utf8_unchecked(key.to_vec()) };

                    let pb_link = summa_proto::proto::dag_pb::PbLink {
                        name: Some(key_str),
                        hash: Some(Cid::try_from(cid).unwrap().to_bytes()),
                        t_size: Some(t_size),
                    };

                    let hamt_link = HamtLink::new(hash.as_ref(), pb_link);
                    tree.add(hamt_link).unwrap();
                }

                let (cid, t_size) = tree.collapse(&self.tree_db.clone());
                summa_proto::proto::dag_pb::PbLink {
                    name: Some(format!("{:02X}", bucket)),
                    hash: Some(cid.to_bytes()),
                    t_size: Some(t_size),
                }
            })
            .collect()
    }

    fn build_small_directory(&mut self) -> io::Result<Cid> {
        let unixfs_proto = summa_proto::proto::unixfs::Data {
            r#type: summa_proto::proto::unixfs::data::DataType::Directory.into(),
            hash_type: Some(self.hasher.into()),
            ..Default::default()
        };

        let mut links: Vec<_> = self
            .hash_db
            .iter()
            .map(|hash_keycid| {
                let hash_keycid = hash_keycid.unwrap();
                let keycid = hash_keycid.1;
                let (key, cid, t_size): (&[u8], &[u8], u64) = bincode::deserialize(&keycid).unwrap();
                summa_proto::proto::dag_pb::PbLink {
                    name: Some(unsafe { String::from_utf8_unchecked(key.to_vec()) }),
                    hash: Some(Cid::try_from(cid).unwrap().to_bytes()),
                    t_size: Some(t_size),
                }
            })
            .collect();
        links.sort_by(|a, b| a.name.partial_cmp(&b.name).unwrap());
        let node_pb = summa_proto::proto::dag_pb::PbNode {
            data: Some(unixfs_proto.encode_to_vec()),
            links: self
                .hash_db
                .iter()
                .map(|hash_keycid| {
                    let hash_keycid = hash_keycid.unwrap();
                    let keycid = hash_keycid.1;
                    let (key, cid, t_size): (&[u8], &[u8], u64) = bincode::deserialize(&keycid).unwrap();
                    summa_proto::proto::dag_pb::PbLink {
                        name: Some(unsafe { String::from_utf8_unchecked(key.to_vec()) }),
                        hash: Some(Cid::try_from(cid).unwrap().to_bytes()),
                        t_size: Some(t_size),
                    }
                })
                .collect(),
        };
        let block = node_pb.encode_to_vec();
        let cid = Cid::new_v1(0x70, self.hasher.digest(&block));
        self.tree_db.insert(cid.to_bytes(), block).unwrap();
        Ok(cid)
    }

    pub fn build<W: Write + 'static>(mut self, writer: W) -> io::Result<Cid> {
        const MAX_BLOCK_SIZE: u64 = 1048576;
        const SAFETY_GAP: u64 = 131702;

        let root_cid = if self.total_size < MAX_BLOCK_SIZE - SAFETY_GAP {
            self.build_small_directory()?
        } else {
            let subtree_cids = self.build_subtrees();
            let tree = IpldHamt::new(self.hasher, self.bucket_size);
            tree.serialize_root_of_subtrees(&self.tree_db, subtree_cids).unwrap()
        };

        let mut headless_car = HeadlessCar::new(Box::new(writer));
        headless_car.add_root_cid(root_cid);
        let mut car = headless_car.write_header()?;

        for entry in self.tree_db.iter() {
            let (cid, block) = entry.unwrap();
            car.write_block(&cid, &block).unwrap();
        }

        self.hash_db.clear()?;
        std::fs::remove_dir_all(self.temporary_data_path.join("hash.db"))?;
        self.tree_db.clear()?;
        std::fs::remove_dir_all(self.temporary_data_path.join("tree.db"))?;
        Ok(root_cid)
    }
}
