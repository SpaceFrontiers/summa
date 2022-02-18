use std::collections::VecDeque;
use std::io;
use std::io::Write;
use std::path::{Path, PathBuf};

use bytes::Bytes;
use cid::multihash::MultihashDigest;
use cid::Cid;
use iroh_unixfs::balanced_tree::{LinkInfo, TreeNode};
use iroh_unixfs::codecs::Codec;
use tantivy::directory::{AntiCallToken, TerminatingWrite};
use tracing::info;

use crate::directories::iroh::directory::{DEFAULT_CODE, DEFAULT_DEGREE};
use crate::directories::iroh::file::IrohFileDescriptor;
use crate::directories::IrohDirectory;

pub struct IrohWriter {
    tail: Bytes,
    path: PathBuf,
    iroh_dd: IrohDirectory,
    store: iroh_store::Store,
    tree: VecDeque<Vec<(Cid, LinkInfo)>>,
    chunk_size: u64,
}

impl IrohWriter {
    // degree = 8
    // VecDeque![ vec![] ]
    // ..
    // VecDeque![ vec![0, 1, 2, 3, 4, 5, 6, 7] ]
    // VecDeque![ vec![8], vec![p0] ]

    // ..

    // VecDeque![ vec![0, 1, 2, 3, 4, 5, 6, 7] vec![p0] ]
    // VecDeque![ vec![], vec![p0, p1]]

    // ..

    // VecDeque![ vec![0, 1, 2, 3, 4, 5, 6, 7] vec![p0, p1, p2, p3, p4, p5, p6, p7], ]
    // VecDeque![ vec![], vec![p0, p1, p2, p3, p4, p5, p6, p7], vec![] ]
    // VecDeque![ vec![8], vec![p8], vec![pp0] ]
    //
    // A vecdeque of vecs, the first vec representing the lowest layer of stem nodes
    // and the last vec representing the root node
    // Since we emit leaf and stem nodes as we go, we only need to keep track of the
    // most "recent" branch, storing the links to that node's children & yielding them
    // when each node reaches `degree` number of links
    pub fn new(store: &iroh_store::Store, iroh_dd: IrohDirectory, path: impl AsRef<Path>, chunk_size: u64) -> Self {
        let mut tree = VecDeque::default();
        tree.push_back(Vec::with_capacity(DEFAULT_DEGREE as usize));
        IrohWriter {
            tail: Bytes::new(),
            path: path.as_ref().to_path_buf(),
            iroh_dd,
            store: store.clone(),
            tree,
            chunk_size,
        }
    }

    fn emit_chunk(&mut self, chunk: &[u8]) -> io::Result<Cid> {
        let tree_len = self.tree.len();
        if self.tree[0].len() as u32 == DEFAULT_DEGREE {
            // if so, iterate through nodes
            for i in 0..tree_len {
                // if we encounter any nodes that are not full, break
                if self.tree[i].len() < DEFAULT_DEGREE as usize {
                    break;
                }

                if i == tree_len - 1 {
                    self.tree.push_back(Vec::with_capacity(DEFAULT_DEGREE as usize));
                }

                // create node, keeping the cid
                let links = std::mem::replace(&mut self.tree[i], Vec::with_capacity(DEFAULT_DEGREE as usize));
                let (block, link_info) = TreeNode::Stem(links).encode(&DEFAULT_CODE).expect("cannot encode stem");
                let (cid, data, links) = block.into_parts();
                self.store.put(cid, data, links).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

                // add link_info to parent node
                self.tree[i + 1].push((cid, link_info));
            }
        }
        let cid = Cid::new_v1(Codec::Raw as _, DEFAULT_CODE.digest(chunk));
        self.tree[0].push((cid, LinkInfo::new(chunk.len() as u64, chunk.len() as u64)));
        self.store.put(cid, chunk, vec![]).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        Ok(cid)
    }

    pub fn terminate(&mut self) -> io::Result<()> {
        let tail = std::mem::replace(&mut self.tail, Bytes::new());
        if !tail.is_empty() || (self.tree.len() == 1 && self.tree[0].is_empty()) {
            self.emit_chunk(&tail)?;
        }

        if self.tree.len() == 1 && self.tree[0].len() == 1 {
            let (cid, link_info) = &self.tree[0][0];
            self.iroh_dd
                .insert(&self.path, IrohFileDescriptor::new(*cid, &self.path, link_info.raw_data_len))?;
            return Ok(());
        }

        // clean up, aka yield the rest of the stem nodes
        // since all the stem nodes are able to receive links
        // we don't have to worry about "overflow"
        info!(action = "tree_depth", depth = self.tree.len());
        while let Some(links) = self.tree.pop_front() {
            info!(action = "emit_links", links = links.len());
            let (block, link_info) = TreeNode::Stem(links).encode(&DEFAULT_CODE).expect("cannot encode");
            let (cid, data, links) = block.into_parts();
            self.store.put(cid, data, links).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

            if let Some(front) = self.tree.front_mut() {
                front.push((cid, link_info));
            } else {
                self.iroh_dd
                    .insert(&self.path, IrohFileDescriptor::new(cid, &self.path, link_info.raw_data_len))?;
            }
        }
        Ok(())
    }
}

impl Write for IrohWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let data = [&self.tail, buf].concat();
        for chunk in data.chunks(self.chunk_size as usize) {
            if chunk.len() as u64 == self.chunk_size {
                self.emit_chunk(chunk)?;
            } else {
                self.tail = Bytes::from(chunk.to_vec())
            }
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl TerminatingWrite for IrohWriter {
    fn terminate_ref(&mut self, _: AntiCallToken) -> io::Result<()> {
        self.terminate()
    }
}
