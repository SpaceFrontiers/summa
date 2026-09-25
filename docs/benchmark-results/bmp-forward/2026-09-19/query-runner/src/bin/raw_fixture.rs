//! Offline benchmark conversion only; no legacy support enters the application.
#![allow(dead_code)]
use std::{
    fs::{File, OpenOptions},
    io::{BufWriter, Write},
    path::Path,
};
#[path = "../../../../../../../summa-core/src/structures/postings/sparse/dimensions.rs"]
mod dimensions;
mod structures {
    pub(crate) mod postings {
        pub(crate) use crate::dimensions as sparse_dimensions;
    }
}
type Result<T> = std::result::Result<T, String>;
fn corrupt(s: &str) -> String {
    s.into()
}
#[path = "../../../../../../../summa-core/src/segment/bmp_forward/codec.rs"]
mod codec;
fn u32_at(b: &[u8], i: usize) -> u32 {
    u32::from_le_bytes(b[i..i + 4].try_into().unwrap())
}
fn u64_at(b: &[u8], i: usize) -> u64 {
    u64::from_le_bytes(b[i..i + 8].try_into().unwrap())
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let source = Path::new(&args[1]);
    let target = Path::new(&args[2]);
    std::fs::create_dir(target).unwrap();
    for entry in std::fs::read_dir(source).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        let output = target.join(entry.file_name());
        if path.extension().is_none_or(|e| e != "sparse") {
            std::fs::hard_link(path, output).unwrap();
            continue;
        }
        let file = File::open(path).unwrap();
        let b = unsafe { memmap2::MmapOptions::new().map(&file).unwrap() };
        let toc = u64_at(&b, b.len() - 16) as usize;
        assert_eq!(u32_at(&b, b.len() - 8), 1);
        assert_eq!(u32_at(&b, toc + 5), 1);
        assert_eq!(u64_at(&b, toc + 17), 0);
        let end = toc - 80;
        assert_eq!(u32_at(&b, end + 76), 0x42504d42);
        let rows = u32_at(&b, end - 16) as usize;
        let old_len = u64_at(&b, end - 8) as usize;
        let directory = end - 16 - rows * 16;
        let payload = directory - old_len;
        let mut w = BufWriter::with_capacity(
            1024 * 1024,
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(output)
                .unwrap(),
        );
        w.write_all(&b[..payload]).unwrap();
        let mut raw_len = 0u64;
        let mut coords = 0;
        let row = |i: usize| {
            let begin = u64_at(&b, directory + i * 16 + 8) as usize;
            let stop = if i + 1 == rows {
                old_len
            } else {
                u64_at(&b, directory + (i + 1) * 16 + 8) as usize
            };
            codec::ForwardVector(&b[payload + begin..payload + stop])
        };
        for i in 0..rows {
            let vector = row(i);
            for (dim, impact) in vector.iter() {
                w.write_all(&dim.to_le_bytes()).unwrap();
                w.write_all(&[impact]).unwrap();
                coords += 1;
                raw_len += 5;
            }
        }
        let mut offset = 0u64;
        for i in 0..rows {
            w.write_all(&b[directory + i * 16..directory + i * 16 + 8])
                .unwrap();
            w.write_all(&offset.to_le_bytes()).unwrap();
            offset += row(i).len() as u64 * 5;
        }
        assert_eq!(offset, raw_len);
        w.write_all(&(rows as u32).to_le_bytes()).unwrap();
        w.write_all(&0u32.to_le_bytes()).unwrap();
        w.write_all(&raw_len.to_le_bytes()).unwrap();
        let mut footer = b[end..end + 80].to_vec();
        footer[76..80].copy_from_slice(&0x41504d42u32.to_le_bytes());
        w.write_all(&footer).unwrap();
        let new_toc = payload as u64 + raw_len + rows as u64 * 16 + 16 + 80;
        let mut table = b[toc..b.len() - 24].to_vec();
        assert_eq!(table.len(), 41);
        table[25..29].copy_from_slice(&(new_toc as u32).to_le_bytes());
        table[29..33].copy_from_slice(&((new_toc >> 32) as u32).to_le_bytes());
        w.write_all(&table).unwrap();
        w.write_all(&new_toc.to_le_bytes()).unwrap();
        w.write_all(&new_toc.to_le_bytes()).unwrap();
        w.write_all(&1u32.to_le_bytes()).unwrap();
        w.write_all(&0x34525053u32.to_le_bytes()).unwrap();
        w.flush().unwrap();
        println!(
            "{}",
            serde_json::json!({"rows":rows,"coordinates":coords,"unchanged_inverted_and_docmap_bytes":payload,"raw_payload":raw_len,"packet_payload":old_len,"raw_sparse_bytes":new_toc+65,"packet_sparse_bytes":b.len()})
        );
    }
}
