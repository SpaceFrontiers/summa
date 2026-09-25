//! Offline benchmark-only transcoder for the single-field version-5 fixture.
//! Uses the production dimension encoder; leaves weight and nomination bytes
//! unchanged. Never use this tool to mutate a live index.
use std::{fs, path::Path};
fn u32_at(b: &[u8], p: usize) -> u32 {
    u32::from_le_bytes(b[p..p + 4].try_into().unwrap())
}
fn u64_at(b: &[u8], p: usize) -> usize {
    u64::from_le_bytes(b[p..p + 8].try_into().unwrap()) as usize
}
fn p32(b: &mut [u8], p: usize, v: usize) {
    b[p..p + 4].copy_from_slice(&(v as u32).to_le_bytes());
}
fn p64(b: &mut [u8], p: usize, v: usize) {
    b[p..p + 8].copy_from_slice(&(v as u64).to_le_bytes());
}
#[path = "../../../summa-core/src/structures/postings/sparse/dimensions.rs"]
mod dimensions;
fn component(b: &[u8], mode: &str, stats: &mut [u64; 4]) -> Vec<u8> {
    let footer = b.len() - 40;
    assert_eq!(u32_at(b, footer + 4), 5);
    if mode == "raw" || u32_at(b, footer + 36) != 0 {
        let mut out = b.to_vec();
        p32(&mut out, footer + 4, 6);
        return out;
    }
    let runs = u32_at(b, footer + 32) as usize;
    let table = footer - runs * 32;
    let mut entries = b[table..footer].to_vec();
    let mut out = Vec::new();
    for i in 0..runs {
        let at = i * 32;
        let off = u64_at(&entries, at);
        let len = u64_at(&entries, at + 8);
        let run = &b[off..off + len];
        let rf = len - 32;
        assert_eq!(u32_at(run, rf + 4), 0);
        let rows = u32_at(run, rf) as usize;
        let dir = u64_at(run, rf + 8);
        let mut directory = run[dir..dir + rows * 24].to_vec();
        let mut encoded = Vec::new();
        for r in 0..rows {
            let at = r * 24;
            let off = u64_at(&directory, at + 8);
            let len = u32_at(&directory, at + 16) as usize;
            let n = u32_at(&directory, at + 20) as usize;
            assert_eq!(directory[at + 6], 0);
            let old = &run[off..off + len];
            let values: Vec<_> = (0..n).map(|i| (u32_at(old, i * 4), 0.0)).collect();
            let (tag, dims) = if mode == "u24" && values.last().is_some_and(|v| v.0 < 1 << 24) {
                (
                    3,
                    values
                        .iter()
                        .flat_map(|v| {
                            let b = v.0.to_le_bytes();
                            [b[0], b[1], b[2]]
                        })
                        .collect(),
                )
            } else if mode == "u16" && values.last().is_some_and(|v| v.0 <= 65535) {
                (
                    1,
                    values
                        .iter()
                        .flat_map(|v| (v.0 as u16).to_le_bytes())
                        .collect(),
                )
            } else {
                dimensions::encode(&values, true)
            };
            assert!(
                dimensions::Dimensions::<false>::new(&dims, n, tag).eq(values.iter().map(|v| v.0))
            );
            stats[0] += (n * 4) as u64;
            stats[1] += dims.len() as u64;
            stats[2] += n as u64;
            stats[3] += 1;
            directory[at + 6] = tag;
            p64(&mut directory, at + 8, encoded.len());
            p32(&mut directory, at + 16, dims.len() + len - n * 4);
            encoded.extend(&dims);
            encoded.extend(&old[n * 4..]);
        }
        let dir = encoded.len();
        encoded.extend(directory);
        let rf2 = encoded.len();
        encoded.extend(&run[rf..]);
        p64(&mut encoded, rf2 + 8, dir);
        p64(&mut encoded, rf2 + 16, rf2);
        p64(&mut entries, at, out.len());
        p64(&mut entries, at + 8, encoded.len());
        out.extend(encoded);
    }
    out.extend(entries);
    let f = out.len();
    out.extend(&b[footer..]);
    p32(&mut out, f + 4, 6);
    out
}
fn file(b: &[u8], mode: &str, stats: &mut [u64; 4]) -> Vec<u8> {
    let toc = u64_at(b, b.len() - 16);
    let off = u64_at(b, toc + 17);
    let len = u64_at(b, toc + 25);
    let enc = component(&b[off..off + len], mode, stats);
    let removed = len - enc.len();
    let mut out = b[..off].to_vec();
    out.extend(&enc);
    out.extend(&b[off + len..]);
    let end = out.len();
    p64(&mut out, toc - removed + 25, enc.len());
    p64(&mut out, end - 16, toc - removed);
    p64(&mut out, end - 24, u64_at(b, b.len() - 24) - removed);
    out
}
fn main() {
    let a: Vec<_> = std::env::args().collect();
    let src = Path::new(&a[1]);
    let dst = Path::new(&a[2]);
    assert!(!dst.exists());
    fs::create_dir_all(dst).unwrap();
    let mode = a.get(3).map_or("dot", String::as_str);
    let mut stats = [0; 4];
    let mut old = 0;
    let mut new = 0;
    for e in fs::read_dir(src).unwrap() {
        let e = e.unwrap();
        let name = e.file_name();
        let p = dst.join(&name);
        if name.to_string_lossy().ends_with(".sparse")
            || name.to_string_lossy().contains(".seismic.")
        {
            let b = fs::read(e.path()).unwrap();
            let enc = file(&b, mode, &mut stats);
            old += b.len();
            new += enc.len();
            fs::write(p, enc).unwrap();
        } else {
            fs::copy(e.path(), p).unwrap();
        }
    }
    println!(
        "{{\"raw_dimensions\":{},\"encoded_dimensions\":{},\"coordinates\":{},\"rows\":{},\"old_sparse\":{old},\"new_sparse\":{new}}}",
        stats[0], stats[1], stats[2], stats[3]
    );
}
