use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter};
use std::str::FromStr;

use cid::Cid;
use ipfs_hamt_directory::StoringItem;
use pyo3::prelude::*;

#[pymodule]
#[pyo3(name = "ipfs_hamt_directory_py")]
fn ipfs_hamt_directory_py(py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    fn from_file(_py: Python, file_path: &str, output_file_path: &str, temporary_data_path: &str) -> PyResult<String> {
        let mut directory_builder = ipfs_hamt_directory::DirectoryBuilder::new(temporary_data_path, multihash::Code::Blake3_256);

        let file = File::open(file_path).unwrap();
        let file = BufReader::with_capacity(128 * 1024, file);

        let data_lines = file.lines().map(|line| {
            let mut line = line.unwrap();

            let mut record = line.split(' ');
            let name = record.next().unwrap();
            let cid = record.next().unwrap();
            let filesize = record.next().unwrap();

            let cid = Cid::from_str(cid).unwrap();
            let filesize: u64 = str::parse(filesize).unwrap();

            // Trick for avoiding new allocations, just truncate line to the name length.
            line.truncate(name.len());
            StoringItem::new(line, cid, filesize)
        });
        directory_builder.add_items(data_lines);
        let file = OpenOptions::new().write(true).create(true).truncate(true).open(output_file_path).unwrap();
        let file = BufWriter::with_capacity(128 * 1024, file);
        let root_cid = directory_builder.build(file)?;

        Ok(root_cid.to_string())
    }
    Ok(())
}
