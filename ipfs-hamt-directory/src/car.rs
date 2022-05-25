use std::io::{self, Write};

use cid::Cid;
use minicbor::encode;
use minicbor::{Encode, Encoder};
use unsigned_varint::encode::{usize, usize_buffer};

pub struct HeadlessCar {
    root_cids: Vec<Cid>,
    file: Box<dyn Write>,
}

impl HeadlessCar {
    pub fn new(file: Box<dyn Write>) -> Self {
        HeadlessCar { root_cids: vec![], file }
    }

    pub fn add_root_cid(&mut self, cid: Cid) {
        self.root_cids.push(cid);
    }

    pub fn write_header(mut self) -> io::Result<Car> {
        let mut header_buf = vec![];
        minicbor::encode(&self, &mut header_buf).unwrap();

        self.file.write_all(usize(header_buf.len(), &mut usize_buffer()))?;
        self.file.write_all(&header_buf)?;

        Ok(Car { file: self.file })
    }
}

impl<C> Encode<C> for HeadlessCar {
    fn encode<W: encode::Write>(&self, e: &mut Encoder<W>, _ctx: &mut C) -> Result<(), encode::Error<W::Error>> {
        e.map(2)?;
        e.str("version")?;
        e.i64(1)?;
        e.str("roots")?;
        e.array(self.root_cids.len() as u64)?;
        for root_cid in &self.root_cids {
            e.tag(minicbor::data::Tag::Unassigned(42))?;
            let mut b = vec![0];
            root_cid.write_bytes(&mut b).unwrap();
            e.bytes(&b)?;
        }
        Ok(())
    }
}

pub struct Car {
    file: Box<dyn Write>,
}

impl Car {
    pub fn write_block(&mut self, cid: &[u8], block: &[u8]) -> io::Result<()> {
        self.file.write_all(usize(cid.len() + block.len(), &mut usize_buffer()))?;
        self.file.write_all(cid)?;
        self.file.write_all(block)?;
        Ok(())
    }
}
