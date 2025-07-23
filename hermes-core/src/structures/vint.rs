//! Shared unsigned variable-length integer encoding.

use std::io::{self, Read, Write};

/// Write an unsigned integer using seven payload bits per byte.
#[inline]
pub fn write_vint<W: Write + ?Sized>(writer: &mut W, mut value: u64) -> io::Result<()> {
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            writer.write_all(&[byte])?;
            return Ok(());
        }
        writer.write_all(&[byte | 0x80])?;
    }
}

/// Read an unsigned integer written by [`write_vint`].
#[inline]
pub fn read_vint<R: Read + ?Sized>(reader: &mut R) -> io::Result<u64> {
    let mut result = 0_u64;
    let mut shift = 0;

    loop {
        let mut encoded = [0_u8; 1];
        reader.read_exact(&mut encoded)?;
        let byte = encoded[0];
        result |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok(result);
        }

        shift += 7;
        if shift >= 64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "varint too long",
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrips_unsigned_boundaries() {
        for value in [0, 1, 0x7f, 0x80, 0x3fff, 0x4000, u32::MAX as u64, u64::MAX] {
            let mut encoded = Vec::new();
            write_vint(&mut encoded, value).unwrap();
            let mut reader = encoded.as_slice();
            assert_eq!(read_vint(&mut reader).unwrap(), value);
            assert!(reader.is_empty());
        }
    }

    #[test]
    fn rejects_truncated_and_overlong_values() {
        let mut truncated = [0x80].as_slice();
        assert_eq!(
            read_vint(&mut truncated).unwrap_err().kind(),
            io::ErrorKind::UnexpectedEof
        );

        let mut overlong = [0x80; 10].as_slice();
        assert_eq!(
            read_vint(&mut overlong).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
    }
}
