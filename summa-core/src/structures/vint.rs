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
// Keep slice cursor updates and Result handling inside dictionary/posting loops.
// Measured tradeoff: docs/benchmark-results/dictionary-decoding-2026-09-26/README.md.
#[inline(always)]
pub fn read_vint<R: Read + ?Sized>(reader: &mut R) -> io::Result<u64> {
    let mut result = 0_u64;
    let mut shift = 0;

    loop {
        let mut encoded = [0_u8; 1];
        reader.read_exact(&mut encoded)?;
        let byte = encoded[0];
        if shift == 63 && byte > 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "varint exceeds u64",
            ));
        }
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

/// [`read_vint`] for byte slices, with the one-byte case (values below 128,
/// such as nearly every dictionary key length) inline: the generic reader
/// costs a `read_exact` per byte in warm dictionary scans.
#[inline(always)]
pub fn read_vint_slice(reader: &mut &[u8]) -> io::Result<u64> {
    match reader.split_first() {
        Some((&byte, rest)) if byte < 0x80 => {
            *reader = rest;
            Ok(u64::from(byte))
        }
        _ => read_vint(reader),
    }
}

/// Advance past one integer written by [`write_vint`] without decoding it.
///
/// Accepts and rejects exactly the inputs [`read_vint`] does, with the same
/// error kinds and cursor position, so value skipping cannot desynchronize a
/// block decoder from value reading.
#[inline(always)]
pub fn skip_vint(reader: &mut &[u8]) -> io::Result<()> {
    let limit = reader.len().min(10);
    for index in 0..limit {
        let byte = reader[index];
        if index == 9 && byte > 1 {
            *reader = &reader[10..];
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "varint exceeds u64",
            ));
        }
        if byte & 0x80 == 0 {
            *reader = &reader[index + 1..];
            return Ok(());
        }
    }
    *reader = &reader[limit..];
    Err(io::Error::new(
        io::ErrorKind::UnexpectedEof,
        "varint truncated",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skip_matches_read_acceptance_error_kind_and_position() {
        let mut inputs: Vec<Vec<u8>> = vec![vec![], vec![0x80, 0, 42], vec![0xff; 12]];
        for value in [0, 1, 127, 128, 16_384, u32::MAX as u64, u64::MAX] {
            let mut encoded = Vec::new();
            write_vint(&mut encoded, value).unwrap();
            for end in 0..=encoded.len() {
                inputs.push(encoded[..end].to_vec());
            }
            encoded.push(7);
            inputs.push(encoded);
        }
        for last in [0, 1, 2, 0x7f] {
            let mut overflow = vec![0x80; 9];
            overflow.extend([last, 42]);
            inputs.push(overflow);
        }
        for input in inputs {
            let (mut read, mut skipped) = (input.as_slice(), input.as_slice());
            let expected = read_vint(&mut read).map(drop).map_err(|error| error.kind());
            let actual = skip_vint(&mut skipped).map_err(|error| error.kind());
            assert_eq!(actual, expected, "{input:?}");
            assert_eq!(skipped, read, "{input:?}");
        }
    }

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

    #[test]
    fn overflowing_tenth_varint_byte_is_rejected_instead_of_wrapping() {
        for last in [2, 3, 0x7f] {
            let mut bytes = [0x80; 10];
            bytes[9] = last;
            assert_eq!(
                read_vint(&mut bytes.as_slice()).unwrap_err().kind(),
                io::ErrorKind::InvalidData
            );
        }
    }

    #[test]
    fn consecutive_values_preserve_encoded_bytes_and_reader_position() {
        let values = [0, 127, 128, 16_383, 16_384, u64::MAX];
        let expected = [
            0, 127, 128, 1, 255, 127, 128, 128, 1, 255, 255, 255, 255, 255, 255, 255, 255, 255, 1,
        ];
        let mut encoded = Vec::new();
        for value in values {
            write_vint(&mut encoded, value).unwrap();
        }
        assert_eq!(encoded, expected);
        let mut slice = encoded.as_slice();
        let mut stream = io::Cursor::new(&encoded);
        for value in values {
            assert_eq!(read_vint(&mut slice).unwrap(), value);
            assert_eq!(read_vint(&mut stream).unwrap(), value);
            assert_eq!(stream.position() as usize, encoded.len() - slice.len());
        }
        assert!(slice.is_empty());
    }

    #[test]
    fn truncated_values_consume_available_bytes_without_reading_the_next_value() {
        for length in 1..=10 {
            let mut encoded = vec![0x80; length - 1];
            encoded.push(1);
            for end in 0..length {
                let mut input = &encoded[..end];
                assert_eq!(
                    read_vint(&mut input).unwrap_err().kind(),
                    io::ErrorKind::UnexpectedEof
                );
                assert!(input.is_empty());
            }
            encoded.push(42);
            let mut input = encoded.as_slice();
            assert_eq!(read_vint(&mut input).unwrap(), 1 << (7 * (length - 1)));
            assert_eq!(input, &[42]);
        }
        // Non-minimal representations remain readable; inlining must not
        // accidentally impose a new canonical-encoding requirement.
        let mut input = [0x80, 0, 42].as_slice();
        assert_eq!(read_vint(&mut input).unwrap(), 0);
        assert_eq!(input, &[42]);
        let mut overflow = [0x80; 11];
        overflow[9] = 2;
        overflow[10] = 42;
        let mut input = overflow.as_slice();
        assert_eq!(
            read_vint(&mut input).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
        assert_eq!(input, &[42]);
    }
}
