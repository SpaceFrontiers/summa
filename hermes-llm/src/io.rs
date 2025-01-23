use anyhow::Result;
use flate2::read::GzDecoder;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

/// Opens a file and returns a buffered reader, automatically decompressing
/// based on file extension (.gz, .zst, .zstd).
pub fn open_file<P: AsRef<Path>>(path: P) -> Result<Box<dyn BufRead>> {
    let path = path.as_ref();
    let file = File::open(path)?;

    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    let reader: Box<dyn Read> = match extension.as_str() {
        "gz" | "gzip" => Box::new(GzDecoder::new(file)),
        "zst" | "zstd" => Box::new(zstd::Decoder::new(file)?),
        _ => Box::new(file),
    };

    Ok(Box::new(BufReader::new(reader)))
}

/// Reads entire file content as string, automatically decompressing if needed.
pub fn read_to_string<P: AsRef<Path>>(path: P) -> Result<String> {
    let mut reader = open_file(path)?;
    let mut content = String::new();
    reader.read_to_string(&mut content)?;
    Ok(content)
}

/// Returns an iterator over lines, automatically decompressing if needed.
pub fn read_lines<P: AsRef<Path>>(path: P) -> Result<impl Iterator<Item = Result<String>>> {
    let reader = open_file(path)?;
    Ok(reader.lines().map(|r| r.map_err(Into::into)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_read_plain_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "hello\nworld").unwrap();

        let content = read_to_string(&path).unwrap();
        assert_eq!(content, "hello\nworld");
    }

    #[test]
    fn test_read_gzip_file() {
        use flate2::Compression;
        use flate2::write::GzEncoder;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.txt.gz");

        let file = File::create(&path).unwrap();
        let mut encoder = GzEncoder::new(file, Compression::default());
        encoder.write_all(b"hello\nworld").unwrap();
        encoder.finish().unwrap();

        let content = read_to_string(&path).unwrap();
        assert_eq!(content, "hello\nworld");
    }

    #[test]
    fn test_read_zstd_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.txt.zst");

        let file = File::create(&path).unwrap();
        let mut encoder = zstd::Encoder::new(file, 0).unwrap();
        encoder.write_all(b"hello\nworld").unwrap();
        encoder.finish().unwrap();

        let content = read_to_string(&path).unwrap();
        assert_eq!(content, "hello\nworld");
    }
}
