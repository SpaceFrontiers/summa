use crate::proto;
use tantivy::store::Compressor;

impl From<proto::Compression> for Compressor {
    fn from(compression: proto::Compression) -> Self {
        match compression {
            proto::Compression::None => Compressor::None,
            proto::Compression::Brotli => Compressor::Brotli,
            proto::Compression::Lz4 => Compressor::Lz4,
            proto::Compression::Snappy => Compressor::Snappy,
            proto::Compression::Zstd => Compressor::Zstd,
        }
    }
}

impl From<Compressor> for proto::Compression {
    fn from(compressor: Compressor) -> Self {
        match compressor {
            Compressor::None => proto::Compression::None,
            Compressor::Brotli => proto::Compression::Brotli,
            Compressor::Lz4 => proto::Compression::Lz4,
            Compressor::Snappy => proto::Compression::Snappy,
            Compressor::Zstd => proto::Compression::Zstd,
        }
    }
}
