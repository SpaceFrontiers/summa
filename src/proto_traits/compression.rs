use crate::proto;

impl From<proto::Compression> for tantivy::store::Compressor {
    fn from(compression: proto::Compression) -> Self {
        match compression {
            proto::Compression::None => tantivy::store::Compressor::None,
            proto::Compression::Brotli => tantivy::store::Compressor::Brotli,
            proto::Compression::Lz4 => tantivy::store::Compressor::Lz4,
            proto::Compression::Snappy => tantivy::store::Compressor::Snappy,
            proto::Compression::Zstd => tantivy::store::Compressor::Zstd,
        }
    }
}

impl Into<proto::Compression> for tantivy::store::Compressor {
    fn into(self) -> proto::Compression {
        match self {
            tantivy::store::Compressor::None => proto::Compression::None,
            tantivy::store::Compressor::Brotli => proto::Compression::Brotli,
            tantivy::store::Compressor::Lz4 => proto::Compression::Lz4,
            tantivy::store::Compressor::Snappy => proto::Compression::Snappy,
            tantivy::store::Compressor::Zstd => proto::Compression::Zstd,
        }
    }
}
