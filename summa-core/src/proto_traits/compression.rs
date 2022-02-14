use summa_proto::proto;
use tantivy::store::{Compressor, ZstdCompressor};

use crate::proto_traits::Wrapper;

impl From<Wrapper<proto::Compression>> for Compressor {
    fn from(compression: Wrapper<proto::Compression>) -> Self {
        match compression.into_inner() {
            proto::Compression::None => Compressor::None,
            proto::Compression::Brotli => Compressor::Brotli,
            proto::Compression::Lz4 => Compressor::Lz4,
            proto::Compression::Snappy => Compressor::Snappy,
            proto::Compression::Zstd => Compressor::Zstd(ZstdCompressor { compression_level: Some(3) }),
            proto::Compression::Zstd7 => Compressor::Zstd(ZstdCompressor { compression_level: Some(7) }),
            proto::Compression::Zstd9 => Compressor::Zstd(ZstdCompressor { compression_level: Some(9) }),
            proto::Compression::Zstd14 => Compressor::Zstd(ZstdCompressor { compression_level: Some(14) }),
            proto::Compression::Zstd19 => Compressor::Zstd(ZstdCompressor { compression_level: Some(19) }),
            proto::Compression::Zstd22 => Compressor::Zstd(ZstdCompressor { compression_level: Some(22) }),
        }
    }
}

impl From<Compressor> for Wrapper<proto::Compression> {
    fn from(compressor: Compressor) -> Self {
        Wrapper::from(match compressor {
            Compressor::None => proto::Compression::None,
            Compressor::Brotli => proto::Compression::Brotli,
            Compressor::Lz4 => proto::Compression::Lz4,
            Compressor::Snappy => proto::Compression::Snappy,
            Compressor::Zstd(ZstdCompressor { compression_level: Some(3) }) => proto::Compression::Zstd,
            Compressor::Zstd(ZstdCompressor { compression_level: Some(7) }) => proto::Compression::Zstd7,
            Compressor::Zstd(ZstdCompressor { compression_level: Some(9) }) => proto::Compression::Zstd9,
            Compressor::Zstd(ZstdCompressor { compression_level: Some(14) }) => proto::Compression::Zstd14,
            Compressor::Zstd(ZstdCompressor { compression_level: Some(19) }) => proto::Compression::Zstd19,
            Compressor::Zstd(ZstdCompressor { compression_level: Some(22) }) => proto::Compression::Zstd22,
            Compressor::Zstd(ZstdCompressor { .. }) => panic!("Unsupported panic"),
        })
    }
}
