//! Vector quantization for IVF indexes
//!
//! Residual Product Quantization with OPQ is the trained float codec;
//! TurboQuant (`tq`) is the training-free per-segment codec.
//!
mod pq;
mod tq;

pub use pq::{DistanceTable, PQCodebook, PQConfig};
#[cfg(feature = "native")]
pub use tq::TqFlatBuilder;
pub use tq::{
    TQ_BLOCK_LANES, TQ_CODEC_VERSION, TqCodec, TqEncodeScratch, TqQueryPlan, tq_block_bytes,
    tq_codes_column_len, tq_codes_column_len_checked, tq_expected_fingerprint, tq_pack_block,
    tq_padded_dim, tq_score_block,
};
