//! Vector quantization for IVF indexes
//!
//! TurboQuant (`tq`) is the float codec: training-free, used by both the
//! per-segment flat scan and the trained-router IVF-TQ leaves.
//!
mod tq;

#[cfg(feature = "native")]
pub use tq::TqFlatBuilder;
pub use tq::{
    TQ_BLOCK_LANES, TQ_CODEC_VERSION, TqCodec, TqEncodeScratch, TqQueryPlan, tq_block_bytes,
    tq_codes_column_len, tq_codes_column_len_checked, tq_expected_fingerprint, tq_ivf_block_bytes,
    tq_ivf_codes_column_len_checked, tq_pack_block, tq_pack_ivf_block, tq_padded_dim,
    tq_score_block, tq_score_ivf_block, tq_shared_codec,
};
