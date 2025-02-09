//! Shared constants for the sparse vector file format.
//!
//! File format (footer-based, data-first):
//! ```text
//! [posting data for all dims across all fields]
//! [TOC: per-field header + per-dim entries]
//! [footer: toc_offset(u64) + num_fields(u32) + magic(u32)]
//! ```
//!
//! TOC per field: field_id(u32) + quantization(u8) + num_dims(u32)
//! TOC per dim:   dim_id(u32) + data_offset(u64) + data_length(u32)

/// Magic number for sparse vectors file footer ("SPR2" in LE)
pub const SPARSE_FOOTER_MAGIC: u32 = 0x32525053;

/// Footer size in bytes: toc_offset(8) + num_fields(4) + magic(4)
pub const SPARSE_FOOTER_SIZE: u64 = 16;
