//! Generic, reproducible corpus preparation.
//!
//! Search engines and canonical record stores are deliberately separate: a
//! [`SearchBackend`] discovers stable record keys, then a
//! [`RecordMaterializer`] resolves authoritative text for those keys. The
//! remaining stages are backend-neutral and produce immutable, tokenized
//! shards plus a complete build manifest.

mod compose;
mod config;
mod materialize;
mod pipeline;
mod recipe;
mod search;

pub use compose::*;
pub use config::*;
pub use materialize::*;
pub use pipeline::*;
pub use recipe::*;
pub use search::*;

#[cfg(test)]
mod tests;
