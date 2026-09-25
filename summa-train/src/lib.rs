// clippy 1.98 added `chunks_exact_to_as_chunks`, which fires on ~40 call sites
// here. Migrating them is a real (if mechanical) improvement — a const-generic
// chunk width lets LLVM drop the per-chunk length check — but most of the sites
// are inside BMP / ScaNN / fast-field wire-format parsers, and rewriting those
// belongs in its own reviewed change rather than riding along with a toolchain
// bump. Tracked as follow-up; see docs/algebraic-float-reductions.md.
#![allow(clippy::chunks_exact_to_as_chunks)]

//! Reusable training workflow, task, and data-preparation contracts.

pub mod acceptance;
/// Internal, security-sensitive artifact I/O shared by the library and the
/// `summa-train` binary. This is public only because Cargo builds those as
/// separate crates; it is not a stable user-facing API.
#[doc(hidden)]
pub mod artifact_io;
pub mod benchmark;
pub mod benchmark_worker;
pub mod builtin_dreaming;
pub mod builtin_sleep_adapters;
pub mod builtin_sleep_runtime;
pub mod corpus;
pub mod device_sampler;
pub mod metrics;
pub mod native_host;
pub mod native_sleep;
pub mod optimizer_artifact;
pub mod posttrain;
pub mod promotion;
#[cfg(unix)]
mod protocol_process;
pub mod qat_candidate;
pub mod quantization;
pub mod resource_worker;
pub mod runtime;
pub mod sleep;
pub mod task;
pub mod tensor_sleep;
pub mod tier_optimizer;
pub mod worker;
pub mod workflow;
