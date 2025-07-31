/// Internal, security-sensitive artifact I/O shared by the library and the
/// `hermes-train` binary. This is public only because Cargo builds those as
/// separate crates; it is not a stable user-facing API.
#[doc(hidden)]
pub mod artifact_io;
pub mod training_accounting;

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

#[cfg(unix)]
mod protocol_process;
pub mod qat_candidate;
pub mod quantization;

pub mod runtime;
pub mod sleep;
pub mod task;
pub mod tensor_sleep;
pub mod tier_optimizer;
pub mod worker;
pub mod workflow;
