//! Generated protobuf/tonic bindings.
//!
//! `summa` is the verbatim summa-server wire contract the broker re-serves;
//! `broker` is the broker-only control surface (separate proto file so the
//! shared contract and its generated Python/TypeScript clients never churn
//! for broker concerns).

// Generated code: prost oneof enums trip clippy::enum_variant_names
// (FieldValue's BytesValue/JsonValue); not ours to rename.
#[allow(clippy::enum_variant_names)]
pub mod summa {
    tonic::include_proto!("summa");
    include!(concat!(env!("OUT_DIR"), "/proto/mutations.rs"));
}

pub mod broker {
    tonic::include_proto!("summa.broker");
}

/// Encoded file descriptor sets for gRPC server reflection.
pub const SUMMA_DESCRIPTOR: &[u8] = tonic::include_file_descriptor_set!("summa_descriptor");
pub const BROKER_DESCRIPTOR: &[u8] = tonic::include_file_descriptor_set!("summa_broker_descriptor");
