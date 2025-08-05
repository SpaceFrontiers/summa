//! Generated protobuf/tonic bindings.
//!
//! `hermes` is the verbatim hermes-server wire contract the broker re-serves;
//! `broker` is the broker-only control surface (separate proto file so the
//! shared contract and its generated Python/TypeScript clients never churn
//! for broker concerns).

// Generated code: prost oneof enums trip clippy::enum_variant_names
// (FieldValue's BytesValue/JsonValue); not ours to rename.
#[allow(clippy::enum_variant_names)]
pub mod hermes {
    tonic::include_proto!("hermes");
}

pub mod broker {
    tonic::include_proto!("hermes.broker");
}

/// Encoded file descriptor sets for gRPC server reflection.
pub const HERMES_DESCRIPTOR: &[u8] = tonic::include_file_descriptor_set!("hermes_descriptor");
pub const BROKER_DESCRIPTOR: &[u8] =
    tonic::include_file_descriptor_set!("hermes_broker_descriptor");
