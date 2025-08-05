fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR")?);
    // File descriptor sets feed gRPC server reflection (grpcurl et al. can
    // then call the broker without local .proto files).
    tonic_prost_build::configure()
        .file_descriptor_set_path(out_dir.join("hermes_descriptor.bin"))
        .compile_protos(&["../hermes-proto/hermes.proto"], &["../hermes-proto"])?;
    tonic_prost_build::configure()
        .file_descriptor_set_path(out_dir.join("hermes_broker_descriptor.bin"))
        .compile_protos(
            &["../hermes-proto/hermes-broker.proto"],
            &["../hermes-proto"],
        )?;
    Ok(())
}
