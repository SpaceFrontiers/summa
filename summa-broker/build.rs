fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR")?);
    let proto_dir = out_dir.join("proto");
    summa_proto::write_sources(&proto_dir)?;
    // File descriptor sets feed gRPC server reflection (grpcurl et al. can
    // then call the broker without local .proto files).
    tonic_prost_build::configure()
        .file_descriptor_set_path(out_dir.join("summa_descriptor.bin"))
        .compile_protos(
            &[proto_dir.join("summa.proto")],
            std::slice::from_ref(&proto_dir),
        )?;
    tonic_prost_build::configure()
        .file_descriptor_set_path(out_dir.join("summa_broker_descriptor.bin"))
        .compile_protos(
            &[proto_dir.join("summa-broker.proto")],
            std::slice::from_ref(&proto_dir),
        )?;
    Ok(())
}
