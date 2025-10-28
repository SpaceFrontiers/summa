fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR")?);
    let proto_dir = out_dir.join("proto");
    summa_proto::write_sources(&proto_dir)?;
    // Only the broker's control surface: topology, backends, refresh.
    tonic_prost_build::configure()
        .build_server(false)
        .compile_protos(
            &[proto_dir.join("summa-broker.proto")],
            std::slice::from_ref(&proto_dir),
        )?;
    Ok(())
}
