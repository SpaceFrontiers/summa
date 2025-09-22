fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto_dir = std::path::PathBuf::from(std::env::var("OUT_DIR")?).join("proto");
    summa_proto::write_sources(&proto_dir)?;
    tonic_prost_build::compile_protos(proto_dir.join("summa.proto"))?;
    Ok(())
}
