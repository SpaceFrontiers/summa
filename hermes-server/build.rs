fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=../hermes-proto/hermes.proto");
    tonic_prost_build::compile_protos("../hermes-proto/hermes.proto")?;
    Ok(())
}
