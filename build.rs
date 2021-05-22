use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = env::var("OUT_DIR").unwrap();
    tonic_build::configure().file_descriptor_set_path(out_dir.to_owned() + "/summa.bin").compile(
        &[
            "summa/proto/consumer_service.proto",
            "summa/proto/index_service.proto",
            "summa/proto/reflection_service.proto",
            "summa/proto/search_service.proto",
        ],
        &["summa/proto"],
    )?;
    Ok(())
}
