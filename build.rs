use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = env::var("OUT_DIR").unwrap();
    tonic_build::configure().file_descriptor_set_path(out_dir + "/summa.bin").compile(
        &[
            "summa/proto/beacon_service.proto",
            "summa/proto/consumer_service.proto",
            "summa/proto/index_service.proto",
            "summa/proto/reflection_service.proto",
            "summa/proto/search_service.proto",
            "summa/proto/utils.proto",
        ],
        &["./", "./summa/proto", "./external/com_google_protobuf/_virtual_imports/empty_proto"],
    )?;
    Ok(())
}
