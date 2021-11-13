fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::env::set_var("PROTOC", protoc_bin_vendored::protoc_bin_path().unwrap());
    let files = &[
        "proto/beacon_service.proto",
        "proto/consumer_service.proto",
        "proto/index_service.proto",
        "proto/query.proto",
        "proto/reflection_service.proto",
        "proto/search_service.proto",
        "proto/utils.proto",
    ];
    #[cfg(feature = "grpc")]
    tonic_build::configure()
        .type_attribute(".", "#[derive(serde::Serialize, serde::Deserialize)]")
        .type_attribute(".", "#[serde(rename_all = \"snake_case\")]")
        .file_descriptor_set_path(std::env::var("OUT_DIR").unwrap() + "/summa.bin")
        .compile(files, &["./proto"])?;
    #[cfg(not(feature = "grpc"))]
    prost_build::Config::new()
        .type_attribute(".", "#[derive(serde::Serialize, serde::Deserialize)]")
        .type_attribute(".", "#[serde(rename_all = \"snake_case\")]")
        .compile_protos(files, &["./proto"], )?;
    Ok(())
}
