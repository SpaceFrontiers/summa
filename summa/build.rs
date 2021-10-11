use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let files = &[
        "proto/beacon_service.proto",
        "proto/consumer_service.proto",
        "proto/index_service.proto",
        "proto/query.proto",
        "proto/reflection_service.proto",
        "proto/search_service.proto",
        "proto/utils.proto",
    ];
    fs::create_dir_all("summa/proto").unwrap();
    for file in files {
        fs::copy(file, &format!("summa/{}", file)).unwrap();
    }
    #[cfg(feature = "grpc")]
    tonic_build::configure()
        .file_descriptor_set_path(std::env::var("OUT_DIR").unwrap() + "/summa.bin")
        .compile(
            &[
                "summa/proto/beacon_service.proto",
                "summa/proto/consumer_service.proto",
                "summa/proto/index_service.proto",
                "summa/proto/query.proto",
                "summa/proto/reflection_service.proto",
                "summa/proto/search_service.proto",
                "summa/proto/utils.proto",
            ],
            &["../external/com_google_protobuf/_virtual_imports/empty_proto"],
        )?;
    #[cfg(not(feature = "grpc"))]
    prost_build::Config::new()
        .type_attribute(".", "#[derive(serde::Serialize, serde::Deserialize)]")
        .type_attribute(".", "#[serde(rename_all = \"snake_case\")]")
        .compile_protos(
            &["summa/proto/index_service.proto", "summa/proto/query.proto", "summa/proto/utils.proto"],
            &["../external/com_google_protobuf/_virtual_imports/empty_proto"],
        )?;
    fs::remove_dir_all("summa").unwrap();
    Ok(())
}
