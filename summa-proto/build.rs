fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::env::set_var("PROTOC", protoc_bin_vendored::protoc_bin_path().unwrap());
    let files = &[
        "proto/consumer_service.proto",
        "proto/dag_pb.proto",
        "proto/index_service.proto",
        "proto/query.proto",
        "proto/reflection_service.proto",
        "proto/search_service.proto",
        "proto/unixfs.proto",
        "proto/utils.proto",
    ];
    let serde_default_structs = &[
        "dag_pb.PBNode",
        "dag_pb.PBLink",
        "summa.proto.HistogramAggregation",
        "summa.proto.IndexAttributes",
        "summa.proto.InflectionConfig",
        "summa.proto.MoreLikeThisQuery",
        "summa.proto.NerMatchConfig",
        "summa.proto.PhraseQuery",
        "summa.proto.QueryParserConfig",
        "summa.proto.ReservoirSamplingCollector",
        "summa.proto.TermsAggregation",
        "summa.proto.TopDocsCollector",
        "unixfs.Data",
    ];
    #[cfg(feature = "grpc")]
    build_tonic(files, serde_default_structs)?;
    #[cfg(not(feature = "grpc"))]
    build_prost(files, serde_default_structs)?;
    Ok(())
}

#[cfg(feature = "grpc")]
fn build_tonic(files: &[&str], serde_default_structs: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
    let builder = tonic_build::configure();
    let mut builder_ref = builder
        .type_attribute(".", "#[derive(serde::Serialize, serde::Deserialize)]")
        .type_attribute(".", "#[serde(rename_all = \"snake_case\")]");
    for serde_default_struct in serde_default_structs {
        builder_ref = builder_ref.type_attribute(serde_default_struct, "#[serde(default)]");
    }
    Ok(builder_ref
        .file_descriptor_set_path(std::env::var("OUT_DIR").unwrap() + "/summa.bin")
        .compile(files, &["./proto"])?)
}

#[cfg(not(feature = "grpc"))]
fn build_prost(files: &[&str], serde_default_structs: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
    let mut builder = prost_build::Config::new();
    let mut builder_ref = builder
        .type_attribute(".", "#[derive(serde::Serialize, serde::Deserialize)]")
        .type_attribute(".", "#[serde(rename_all = \"snake_case\")]");
    for serde_default_struct in serde_default_structs {
        builder_ref = builder_ref.type_attribute(serde_default_struct, "#[serde(default)]");
    }
    Ok(builder_ref.compile_protos(files, &["./proto"])?)
}
