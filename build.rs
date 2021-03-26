fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure().compile(
        &[
            "summa/proto/consumer.proto",
            "summa/proto/consumer_service.proto",
            "summa/proto/index.proto",
            "summa/proto/index_service.proto",
            "summa/proto/search_service.proto",
        ],
        &["./", "../"],
    )?;
    Ok(())
}
