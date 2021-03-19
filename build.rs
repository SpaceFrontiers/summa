fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure().compile(
        &[
            "proto/consumer.proto",
            "proto/consumer_service.proto",
            "proto/index.proto",
            "proto/index_service.proto",
            "proto/search_service.proto",
        ],
        &["./"],
    )?;
    Ok(())
}
