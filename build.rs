use std::fs::{create_dir_all, remove_dir_all};
use std::os::unix::fs::symlink;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto_prefix = "summa";
    create_dir_all(proto_prefix).unwrap();
    symlink("../proto", proto_prefix.to_string() + "/proto").unwrap();
    tonic_build::configure().compile(
        &[
            "summa/proto/consumer.proto",
            "summa/proto/consumer_service.proto",
            "summa/proto/index.proto",
            "summa/proto/index_service.proto",
            "summa/proto/search_service.proto",
        ],
        &["./"],
    )?;
    remove_dir_all(proto_prefix).unwrap();
    Ok(())
}
