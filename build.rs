use std::env;
use std::fs::{canonicalize, create_dir_all, remove_dir_all};
use std::os::unix::fs::symlink;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = env::var("OUT_DIR").unwrap();
    let proto_prefix = format!("{}/{}", out_dir, "/summa");
    let original_path = canonicalize("proto").unwrap();
    create_dir_all(&proto_prefix).unwrap();
    symlink(original_path, proto_prefix.to_string() + "/proto").unwrap();
    tonic_build::configure().compile(
        &[
            "summa/proto/consumer.proto",
            "summa/proto/consumer_service.proto",
            "summa/proto/index.proto",
            "summa/proto/index_service.proto",
            "summa/proto/search_service.proto",
        ],
        &[&out_dir],
    )?;
    remove_dir_all(proto_prefix).unwrap();
    Ok(())
}
