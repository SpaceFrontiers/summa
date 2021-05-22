use std::env;
use std::fs::{canonicalize, create_dir_all, remove_dir_all};
use std::os::unix::fs::symlink;

/// In `Izihawa` `Summa` repository keeped in `./summa` subdirectory and is the part of the larger monorepository
/// `*.proto` files are an external interface for other parts inside this monorepository
/// and therefore referenced by other parts using their full path `summa/proto/*.proto`. It is the reason why
/// imports in these files are done by their full path. It makes possible to build project from the root of monorepository
/// but causes issues when you try to build `summa` staying inside subdirectory.
/// That is the reason of this trick with moving `*.proto` files into subdirectory during building of the exported version of `summa`
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = env::var("OUT_DIR").unwrap();
    let proto_prefix = format!("{}/{}", out_dir, "/summa");
    let original_path = canonicalize("proto").unwrap();
    create_dir_all(&proto_prefix).unwrap();
    symlink(original_path, proto_prefix.to_string() + "/proto").unwrap();
    tonic_build::configure().file_descriptor_set_path(out_dir.to_owned() + "/summa.bin").compile(
        &[
            "summa/proto/consumer.proto",
            "summa/proto/consumer_service.proto",
            "summa/proto/index.proto",
            "summa/proto/index_service.proto",
            "summa/proto/reflection_service.proto",
            "summa/proto/search_service.proto",
        ],
        &[&out_dir],
    )?;
    remove_dir_all(proto_prefix).unwrap();
    Ok(())
}
