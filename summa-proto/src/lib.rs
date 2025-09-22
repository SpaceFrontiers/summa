//! Canonical build-time protocol sources for Summa servers and brokers.

use std::io;
use std::path::Path;

/// Materialize the schemas and shared Rust validation in a private build
/// directory for compilation with the consumer's protobuf generator.
pub fn write_sources(directory: &Path) -> io::Result<()> {
    std::fs::create_dir_all(directory)?;
    for (name, contents) in [
        ("summa.proto", include_str!("../summa.proto")),
        ("summa-broker.proto", include_str!("../summa-broker.proto")),
        ("mutations.rs", include_str!("../mutations.rs")),
    ] {
        std::fs::write(directory.join(name), contents)?;
    }
    Ok(())
}
