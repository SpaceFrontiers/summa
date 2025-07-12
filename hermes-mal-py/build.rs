//! On macOS, a PyO3 `extension-module` cdylib links against Python symbols that
//! are only present in the host interpreter at load time. maturin injects the
//! `-undefined dynamic_lookup` linker flag for us, but a bare `cargo build`
//! (e.g. a workspace build) does not — so emit it here to keep the whole
//! workspace buildable without maturin. No-op on other platforms.
fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "macos" || target_os == "ios" {
        println!("cargo:rustc-cdylib-link-arg=-undefined");
        println!("cargo:rustc-cdylib-link-arg=dynamic_lookup");
    }
}
