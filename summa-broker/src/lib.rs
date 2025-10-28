//! Library surface of the broker shared with deployment tooling
//! (`summa-k8s-manager`): placement rule syntax and the generated protocol
//! bindings. The broker binary itself lives in `main.rs`.

pub mod placement_rules;
pub mod proto;
