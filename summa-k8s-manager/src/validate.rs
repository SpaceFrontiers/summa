//! `SearchCluster` validation. The operator refuses to act on an invalid
//! cluster and the API refuses to save one; both call [`validate`].

use std::collections::BTreeSet;

use serde::Serialize;
use summa_broker::placement_rules;

use crate::crd::{ImageSpec, SearchClusterSpec};
use crate::render::{
    BROKER_RESERVED_FLAGS, SHARD_RESERVED_FLAGS, flag_names, routing, shard_paths,
};

/// Longest cluster name: leaves room for `-broker` and shard suffixes
/// within a 63-character Deployment name.
const MAX_NAME_LEN: usize = 50;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FieldError {
    /// Path in the resource, e.g. `spec.shards[1].dataPath`.
    pub field: String,
    pub message: String,
}

/// A host directory a shard opens, for the shared-directory check.
pub struct HostPath {
    pub node: String,
    pub path: String,
    /// Who uses it, e.g. `social shard 0`.
    pub owner: String,
}

/// Directories a cluster's shards open.
pub fn host_paths(cluster: &str, spec: &SearchClusterSpec) -> Vec<HostPath> {
    spec.shards
        .iter()
        .enumerate()
        .flat_map(|(i, shard)| {
            shard_paths(shard).map(|path| HostPath {
                node: shard.node.clone(),
                path: normalize_path(&path),
                owner: format!("{cluster} shard {i}"),
            })
        })
        .collect()
}

#[derive(Default)]
struct Errors(Vec<FieldError>);

impl Errors {
    fn push(&mut self, field: impl Into<String>, message: impl Into<String>) {
        self.0.push(FieldError {
            field: field.into(),
            message: message.into(),
        });
    }
}

/// Validate a cluster against its fixed routing (`pinned`, set once the index
/// exists) and the directories of the other clusters in the namespace.
pub fn validate(
    cluster: &str,
    spec: &SearchClusterSpec,
    pinned: Option<&str>,
    others: &[HostPath],
) -> Vec<FieldError> {
    let mut errors = Errors::default();
    if let Err(message) = check_cluster_name(cluster) {
        errors.push("metadata.name", message);
    }
    validate_image(&mut errors, "spec.shard", &spec.shard.image);
    validate_image(&mut errors, "spec.broker", &spec.broker.image);
    if spec
        .schema
        .as_ref()
        .is_some_and(|schema| schema.trim().is_empty())
    {
        errors.push("spec.schema", "empty; omit it for an existing index");
    }
    validate_routing(&mut errors, cluster, spec, pinned);
    validate_shards(&mut errors, spec, others);
    validate_broker(&mut errors, spec);
    errors.0
}

fn validate_image(errors: &mut Errors, field: &str, image: &ImageSpec) {
    if image.image.trim().is_empty() {
        errors.push(format!("{field}.image"), "required");
    }
    if let Some(policy) = &image.image_pull_policy
        && !["Always", "IfNotPresent", "Never"].contains(&policy.as_str())
    {
        errors.push(
            format!("{field}.imagePullPolicy"),
            format!("'{policy}' is not Always, IfNotPresent or Never"),
        );
    }
}

fn validate_routing(
    errors: &mut Errors,
    cluster: &str,
    spec: &SearchClusterSpec,
    pinned: Option<&str>,
) {
    if spec.shards.is_empty() {
        errors.push("spec.shards", "at least one shard");
        return;
    }
    let rule = routing(cluster, spec);
    if let Err(message) = rule.validate() {
        errors.push("spec.shards", message);
        return;
    }
    let Some(pinned) = pinned else {
        return;
    };
    let fixed = placement_rules::parse(pinned).map(|rule| rule.routing());
    if fixed.as_ref().ok() != Some(&rule.routing()) {
        errors.push(
            "spec.shards",
            format!(
                "the index exists with routing {pinned}; shard count, order and weights \
                 decide where every primary key lives and cannot change. Create a new cluster \
                 to repartition"
            ),
        );
    }
}

fn validate_shards(errors: &mut Errors, spec: &SearchClusterSpec, others: &[HostPath]) {
    reserved_flags(
        errors,
        "spec.shard.args",
        &spec.shard.args,
        SHARD_RESERVED_FLAGS,
    );
    let server_flags: BTreeSet<&str> = flag_names(&spec.shard.args).collect();
    let mut claimed: Vec<HostPath> = Vec::new();
    for (i, shard) in spec.shards.iter().enumerate() {
        let field = format!("spec.shards[{i}]");
        if shard.node.trim().is_empty() {
            errors.push(format!("{field}.node"), "required");
        }
        reserved_flags(
            errors,
            &format!("{field}.args"),
            &shard.args,
            SHARD_RESERVED_FLAGS,
        );
        for flag in flag_names(&shard.args).filter(|flag| server_flags.contains(flag)) {
            errors.push(
                format!("{field}.args"),
                format!("{flag} is already set in spec.shard.args"),
            );
        }
        for (key, path) in ["dataPath", "cachePath"]
            .into_iter()
            .zip(shard_paths(shard))
        {
            if let Err(message) = check_path(&path) {
                errors.push(format!("{field}.{key}"), message);
                continue;
            }
            let claim = HostPath {
                node: shard.node.clone(),
                path: normalize_path(&path),
                owner: format!("{field}.{key}"),
            };
            if let Some(other) = claimed
                .iter()
                .chain(others)
                .find(|other| other.node == claim.node && paths_overlap(&other.path, &claim.path))
            {
                errors.push(
                    &claim.owner,
                    format!(
                        "'{path}' on node '{}' overlaps {}: two summa-server processes must \
                         never share a directory",
                        claim.node, other.owner
                    ),
                );
            }
            claimed.push(claim);
        }
    }
}

/// Absolute, without empty, `.` or `..` components, so string comparison
/// of two paths is a comparison of directories.
fn check_path(path: &str) -> Result<(), String> {
    let Some(relative) = path.strip_prefix('/') else {
        return Err(format!("'{path}' is not an absolute path"));
    };
    if relative
        .trim_end_matches('/')
        .split('/')
        .any(|part| part.is_empty() || part == "." || part == "..")
    {
        return Err(format!("'{path}' has an empty, '.' or '..' component"));
    }
    Ok(())
}

fn validate_broker(errors: &mut Errors, spec: &SearchClusterSpec) {
    let broker = &spec.broker;
    if broker.replicas < 1 {
        errors.push(
            "spec.broker.replicas",
            "at least 1: every request goes through the broker",
        );
    }
    if broker.service_account_name.trim().is_empty() {
        errors.push("spec.broker.serviceAccountName", "required");
    }
    if broker.backend_max_searches == 0 {
        errors.push("spec.broker.backendMaxSearches", "must be greater than 0");
    } else if i64::from(broker.backend_max_searches) < i64::from(broker.replicas) {
        errors.push(
            "spec.broker.backendMaxSearches",
            format!(
                "{} cannot be split across {} broker replicas",
                broker.backend_max_searches, broker.replicas
            ),
        );
    }
    reserved_flags(
        errors,
        "spec.broker.args",
        &broker.args,
        BROKER_RESERVED_FLAGS,
    );
}

fn reserved_flags(errors: &mut Errors, field: &str, args: &[String], reserved: &[&str]) {
    for flag in flag_names(args).filter(|flag| reserved.contains(flag)) {
        errors.push(field, format!("{flag} is set by the operator"));
    }
}

/// A DNS label short enough for the derived Deployment names; it also names
/// the index (`-` as `_`).
fn check_cluster_name(name: &str) -> Result<(), String> {
    let bytes = name.as_bytes();
    let valid = !bytes.is_empty()
        && bytes
            .iter()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || *b == b'-')
        && bytes[0].is_ascii_lowercase()
        && bytes[bytes.len() - 1].is_ascii_alphanumeric();
    if !valid || name.len() > MAX_NAME_LEN {
        return Err(format!(
            "'{name}' must be 1 to {MAX_NAME_LEN} lowercase letters, digits or '-', starting \
             with a letter and ending with a letter or digit"
        ));
    }
    Ok(())
}

pub fn normalize_path(path: &str) -> String {
    match path.trim_end_matches('/') {
        "" => "/".to_string(),
        trimmed => trimmed.to_string(),
    }
}

/// True when one directory is the other or contains it.
pub fn paths_overlap(a: &str, b: &str) -> bool {
    let contains = |outer: &str, inner: &str| {
        outer == inner
            || outer == "/"
            || inner
                .strip_prefix(outer)
                .is_some_and(|rest| rest.starts_with('/'))
    };
    contains(a, b) || contains(b, a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::tests::spec;

    fn fields(errors: &[FieldError]) -> Vec<&str> {
        errors.iter().map(|e| e.field.as_str()).collect()
    }

    #[test]
    fn sample_spec_is_valid() {
        assert_eq!(validate("documents-1", &spec(), None, &[]), []);
    }

    #[test]
    fn partitions_never_share_or_nest_a_directory_on_a_node() {
        let mut nested = spec();
        nested.shards[1].data_path = "/data2/summa/documents_1/inner".into();
        let errors = validate("documents-1", &nested, None, &[]);
        // Its default cache path nests in shard 0's directory too.
        assert_eq!(
            fields(&errors),
            ["spec.shards[1].dataPath", "spec.shards[1].cachePath"]
        );
        assert!(errors[0].message.contains("spec.shards[0].dataPath"));

        // Another cluster's directory on the same node is refused too; the
        // same path on another node is a different directory.
        let social = host_paths("social", &spec());
        let errors = validate("documents-2", &spec(), None, &social);
        assert!(errors[0].message.contains("social shard 0"));
        let mut elsewhere = spec();
        for shard in &mut elsewhere.shards {
            shard.node = "node-b".into();
        }
        assert_eq!(validate("documents-2", &elsewhere, None, &social), []);
    }

    #[test]
    fn routing_is_fixed_once_the_index_exists() {
        let pinned = Some("documents_1=0:5,1:4");
        assert_eq!(validate("documents-1", &spec(), pinned, &[]), []);
        // Scaling every weight keeps the routing.
        let mut scaled = spec();
        scaled.shards[0].weight = Some(10);
        scaled.shards[1].weight = Some(8);
        assert_eq!(validate("documents-1", &scaled, pinned, &[]), []);
        let mut reweighted = spec();
        reweighted.shards[1].weight = Some(5);
        assert_eq!(
            fields(&validate("documents-1", &reweighted, pinned, &[])),
            ["spec.shards"]
        );
        let mut grown = spec();
        grown.shards.push(grown.shards[0].clone());
        grown.shards[2].data_path = "/data4/summa/documents_1".into();
        assert_eq!(
            fields(&validate("documents-1", &grown, pinned, &[])),
            ["spec.shards"]
        );
        // Moving a shard keeps the routing.
        let mut moved = spec();
        moved.shards[0].node = "node-b".into();
        assert_eq!(validate("documents-1", &moved, pinned, &[]), []);
    }

    #[test]
    fn paths_are_compared_as_directories() {
        let mut spec = spec();
        spec.shards[1].data_path = "/data2//summa/documents_1".into();
        spec.shards[1].cache_path = Some("/data2/summa/../summa/x".into());
        assert_eq!(
            fields(&validate("documents-1", &spec, None, &[])),
            ["spec.shards[1].dataPath", "spec.shards[1].cachePath"]
        );
    }

    #[test]
    fn names_and_flags_are_checked() {
        assert_eq!(
            fields(&validate("Documents_1", &spec(), None, &[])),
            ["metadata.name"]
        );
        // A Service name starts with a letter.
        assert_eq!(
            fields(&validate("2026-docs", &spec(), None, &[])),
            ["metadata.name"]
        );
        let mut spec = spec();
        spec.shards[0].args = vec![
            "--data-dir=/x".into(),
            "--search-threads".into(),
            "4".into(),
        ];
        spec.shard.args = vec![
            "--max-concurrent-searches".into(),
            "8".into(),
            "--search-threads".into(),
            "8".into(),
        ];
        spec.broker.args = vec![
            "--placement".into(),
            "x=0".into(),
            "--backend-port=1".into(),
        ];
        assert_eq!(
            fields(&validate("documents-1", &spec, None, &[])),
            [
                "spec.shard.args",
                "spec.shards[0].args",
                "spec.shards[0].args",
                "spec.broker.args",
                "spec.broker.args"
            ]
        );
        let mut single = crate::render::tests::spec();
        single.shards.truncate(1);
        let errors = validate("documents-1", &single, None, &[]);
        assert!(errors[0].message.contains("multi-shard"));
    }
}
