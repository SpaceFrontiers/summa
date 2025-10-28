//! Static `SearchCluster` validation: everything that can be decided from the
//! spec alone. Live checks against the running broker are in `safety`.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;

use crate::crd::SearchClusterSpec;
use crate::render::{
    BROKER_RESERVED_FLAGS, SERVER_RESERVED_FLAGS, broker_name, flag_names, flag_value,
    replica_name, rule_spec, server_args,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FieldError {
    /// Path in the spec, e.g. `spec.shards[1].replicas[0].dataPath`.
    pub field: String,
    pub message: String,
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

pub fn validate(cluster: &str, spec: &SearchClusterSpec) -> Vec<FieldError> {
    let mut errors = Errors::default();
    if spec.image.trim().is_empty() {
        errors.push("spec.image", "required");
    }
    if let Some(policy) = &spec.image_pull_policy
        && !["Always", "IfNotPresent", "Never"].contains(&policy.as_str())
    {
        errors.push(
            "spec.imagePullPolicy",
            format!("'{policy}' is not Always, IfNotPresent or Never"),
        );
    }
    validate_reserved(
        &mut errors,
        "spec.server.args",
        &spec.server.args,
        SERVER_RESERVED_FLAGS,
    );
    validate_shards(&mut errors, cluster, spec);
    validate_broker(&mut errors, cluster, spec);
    validate_placement(&mut errors, spec);
    errors.0
}

fn validate_reserved(errors: &mut Errors, field: &str, args: &[String], reserved: &[&str]) {
    for flag in flag_names(args) {
        if reserved.contains(&flag) {
            errors.push(field, format!("{flag} is set by the operator"));
        }
    }
}

fn validate_shards(errors: &mut Errors, cluster: &str, spec: &SearchClusterSpec) {
    let mut ids = BTreeSet::new();
    let mut names: BTreeMap<String, String> = BTreeMap::new();
    names.insert(broker_name(cluster, spec), "spec.broker.name".into());
    // (node, path) -> field that claimed it.
    let mut paths: Vec<(String, String, String)> = Vec::new();
    let server_flags: BTreeSet<&str> = flag_names(&spec.server.args).into_iter().collect();

    for (i, shard) in spec.shards.iter().enumerate() {
        let field = format!("spec.shards[{i}]");
        if let Err(message) = check_label_value(&shard.id) {
            errors.push(format!("{field}.id"), message);
        } else if shard.id.contains([',', ':', '=']) {
            errors.push(
                format!("{field}.id"),
                "must not contain ',', ':' or '=' (placement rule syntax)",
            );
        }
        if !ids.insert(shard.id.as_str()) {
            errors.push(
                format!("{field}.id"),
                format!("duplicate shard id '{}'", shard.id),
            );
        }
        validate_reserved(
            errors,
            &format!("{field}.args"),
            &shard.args,
            SERVER_RESERVED_FLAGS,
        );
        for flag in flag_names(&shard.args) {
            if server_flags.contains(flag) {
                errors.push(
                    format!("{field}.args"),
                    format!("{flag} is already set in spec.server.args"),
                );
            }
        }
        let backend_budget = spec.broker.backend_max_searches;
        if let Some(value) = flag_value(&server_args(spec, shard), "--max-concurrent-searches") {
            match value.parse::<u32>() {
                Ok(limit) if limit < backend_budget => errors.push(
                    format!("{field}.args"),
                    format!(
                        "--max-concurrent-searches {limit} is below spec.broker.backendMaxSearches \
                         {backend_budget}: the broker would admit searches the server refuses"
                    ),
                ),
                Ok(_) => {}
                Err(_) => errors.push(
                    format!("{field}.args"),
                    format!("--max-concurrent-searches '{value}' is not a number"),
                ),
            }
        }
        match shard.replicas.len() {
            0 => errors.push(format!("{field}.replicas"), "a shard needs one replica"),
            1 => {}
            _ => errors.push(
                format!("{field}.replicas"),
                "exactly one replica per shard: summa-server has no follower replication yet, \
                 so another replica would serve data that never receives writes",
            ),
        }
        for (j, replica) in shard.replicas.iter().enumerate() {
            let field = format!("{field}.replicas[{j}]");
            let name = replica_name(cluster, shard, replica);
            if let Err(message) = check_dns_label(&name) {
                errors.push(
                    format!("{field}.name"),
                    format!("Deployment name '{name}': {message}"),
                );
            }
            if let Some(previous) = names.insert(name.clone(), format!("{field}.name")) {
                errors.push(
                    format!("{field}.name"),
                    format!("Deployment name '{name}' is already used by {previous}"),
                );
            }
            if replica.node.trim().is_empty() {
                errors.push(format!("{field}.node"), "required");
            }
            for (key, path) in [
                ("dataPath", &replica.data_path),
                ("cachePath", &replica.cache_path),
            ] {
                let path_field = format!("{field}.{key}");
                if !path.starts_with('/') {
                    errors.push(path_field, format!("'{path}' is not an absolute path"));
                    continue;
                }
                let normalized = normalize_path(path);
                if let Some((_, _, other)) = paths.iter().find(|(node, other_path, _)| {
                    node == &replica.node && paths_overlap(other_path, &normalized)
                }) {
                    errors.push(
                        path_field.clone(),
                        format!(
                            "'{path}' on node '{}' overlaps {other}: two summa-server \
                             processes must never share a directory",
                            replica.node
                        ),
                    );
                }
                paths.push((replica.node.clone(), normalized, path_field));
            }
        }
    }
}

fn validate_broker(errors: &mut Errors, cluster: &str, spec: &SearchClusterSpec) {
    let broker = &spec.broker;
    if let Err(message) = check_dns_label(&broker_name(cluster, spec)) {
        errors.push("spec.broker.name", message);
    }
    if broker
        .image
        .as_ref()
        .is_some_and(|image| image.trim().is_empty())
    {
        errors.push("spec.broker.image", "empty; omit it to use spec.image");
    }
    if broker.replicas < 1 {
        errors.push(
            "spec.broker.replicas",
            "at least 1: every search goes through the broker",
        );
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
    validate_reserved(
        errors,
        "spec.broker.args",
        &broker.args,
        BROKER_RESERVED_FLAGS,
    );
}

fn validate_placement(errors: &mut Errors, spec: &SearchClusterSpec) {
    let shards: BTreeSet<&str> = spec.shards.iter().map(|s| s.id.as_str()).collect();
    let mut patterns: BTreeMap<&str, usize> = BTreeMap::new();
    for (i, rule) in spec.placement.rules.iter().enumerate() {
        let field = format!("spec.placement.rules[{i}]");
        if let Err(message) = rule_spec(rule).validate() {
            errors.push(field.clone(), message);
        }
        if let Some(first) = patterns.insert(rule.pattern.as_str(), i) {
            errors.push(
                format!("{field}.pattern"),
                format!(
                    "'{}' never matches: rules[{first}] has the same pattern",
                    rule.pattern
                ),
            );
            patterns.insert(rule.pattern.as_str(), first);
        }
        for (j, partition) in rule.partitions.iter().enumerate() {
            if !shards.contains(partition.shard.as_str()) {
                errors.push(
                    format!("{field}.partitions[{j}].shard"),
                    format!("shard '{}' is not in spec.shards", partition.shard),
                );
            }
        }
    }
}

/// Kubernetes label value: at most 63 characters, alphanumeric at both ends,
/// `-`, `_` and `.` inside.
fn check_label_value(value: &str) -> Result<(), String> {
    if value.is_empty() {
        return Err("required".into());
    }
    if value.len() > 63 {
        return Err(format!("'{value}' is longer than 63 characters"));
    }
    let bytes = value.as_bytes();
    let valid_inner = bytes
        .iter()
        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'));
    if !valid_inner
        || !bytes[0].is_ascii_alphanumeric()
        || !bytes[bytes.len() - 1].is_ascii_alphanumeric()
    {
        return Err(format!(
            "'{value}' must be alphanumeric with '-', '_' or '.' inside"
        ));
    }
    Ok(())
}

/// RFC 1123 label, as Deployment and Service names require.
fn check_dns_label(value: &str) -> Result<(), String> {
    if value.is_empty() || value.len() > 63 {
        return Err("must be 1 to 63 characters".into());
    }
    let bytes = value.as_bytes();
    if !bytes
        .iter()
        .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || *b == b'-')
        || bytes[0] == b'-'
        || bytes[bytes.len() - 1] == b'-'
    {
        return Err(
            "must be lowercase alphanumeric or '-', starting and ending alphanumeric".into(),
        );
    }
    Ok(())
}

pub fn normalize_path(path: &str) -> String {
    let trimmed = path.trim_end_matches('/');
    if trimmed.is_empty() {
        "/".to_string()
    } else {
        trimmed.to_string()
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
    use crate::crd::{PartitionRef, PlacementRuleSpec, ReplicaSpec};
    use crate::render::tests::sample_spec;

    fn fields(errors: &[FieldError]) -> Vec<&str> {
        errors.iter().map(|e| e.field.as_str()).collect()
    }

    #[test]
    fn sample_spec_is_valid() {
        assert_eq!(validate("summa", &sample_spec()), []);
    }

    #[test]
    fn two_replicas_sharing_or_nesting_a_directory_on_one_node_are_refused() {
        let mut spec = sample_spec();
        spec.shards[1].replicas[0].data_path = "/data2/summa-server/".into();
        let errors = validate("summa", &spec);
        assert_eq!(fields(&errors), ["spec.shards[1].replicas[0].dataPath"]);
        assert!(
            errors[0]
                .message
                .contains("spec.shards[0].replicas[0].dataPath")
        );

        let mut spec = sample_spec();
        spec.shards[1].replicas[0].cache_path = "/data2".into();
        assert_eq!(
            fields(&validate("summa", &spec)),
            ["spec.shards[1].replicas[0].cachePath"]
        );

        // The same directory on another node is a different directory.
        let mut spec = sample_spec();
        spec.shards[1].replicas[0].node = "node-b".into();
        spec.shards[1].replicas[0].data_path = "/data2/summa-server".into();
        spec.shards[1].replicas[0].cache_path = "/data2/summa-cache".into();
        assert_eq!(validate("summa", &spec), []);
    }

    #[test]
    fn follower_replicas_are_refused_until_the_server_replicates() {
        let mut spec = sample_spec();
        spec.shards[0].replicas.push(ReplicaSpec {
            name: Some("summa-server-s2-b".into()),
            node: "node-b".into(),
            data_path: "/data/summa-server".into(),
            cache_path: "/data/summa-cache".into(),
        });
        assert_eq!(
            fields(&validate("summa", &spec)),
            ["spec.shards[0].replicas"]
        );
        spec.shards[0].replicas.clear();
        assert_eq!(
            fields(&validate("summa", &spec)),
            ["spec.shards[0].replicas"]
        );
    }

    #[test]
    fn placement_must_name_existing_shards_and_reachable_patterns() {
        let mut spec = sample_spec();
        spec.placement.rules.push(PlacementRuleSpec {
            pattern: "social*".into(),
            partitions: vec![PartitionRef {
                shard: "9".into(),
                weight: None,
            }],
        });
        spec.placement.rules.push(spec.placement.rules[0].clone());
        assert_eq!(
            fields(&validate("summa", &spec)),
            [
                "spec.placement.rules[1].partitions[0].shard",
                "spec.placement.rules[2].pattern"
            ]
        );
        let mut spec = sample_spec();
        spec.placement.rules[0].partitions.truncate(1);
        let errors = validate("summa", &spec);
        assert_eq!(fields(&errors), ["spec.placement.rules[0]"]);
        assert!(errors[0].message.contains("multi-shard"));
    }

    #[test]
    fn operator_owned_and_duplicated_flags_are_refused() {
        let mut spec = sample_spec();
        spec.shards[0].args = vec![
            "--data-dir=/x".into(),
            "--max-concurrent-searches".into(),
            "32".into(),
        ];
        spec.broker.args = vec!["--placement".into(), "x=1".into()];
        assert_eq!(
            fields(&validate("summa", &spec)),
            [
                "spec.shards[0].args",
                "spec.shards[0].args",
                "spec.broker.args"
            ]
        );
    }

    #[test]
    fn broker_search_budget_must_cover_replicas_and_fit_the_servers() {
        let mut spec = sample_spec();
        spec.broker.backend_max_searches = 32;
        let errors = validate("summa", &spec);
        // Both shards accept 16 concurrent searches.
        assert_eq!(
            fields(&errors),
            ["spec.shards[0].args", "spec.shards[1].args"]
        );
        let mut spec = sample_spec();
        spec.broker.replicas = 17;
        assert_eq!(
            fields(&validate("summa", &spec)),
            ["spec.broker.backendMaxSearches"]
        );
    }

    #[test]
    fn names_and_ids_are_kubernetes_safe() {
        let mut spec = sample_spec();
        spec.shards[0].id = "a,b".into();
        spec.shards[1].replicas[0].name = Some("Shard_3".into());
        spec.placement.rules.clear();
        assert_eq!(
            fields(&validate("summa", &spec)),
            ["spec.shards[0].id", "spec.shards[1].replicas[0].name"]
        );
        let mut spec = sample_spec();
        spec.shards[1].replicas[0].name = Some("summa-broker".into());
        assert_eq!(
            fields(&validate("summa", &spec)),
            ["spec.shards[1].replicas[0].name"]
        );
    }
}
