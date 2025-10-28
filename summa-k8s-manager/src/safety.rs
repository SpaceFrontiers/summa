//! Checks of a proposed spec against the running cluster: placement changes
//! that would reroute live indexes, and shard removals that would drop
//! indexes. The operator gates its own actions on them and the console shows
//! them before saving.

use serde::Serialize;
use summa_broker::placement_rules::{self, RuleSpec};

use crate::broker::{Backend, LiveIndex};

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PlacementChange {
    pub index: String,
    /// Routing under the running rules (`shard[:weight],...`), if any.
    pub current: Option<String>,
    /// Routing under the proposed rules, if any.
    pub proposed: Option<String>,
    /// Why the change would break the index; `None` when it is safe.
    pub blocked: Option<String>,
    pub note: Option<String>,
}

/// Parse running `--placement` values, as the broker did at startup.
pub fn parse_rules(values: &[String]) -> anyhow::Result<Vec<RuleSpec>> {
    values.iter().map(|v| placement_rules::parse(v)).collect()
}

/// Live indexes whose routing differs between the running and proposed
/// rules. Partition order and weights are part of a partitioned generation's
/// routing contract; a single-shard rule only decides where the next
/// `CreateIndex` lands and which copy wins while an index is on two shards.
pub fn placement_changes(
    indexes: &[LiveIndex],
    current: &[RuleSpec],
    proposed: &[RuleSpec],
) -> Vec<PlacementChange> {
    let mut changes = Vec::new();
    for index in indexes {
        let old = placement_rules::resolve(current, &index.name).map(RuleSpec::routing);
        let new = placement_rules::resolve(proposed, &index.name).map(RuleSpec::routing);
        if old == new {
            continue;
        }
        let describe = |routing: &Option<placement_rules::Routing>| {
            routing
                .as_ref()
                .map_or("no rule".to_string(), ToString::to_string)
        };
        let (blocked, note) = if index.partitioned {
            (
                Some(format!(
                    "'{}' is partitioned as {}; under the new rules it would be routed as {}, \
                     so upserts and deletes would reach the wrong partition. Pin it with an \
                     exact rule that keeps its current routing first",
                    index.name,
                    describe(&old),
                    describe(&new),
                )),
                None,
            )
        } else if new.as_ref().is_some_and(|r| r.is_partitioned()) {
            (
                Some(format!(
                    "'{}' lives on shard {} only; a multi-shard rule would fail every read on \
                     the partitions that do not host it. Rebuild it under a new name instead",
                    index.name,
                    index.hosts.join(", "),
                )),
                None,
            )
        } else {
            let note = match &new {
                Some(routing) if !index.hosts.contains(&routing.shards[0]) => format!(
                    "stays on shard {}; only a new CreateIndex of '{}' lands on shard {}",
                    index.hosts.join(", "),
                    index.name,
                    routing.shards[0]
                ),
                Some(routing) => format!(
                    "reads and writes pin to shard {} while the index is on several shards",
                    routing.shards[0]
                ),
                None => format!("stays on shard {}", index.hosts.join(", ")),
            };
            (None, Some(note))
        };
        changes.push(PlacementChange {
            index: index.name.clone(),
            current: old.map(|r| r.to_string()),
            proposed: new.map(|r| r.to_string()),
            blocked,
            note,
        });
    }
    changes
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ShardRemoval {
    pub shard: String,
    pub indexes: Vec<String>,
    pub blocked: Option<String>,
}

/// Whether removing these shards would drop data the broker serves.
pub fn shard_removals(backends: &[Backend], removed: &[String]) -> Vec<ShardRemoval> {
    removed
        .iter()
        .map(|shard| {
            let members: Vec<&Backend> = backends.iter().filter(|b| &b.shard_id == shard).collect();
            let mut indexes: Vec<String> = members
                .iter()
                .flat_map(|b| b.indexes.iter().cloned())
                .collect();
            indexes.sort();
            indexes.dedup();
            let blocked = if !indexes.is_empty() {
                Some(format!(
                    "shard {shard} hosts {}: delete or move them first",
                    indexes.join(", ")
                ))
            } else {
                members
                    .iter()
                    .find(|b| !b.ready || b.health != "healthy")
                    .map(|unknown| {
                        format!(
                            "backend {} of shard {shard} is {} ({}), so the indexes it holds \
                             are unknown",
                            unknown.backend_id,
                            unknown.health,
                            if unknown.ready { "ready" } else { "not ready" }
                        )
                    })
            };
            ShardRemoval {
                shard: shard.clone(),
                indexes,
                blocked,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rules(values: &[&str]) -> Vec<RuleSpec> {
        parse_rules(&values.iter().map(|v| v.to_string()).collect::<Vec<_>>()).unwrap()
    }

    fn index(name: &str, hosts: &[&str]) -> LiveIndex {
        LiveIndex {
            name: name.into(),
            partitioned: hosts.len() > 1,
            hosts: hosts.iter().map(|h| h.to_string()).collect(),
        }
    }

    #[test]
    fn reweighting_a_family_is_refused_until_live_generations_are_pinned() {
        let live = [
            index("documents_20260923", &["2", "3", "4"]),
            index("social", &["1"]),
        ];
        let current = rules(&["documents*=2,3,4", "social*=1"]);

        let reweighted = rules(&["documents*=2:5,3:5,4:4", "social*=1"]);
        let changes = placement_changes(&live, &current, &reweighted);
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].current.as_deref(), Some("2:1,3:1,4:1"));
        assert_eq!(changes[0].proposed.as_deref(), Some("2:5,3:5,4:4"));
        assert!(changes[0].blocked.is_some());

        let pinned = rules(&[
            "documents_20260923=2,3,4",
            "documents*=2:5,3:5,4:4",
            "social*=1",
        ]);
        assert_eq!(placement_changes(&live, &current, &pinned), []);

        // Same ratio at another scale is the same routing.
        let rescaled = rules(&["documents*=2:3,3:3,4:3", "social*=1"]);
        assert_eq!(placement_changes(&live, &current, &rescaled), []);
    }

    #[test]
    fn moving_an_unpartitioned_family_is_allowed_and_partitioning_it_is_not() {
        let live = [index("social", &["1"])];
        let current = rules(&["social*=1"]);
        let moved = placement_changes(&live, &current, &rules(&["social*=5"]));
        assert_eq!(moved.len(), 1);
        assert!(moved[0].blocked.is_none());
        assert!(
            moved[0]
                .note
                .as_deref()
                .unwrap()
                .contains("stays on shard 1")
        );

        let partitioned = placement_changes(&live, &current, &rules(&["social*=1,5"]));
        assert!(partitioned[0].blocked.is_some());
    }

    #[test]
    fn dropping_the_rule_of_a_partitioned_index_is_refused() {
        let live = [index("documents_20260923", &["2", "3", "4"])];
        let changes = placement_changes(&live, &rules(&["documents*=2,3,4"]), &[]);
        assert_eq!(changes[0].proposed, None);
        assert!(changes[0].blocked.is_some());
    }

    fn backend(id: &str, shard: &str, indexes: &[&str], ready: bool, health: &str) -> Backend {
        Backend {
            backend_id: id.into(),
            address: format!("{id}:50051"),
            shard_id: shard.into(),
            role: "master".into(),
            health: health.into(),
            indexes: indexes.iter().map(|i| i.to_string()).collect(),
            index_map_age_ms: 1_000,
            consecutive_failures: 0,
            ready,
        }
    }

    #[test]
    fn a_shard_is_removable_only_when_it_is_known_to_be_empty() {
        let backends = [
            backend("s1", "1", &["social"], true, "healthy"),
            backend("s5", "5", &[], true, "healthy"),
            backend("s6", "6", &[], false, "healthy"),
        ];
        let removals = shard_removals(&backends, &["1".into(), "5".into(), "6".into(), "7".into()]);
        assert_eq!(removals[0].indexes, ["social"]);
        assert!(removals[0].blocked.is_some());
        assert!(removals[1].blocked.is_none());
        assert!(
            removals[2]
                .blocked
                .as_deref()
                .unwrap()
                .contains("not ready")
        );
        // No pod at all: nothing is served from it, its directory stays.
        assert!(removals[3].blocked.is_none());
    }
}
