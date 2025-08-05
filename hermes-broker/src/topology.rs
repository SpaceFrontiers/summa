//! Topology model: discovered backends × learned index maps → routing.
//!
//! The broker never configures which backend hosts which index — it learns
//! that by polling `ListIndexes` per backend (see `poller`). What IS
//! configured is shard identity (a pod label in kubernetes discovery) and
//! placement rules for `CreateIndex` / migration tiebreaks.
//!
//! Snapshots are immutable and swapped atomically (`arc_swap`) so request
//! handlers route against a consistent view with no locks on the hot path.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::time::Instant;

use tonic::Status;

use crate::placement::{PlacementDefault, PlacementRules};

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BackendId(pub String);

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ShardId(pub String);

/// Replica role within a shard. `None` in `DiscoveredEndpoint.role` means
/// unlabeled: a shard whose only member is unlabeled treats that member as
/// the implicit master (today's single-pod-per-shard world needs no labels).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Master,
    Follower,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Health {
    Healthy,
    /// Recently failed or never successfully polled; still eligible for reads
    /// as a last resort (off its last-known index map) until the grace period
    /// evicts it.
    Suspect,
    /// Out of rotation entirely.
    Evicted,
}

impl Health {
    pub fn as_str(&self) -> &'static str {
        match self {
            Health::Healthy => "healthy",
            Health::Suspect => "suspect",
            Health::Evicted => "evicted",
        }
    }
}

/// One discovered hermes-server endpoint (a pod in kubernetes mode, a
/// configured backend in static mode).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiscoveredEndpoint {
    pub id: BackendId,
    /// `host:port`, plaintext h2c — same addressing hermes clients use.
    pub addr: String,
    pub shard: ShardId,
    pub role: Option<Role>,
    /// Pod readiness; unready endpoints are visible in the admin surface but
    /// never routed to and never polled.
    pub ready: bool,
}

/// Poll-time knowledge about one backend.
#[derive(Clone, Debug)]
pub struct BackendKnowledge {
    pub health: Health,
    /// Last successful `ListIndexes`, sorted.
    pub indexes: Vec<String>,
    pub last_index_refresh: Option<Instant>,
    pub consecutive_failures: u64,
}

impl Default for BackendKnowledge {
    fn default() -> Self {
        Self {
            // A backend we have never successfully polled must not be treated
            // as healthy; it also advertises no indexes, so it cannot be
            // routed to until its first successful poll promotes it.
            health: Health::Suspect,
            indexes: Vec::new(),
            last_index_refresh: None,
            consecutive_failures: 0,
        }
    }
}

/// A backend as the snapshot sees it: discovery identity + poll knowledge.
#[derive(Clone, Debug)]
pub struct Backend {
    pub endpoint: DiscoveredEndpoint,
    pub know: BackendKnowledge,
}

impl Backend {
    /// Eligible to carry traffic at all (Suspect still qualifies — grace).
    pub fn routable(&self) -> bool {
        self.endpoint.ready && self.know.health != Health::Evicted
    }
}

#[derive(Clone, Debug, Default)]
pub struct ShardGroup {
    /// Sorted member ids (all discovered members, any health/readiness).
    pub members: Vec<BackendId>,
    /// Members explicitly labeled master.
    pub labeled_masters: Vec<BackendId>,
}

#[derive(Clone, Debug)]
pub struct IndexRoute {
    /// Shard ids on which routable backends advertise this index (sorted).
    pub shards: BTreeSet<ShardId>,
    /// Placement rule match, if any.
    pub ruled_shard: Option<ShardId>,
}

impl IndexRoute {
    /// Seen on several shards with no rule pinning one — migration transient.
    pub fn ambiguous(&self) -> bool {
        self.shards.len() > 1
            && !self
                .ruled_shard
                .as_ref()
                .is_some_and(|r| self.shards.contains(r))
    }

    /// The shard a read goes to. Ambiguous routes resolve to the
    /// lexicographically-first shard so behavior stays deterministic during
    /// migrations; the bool reports the ambiguity for metrics.
    fn read_shard(&self) -> (&ShardId, bool) {
        if let Some(ruled) = &self.ruled_shard
            && let Some(shard) = self.shards.get(ruled)
        {
            return (shard, false);
        }
        let first = self.shards.iter().next().expect("route has >=1 shard");
        (first, self.shards.len() > 1)
    }
}

#[derive(Clone, Debug, Default)]
pub struct TopologySnapshot {
    pub backends: BTreeMap<BackendId, Backend>,
    pub shards: BTreeMap<ShardId, ShardGroup>,
    pub indexes: BTreeMap<String, IndexRoute>,
}

/// Outcome of read-path backend selection.
#[derive(Debug)]
pub struct ReadSelection<'a> {
    pub backend: &'a Backend,
    /// Selected backend is Suspect: serving off a possibly-stale index map.
    pub stale: bool,
    /// Route was ambiguous (multi-shard, no rule) and a deterministic shard
    /// was picked.
    pub ambiguous: bool,
}

impl TopologySnapshot {
    /// Build a snapshot from discovered endpoints, per-backend poll
    /// knowledge, and placement rules. Endpoints missing from `knowledge`
    /// get `BackendKnowledge::default()` (Suspect, no indexes).
    pub fn assemble(
        endpoints: &[DiscoveredEndpoint],
        knowledge: &HashMap<BackendId, BackendKnowledge>,
        placement: &PlacementRules,
    ) -> Self {
        let mut backends = BTreeMap::new();
        let mut shards: BTreeMap<ShardId, ShardGroup> = BTreeMap::new();
        for ep in endpoints {
            let know = knowledge.get(&ep.id).cloned().unwrap_or_default();
            let group = shards.entry(ep.shard.clone()).or_default();
            group.members.push(ep.id.clone());
            if ep.role == Some(Role::Master) {
                group.labeled_masters.push(ep.id.clone());
            }
            backends.insert(
                ep.id.clone(),
                Backend {
                    endpoint: ep.clone(),
                    know,
                },
            );
        }
        for group in shards.values_mut() {
            group.members.sort();
            group.labeled_masters.sort();
        }

        let mut indexes: BTreeMap<String, IndexRoute> = BTreeMap::new();
        for backend in backends.values().filter(|b| b.routable()) {
            for name in &backend.know.indexes {
                indexes
                    .entry(name.clone())
                    .or_insert_with(|| IndexRoute {
                        shards: BTreeSet::new(),
                        ruled_shard: placement.shard_for(name).cloned(),
                    })
                    .shards
                    .insert(backend.endpoint.shard.clone());
            }
        }

        Self {
            backends,
            shards,
            indexes,
        }
    }

    pub fn any_healthy(&self) -> bool {
        self.backends
            .values()
            .any(|b| b.endpoint.ready && b.know.health == Health::Healthy)
    }

    /// Union of index names across routable backends.
    pub fn all_index_names(&self) -> Vec<String> {
        self.indexes.keys().cloned().collect()
    }

    fn route(&self, index_name: &str) -> Result<&IndexRoute, Status> {
        self.indexes.get(index_name).ok_or_else(|| {
            Status::not_found(format!(
                "index '{index_name}' is not present on any healthy backend"
            ))
        })
    }

    /// Pick the backend serving a read for `index_name`. `rotation` spreads
    /// load across replicas of the shard (callers pass an incrementing
    /// counter; with one replica it is a no-op).
    pub fn select_read_backend(
        &self,
        index_name: &str,
        rotation: usize,
    ) -> Result<ReadSelection<'_>, Status> {
        let route = self.route(index_name)?;
        let (shard, ambiguous) = route.read_shard();
        let group = self.shards.get(shard).ok_or_else(|| {
            Status::internal(format!("shard '{}' vanished from snapshot", shard.0))
        })?;
        let candidates: Vec<&Backend> = group
            .members
            .iter()
            .filter_map(|id| self.backends.get(id))
            .filter(|b| b.routable() && b.know.indexes.iter().any(|n| n == index_name))
            .collect();
        let healthy: Vec<&&Backend> = candidates
            .iter()
            .filter(|b| b.know.health == Health::Healthy)
            .collect();
        if let Some(b) = healthy.get(rotation % healthy.len().max(1)).copied() {
            return Ok(ReadSelection {
                backend: b,
                stale: false,
                ambiguous,
            });
        }
        // Grace path: no healthy replica; better a possibly-stale answer from
        // a Suspect backend than none.
        if let Some(b) = candidates.first() {
            return Ok(ReadSelection {
                backend: b,
                stale: true,
                ambiguous,
            });
        }
        Err(Status::unavailable(format!(
            "index '{index_name}': no routable replica on shard '{}'",
            shard.0
        )))
    }

    /// The unique master of a shard, or why there is none. A single unlabeled
    /// member is the implicit master; with several members, exactly one must
    /// be labeled master.
    pub fn shard_master(&self, shard: &ShardId) -> Result<&Backend, Status> {
        let group = self
            .shards
            .get(shard)
            .ok_or_else(|| Status::failed_precondition(format!("unknown shard '{}'", shard.0)))?;
        let master_id = match group.labeled_masters.as_slice() {
            [only] => only,
            [] => {
                if let [only] = group.members.as_slice() {
                    let backend = self.backends.get(only).expect("member is a backend");
                    if backend.endpoint.role == Some(Role::Follower) {
                        return Err(Status::failed_precondition(format!(
                            "shard '{}' has no master: its only member is labeled follower",
                            shard.0
                        )));
                    }
                    only
                } else {
                    return Err(Status::failed_precondition(format!(
                        "shard '{}' has no master among {} members; label exactly one pod \
                         with role=master",
                        shard.0,
                        group.members.len()
                    )));
                }
            }
            many => {
                return Err(Status::failed_precondition(format!(
                    "shard '{}' has {} masters; writes refused until exactly one remains",
                    shard.0,
                    many.len()
                )));
            }
        };
        Ok(self.backends.get(master_id).expect("master is a backend"))
    }

    /// Pick the backend for a write RPC against an existing index: the master
    /// of the shard hosting it. An index on several shards requires a
    /// placement rule pinning the writable shard.
    pub fn select_write_backend(&self, index_name: &str) -> Result<&Backend, Status> {
        let route = self.route(index_name)?;
        let shard = if route.shards.len() == 1 {
            route.shards.iter().next().expect("len checked")
        } else if let Some(ruled) = &route.ruled_shard
            && route.shards.contains(ruled)
        {
            route.shards.get(ruled).expect("contains checked")
        } else {
            return Err(Status::failed_precondition(format!(
                "index '{index_name}' exists on {} shards with no placement rule pinning the \
                 writable one; add --placement",
                route.shards.len()
            )));
        };
        let master = self.shard_master(shard)?;
        if !master.routable() {
            return Err(Status::unavailable(format!(
                "index '{index_name}': shard '{}' master '{}' is unavailable",
                shard.0, master.endpoint.id.0
            )));
        }
        Ok(master)
    }

    /// Pick the backend for CreateIndex: the master of the placement-ruled
    /// shard, or the default policy's shard for unmatched names.
    pub fn select_create_backend(
        &self,
        index_name: &str,
        placement: &PlacementRules,
    ) -> Result<&Backend, Status> {
        let shard = match placement.shard_for(index_name) {
            Some(ruled) => {
                if !self.shards.contains_key(ruled) {
                    return Err(Status::failed_precondition(format!(
                        "placement rule for '{index_name}' targets shard '{}' but no backend \
                         carries that shard id",
                        ruled.0
                    )));
                }
                ruled.clone()
            }
            None => match placement.default {
                PlacementDefault::Reject => {
                    return Err(Status::failed_precondition(format!(
                        "no placement rule matches index '{index_name}' and \
                         --placement-default is 'reject'"
                    )));
                }
                PlacementDefault::Single => {
                    // Shard hosting the fewest indexes; lexicographic tiebreak
                    // (BTreeMap iteration order makes the first minimum win).
                    let mut counts: BTreeMap<&ShardId, usize> =
                        self.shards.keys().map(|s| (s, 0)).collect();
                    for route in self.indexes.values() {
                        for shard in &route.shards {
                            if let Some(c) = counts.get_mut(shard) {
                                *c += 1;
                            }
                        }
                    }
                    counts
                        .into_iter()
                        .min_by_key(|(_, c)| *c)
                        .map(|(s, _)| s.clone())
                        .ok_or_else(|| {
                            Status::unavailable("no backends discovered; cannot place index")
                        })?
                }
            },
        };
        let master = self.shard_master(&shard)?;
        if !master.routable() {
            return Err(Status::unavailable(format!(
                "shard '{}' master '{}' is unavailable",
                shard.0, master.endpoint.id.0
            )));
        }
        Ok(master)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placement::{PlacementDefault, PlacementRules, parse_placement};

    fn ep(id: &str, shard: &str, role: Option<Role>, ready: bool) -> DiscoveredEndpoint {
        DiscoveredEndpoint {
            id: BackendId(id.to_string()),
            addr: format!("10.0.0.{}:50051", id.len()),
            shard: ShardId(shard.to_string()),
            role,
            ready,
        }
    }

    fn know(health: Health, indexes: &[&str]) -> BackendKnowledge {
        BackendKnowledge {
            health,
            indexes: indexes.iter().map(|s| s.to_string()).collect(),
            last_index_refresh: Some(Instant::now()),
            consecutive_failures: 0,
        }
    }

    fn no_rules() -> PlacementRules {
        PlacementRules::new(vec![], PlacementDefault::Single)
    }

    fn rules(specs: &[&str]) -> PlacementRules {
        PlacementRules::new(
            specs.iter().map(|s| parse_placement(s).unwrap()).collect(),
            PlacementDefault::Single,
        )
    }

    #[test]
    fn single_shard_routes_reads_and_writes_to_its_only_member() {
        let eps = vec![ep("a", "0", None, true)];
        let mut k = HashMap::new();
        k.insert(
            BackendId("a".into()),
            know(Health::Healthy, &["documents", "social"]),
        );
        let snap = TopologySnapshot::assemble(&eps, &k, &no_rules());

        let read = snap.select_read_backend("documents", 0).unwrap();
        assert_eq!(read.backend.endpoint.id.0, "a");
        assert!(!read.stale);
        assert!(!read.ambiguous);
        assert_eq!(
            snap.select_write_backend("social").unwrap().endpoint.id.0,
            "a"
        );
        assert_eq!(snap.all_index_names(), vec!["documents", "social"]);
    }

    #[test]
    fn unknown_index_is_not_found() {
        let eps = vec![ep("a", "0", None, true)];
        let mut k = HashMap::new();
        k.insert(BackendId("a".into()), know(Health::Healthy, &["documents"]));
        let snap = TopologySnapshot::assemble(&eps, &k, &no_rules());
        let err = snap.select_read_backend("nope", 0).unwrap_err();
        assert_eq!(err.code(), tonic::Code::NotFound);
        let err = snap.select_write_backend("nope").unwrap_err();
        assert_eq!(err.code(), tonic::Code::NotFound);
    }

    #[test]
    fn never_polled_backend_serves_nothing() {
        let eps = vec![ep("a", "0", None, true)];
        let snap = TopologySnapshot::assemble(&eps, &HashMap::new(), &no_rules());
        assert!(snap.indexes.is_empty());
        assert!(!snap.any_healthy());
    }

    #[test]
    fn suspect_backend_serves_reads_as_stale_last_resort() {
        let eps = vec![ep("a", "0", None, true)];
        let mut k = HashMap::new();
        k.insert(BackendId("a".into()), know(Health::Suspect, &["documents"]));
        let snap = TopologySnapshot::assemble(&eps, &k, &no_rules());
        let read = snap.select_read_backend("documents", 0).unwrap();
        assert!(read.stale);
    }

    #[test]
    fn evicted_backend_drops_out_of_routes() {
        let eps = vec![ep("a", "0", None, true)];
        let mut k = HashMap::new();
        k.insert(BackendId("a".into()), know(Health::Evicted, &["documents"]));
        let snap = TopologySnapshot::assemble(&eps, &k, &no_rules());
        assert_eq!(
            snap.select_read_backend("documents", 0).unwrap_err().code(),
            tonic::Code::NotFound
        );
    }

    #[test]
    fn unready_pod_is_not_routed() {
        let eps = vec![ep("a", "0", None, false)];
        let mut k = HashMap::new();
        k.insert(BackendId("a".into()), know(Health::Healthy, &["documents"]));
        let snap = TopologySnapshot::assemble(&eps, &k, &no_rules());
        assert!(snap.indexes.is_empty());
    }

    #[test]
    fn ambiguous_index_reads_pick_first_shard_and_report() {
        let eps = vec![ep("a", "0", None, true), ep("bb", "1", None, true)];
        let mut k = HashMap::new();
        k.insert(BackendId("a".into()), know(Health::Healthy, &["social"]));
        k.insert(BackendId("bb".into()), know(Health::Healthy, &["social"]));
        let snap = TopologySnapshot::assemble(&eps, &k, &no_rules());

        let read = snap.select_read_backend("social", 0).unwrap();
        assert!(read.ambiguous);
        assert_eq!(read.backend.endpoint.shard.0, "0"); // lexicographic-first

        let err = snap.select_write_backend("social").unwrap_err();
        assert_eq!(err.code(), tonic::Code::FailedPrecondition);
    }

    #[test]
    fn placement_rule_resolves_migration_ambiguity() {
        let eps = vec![ep("a", "0", None, true), ep("bb", "1", None, true)];
        let mut k = HashMap::new();
        k.insert(BackendId("a".into()), know(Health::Healthy, &["social"]));
        k.insert(BackendId("bb".into()), know(Health::Healthy, &["social"]));
        let snap = TopologySnapshot::assemble(&eps, &k, &rules(&["social*=1"]));

        let read = snap.select_read_backend("social", 0).unwrap();
        assert!(!read.ambiguous);
        assert_eq!(read.backend.endpoint.shard.0, "1");
        assert_eq!(
            snap.select_write_backend("social")
                .unwrap()
                .endpoint
                .shard
                .0,
            "1"
        );
    }

    #[test]
    fn rule_pointing_at_shard_without_the_index_falls_back_to_reality() {
        // Rule says social*=1 but the index only exists on shard 0 (pre-copy
        // migration state): reads and writes go where the data is.
        let eps = vec![ep("a", "0", None, true), ep("bb", "1", None, true)];
        let mut k = HashMap::new();
        k.insert(BackendId("a".into()), know(Health::Healthy, &["social"]));
        k.insert(BackendId("bb".into()), know(Health::Healthy, &[]));
        let snap = TopologySnapshot::assemble(&eps, &k, &rules(&["social*=1"]));
        assert_eq!(
            snap.select_read_backend("social", 0)
                .unwrap()
                .backend
                .endpoint
                .shard
                .0,
            "0"
        );
        assert_eq!(
            snap.select_write_backend("social")
                .unwrap()
                .endpoint
                .shard
                .0,
            "0"
        );
    }

    #[test]
    fn multi_member_shard_requires_exactly_one_labeled_master() {
        let eps = vec![
            ep("a", "0", Some(Role::Master), true),
            ep("bb", "0", Some(Role::Follower), true),
        ];
        let mut k = HashMap::new();
        k.insert(BackendId("a".into()), know(Health::Healthy, &["documents"]));
        k.insert(
            BackendId("bb".into()),
            know(Health::Healthy, &["documents"]),
        );
        let snap = TopologySnapshot::assemble(&eps, &k, &no_rules());
        assert_eq!(
            snap.select_write_backend("documents")
                .unwrap()
                .endpoint
                .id
                .0,
            "a"
        );

        // No labels at all on a two-member shard: zero masters, writes refused.
        let eps = vec![ep("a", "0", None, true), ep("bb", "0", None, true)];
        let snap = TopologySnapshot::assemble(&eps, &k, &no_rules());
        let err = snap.select_write_backend("documents").unwrap_err();
        assert_eq!(err.code(), tonic::Code::FailedPrecondition);
    }

    #[test]
    fn sole_member_labeled_follower_is_not_an_implicit_master() {
        let eps = vec![ep("a", "0", Some(Role::Follower), true)];
        let mut k = HashMap::new();
        k.insert(BackendId("a".into()), know(Health::Healthy, &["documents"]));
        let snap = TopologySnapshot::assemble(&eps, &k, &no_rules());
        let err = snap.select_write_backend("documents").unwrap_err();
        assert_eq!(err.code(), tonic::Code::FailedPrecondition);
    }

    #[test]
    fn reads_rotate_across_healthy_replicas() {
        let eps = vec![
            ep("a", "0", Some(Role::Master), true),
            ep("bb", "0", Some(Role::Follower), true),
        ];
        let mut k = HashMap::new();
        k.insert(BackendId("a".into()), know(Health::Healthy, &["documents"]));
        k.insert(
            BackendId("bb".into()),
            know(Health::Healthy, &["documents"]),
        );
        let snap = TopologySnapshot::assemble(&eps, &k, &no_rules());
        let first = snap
            .select_read_backend("documents", 0)
            .unwrap()
            .backend
            .endpoint
            .id
            .0
            .clone();
        let second = snap
            .select_read_backend("documents", 1)
            .unwrap()
            .backend
            .endpoint
            .id
            .0
            .clone();
        assert_ne!(first, second);
    }

    #[test]
    fn create_follows_placement_rule_and_validates_shard_exists() {
        let eps = vec![ep("a", "0", None, true), ep("bb", "1", None, true)];
        let mut k = HashMap::new();
        k.insert(BackendId("a".into()), know(Health::Healthy, &["documents"]));
        k.insert(BackendId("bb".into()), know(Health::Healthy, &[]));
        let snap = TopologySnapshot::assemble(&eps, &k, &rules(&["social*=1"]));

        let placement = rules(&["social*=1"]);
        assert_eq!(
            snap.select_create_backend("social_20260810", &placement)
                .unwrap()
                .endpoint
                .shard
                .0,
            "1"
        );

        let bad = rules(&["social*=9"]);
        let err = snap.select_create_backend("social_x", &bad).unwrap_err();
        assert_eq!(err.code(), tonic::Code::FailedPrecondition);
    }

    #[test]
    fn create_default_single_picks_least_loaded_shard() {
        let eps = vec![ep("a", "0", None, true), ep("bb", "1", None, true)];
        let mut k = HashMap::new();
        k.insert(BackendId("a".into()), know(Health::Healthy, &["documents"]));
        k.insert(BackendId("bb".into()), know(Health::Healthy, &[]));
        let snap = TopologySnapshot::assemble(&eps, &k, &no_rules());
        let placement = no_rules();
        assert_eq!(
            snap.select_create_backend("fresh", &placement)
                .unwrap()
                .endpoint
                .shard
                .0,
            "1"
        );
    }

    #[test]
    fn create_default_reject_refuses_unmatched() {
        let eps = vec![ep("a", "0", None, true)];
        let mut k = HashMap::new();
        k.insert(BackendId("a".into()), know(Health::Healthy, &[]));
        let snap = TopologySnapshot::assemble(&eps, &k, &no_rules());
        let placement = PlacementRules::new(vec![], PlacementDefault::Reject);
        let err = snap.select_create_backend("fresh", &placement).unwrap_err();
        assert_eq!(err.code(), tonic::Code::FailedPrecondition);
    }
}
