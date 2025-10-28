//! CreateIndex placement rules and read/write tiebreaks.
//!
//! A rule maps an index-name glob to a shard id: `--placement "documents*=0"`.
//! First matching rule wins, so dated full-build names (`documents_20260724`)
//! follow their family. Rule order is also the partition order once an index
//! is partitioned across shards (phase 2), which makes the ordering an
//! immutable routing contract — reorder rules only together with a rebuild.

use summa_broker::placement_rules::{self, glob_match};

use crate::partition::PartitionWeights;
use crate::topology::ShardId;

#[derive(Debug, Clone)]
pub struct PlacementRule {
    pub pattern: String,
    /// One shard pins the index to it; several shards partition the index
    /// across them, in this (immutable) order: writes hash the primary key
    /// to a position in this list, reads fan out to all of them.
    pub shards: Vec<ShardId>,
    /// Share of rows each partition receives (`shard:weight`, default 1).
    pub weights: PartitionWeights,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum PlacementDefault {
    /// Unmatched CreateIndex lands on the shard hosting the fewest indexes
    /// (lexicographic shard-id tiebreak).
    Single,
    /// Unmatched CreateIndex is refused with FAILED_PRECONDITION.
    Reject,
}

#[derive(Debug, Clone)]
pub struct PlacementRules {
    rules: Vec<PlacementRule>,
    pub default: PlacementDefault,
}

impl PlacementRules {
    pub fn new(rules: Vec<PlacementRule>, default: PlacementDefault) -> Self {
        Self { rules, default }
    }

    /// First rule whose glob matches the index name.
    /// Shards of the first matching rule.
    pub fn shards_for(&self, index_name: &str) -> Option<&[ShardId]> {
        self.rules
            .iter()
            .find(|r| glob_match(&r.pattern, index_name))
            .map(|r| r.shards.as_slice())
    }

    /// Partition weights of the first matching rule.
    pub fn weights_for(&self, index_name: &str) -> Option<&PartitionWeights> {
        self.rules
            .iter()
            .find(|r| glob_match(&r.pattern, index_name))
            .map(|r| &r.weights)
    }

    /// First shard of the first matching rule (the whole placement for an
    /// unpartitioned index).
    pub fn shard_for(&self, index_name: &str) -> Option<&ShardId> {
        self.shards_for(index_name)
            .and_then(|shards| shards.first())
    }
}

/// Parse one `--placement` argument (syntax in [`placement_rules::parse`]).
pub fn parse_placement(s: &str) -> anyhow::Result<PlacementRule> {
    let rule = placement_rules::parse(s)?;
    let weights = rule.weights();
    Ok(PlacementRule {
        pattern: rule.pattern,
        shards: rule
            .partitions
            .into_iter()
            .map(|p| ShardId(p.shard))
            .collect(),
        weights: PartitionWeights::new(weights),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placement_first_match_wins() {
        let rules = PlacementRules::new(
            vec![
                parse_placement("documents*=0").unwrap(),
                parse_placement("*=9").unwrap(),
            ],
            PlacementDefault::Single,
        );
        assert_eq!(rules.shard_for("documents_20260724").unwrap().0, "0");
        assert_eq!(rules.shard_for("social").unwrap().0, "9");
    }

    #[test]
    fn placement_parse_rejects_malformed() {
        assert!(parse_placement("no-separator").is_err());
        assert!(parse_placement("=0").is_err());
        assert!(parse_placement("documents*=").is_err());
        let rule = parse_placement("social*=fin2").unwrap();
        assert_eq!(rule.pattern, "social*");
        assert_eq!(rule.shards.len(), 1);
        assert_eq!(rule.shards[0].0, "fin2");
    }

    #[test]
    fn placement_parses_partition_weights() {
        let rule = parse_placement("documents_2026*=2:5,3:5,4:4").unwrap();
        let shards: Vec<&str> = rule.shards.iter().map(|s| s.0.as_str()).collect();
        assert_eq!(shards, vec!["2", "3", "4"]);
        assert_eq!(rule.weights, PartitionWeights::new(vec![5, 5, 4]));
        let spaced = parse_placement("documents_2026*=2:5, 3:5 ,4: 4").unwrap();
        assert_eq!(spaced.weights, rule.weights);
        // Unlisted weights default to 1.
        assert_eq!(
            parse_placement("documents*=2,3,4:2").unwrap().weights,
            PartitionWeights::new(vec![1, 1, 2])
        );
        // No weights: the unweighted routing every existing generation uses.
        assert_eq!(
            parse_placement("documents*=2,3,4").unwrap().weights,
            PartitionWeights::uniform(3)
        );
        for bad in [
            "documents*=2,3,4:0",
            "documents*=2,3,4:-1",
            "documents*=2,3,4:+4",
            "documents*=2,3,4:0.8",
            "documents*=2,3,4:1.0",
            "documents*=2,3,4:abc",
            "documents*=2,3,4:",
            "documents*=2,3,4:1e3",
            "documents*=2,3,4:4294967296",
            "documents*=2:5",
            "documents*=:5,3",
        ] {
            assert!(parse_placement(bad).is_err(), "{bad} must be refused");
        }
        let rules = PlacementRules::new(
            vec![
                parse_placement("documents_20261020=2:5,3:5,4:4").unwrap(),
                parse_placement("documents*=2,3,4").unwrap(),
            ],
            PlacementDefault::Single,
        );
        assert!(
            !rules
                .weights_for("documents_20261020")
                .unwrap()
                .is_uniform()
        );
        assert!(
            rules
                .weights_for("documents_20260905")
                .unwrap()
                .is_uniform()
        );
        assert!(rules.weights_for("social").is_none());
    }

    #[test]
    fn placement_parses_partition_lists() {
        let rule = parse_placement("documents_2026*=2, 3,4").unwrap();
        let shards: Vec<&str> = rule.shards.iter().map(|s| s.0.as_str()).collect();
        assert_eq!(shards, vec!["2", "3", "4"]);
        assert!(parse_placement("documents*=2,,3").is_err());
        assert!(parse_placement("documents*=2,2").is_err());
        let rules = PlacementRules::new(vec![rule], PlacementDefault::Single);
        assert_eq!(rules.shards_for("documents_20260905").unwrap().len(), 3);
        assert_eq!(rules.shard_for("documents_20260905").unwrap().0, "2");
        assert!(rules.shards_for("social").is_none());
    }
}
