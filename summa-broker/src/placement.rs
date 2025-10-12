//! CreateIndex placement rules and read/write tiebreaks.
//!
//! A rule maps an index-name glob to a shard id: `--placement "documents*=0"`.
//! First matching rule wins, so dated full-build names (`documents_20260724`)
//! follow their family. Rule order is also the partition order once an index
//! is partitioned across shards (phase 2), which makes the ordering an
//! immutable routing contract — reorder rules only together with a rebuild.

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

/// Parse one `--placement "pattern=shard"` argument. The shard id follows the
/// last `=` so patterns themselves may not contain `=` (index names cannot
/// either, per summa-server's index-name validation).
/// `pattern=shard` or `pattern=shardA,shardB,...` (partitions in order).
/// A partition may carry a positive integer weight, `pattern=a:5,b:5,c:4`:
/// partitions receive rows in proportion to their weights (default 1).
pub fn parse_placement(s: &str) -> anyhow::Result<PlacementRule> {
    let (pattern, shards) = s.rsplit_once('=').ok_or_else(|| {
        anyhow::anyhow!("invalid --placement '{s}': expected 'pattern=shard[:weight][,shard...]'")
    })?;
    if pattern.is_empty() || shards.is_empty() {
        anyhow::bail!("invalid --placement '{s}': empty pattern or shard");
    }
    let mut parsed: Vec<ShardId> = Vec::new();
    let mut weights = Vec::new();
    let mut weighted = false;
    for entry in shards.split(',') {
        let (shard, weight) = match entry.split_once(':') {
            Some((shard, weight)) => {
                weighted = true;
                (
                    shard.trim(),
                    parse_weight(weight.trim()).map_err(|e| {
                        anyhow::anyhow!(
                            "invalid --placement '{s}': weight of shard '{}': {e}",
                            shard.trim()
                        )
                    })?,
                )
            }
            None => (entry.trim(), 1),
        };
        if shard.is_empty() {
            anyhow::bail!("invalid --placement '{s}': empty shard id");
        }
        if parsed.iter().any(|p| p.0 == shard) {
            anyhow::bail!("invalid --placement '{s}': shard '{shard}' listed twice");
        }
        parsed.push(ShardId(shard.to_string()));
        weights.push(weight);
    }
    if weighted && parsed.len() == 1 {
        anyhow::bail!(
            "invalid --placement '{s}': a weight only applies to a partition of a multi-shard rule"
        );
    }
    Ok(PlacementRule {
        pattern: pattern.to_string(),
        shards: parsed,
        weights: PartitionWeights::new(weights),
    })
}

/// A positive integer weight. Decimal fractions are refused: weights decide
/// where every key is routed, so they are compared exactly.
fn parse_weight(weight: &str) -> anyhow::Result<u32> {
    if weight.is_empty() || !weight.bytes().all(|b| b.is_ascii_digit()) {
        anyhow::bail!("'{weight}' is not a positive integer such as 4");
    }
    match weight.parse::<u32>() {
        Ok(0) => anyhow::bail!("'{weight}' must be greater than 0"),
        Ok(weight) => Ok(weight),
        Err(_) => anyhow::bail!("'{weight}' is larger than {}", u32::MAX),
    }
}

/// Glob match supporting `*` (any run of characters, including empty).
/// Iterative backtracking matcher — no recursion, O(pattern × name).
pub fn glob_match(pattern: &str, name: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let n: Vec<char> = name.chars().collect();
    let (mut pi, mut ni) = (0usize, 0usize);
    let mut star: Option<(usize, usize)> = None; // (pattern idx after '*', name idx it consumed to)
    while ni < n.len() {
        if pi < p.len() && (p[pi] == n[ni]) {
            pi += 1;
            ni += 1;
        } else if pi < p.len() && p[pi] == '*' {
            star = Some((pi + 1, ni));
            pi += 1;
        } else if let Some((star_pi, star_ni)) = star {
            // Let the last '*' swallow one more character and retry.
            pi = star_pi;
            ni = star_ni + 1;
            star = Some((star_pi, star_ni + 1));
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == '*' {
        pi += 1;
    }
    pi == p.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glob_matches() {
        assert!(glob_match("documents*", "documents"));
        assert!(glob_match("documents*", "documents_20260724"));
        assert!(glob_match("*", "anything"));
        assert!(glob_match("a*b*c", "aXXbYYc"));
        assert!(glob_match("a*b*c", "abc"));
        assert!(!glob_match("documents*", "social"));
        assert!(!glob_match("documents", "documents_20260724"));
        assert!(!glob_match("a*b", "aXc"));
        assert!(glob_match("", ""));
        assert!(!glob_match("", "x"));
    }

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
