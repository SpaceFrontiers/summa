//! Placement rule syntax: `pattern=shard[:weight][,shard[:weight]...]`.
//!
//! The broker parses its `--placement` arguments here, and `summa-k8s-manager`
//! renders and validates the same rules from a `SearchCluster` resource, so
//! both sides agree on what a rule means. Routing semantics (first match wins,
//! partition order, weights reduced by their greatest common divisor) live
//! here too, because the manager compares live routing against a proposed rule
//! set before the broker is restarted with it.

use std::fmt;

/// One rule as written: an index-name glob and its partitions in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuleSpec {
    pub pattern: String,
    pub partitions: Vec<PartitionSpec>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionSpec {
    pub shard: String,
    /// Explicit weight (`shard:weight`). Unlisted weights default to 1.
    pub weight: Option<u32>,
}

/// How a rule routes an index: shards in partition order and their weights
/// reduced by the greatest common divisor. Two routings are interchangeable
/// for a live index exactly when they are equal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Routing {
    pub shards: Vec<String>,
    pub weights: Vec<u32>,
}

impl Routing {
    pub fn is_partitioned(&self) -> bool {
        self.shards.len() > 1
    }
}

impl fmt::Display for Routing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.is_partitioned() {
            return write!(f, "{}", self.shards.join(","));
        }
        let parts: Vec<String> = self
            .shards
            .iter()
            .zip(&self.weights)
            .map(|(shard, weight)| format!("{shard}:{weight}"))
            .collect();
        write!(f, "{}", parts.join(","))
    }
}

impl RuleSpec {
    /// Check the rule against the broker's constraints. Messages name the
    /// offending part without the `--placement` prefix.
    pub fn validate(&self) -> Result<(), String> {
        if self.pattern.is_empty() {
            return Err("empty pattern".to_string());
        }
        if self.pattern.contains('=') {
            return Err(format!("pattern '{}' contains '='", self.pattern));
        }
        if self.partitions.is_empty() {
            return Err("empty shard".to_string());
        }
        for (i, partition) in self.partitions.iter().enumerate() {
            let shard = &partition.shard;
            if shard.is_empty() {
                return Err("empty shard id".to_string());
            }
            if shard.trim() != shard || shard.contains([',', ':', '=']) {
                return Err(format!(
                    "shard id '{shard}' contains whitespace, ',', ':' or '='"
                ));
            }
            if self.partitions[..i].iter().any(|p| &p.shard == shard) {
                return Err(format!("shard '{shard}' listed twice"));
            }
            if partition.weight == Some(0) {
                return Err(format!("weight of shard '{shard}': must be greater than 0"));
            }
        }
        if self.partitions.len() == 1 && self.partitions[0].weight.is_some() {
            return Err("a weight only applies to a partition of a multi-shard rule".to_string());
        }
        Ok(())
    }

    /// Weights in partition order, unlisted ones as 1.
    pub fn weights(&self) -> Vec<u32> {
        self.partitions
            .iter()
            .map(|p| p.weight.unwrap_or(1))
            .collect()
    }

    pub fn routing(&self) -> Routing {
        Routing {
            shards: self.partitions.iter().map(|p| p.shard.clone()).collect(),
            weights: reduced_weights(&self.weights()),
        }
    }
}

/// The `--placement` argument for this rule; [`parse`] reads it back.
impl fmt::Display for RuleSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let parts: Vec<String> = self
            .partitions
            .iter()
            .map(|p| match p.weight {
                Some(weight) => format!("{}:{weight}", p.shard),
                None => p.shard.clone(),
            })
            .collect();
        write!(f, "{}={}", self.pattern, parts.join(","))
    }
}

/// Parse one `--placement "pattern=shard"` argument. The shard id follows the
/// last `=` so patterns themselves may not contain `=` (index names cannot
/// either, per summa-server's index-name validation).
/// `pattern=shard` or `pattern=shardA,shardB,...` (partitions in order).
/// A partition may carry a positive integer weight, `pattern=a:5,b:5,c:4`:
/// partitions receive rows in proportion to their weights (default 1).
pub fn parse(s: &str) -> anyhow::Result<RuleSpec> {
    let (pattern, shards) = s.rsplit_once('=').ok_or_else(|| {
        anyhow::anyhow!("invalid --placement '{s}': expected 'pattern=shard[:weight][,shard...]'")
    })?;
    if pattern.is_empty() || shards.is_empty() {
        anyhow::bail!("invalid --placement '{s}': empty pattern or shard");
    }
    let mut partitions = Vec::new();
    for entry in shards.split(',') {
        let partition = match entry.split_once(':') {
            Some((shard, weight)) => PartitionSpec {
                shard: shard.trim().to_string(),
                weight: Some(parse_weight(weight.trim()).map_err(|e| {
                    anyhow::anyhow!(
                        "invalid --placement '{s}': weight of shard '{}': {e}",
                        shard.trim()
                    )
                })?),
            },
            None => PartitionSpec {
                shard: entry.trim().to_string(),
                weight: None,
            },
        };
        partitions.push(partition);
    }
    let rule = RuleSpec {
        pattern: pattern.to_string(),
        partitions,
    };
    rule.validate()
        .map_err(|e| anyhow::anyhow!("invalid --placement '{s}': {e}"))?;
    Ok(rule)
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

/// First rule whose glob matches the index name.
pub fn resolve<'a>(rules: &'a [RuleSpec], index_name: &str) -> Option<&'a RuleSpec> {
    rules.iter().find(|r| glob_match(&r.pattern, index_name))
}

/// Weights divided by their greatest common divisor: only the ratio routes
/// keys, so `2:10,3:10,4:8` and `2:5,3:5,4:4` are the same partitioning and
/// equal weights reduce to all ones (plain `fnv1a64 % n`).
pub fn reduced_weights(weights: &[u32]) -> Vec<u32> {
    let divisor = weights.iter().copied().fold(0, gcd);
    if divisor == 0 {
        return weights.to_vec();
    }
    weights.iter().map(|w| w / divisor).collect()
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd(b, a % b) }
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
    fn rendered_rules_parse_back_to_the_same_rule() {
        for arg in [
            "social*=1",
            "documents_20260923=2,3,4",
            "documents*=2:5,3:5,4:4",
            "documents*=2,3,4:2",
        ] {
            let rule = parse(arg).unwrap();
            assert_eq!(rule.to_string(), arg);
            assert_eq!(parse(&rule.to_string()).unwrap(), rule);
        }
    }

    #[test]
    fn validation_refuses_rules_the_parser_cannot_express() {
        let rule = |pattern: &str, shards: &[(&str, Option<u32>)]| RuleSpec {
            pattern: pattern.to_string(),
            partitions: shards
                .iter()
                .map(|(shard, weight)| PartitionSpec {
                    shard: shard.to_string(),
                    weight: *weight,
                })
                .collect(),
        };
        assert!(rule("a=b", &[("1", None)]).validate().is_err());
        assert!(rule("docs*", &[]).validate().is_err());
        assert!(rule("docs*", &[("1,2", None)]).validate().is_err());
        assert!(rule("docs*", &[("1:2", None)]).validate().is_err());
        assert!(rule("docs*", &[(" 1", None)]).validate().is_err());
        assert!(rule("docs*", &[("1", Some(3))]).validate().is_err());
        assert!(
            rule("docs*", &[("1", Some(0)), ("2", None)])
                .validate()
                .is_err()
        );
        assert!(
            rule("docs*", &[("1", None), ("1", None)])
                .validate()
                .is_err()
        );
        assert!(
            rule("docs*", &[("1", Some(5)), ("2", None)])
                .validate()
                .is_ok()
        );
    }

    #[test]
    fn routing_compares_weight_ratios_not_their_scale() {
        let weighted = parse("documents*=2:5,3:5,4:4").unwrap().routing();
        assert_eq!(weighted.weights, vec![5, 5, 4]);
        assert_eq!(
            parse("documents*=2:10,3:10,4:8").unwrap().routing(),
            weighted
        );
        // Equal weights are the unweighted routing.
        assert_eq!(
            parse("documents*=2:7,3:7,4:7").unwrap().routing(),
            parse("documents*=2,3,4").unwrap().routing()
        );
        assert_ne!(
            parse("documents*=3,2,4").unwrap().routing(),
            parse("documents*=2,3,4").unwrap().routing()
        );
        assert_eq!(weighted.to_string(), "2:5,3:5,4:4");
        assert_eq!(parse("social*=1").unwrap().routing().to_string(), "1");
    }

    #[test]
    fn first_matching_rule_resolves_an_index() {
        let rules = vec![
            parse("documents_20260923=2,3,4").unwrap(),
            parse("documents*=2:5,3:5,4:4").unwrap(),
        ];
        assert_eq!(
            resolve(&rules, "documents_20260923").unwrap().pattern,
            "documents_20260923"
        );
        assert_eq!(
            resolve(&rules, "documents_20261101").unwrap().pattern,
            "documents*"
        );
        assert!(resolve(&rules, "social").is_none());
    }
}
