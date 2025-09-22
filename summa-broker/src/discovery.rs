//! Static backend discovery: `--backend "id=..,addr=..,shard=..[,role=..]"`.
//!
//! Static mode feeds the same watch channel the kubernetes watcher does, so
//! everything downstream (poller, topology, services) is identical. Used for
//! local development and the integration tests.

use crate::topology::{BackendId, DiscoveredEndpoint, Role, ShardId};

pub fn parse_static_backend(s: &str) -> anyhow::Result<DiscoveredEndpoint> {
    let mut id = None;
    let mut addr = None;
    let mut shard = None;
    let mut role = None;
    for part in s.split(',') {
        let (key, value) = part
            .split_once('=')
            .ok_or_else(|| anyhow::anyhow!("invalid --backend '{s}': '{part}' is not key=value"))?;
        match key {
            "id" => id = Some(value.to_string()),
            "addr" => addr = Some(value.to_string()),
            "shard" => shard = Some(value.to_string()),
            "role" => {
                role = Some(match value {
                    "master" => Role::Master,
                    "follower" => Role::Follower,
                    other => anyhow::bail!(
                        "invalid --backend '{s}': role '{other}' is not master|follower"
                    ),
                })
            }
            other => anyhow::bail!("invalid --backend '{s}': unknown key '{other}'"),
        }
    }
    let (Some(id), Some(addr), Some(shard)) = (id, addr, shard) else {
        anyhow::bail!("invalid --backend '{s}': id, addr and shard are required");
    };
    if id.is_empty() || addr.is_empty() || shard.is_empty() {
        anyhow::bail!("invalid --backend '{s}': id, addr and shard must be non-empty");
    }
    Ok(DiscoveredEndpoint {
        id: BackendId(id),
        addr,
        shard: ShardId(shard),
        role,
        ready: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_full_backend_spec() {
        let ep = parse_static_backend("id=hs-a,addr=127.0.0.1:50051,shard=0,role=master").unwrap();
        assert_eq!(ep.id.0, "hs-a");
        assert_eq!(ep.addr, "127.0.0.1:50051");
        assert_eq!(ep.shard.0, "0");
        assert_eq!(ep.role, Some(Role::Master));
        assert!(ep.ready);
    }

    #[test]
    fn role_is_optional_and_validated() {
        let ep = parse_static_backend("id=a,addr=h:1,shard=0").unwrap();
        assert_eq!(ep.role, None);
        assert!(parse_static_backend("id=a,addr=h:1,shard=0,role=primary").is_err());
    }

    #[test]
    fn missing_required_keys_fail() {
        assert!(parse_static_backend("id=a,shard=0").is_err());
        assert!(parse_static_backend("addr=h:1,shard=0").is_err());
        assert!(parse_static_backend("id=a,addr=h:1").is_err());
        assert!(parse_static_backend("garbage").is_err());
    }
}
