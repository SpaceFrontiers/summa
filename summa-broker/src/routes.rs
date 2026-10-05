//! Route resolution shared by the search and index services: one backend
//! for an unpartitioned index, one backend per partition for a partitioned
//! one, plus the per-index primary-key field a partitioned write hashes.

use std::time::Instant;

use tonic::{Request, Status};

use crate::client::BackendChannels;
use crate::context::{BrokerContext, code_label};
use crate::metrics as m;
use crate::partition;
use crate::proto::summa::GetIndexInfoRequest;

/// One backend a request goes to.
#[derive(Clone)]
pub struct Target {
    pub shard: String,
    pub backend_id: String,
    pub channels: BackendChannels,
}

/// Where a request for an index goes.
#[derive(Clone)]
pub enum Route {
    Single(Box<Target>),
    /// One target per partition, in placement-rule order. A partial read
    /// (`--partial-partition-reads`) lists the shards of the partitions it
    /// leaves out in `missing`; every other route has all of them.
    Partitioned {
        targets: Vec<Target>,
        missing: Vec<String>,
    },
}

impl Route {
    pub fn targets(&self) -> &[Target] {
        match self {
            Route::Single(target) => std::slice::from_ref(target.as_ref()),
            Route::Partitioned { targets, .. } => targets,
        }
    }

    pub fn is_partitioned(&self) -> bool {
        matches!(self, Route::Partitioned { .. })
    }

    /// Shards of the partitions this read leaves out.
    pub fn missing(&self) -> &[String] {
        match self {
            Route::Single(_) => &[],
            Route::Partitioned { missing, .. } => missing,
        }
    }

    /// The same read without the partitions on `lost` shards, which then
    /// count as missing.
    pub fn without(&self, lost: &[String]) -> Route {
        match self {
            Route::Partitioned { targets, missing } if !lost.is_empty() => Route::Partitioned {
                targets: targets
                    .iter()
                    .filter(|target| !lost.contains(&target.shard))
                    .cloned()
                    .collect(),
                missing: missing.iter().chain(lost).cloned().collect(),
            },
            route => route.clone(),
        }
    }
}

/// Count each partition a partial read left out (bounded labels: index
/// families and shard ids).
pub fn record_partial(index_name: &str, rpc: &'static str, missing: &[String]) {
    for shard in missing {
        metrics::counter!(
            m::PARTIAL_READS,
            "index" => index_name.to_string(),
            "rpc" => rpc,
            "shard" => shard.clone(),
        )
        .increment(1);
    }
}

pub fn record_backend(backend: &str, rpc: &'static str, started: Instant, code: tonic::Code) {
    metrics::histogram!(
        m::BACKEND_DURATION,
        "backend" => backend.to_string(),
        "rpc" => rpc,
    )
    .record(started.elapsed().as_secs_f64());
    metrics::counter!(
        m::BACKEND_REQUESTS,
        "backend" => backend.to_string(),
        "rpc" => rpc,
        "code" => code_label(code),
    )
    .increment(1);
}

impl BrokerContext {
    /// Backend(s) serving a read for `index_name`.
    pub fn read_route(&self, index_name: &str) -> Result<Route, Status> {
        let snapshot = self.snapshot.load();
        if let Some((partitions, unavailable)) = snapshot.partition_availability(index_name)? {
            if !self.partial_partition_reads {
                // Strict: every partition must be routable.
                snapshot.partitions(index_name)?;
            }
            let mut targets = Vec::with_capacity(partitions.len());
            let mut missing = Vec::new();
            for shard in partitions {
                if unavailable.contains(&shard) {
                    missing.push(shard.0.clone());
                    continue;
                }
                let selection = match snapshot.select_read_backend_on(
                    index_name,
                    shard,
                    self.next_rotation(),
                ) {
                    Ok(selection) => selection,
                    Err(status)
                        if self.partial_partition_reads
                            && status.code() == tonic::Code::Unavailable =>
                    {
                        missing.push(shard.0.clone());
                        continue;
                    }
                    Err(status) => return Err(status),
                };
                if selection.stale {
                    metrics::counter!(
                        m::STALE_TOPOLOGY_SERVES,
                        "backend" => selection.backend.endpoint.id.0.clone()
                    )
                    .increment(1);
                }
                targets.push(Target {
                    shard: shard.0.clone(),
                    backend_id: selection.backend.endpoint.id.0.clone(),
                    channels: self.pool.get(&selection.backend.endpoint.addr)?,
                });
            }
            if targets.is_empty() {
                return Err(Status::unavailable(format!(
                    "index '{index_name}': none of its {} partitions has a routable replica",
                    partitions.len()
                )));
            }
            return Ok(Route::Partitioned { targets, missing });
        }
        let selection = snapshot.select_read_backend(index_name, self.next_rotation())?;
        if selection.ambiguous {
            metrics::counter!(m::AMBIGUOUS_INDEX, "index" => index_name.to_string()).increment(1);
        }
        if selection.stale {
            metrics::counter!(
                m::STALE_TOPOLOGY_SERVES,
                "backend" => selection.backend.endpoint.id.0.clone()
            )
            .increment(1);
        }
        Ok(Route::Single(Box::new(Target {
            shard: selection.backend.endpoint.shard.0.clone(),
            backend_id: selection.backend.endpoint.id.0.clone(),
            channels: self.pool.get(&selection.backend.endpoint.addr)?,
        })))
    }

    /// Master(s) a write for `index_name` goes to.
    pub fn write_route(&self, index_name: &str) -> Result<Route, Status> {
        let snapshot = self.snapshot.load();
        let reject = |status: Status| {
            metrics::counter!(
                m::WRITE_REJECTED,
                "index" => index_name.to_string(),
                "reason" => code_label(status.code()),
            )
            .increment(1);
            status
        };
        if let Some(partitions) = snapshot.partitions(index_name).map_err(reject)? {
            let mut targets = Vec::with_capacity(partitions.len());
            for shard in partitions {
                let master = snapshot
                    .select_write_backend_on(index_name, shard)
                    .map_err(reject)?;
                targets.push(Target {
                    shard: shard.0.clone(),
                    backend_id: master.endpoint.id.0.clone(),
                    channels: self.pool.get(&master.endpoint.addr)?,
                });
            }
            return Ok(Route::Partitioned {
                targets,
                missing: Vec::new(),
            });
        }
        let master = snapshot.select_write_backend(index_name).map_err(reject)?;
        Ok(Route::Single(Box::new(Target {
            shard: master.endpoint.shard.0.clone(),
            backend_id: master.endpoint.id.0.clone(),
            channels: self.pool.get(&master.endpoint.addr)?,
        })))
    }

    /// The field a partitioned index hashes documents by, read once from
    /// the schema of one partition and cached.
    pub async fn primary_key_field(
        &self,
        index_name: &str,
        target: &Target,
    ) -> Result<String, Status> {
        if let Some(field) = self.primary_keys.read().get(index_name) {
            return Ok(field.clone());
        }
        let started = Instant::now();
        let result = target
            .channels
            .search
            .clone()
            .get_index_info(Request::new(GetIndexInfoRequest {
                index_name: index_name.to_string(),
            }))
            .await;
        let code = result
            .as_ref()
            .map(|_| tonic::Code::Ok)
            .unwrap_or_else(|s| s.code());
        record_backend(&target.backend_id, "get_index_info", started, code);
        let info = result?.into_inner();
        let field = partition::primary_key_field(&info.schema).ok_or_else(|| {
            Status::failed_precondition(format!(
                "index '{index_name}' is partitioned but its schema declares no primary key field"
            ))
        })?;
        self.primary_keys
            .write()
            .insert(index_name.to_string(), field.clone());
        Ok(field)
    }

    /// Cached primary-key field, if a write has resolved it.
    pub fn primary_key_cached(&self, index_name: &str) -> Option<String> {
        self.primary_keys.read().get(index_name).cloned()
    }

    pub fn forget_index(&self, index_name: &str) {
        self.primary_keys.write().remove(index_name);
    }
}
