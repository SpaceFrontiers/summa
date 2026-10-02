//! Directory leases bridge read completion to the existing unlink lifecycle.
use std::{io, sync::Arc};
use tokio::sync::{OwnedRwLockReadGuard, OwnedRwLockWriteGuard, RwLock};

pub(super) type ReadLease = OwnedRwLockReadGuard<bool>;

#[derive(Default)]
pub(in crate::directories) struct Scope {
    retired: Arc<RwLock<bool>>,
}
impl Scope {
    pub(super) async fn read(&self) -> io::Result<ReadLease> {
        let guard = self.retired.clone().read_owned().await;
        if *guard {
            return Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "payload directory is retired",
            ));
        }
        Ok(guard)
    }
    pub(in crate::directories) async fn exclusive(&self) -> OwnedRwLockWriteGuard<bool> {
        self.retired.clone().write_owned().await
    }
}

/// Weak gate identities survive cancelled open futures through the worker's
/// owned guard, without retaining every directory ever attached to a service.
#[derive(Default)]
pub(super) struct Registry {
    gates: parking_lot::Mutex<
        std::collections::HashMap<std::path::PathBuf, std::sync::Weak<RwLock<bool>>>,
    >,
}
impl Registry {
    pub(super) fn attach(&self, root: &std::path::Path) -> Arc<Scope> {
        let mut gates = self.gates.lock();
        gates.retain(|_, gate| gate.strong_count() != 0);
        let gate = gates
            .get(root)
            .and_then(std::sync::Weak::upgrade)
            .filter(|gate| gate.try_read().map_or(true, |retired| !*retired))
            .unwrap_or_else(|| {
                let gate = Arc::new(RwLock::new(false));
                gates.insert(root.to_path_buf(), Arc::downgrade(&gate));
                gate
            });
        Arc::new(Scope { retired: gate })
    }
}
