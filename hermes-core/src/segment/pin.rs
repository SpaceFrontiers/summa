//! Hot-metadata pinning: budgeted residency for per-query-mandatory
//! structures (a meta/data residency split).
//!
//! Every query must touch certain small metadata sections — BMP block-offset
//! tables, sparse skip sections, doc-id maps, superblock grids. Under memory
//! pressure the kernel evicts them like bulk data, and queries then pay major
//! faults on structures they cannot skip. This module pins them, in priority
//! order (smallest/hottest first), until a per-segment budget is exhausted.
//!
//! Design: `docs/hot-metadata-pinning.md`. Bulk data (BMP 4-bit grid, block
//! data, raw vectors) is never pinned — it is covered by the
//! `MADV_RANDOM`/`MADV_WILLNEED` discipline instead.

use std::sync::OnceLock;

use crate::directories::OwnedBytes;

/// How pinned bytes are kept resident.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PinMode {
    /// `mlock` the mmap pages in place — zero-copy, but requires
    /// RLIMIT_MEMLOCK headroom (containers often need CAP_IPC_LOCK or an
    /// explicit ulimit). Failures are logged and counted, never fatal.
    Mlock,
    /// Copy the section to the heap — no permissions needed, duplicates the
    /// bytes. Immune to page-cache eviction (production runs swapless).
    Copy,
}

/// Per-segment metadata pinning policy.
#[derive(Debug, Clone, Copy)]
pub struct PinPolicy {
    /// Metadata bytes to pin per segment. 0 = pinning disabled (default).
    pub budget_bytes: u64,
    pub mode: PinMode,
}

impl PinPolicy {
    pub const fn disabled() -> Self {
        Self {
            budget_bytes: 0,
            mode: PinMode::Mlock,
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.budget_bytes > 0
    }

    /// Read policy from environment:
    /// `HERMES_PIN_METADATA_BUDGET_MB` (default 0 = off),
    /// `HERMES_PIN_MODE` = `mlock` (default) | `copy`.
    fn from_env() -> Self {
        let budget_mb: u64 = std::env::var("HERMES_PIN_METADATA_BUDGET_MB")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let mode = match std::env::var("HERMES_PIN_MODE").as_deref() {
            Ok("copy") => PinMode::Copy,
            Ok("mlock") | Err(_) => PinMode::Mlock,
            Ok(other) => {
                log::warn!("HERMES_PIN_MODE '{}' unknown; using mlock", other);
                PinMode::Mlock
            }
        };
        Self {
            budget_bytes: budget_mb * 1024 * 1024,
            mode,
        }
    }
}

static PIN_POLICY: OnceLock<PinPolicy> = OnceLock::new();

/// Override the process-wide pin policy. Must be called before the first
/// segment is opened; returns false (and warns) if the policy was already
/// initialized.
pub fn set_pin_policy(policy: PinPolicy) -> bool {
    let ok = PIN_POLICY.set(policy).is_ok();
    if !ok {
        log::warn!("pin policy already initialized; set_pin_policy ignored");
    }
    ok
}

/// The process-wide pin policy (env-initialized on first use).
pub fn pin_policy() -> &'static PinPolicy {
    PIN_POLICY.get_or_init(PinPolicy::from_env)
}

/// Accumulates pin accounting for one segment.
#[derive(Debug, Default, Clone, Copy)]
pub struct PinReport {
    /// Bytes of pinnable metadata found (regardless of budget/failures)
    pub intended_bytes: u64,
    /// Bytes actually pinned
    pub pinned_bytes: u64,
    /// Bytes skipped because the budget was exhausted
    pub skipped_budget_bytes: u64,
    /// Bytes where mlock failed (RLIMIT_MEMLOCK etc.)
    pub failed_bytes: u64,
}

/// Pin one metadata section, updating `remaining` budget and the report.
///
/// In `Copy` mode the section is replaced with a heap copy (heap memory is
/// not page-cache-evictable). In `Mlock` mode the mmap pages are locked in
/// place. Non-mmap-backed sections (RAM directories) are already resident
/// and are skipped silently.
pub(crate) fn pin_section(
    bytes: &mut OwnedBytes,
    label: &str,
    mode: PinMode,
    remaining: &mut u64,
    report: &mut PinReport,
) {
    if !bytes.is_mmap() || bytes.is_empty() {
        return;
    }
    let len = bytes.len() as u64;
    report.intended_bytes += len;

    if len > *remaining {
        report.skipped_budget_bytes += len;
        log::debug!(
            "[pin] budget exhausted: skipping {} ({} bytes, {} remaining)",
            label,
            len,
            *remaining
        );
        return;
    }

    match mode {
        PinMode::Mlock => {
            if bytes.mlock() {
                *remaining -= len;
                report.pinned_bytes += len;
            } else {
                report.failed_bytes += len;
                log::warn!(
                    "[pin] mlock failed for {} ({} bytes) — check RLIMIT_MEMLOCK; \
                     continuing unpinned",
                    label,
                    len
                );
            }
        }
        PinMode::Copy => {
            *bytes = OwnedBytes::new(bytes.to_vec());
            *remaining -= len;
            report.pinned_bytes += len;
        }
    }
}
