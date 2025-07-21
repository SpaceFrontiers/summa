//! Hot-metadata pinning: budgeted residency for per-query-mandatory
//! structures (a meta/data residency split).
//!
//! Every query must touch certain small metadata sections — BMP block-offset
//! tables, sparse skip sections, doc-id maps, and the coarse BMP hierarchy. Under memory
//! pressure the kernel evicts them like bulk data, and queries then pay major
//! faults on structures they cannot skip. This module pins them, in priority
//! order (smallest/hottest first), until a per-segment budget is exhausted.
//!
//! Design: `docs/hot-metadata-pinning.md`. Bulk data (BMP 4-bit grid, block
//! data, raw vectors) is never pinned — it is covered by the
//! `MADV_RANDOM`/`MADV_WILLNEED` discipline instead.

use std::sync::{Arc, OnceLock};

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
    ///
    /// Used as the default initializer for [`pin_policy`]. Consumers such as
    /// hermes-server expose CLI flags and honor these environment values when
    /// the corresponding flags are unset.
    pub fn from_env() -> Self {
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
    /// Additional heap allocated by `PinMode::Copy`. Already-heap ANN routing
    /// structures are resident but do not contribute here.
    pub heap_copy_bytes: u64,
}

/// RAII owner for heap pages locked on behalf of one immutable ANN artifact
/// generation. The referenced allocations are owned by the same
/// `TrainedVectorStructures`; its field order drops this set before the
/// artifact `Arc`s, so every address remains valid through `munlock`.
struct HeapPinGuard {
    page_start: *mut libc::c_void,
    page_len: usize,
}

// The guard never dereferences its pointer. The immutable artifact allocations
// it describes are safe to share, and mlock/munlock operate on process mappings.
unsafe impl Send for HeapPinGuard {}
unsafe impl Sync for HeapPinGuard {}

impl Drop for HeapPinGuard {
    fn drop(&mut self) {
        if unsafe { libc::munlock(self.page_start, self.page_len) } != 0 {
            log::warn!(
                "[pin] munlock failed for {} of ANN heap: {}",
                crate::format_bytes(self.page_len as u64),
                std::io::Error::last_os_error()
            );
        }
    }
}

/// Locked heap allocations associated with one index-global ANN generation.
/// Segment-local vector/code payloads are intentionally excluded.
#[derive(Default)]
pub(crate) struct HeapPinSet {
    guards: Vec<HeapPinGuard>,
    /// Keep every allocation owner alive until after its guards are dropped,
    /// even if a cloned `TrainedVectorStructures` has its public maps mutated.
    owners: Vec<Arc<dyn std::any::Any + Send + Sync>>,
    report: PinReport,
}

impl HeapPinSet {
    pub(crate) fn report(&self) -> PinReport {
        self.report
    }

    pub(crate) fn retain_owner<T: std::any::Any + Send + Sync>(&mut self, owner: Arc<T>) {
        self.owners.push(owner);
    }

    /// Keep one immutable heap slice resident, subject to the generation
    /// budget. `Copy` mode needs no allocation: trained artifacts are already
    /// heap-owned, which is exactly the residency guarantee that mode provides
    /// on the supported swapless deployment.
    pub(crate) fn pin_slice<T>(
        &mut self,
        slice: &[T],
        label: &str,
        mode: PinMode,
        remaining: &mut u64,
    ) {
        let len = std::mem::size_of_val(slice);
        if len == 0 {
            return;
        }
        let Ok(len_u64) = u64::try_from(len) else {
            self.report.failed_bytes = u64::MAX;
            log::warn!("[pin] ANN region {label} is too large to account");
            return;
        };
        self.report.intended_bytes = self.report.intended_bytes.saturating_add(len_u64);
        if len_u64 > *remaining {
            self.report.skipped_budget_bytes =
                self.report.skipped_budget_bytes.saturating_add(len_u64);
            log::debug!(
                "[pin] ANN budget exhausted: skipping {} ({}, {} remaining)",
                label,
                crate::format_bytes(len_u64),
                crate::format_bytes(*remaining)
            );
            return;
        }

        if mode == PinMode::Copy {
            *remaining -= len_u64;
            self.report.pinned_bytes = self.report.pinned_bytes.saturating_add(len_u64);
            return;
        }

        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        let page_size = usize::try_from(page_size).ok().filter(|&size| size > 0);
        let Some(page_size) = page_size else {
            self.report.failed_bytes = self.report.failed_bytes.saturating_add(len_u64);
            log::warn!("[pin] cannot determine page size while locking {label}");
            return;
        };
        let address = slice.as_ptr() as usize;
        let page_start = address / page_size * page_size;
        let Some(end) = address.checked_add(len) else {
            self.report.failed_bytes = self.report.failed_bytes.saturating_add(len_u64);
            log::warn!("[pin] ANN region address overflow while locking {label}");
            return;
        };
        let Some(rounded_end) = end
            .checked_add(page_size - 1)
            .map(|value| value / page_size * page_size)
        else {
            self.report.failed_bytes = self.report.failed_bytes.saturating_add(len_u64);
            log::warn!("[pin] ANN region page range overflow while locking {label}");
            return;
        };
        let page_len = rounded_end - page_start;
        let page_start = page_start as *mut libc::c_void;
        if unsafe { libc::mlock(page_start.cast_const(), page_len) } == 0 {
            self.guards.push(HeapPinGuard {
                page_start,
                page_len,
            });
            *remaining -= len_u64;
            self.report.pinned_bytes = self.report.pinned_bytes.saturating_add(len_u64);
        } else {
            self.report.failed_bytes = self.report.failed_bytes.saturating_add(len_u64);
            log::warn!(
                "[pin] mlock failed for ANN {} ({}): {} — check RLIMIT_MEMLOCK/CAP_IPC_LOCK; continuing unpinned",
                label,
                crate::format_bytes(len_u64),
                std::io::Error::last_os_error()
            );
        }
    }
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
            "[pin] budget exhausted: skipping {} ({}, {} remaining)",
            label,
            crate::format_bytes(len),
            crate::format_bytes(*remaining)
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
                    "[pin] mlock failed for {} ({}) — check RLIMIT_MEMLOCK; \
                     continuing unpinned",
                    label,
                    crate::format_bytes(len)
                );
            }
        }
        PinMode::Copy => {
            *bytes = OwnedBytes::new(bytes.to_vec());
            *remaining -= len;
            report.pinned_bytes += len;
            report.heap_copy_bytes += len;
        }
    }
}
