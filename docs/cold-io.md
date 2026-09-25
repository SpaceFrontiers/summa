# Cold IO for merges (hot-metadata-pinning Phase 2)

Status: design (2026-07-09), implemented.

These writers use buffered filesystem I/O and cache advice, not io_uring. The
[September 20 I/O audit](iresearch-optimization-audit.md#io_uring-source-and-running-host-findings)
traces the active server path and separates proposed async payload reads/writes
from the existing mmap reader.

## Problem

Phase 1 (`docs/hot-metadata-pinning.md`) pins per-query metadata so the
kernel cannot evict it. The remaining eviction source is bulk one-shot IO:

- **Merge/reorder writes**: writing a multi-GB merged segment dirties the
  entire output in page cache. The kernel makes room by evicting the
  currently-serving segments' warm pages — the merge output has zero reuse
  value until the swap-in, and even then only its hot subset matters.
- **Merge/reorder reads**: mostly handled already (`MADV_SEQUENTIAL` +
  `MADV_DONTNEED` on source sections after copy), except the whole-file
  copies in standalone reorder, which faulted entire source files in and
  left them resident.

## Mechanism: write-behind cache drop, not literal O_DIRECT

`O_DIRECT` requires sector-aligned buffers, offsets, and lengths, needs
special handling for the final unaligned tail, and silently degrades to
buffered IO on some filesystems. Summa instead limits cache pollution with a
write-behind discipline; writeback and eviction advice do not provide the same
cache-bypass guarantee as successful direct I/O:

- **Linux**: each completed 64 MB window starts asynchronous writeback with
  `sync_file_range(WRITE)`. One window later, `posix_fadvise(DONTNEED)` drops
  the preceding clean pages; `finish()` fsyncs and drops the tail. Steady-state
  intended page-cache footprint is bounded instead of growing with the segment;
  actual eviction depends on writeback progress and the kernel/filesystem.
- **Linux byte-identical ranges**: BMP block payloads and ordinal maps use
  `copy_file_range`, so they do not cross a userspace buffer or fault the
  source mmap merely to copy it. Filesystems that do not support the syscall
  fall back to bounded local-file reads before emitting bytes. Abstract
  directories retain the mapped-byte streaming-write path.
- **macOS**: `fcntl(fd, F_NOCACHE, 1)` at creation — the kernel bypasses the
  buffer cache for this file descriptor.
- Other platforms / non-fs directories: plain buffered writer (loudly
  logged once).

Reads: the standalone-reorder whole-file copy now drops source pages behind
the copy cursor (`MADV_DONTNEED` per copied chunk).

## Wiring

Partitioned Seismic output may hold 16 cold writers at once. The directory
owner provides `streaming_writer_cold_with_capacity` so local writers use
64 KiB each (1 MiB total), rather than allocating sixteen default 8 MiB buffers.
The ordinary cold writer retains its 8 MiB default. Filesystem capacities are
clamped to 1 byte through 8 MiB; large writes still bypass the userspace buffer.
Directory wrappers forward this policy and invalidate cached destinations.
Custom backends may use the default delegation; the capacity is a buffer hint,
not a limit on a RAM directory's owned output or on backend-specific caches.

- `DirectoryWriter::streaming_writer_cold(path)` — call sites declare
  intent ("bulk one-shot data"); the default impl delegates to the normal
  `streaming_writer` (RAM/HTTP directories). `FsDirectory`/`MmapDirectory`
  always return a `ColdStreamingWriter` — this is the **default and only**
  behaviour for merge/reorder output, with no configuration.
- Used by all merge output files (postings, positions, term dict, store,
  fast, vectors, sparse) and all reorder output files.
- Observability: `summa_cold_write_bytes_total` counter (metrics feature)
  and a per-file debug log of dropped bytes; the mechanism logs once at
  first use.

## Trade-off

A freshly merged segment starts serving with a cold page cache and warms on
demand — metadata is pinned at open by Phase 1, while BMP and clustered ANN
query paths prefetch only the selected scoring ranges (flat TQ retains its
deliberate sequential scan). That is the right trade under memory pressure:
bounded on-demand reads on the new segment instead of evicting the entire
serving working set during the merge.

## Shared local range copying

Encoded runs and temporary sparse directories use the same range-copy helper.
It prefers kernel-assisted copying. If that is unsupported before any bytes are
emitted, it reads from the same open source file through a staging buffer of at
most 4 MiB and streams those bytes to the cold writer. Abstract directories
retain their mapped-byte fallback. Scratch is bounded per active copy,
independent of corpus size; copying local files need not fault their source mmap.
The OS still controls page-cache residency.

The helper handles short and interrupted I/O, checks cancellation between chunks
and before writing a completed read, and never restarts after partial kernel
output. Source paths retain the existing contract that they name the same
immutable bytes as their admitted directory view. This does not change output
formats, cold-writer policy, admission or atomic publication.

## Seismic admission read-ahead

Opening completed cold output for structural admission and lifecycle statistics
can cost more than writing it. Seismic admission checks term-directory entries
and payload coverage first, then visits terms in physical file order. Maintenance
can write payloads in priority order while the term directory stays sorted by
dimension. The existing transient extent scratch also holds expected cluster
counts: 24 rather than 16 bytes per term on 64-bit hosts. Every term still receives
the same structural checks.

Linux mmap admission hints the current 4 MiB window and one 4 MiB lookahead using
`OwnedBytes::madvise_range`. Requests are at most 128 KiB, disjoint, and clamped to
validated payload extents. Small requests avoid treating a large accepted hint
as completed read-ahead when the kernel caps the individual request. Large terms
still receive all checks even when they exceed the lookahead. Heap, WASM and
other operating systems issue no new advice. There is no persistent mapping
policy change, payload heap copy, or bypass of source/output admission.

Measured whole-merge results and the Linux read-ahead diagnosis are recorded in
[the performance review](search-performance-review.md).
