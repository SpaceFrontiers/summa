# Cold IO for merges (hot-metadata-pinning Phase 2)

Status: design (2026-07-09), implemented.

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
buffered IO on some filesystems. The same eviction guarantee is achieved
with the write-behind discipline used by rsync/borg/RocksDB:

- **Linux**: every 64 MB window, `sync_file_range(WAIT_BEFORE|WRITE|WAIT_AFTER)`
  forces the window to storage, then `posix_fadvise(POSIX_FADV_DONTNEED)`
  drops its (now clean) pages. Steady-state page-cache footprint of a merge
  write is one window instead of the whole segment.
- **macOS**: `fcntl(fd, F_NOCACHE, 1)` at creation — the kernel bypasses the
  buffer cache for this file descriptor.
- Other platforms / non-fs directories: plain buffered writer (loudly
  logged once).

Reads: the standalone-reorder whole-file copy now drops source pages behind
the copy cursor (`MADV_DONTNEED` per copied chunk).

## Wiring

- `DirectoryWriter::streaming_writer_cold(path)` — call sites declare
  intent ("bulk one-shot data"); the default impl delegates to the normal
  `streaming_writer` (RAM/HTTP directories). `FsDirectory`/`MmapDirectory`
  always return a `ColdStreamingWriter` — this is the **default and only**
  behaviour for merge/reorder output, with no configuration.
- Used by all merge output files (postings, positions, term dict, store,
  fast, vectors, sparse) and all reorder output files.
- Observability: `hermes_cold_write_bytes_total` counter (metrics feature)
  and a per-file debug log of dropped bytes; the mechanism logs once at
  first use.

## Trade-off

A freshly merged segment starts serving with a cold page cache and warms on
demand — metadata is pinned at open by Phase 1, while BMP and clustered ANN
query paths prefetch only the selected scoring ranges (flat TQ retains its
deliberate sequential scan). That is the right trade under memory pressure:
bounded on-demand reads on the new segment instead of evicting the entire
serving working set during the merge.
