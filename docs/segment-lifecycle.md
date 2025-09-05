# Segment Lifecycle and Recovery

Status: design and operational contract (2026-07-15), implemented.

Hermes treats publication, replacement, cleanup, and deletion as one segment
ownership protocol. This document defines that protocol and the operator-facing
failure behavior. The central rule is simple:

> Every on-disk `seg_<id>.*` file set must have at least one lifecycle owner.
> Cleanup may delete an ID only after every owner is absent.

## Owners and transitions

| Owner            | Protects                                                                     | Released when                                                                    |
| ---------------- | ---------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| metadata         | A live, searchable segment in `metadata.json`                                | An atomic replacement generation is published                                    |
| active operation | Indexing output, merge/reorder sources, and its unpublished output           | Publication or the operation finishes; ordinary aborts retain it through cleanup |
| segment tracker  | Retired files still reachable by an existing reader, plus scheduled deletion | The last reader drops and deletion completes                                     |

The required transitions are:

```text
indexing: active(output) -> metadata + tracker -> release active

rewrite:  metadata(sources) + active(sources, output)
          -> metadata(output) + tracker(output, retired sources)
          -> release active -> delete sources after last reader

abort:    active(output) -> tracked deletion -> release active
panic:    active(output) -> release active + tracked idempotent cleanup/sweep
```

The next owner is installed before the previous owner is released. Merge and
reorder also claim all sources, which prevents overlapping rewrites of the same
segment. Newly generated outputs are claimed before their first file write, so
partial builders are protected from an orphan sweep too.

Row deletion uses this same protocol. Each immutable `.del` generation has an
independent tracked ID referenced by segment metadata; snapshots retain both the
data segment and its exact mask. Layout-preserving rewrites carry the latest
mask, while compaction checks its captured visibility before replacement. The
`.rowstats` file belongs to its data segment's ordinary file set. See
[row deletion and compaction](row-deletion.md) for the format and API contract.

## Metadata is the commit point

Metadata mutations are serialized. A generation is written and fsynced as
`metadata.json.tmp`, then atomically renamed over `metadata.json`. The rename is
the logical commit point. The matching in-memory state and tracker transition
continues even if the requesting RPC is canceled; shutdown tracks and drains
that transaction.

A directory-fsync error after a successful rename is reported as degraded
crash durability, not as a rolled-back commit. Returning a pre-commit error at
that point would be unsafe: disk readers could already observe metadata that
references an output which an error path then deletes.

Indexing flush generations are all-or-nothing. The first worker build error or
panic marks the generation failed, the other workers drain their queues, every
successful sibling output is abandoned through the same tracked cleanup path,
and no partial generation is published. A timeout leaves the writer paused on
the same generation so a late worker cannot acknowledge the next commit by
mistake; core returns `CommitFlushTimeout` and the caller retries commit.

The server acquires an owned writer guard before starting `Commit` and moves
it into a task that outlives the RPC. That task retries only worker-flush
timeouts, observing the same finite queued generation at five-minute intervals;
it does not spin, admit another generation, or restart failed builds. A client
deadline/disconnect cancels the waiter, not the accepted commit. Reader reload
is included in the owned operation. Build/publication failures are logged and
returned, not retried indefinitely. Concurrent commits wait for the same writer
lock, and cancellation before acquiring that lock starts no operation.

Once a prepared indexing commit starts, an owned finalizer—not the requesting
RPC—holds its segment guards. It completes metadata publication, refreshes the
primary-key view, and resumes workers in that order even if the request is
canceled. Ingestion receives explicit backpressure until the finalizer ends.
If publication fails before the atomic rename, the same prepared segments stay
owned and workers remain paused for a lossless retry; a new indexing generation
cannot be mixed into them. After the rename, optional cache-refresh failures
are fail-closed and cannot turn a durable commit into an apparent abort.

## Cancellation and shutdown

Tokio cannot cancel a `spawn_blocking` closure after it has started. Dropping or
aborting only its async wrapper while deleting an index would let filesystem
writes continue into a removed directory. Hermes therefore drains blocking
merge/reorder work instead of pretending to cancel it.

Process shutdown first rejects new registry requests and stops optimizer scans.
It waits for accepted commits to release their owned writer guards before
closing segment-build admission and stopping writers/managers. Manager shutdown
signals BP cancellation before joining the optimizer supervisor. This preserves
pending commit flushes without waiting for an uncancelled full BP pass. Forced
process termination is still outside the guarantee; deployments must allow
adequate termination grace and drain/commit ingestion before restarting.

Index deletion follows this order:

1. Acquire the per-index registry lease and create `.deleting`, serializing
   against open/create.
2. Stop accepting new lifecycle claims and signal maintenance cancellation,
   before waiting for issued handles. A Reorder can hold an index handle while
   waiting for shared BP capacity; waiting for that handle first prevents the
   cancellation that would release it.
3. Drain search and writer handles already issued by the registry.
4. Signal and join indexing OS threads.
5. Drop unpublished prepared segments and cached readers/writer handles.
6. Drain merge/reorder operations, metadata transactions, and deferred deletes.
7. Remove the index directory.

The delete transaction is detached from the requesting RPC after the marker is
installed, so client cancellation does not abandon a live writer. If the
process exits during step 7, the `.deleting` marker causes the remaining
directory to be removed on the next server startup.

An evicted index's marker is checked before waiting for its open/delete lease,
and again after acquiring the lease. A request that obtained an index handle
just before deletion must fail its later writer lookup promptly; it cannot wait
for deletion while retaining a handle that deletion needs to drain.

Manual Reorder commits the admitted indexing generation while holding the
writer lock, then releases that lock before waiting for maintenance capacity or
rewriting segments. Core retains the shared writer handle and uses the existing
segment claims, publication, snapshot refresh and primary-key refresh path.
Manual BP uses the same bounded CPU pool as background maintenance.
Concurrent ingestion can publish new segments while the bounded maintenance
snapshot is processed. Deletion still drains the operation before unlinking its
files; neither a client timeout nor cancellation is evidence that blocking work
has stopped.

## Orphans versus corruption

An orphan is a segment ID absent from metadata, active operations, and the
reader/deletion tracker. Exclusive writer open and the background optimizer may
sweep it. The sweep deletes every discovered path belonging to that ID,
including unknown legacy or partial suffixes, and safely ignores malformed
names. Read-only `Index::open` does not mutate the directory.

Registry open uses core's locked writer opener: it acquires the OS single-writer
lock before reading metadata or sweeping crash leftovers. The returned index
and writer share that same segment manager, so there is no stale pre-lock
metadata snapshot or second lifecycle owner.
Its per-name mutex serializes callers within one server process; it cannot prove
that another process has stopped producing files in the same directory.

A missing file for a **metadata-live** segment is not an orphan. Normal cleanup
must never make the metadata internally consistent by silently dropping its
documents. Merge preflight validates every mandatory source file before costly
CPU work; deterministic missing/corrupt sources are quarantined for the process
lifetime and excluded from candidate selection. The entry remains visible for
diagnosis and explicit recovery.

Other merge failures back off exponentially from 30 seconds to 30 minutes and
schedule their own wakeup. A failed output is deleted only after rechecking
under the metadata lock that it was not published. This prevents both an
immediate retry/core-saturation loop and cleanup of an already committed output.
Standalone optimizer failures use the same exponential delay per source,
measured from pass completion; otherwise a pass longer than the scan interval
would restart almost immediately. Deterministic reorder corruption enters the
same process-lifetime quarantine as a corrupt merge source.

Reorder copies distinguish absent optional files from failures: only a genuine
`NotFound` for a file the source reader did not observe is skipped. Required
files, permission/storage errors, short copies, and invalid optional formats
fail the output. Before replacement publication, Hermes opens the complete
output segment and verifies its document count; metadata is never switched to
an output that only passed a shallow `.meta` check.

Budget-truncated BP is a successful lifecycle transition, not a failure retry.
Its replacement metadata carries `bp_unconverged_passes`; the optimizer admits
only lineages below `--optimizer-max-unconverged-passes`. Thus both failure
retries and successful deepening have explicit, finite scheduling bounds.

## Operator recovery

Repeated “quarantined metadata-live segment” or mandatory-file errors mean the
index already has a broken metadata reference. Preserve/copy the directory
before repair if the documents are not reproducible. Then stop normal traffic
and run the server once with:

```bash
hermes-server --data-dir /data --doctor
```

Doctor opens every metadata-live segment, removes entries that cannot be
validated, atomically saves the repaired metadata, and deletes their remaining
files. This is intentionally destructive and may reduce the document count; it
is not part of normal startup cleanup.

For BP CPU and memory sizing, see [Budgeted reordering](budgeted-reorder.md).
For page-cache behavior during lifecycle rewrites, see [Cold IO](cold-io.md).

## Maintainer checklist

Any new segment-producing or deleting path must answer all of these:

- Is the output claimed before the first write?
- Does ownership survive success, error, cancellation, and panic?
- Is publication one durable metadata transaction with matching tracker state?
- Does abandoned-output cleanup recheck metadata before deletion?
- Are blocking tasks registered before shutdown can observe the task list empty?
- Does index deletion drain the task rather than only aborting its wrapper?
- Can the normal orphan sweep prove that metadata, active operations, and
  readers/deferred deletion no longer own the ID?
- Is a deterministic corrupt source quarantined while transient failures back
  off and wake themselves?

## Compaction admission and visibility

Ordinary merge publication concatenates current source visibility masks while
retaining physical row counts. Address-preserving single-source rewrites reuse
the exact immutable mask generation, transferring its metadata ownership before
retiring the source. Explicit compaction claims the source/output and snapshots
its mask; publication rejects a changed visibility generation. It writes directly
from the source and carries global/local maintenance and optimizer-class permits
inside the owned lifecycle task until blocking work drains. The server's existing
optimizer uses nonblocking admission and a completion cooldown for automatic
compaction. Force-merge RPC admission remains cancellable while waiting for the
writer; after acquiring it, the owned operation retains its guard through reader
refresh even after client cancellation. CLI row-mutation and merge commands stop
workers, release writer snapshots, and await core cleanup before exiting their
runtime, on both successful and failed maintenance. See
[row deletion](row-deletion.md) for the API and ordering rules.
