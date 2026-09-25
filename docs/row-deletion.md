# Row deletion, upserts, and compaction

Implemented in native/portable core, CLI, gRPC server/broker, Python/TypeScript,
and WASM LocalIndex. Current metadata is **format 9**; formats 6–8 upgrade on
open without rewriting segments. Readers upgrade in memory and warn; writers
persist the stamp, after which older builds cannot open the index. Formats 7,
8, and 9 introduced deletion masks, content hashes, and compact text/norm flags.
Segments written before 1.8.125 still need rebuilding (BMP blob magic);
segments without `.rowstats` require a merge before compaction.
Native mutation APIs require an initialized primary-key index; server and WASM
writers initialize it automatically. Cross-shard atomic upserts are not supported.

## Feasibility and identity

Tantivy's immutable live-document bitset applies to Summa. A deletion changes
visibility, not postings or stored values. A replacement document is an insertion
plus a deletion published in the same metadata transaction. Primary keys remain
stable through merges; `(segment_id, doc_id)` addresses are snapshot-local.

Summa differs from Tantivy in three consequential ways: primary keys have a
uniqueness index, fields can use independent chunk/virtual IDs, and trained ANN
payloads must retain their global artifact identity. Filtering only final hits
would underfill top-k, let dead candidates raise pruning thresholds, and affect
fusion. Reindexing the document store cannot implement general compaction:
indexed-only fields are absent from it, and merge must not retrain ANN artifacts.
All current field representations can be compacted directly, including fields
with `indexed: true, stored: false`.

## Mutation API

The replacement operation is named **upsert**: insert when the primary key is
absent, otherwise replace the complete document and all its chunks. The native
and portable writer expose async `upsert_document`; clients expose their corresponding
single/batch helpers; the CLI command is `upsert`; IndexService exposes
`UpsertDocuments(UpsertDocumentsRequest)`.

```rust,ignore
let mut writer = index.writer();
writer.init_primary_key_dedup().await?;
writer.delete_primary_key("old-key")?;
writer.upsert_document(replacement).await?; // complete document, including its key
writer.commit().await?;
reader.reload().await?;

writer.compact(256 * 1024 * 1024).await?; // each dirty segment separately
writer.compact_segment(&segment_id, 256 * 1024 * 1024).await?;
writer.force_merge().await?; // combines segments, preserving tombstones
writer.force_merge_with_compaction(true).await?; // merge, then compact final outputs
```

Deleting a missing key is idempotent. Upserting a missing key inserts it. Upserts
replace the entire document; they are not partial field patches. A rejected
queue admission rolls back that call's staged deletion. Commit publishes all
staged deletions and replacement insertions together; prepared-commit abort
rolls them back together. Keys are exact, case-sensitive primary-field values.

With an optional [content-hash field](content-deduplication.md), a hash matching
this key's latest staged or committed live row is an accepted no-op. A pending
delete disables comparison until another replacement is accepted.

Upsert and delete apply to the latest accepted version, including queued,
building, and flushed-but-unpublished rows. Multiple replacements of the same
key can share one commit; ordinary add still rejects duplicates. A rejected
replacement preserves the previous staged row. Commit publishes only the latest
live version, while abort discards the complete pending sequence. Rows already
encoded are hidden using the existing masks; queued cancelled rows can be skipped.
Pending deletions are bounded at 100,000 distinct keys and 8 MiB of key bytes;
each mutation key must contain 1–65,536 bytes. Pending primary-key metadata,
including retained content hashes, is separately bounded at 64 MiB.

Primary-key admission requires exactly one text value, including ordinary
inserts with deduplication enabled. Inserts and upserts share the same key
validation and size bound, before reserving a key or staging deletion. This
prevents the reservation key from differing from the persisted single-value
column, and ensures that every admitted key can later be deleted.

## Mutation surfaces

`DeleteDocuments(index_name, primary_keys)` and
`UpsertDocuments(index_name, documents)` stage operations through IndexService.
Both return `DocumentMutationResponse { accepted_count, errors }`; errors retain
zero-based input positions. Accepted counts describe operations, not affected
rows: deleting a missing key is accepted. Every input is either accepted or has
an error. Partial acceptance is explicit and a subsequent `Commit` publishes the
accepted operations. A failed RPC can have an unknown outcome; callers must not
blindly retry upserts before resolving/committing pending work.

Deletion requests admit at most 100,000 keys and 8 MiB of key bytes. Replacement
requests admit at most 1,000 documents and 32 MiB of encoded protobuf data.
A singleton replacement may use 200 MiB including its request envelope; batch
concurrency and mutation admission remain unchanged. Both
server and broker check envelope limits before lookup, conversion or admission.
The core's cumulative pending-key limits still apply across requests. Server
mutation batches hold the existing exclusive writer lock through
staging on blocking workers (async stored-hash reads are driven by their runtime handle); cancellation while waiting starts no operation.
Conversion and staging share four application-wide admission permits, retained
by started workers even if their request is cancelled. Commit retains its
existing owned finalizer and reader publication. Batch conversion is bounded and
runs off the async executor.

The broker routes deletion keys with the same FNV-1a partition hash used for
insertions and routes replacements by their primary field. It maps per-shard
errors to original input positions. There is no transaction across partitions;
a backend failure can leave work staged on successful partitions. No automatic
mutation retry is introduced.

Python exposes `delete_document(s)` and `upsert_document(s)`; TypeScript exposes
`deleteDocument(s)` and `upsertDocument(s)`. Single-item helpers raise on rejected
operations; batch helpers return `DocumentMutationResult` (`accepted_count` in Python,
`acceptedCount` in TypeScript, plus `errors: [{index, error}]`).
All require an explicit commit. The CLI and native writer retain their APIs.

WASM LocalIndex provides primary-key enforcement, `deleteDocument(s)`,
`upsertDocument(s)`, and `abort`. Its portable core writer reuses the native key
reservation/Bloom component, fast-column deletion scanner and visibility encoder.
It runs inline with exclusive mutable access. A failed/cancelled builder flush
poisons the pending transaction until abort, so a missing replacement can never
publish its deletion. Visibility and replacement files precede the atomic
metadata publication; PK refresh is prepared before publication. Storage-adapter
commit failures can be retried without replaying row mutations. Old masks are
retired only after publication. A cancelled metadata save reconciles the durable
publication generation before any cleanup or replay. Reopen reclaims unreferenced
segment artifacts left by failed writers. LocalIndex tracks attempted storage
writes through failure and retries synchronization even when core commit has no
new mutations. The storage adapter must atomically replace each file and have one
active writable LocalIndex per storage namespace; RemoteIndex and IpfsIndex remain read-only
readers, consuming the same masks as native indexes.

## Visibility and publication

`SegmentMeta.num_docs` remains the physical address-space bound.
`SegmentReader::num_live_docs` and searcher counts subtract tombstones. Search,
hydration, candidate validation, and primary-key checks use the same immutable
visibility generation. Existing searchers retain their old visibility until
released. BM25 statistics remain physical until compaction. Compaction changes
corpus/term statistics, so surviving BM25 scores can then change.

Each nonempty generation is `seg_<independent-id>.del`: an eight-byte versioned
magic, physical and deleted u32 counts, little-endian u64 live-bit words, and an
eight-byte FNV-1a integrity checksum. A set bit means live. The file occupies
`24 + 8 * ceil(physical_rows / 64)` bytes. Readers validate identity, version,
length, population, tail padding, and integrity. An absent metadata reference
means all rows live. A missing/corrupt referenced file fails the open.

Bitset IDs participate in the existing active-operation, metadata, and reader
tracker ownership protocol. The owned commit finalizer publishes references
atomically with newly built segments. Snapshots protect exact file generations;
ownership transfers before retirement. Started compaction workers retain claims
and admission permits through cancellation; shutdown drains them before removal.
The existing doctor validates deletion files when checking a segment.

Raw `SegmentReader::open` opens physical segment files without index metadata.
Use `IndexReader` for snapshot ownership, or `open_with_deletions` with a known
metadata generation when inspecting standalone files.

## Bloom filters and primary keys

Bloom filters remain monotonic negative filters; deletion never clears Bloom
bits. A positive is checked against the exact fast-field dictionary and, for a
dirty segment, a bitmap of live dictionary ordinals. That bitmap is built once
per visibility refresh by scanning the primary-key column. Duplicate checks
remain dictionary lookup plus constant-time membership, without a row scan on
each insertion. Deleted Bloom entries are harmless false positives.

Visibility identity is part of PK cache reuse, even when the data segment ID
has not changed. Bloom-cache reopen reconstructs live-key bitmaps from committed
masks. Merge/compaction refreshes the topology while preserving pending key
reservations. Published deletion requests are cleared independently of fallible
PK cache refresh, preventing a retry from deleting the replacement row itself.

## Merge and compaction

Ordinary merges, including force merge, copy compatible encoded representations
and carry the latest source masks into the output. They do not physically remove
deleted rows. `ForceMergeRequest.compact = true` explicitly compacts final outputs
once, after the normal merge hierarchy, including a singleton. CLI `merge --compact`
and client helpers expose the same flag; omitted/false preserves the cheap default.

Compaction writes directly from each final source, without a temporary copied
segment. A concurrent deletion invalidates captured visibility and prevents stale
publication. Single-segment `compact` preserves segment separation. An all-deleted
segment becomes an empty segment. Old row addresses require their original reader.
The physical-copy primitive rejects masked readers; the segment manager owns mask
remapping and publication for normal merges.

Compaction preserves the relative order of surviving physical rows, chunk/value
ordinals, and BMP records in each field's current virtual order. Dense labels are
remapped monotonically; trained ANN codes and routing artifacts are unchanged.
The normal merge planner chooses source concatenation order; it does not promise
global insertion or field-sorted order. Configured merge-time BP may change BMP
order before compaction; compaction preserves that resulting order.
BMP is identity-reblocked, so block membership and pruning statistics are rebuilt.
A previously reordered segment remains `reordered = true`, but a surviving BMP
layout becomes `bp_converged = false`; compaction does not increment or reset its
`bp_unconverged_passes`. Non-BMP/empty outputs retain their existing BP state.
Normal field-only reorder preserves physical row IDs and carries deletion masks;
a later BP pass records convergence and attempt counts through the existing rules.

## Automatic compaction and observability

Implemented by the server's existing periodic optimizer, including indexes without
reorder fields. The configurable deleted-row ratio threshold defaults to 0.30;
zero disables automatic compaction. Selection uses committed metadata counts,
not a payload scan. Eligible segments are considered before fresh BP work, while
the existing cooldown-eligible deepening slot retains its priority. A segment
selected for compaction is excluded from BP candidates in that scan.

Compaction shares optimizer task slots, the process-wide background CPU pool,
merge capacity, and the whole-pass reorder gate. At most one automatic compaction
runs globally, with a configurable completion-based cooldown (default 60 seconds).
Admission is nonblocking; manual force merges and busy/quarantined sources are
skipped. Existing capped exponential retry backoff also applies to compaction failures
(30 seconds initially, up to 30 minutes; retries continue while the source remains eligible).
Retry entries belong only to current committed segments. Publication removes
retired entries under the same state lock used to admit a failure record, so a
late failure cannot recreate backoff for an already replaced source.
The default scratch cap is 256 MiB (`--compaction-memory-budget-mb`), separate from
the much larger BP budget. Cancellation retains source/output ownership and permits
until blocking writes drain. No new scheduler loop or worker pool is introduced.

Per-segment metadata already records physical and deleted counts. Core helpers
report live/deleted rows and deleted ratio; index-info RPC reports aggregate
physical/deleted counts and their weighted ratio. Broker aggregation sums counts
before computing the ratio. Compaction completion logs record input rows, removed
rows, share, output rows and elapsed time; failures remain observable.

| Representation                          | Treatment                                                                                                       |
| --------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| Text/numeric postings                   | Skip dead blocks; copy compatible gaps/TFs and position blocks; rebuild affected blocks and conservative bounds |
| Chunked text                            | Drop dead rows' chunks, preserve value ordinals, remap virtual IDs and lengths                                  |
| Fast fields                             | Copy intact blocks/local dictionaries; batch-rebuild mixed blocks with missing/multi-value semantics            |
| Stored fields                           | Copy intact dictionary-free compressed blocks; recompress affected or dictionary-dependent blocks               |
| BMP                                     | Filter record maps and reuse the bounded BMP writer with identity reblocking; copy surviving forward payloads   |
| MaxScore sparse                         | Remap addresses; preserve quantized weight bits and their original affine scale/minimum                         |
| Flat vectors                            | Copy surviving encoded values and remap document labels and ordinals                                            |
| TQ, IVF-TQ, binary IVF, ScaNN AH/binary | Preserve codes, assignments, scales, codebooks and fingerprints; repack partial SIMD blocks                     |

New `.rowstats` numeric columns preserve information absent from the old formats:
for each indexed text field, zero means missing and `token_count + 1` records
presence and the full per-row token count; sparse fields record vector counts,
including empty/pruned vectors. Existing u16 norms cannot distinguish missing
from empty text or recover long token counts. These columns use the existing
fast-field format, remain evictable, and are copied by normal merge/reorder.
Legacy standalone segments without required row statistics explicitly refuse
compaction; rebuilding from stored fields would lose indexed-only information.

## Cost and validation

### Compaction copy and streaming paths

Compaction retains the current metadata format, stable survivor ordering, exact encoded vector
values, and the existing publication/cancellation protocol. Physical row mapping
shares the reader's visibility bitmap and adds one u32 population-count prefix
per 64 rows plus a sentinel: `4 * (ceil(physical_rows / 64) + 1)` bytes for mixed
visibility. Forward mapping uses a rank lookup. Survivor traversal scans set bits;
posting bounds are computed from the source rows without reverse lookups. Identity
and empty mappings retain no map allocation. Chunk mappings own a bitmap plus
the same prefix table, about 0.1875 bytes per physical chunk. Their surviving
chunk labels/norms are admitted separately before allocating output builders.

Column compaction preserves complete live source blocks and their local
dictionaries, omits dead blocks, and batch-decodes mixed blocks in source ranges
of at most 4096 rows. Text values use the source block's dictionary without
materializing a merged global dictionary. The existing column encoder owns all
rebuilding. Block boundaries may change, but copied payload bytes and decoded
values are preserved.

Posting compaction retains admitted encoded directories and reads payload blocks
on demand. Entirely dead document intervals skip payload I/O. A fully live
interval has a constant row-ID shift, so its header/directory can be remapped
while copying encoded gaps and frequencies. A deletion inside a block's address
span requires rebuilding gaps even when the deleted row did not match that term.
Mixed blocks decode at most 128 postings with the existing codec. Their survivors
fill a shared 128-posting output buffer across mixed source blocks; the buffer
flushes before a copied block or at the end of the term. This avoids retaining
one tiny block per sparse group of survivors. A fixed 16 KiB reserve covers
posting/position codec buffers before the remaining budget is divided among
encoded directories. Current position
streams copy complete selected blocks and decode at most one partial block at a
time. Output cursors, directories and conservative score bounds are rebuilt.
Decoded survivors/positions are never retained for an entire current-format term.
Encoded directories remain budgeted. Unsupported position formats are rejected;
old indexes must be rebuilt.

Flat-vector payloads use bounded batches (at most 4 MiB), skip all-dead batches
before I/O, and copy contiguous surviving slices within mixed batches. ANN
ordinals and binary codes also use interval copies. Clean ANN runs and aligned
TQ/ScaNN packed groups preserve bytes; unaligned groups use the owning repacker.
BMP forward payloads copy survivor intervals. Identity BMP reblocking skips the
unused inverse map and forward graph; its surviving record map and identity
permutation remain budgeted allocations. BMP block membership/pruning metadata
still require rebuilding.

Ordinary merges, background thresholds, concurrency and ANN training are
unchanged. Fusing the final merge and compaction remains separate work.

Visibility takes one bit per physical row per held reader generation, plus the
writer's mask and live-key bitmap for dirty PK segments. Mask loading has a
transient encoded input buffer. Masks are mandatory heap state, separate from
the optional metadata pin budget; ordinary heap residency does not imply mlock.
New builder scratch adds eight bytes per row/statistic field. Encoded row-statistic
values are compressed and file-backed; their block directories are accounted in
reader heap diagnostics.

Deletion resolves each requested key to a dictionary ordinal once per segment,
then batch-scans affected primary-key columns using the existing decoder and
global ordinal remapping. Its target set is bounded by the pending-key limit;
it avoids decoding and hashing strings for every row. Cancellation interrupts
the scan, and the manager retains the publication lock until the atomic commit
completes. Deletion writes masks/metadata, not corpus payloads.
The physical row space is bounded by u32. Compaction requires at least 1 MiB
scratch; at most a quarter is admitted for the physical map's prefix directory.
Chunk maps, column blocks, store decompression, posting/position directories,
sparse directories and ANN packing have additional budget checks. Oversized work
errors before publication rather than silently truncating data. Shared maintenance
capacity bounds concurrent compactions. The scratch limit does not include the
immutable input reader's existing residency or the output directory's storage
(for example, output bytes retained by a RAM directory).

Compaction retains a bounded prefix of rebuilt fast-column chunks to avoid
encoding them again after writing the leading block directory. It reserves at
most a quarter of remaining scratch for that cache (including vector capacities),
a quarter for descriptors, and half for the chunk encoder. Uncached mixed chunks
retain the deterministic two-pass fallback. Cached and uncached paths produce
identical output bytes. Clean blocks require neither pass through the encoder.
Profiling the small numeric-only merge fixture also found repeated construction
of the FST registry for an empty term dictionary. The canonical empty block
index bytes are cached once through the existing encoder, copying this fixed-size artifact for
subsequent empty outputs. Nonempty dictionaries keep their existing build path.

Regression coverage includes old searchers, reopen, indexed-only text and vectors,
missing/multi-value fields, chunk ordinals/positions, sparse scoring, dense top-k,
single/all-deleted compaction, Bloom reuse, abort/retry/cancellation, and concurrent
delete/compaction. ANN tests compare complete output bytes for all five encoded
kinds; sparse tests compare weight bits across all four quantizations.
Performance evidence and remaining work are recorded in the
[performance review](search-performance-review.md).

References: [search contract](search-system-contract.md),
[segment lifecycle](segment-lifecycle.md),
[Tantivy architecture](https://github.com/quickwit-oss/tantivy/blob/main/ARCHITECTURE.md).
