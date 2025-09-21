# Content-hash deduplication

Implemented: opt-in `content_hash` on one stored, single-valued
text, bytes, or u64 field, alongside a text primary key. The hash is supplied by
the caller and compared exactly, without normalization or hash computation.
Equal hashes assert equality of the complete document, including indexed-only
fields. Missing hashes on either side cause normal replacement; malformed or
multiple hash values are rejected before mutation. Ordinary insertion retains
its existing duplicate-key behavior.

Upsert compares the latest accepted version of the same primary key: a staged
row takes precedence over the committed snapshot. Equal hashes on different
keys do not deduplicate those documents. A pending deletion disables comparison:
its replacement must be inserted. Upserts can replace staged rows repeatedly;
ordinary add still rejects duplicate keys.
Equal hashes return success without queuing a document or staging a deletion.
They count as accepted operations in RPC responses and emit a debug log.
Changed/missing hashes follow the existing atomic deletion-plus-insertion path.
Abort, failed publication, refresh recovery, and merge preserve their existing
ownership and reservation rules.

Native upsert becomes async, matching portable upsert, because the stored field
may require I/O for committed content. Staged comparison uses its retained hash
without reading stored blocks. Lookup retains the primary-key snapshot through the read; no
parking-lot guard is held across await. All I/O precedes mutation; cancellation
or corruption cannot stage a deletion. The server retains its bounded blocking
workers and writer lease, driving the async call from the worker's runtime
handle. No transport messages or client signatures change.

The PK component owns exact key-to-live-row resolution. For opted-in schemas it
builds an ordinal-to-row table from the existing fast-column decoder at snapshot
load, with four bytes per dictionary entry up to 64 MiB per segment (16,777,216 keys). Larger
segments log a warning and resolve rows with a constant-scratch column scan.
The table is compact heap metadata; old and new tables overlap during refresh.
No committed-corpus hash cache is introduced. Stored fields use the existing lazy reader,
field-selective deserializer, bounded decompressor and store cache (32 MiB shared
by native content-hash readers; portable reads do not retain decompressed blocks). At most four PK segment loads/builds run concurrently.
Lookup costs one dictionary search per candidate segment, constant-time row
resolution when the table fits, and one stored block fetch/decompression on a
cache miss. Startup/visibility refresh adds a column scan. These are cost models,
not measured latency claims. The fallback costs a column scan per matching key.

Schema serialization uses a defaulted optional field marker, introduced in
metadata format 8. Current format 9 upgrades formats 6–8 without rewriting
segment payloads; older writers cannot silently drop the setting. Validation
applies on SDL/JSON parsing, core creation, and metadata load. Hashes use the
existing store; no sidecar, second writer, or hash index is introduced.

Validation covers physical row/mask stability for equal hashes, changed/missing
hashes, pending insertions/deletions, old readers, abort, reopen, topology changes,
read failures and cancellation, schema round trips, and portable parity.

## Enabling an existing stored hash

An existing index with a compatible primary key and stored hash needs only a
metadata change; its segment payloads already contain everything required.
Close the index and its writers before editing `metadata.json`. Add
`"content_hash": true` to exactly one existing entry in `schema.fields` and set
the top-level `version` to `9` (never downgrade a newer format). Preserve field
order, types, name mappings, and all other metadata. Reopen with the new build to load the updated schema and
build the primary-key lookup. Do not patch metadata while a writer or background
maintenance can publish another generation.

The existing hash field must already be stored, single-valued, and text, bytes,
or u64; the existing primary key must be single-valued text with indexed and fast
storage. Merely changing those storage flags cannot add missing encoded columns.
Rows without a stored hash are replaced normally on their next upsert. Automatic
format migration from versions 6–8 leaves deduplication disabled unless the
field is explicitly marked. No enablement command or RPC is required.

## Stored-block allocation correction

The first fixture found that bounded Zstd decoding reserved the entire 256 MiB
safety limit for each small store block. Retained capacity consequently exceeded
the store cache's admission limit, forcing repeated decompression. The correction
uses the frame's declared size (when available) capped by the caller's hard
limit; unknown sizes start with the existing 512 KiB capacity. Unknown frames
that outgrow that initial buffer retain the bounded streaming fallback. The output limit and all encoded bytes stay unchanged.
This belongs in the existing compression component, not a hash-specific decoder.

## Latest staged version

The writer compares the hash for the latest accepted version of the same
primary key, including queued, building, and flushed-but-unpublished rows.
Different primary keys remain independent. Ordinary add still rejects duplicate
keys; upsert and delete may replace or remove a staged row without committing.
A rejected admission must preserve the previous staged version. Replacing a staged ID never forces a commit; unrelated pending work stays unpublished.

Each pending key owns a small row handle and its optional exact hash under a
64 MiB pending-key metadata budget. Payload documents still move through the
existing bounded queue and builders. Replaced rows not yet consumed can be
skipped; rows already encoded are hidden by the existing deletion-mask format.
Handles attach to a per-builder list of cancelled row IDs, which remains mutable
while the segment is unpublished. Commit freezes the resulting visibility and publishes
it with committed-row deletions in the existing owned transaction. Abort releases
all handles and unpublished outputs. No segment or metadata format changes are
required. Each superseded staged row reserves sixteen bytes in the same 64 MiB
budget, covering row-ID storage and vector growth even if the row is still queued.
Commit materializes one visibility bitmap at a time, with one bit per physical
row in that segment; clean segments allocate no bitmap. Published masks retain
the existing reader-residency cost.

The admission lock covers reservation checks, bounded queue admission, and the
latest-version change. Only accepted replacements invalidate the old row. A row
handle serializes worker attachment against cancellation, with lock order
primary-key state → row handle → cancellation list. Workers never acquire primary-
key state. Portable writers use the same handles and preserve their poison and
publication-recovery rules. Prepared native generations reject further mutation;
failed publication retains the frozen cancellation lists for retry. A successful publication
clears pending state independently of fallible cache refresh, as before.

The metadata budget conservatively charges keys, hashes, handles, superseded-row
records, and hash-table capacity (including capacity retained after commit) before cloning a hash or admitting a document. Exceeding 64 MiB returns
an explicit commit-required error and leaves the latest staged version intact.
Encoded cancelled rows remain physical until compaction; their postings, indexed-
only fields, vectors, and chunks use the existing visibility filter. Matching
staged hashes add neither rows nor tombstones.
