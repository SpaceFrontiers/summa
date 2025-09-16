# Compact text storage and quantized norms

## Status and invariant

Implementation experiment, not a measured replacement for the current default.
The baseline is the September 16 admission/position-directory build. RGB remains
disabled for comparisons. Existing formats remain readable. Unknown revisions
fail before query execution. Encoded payloads are copied during ordinary merges;
only small addressing metadata is remapped. Scratch and retained caches remain
bounded. Native, async and portable readers share the format owners.

## Position directory

POS4 separates packed payloads from structural metadata. Every block has a
two-byte descriptor: seven bits for count minus one, six for width, and one for
the existing full-block SIMD codec. The remaining two bits must be zero.
Every eight blocks have a checkpoint `(payload_offset: u32, value_start: u64)`.
The existing 16-byte footer stores block count, total values and POS4 magic.
Descriptors follow checkpoints. A directory costs `12*ceil(blocks/8)+2*blocks`
bytes instead of POS3's 16 bytes per block (header plus directory).

Explicit POS4 deserialization checks descriptors, checkpoint adjacency, exact
payload extent and total values. Normal query opens trust those invariants. Seeking needs at most seven local
descriptor additions after a checkpoint lookup. Fresh full-block streams retain
direct logical cursor addressing. Copied short interior blocks remain legal.
The existing position encoder owns both legacy and compact output. Merging
POS3 into POS4 relocates headers into descriptors and copies each packed payload;
there is no value decode. A single-source copy may retain its original format.
Range compaction decodes only partial retained blocks, as today.

## Posting directory

The first posting revision retains the 16-byte L0 directory and moves the block
header into a four-byte descriptor `(count:u16, doc_codec_width:u8, tf_width:u8)`
after L0. The first document is already in L0 and is no longer duplicated in the
payload. A footer flag distinguishes this layout. A second flag selects four-byte
position cursors when the total position count fits u32; larger streams retain
eight-byte cursors. This saves eight bytes per positioned block in the common
case. Fixed-width structural admission only reads directory pages. Pfor sources
retain the old layout for now because their exception framing lives in payloads.
The SIMD first-gap check applies to explicitly deserialized lists, not trusted
query views.

Existing decoders receive the descriptor and borrowed payload; they do not
materialize a legacy term-sized byte buffer. Compact descriptors share the
existing borrowed L0 slice. Decoding and content validation reuse one header and
payload lookup per block, rather than keeping a second reference-counted slice
or repeating those lookups. Merges may rebuild headers while
copying the encoded arrays when combining old and compact streams. Payload
bytes and codec choices remain identical. Further short-list/footer compression
requires separate measurements; this revision does not claim Tantivy-sized
postings or a faster decoder.

## Quantized normalization

A versioned byte norm uses the monotone 256-entry byte4 representatives, rounded
down. Exact total token statistics and physical chunk geometry are preserved.
Old u16 sections retain their original semantics. New sections have an explicit
kind and file version; no reader-open corpus-sized conversion is allowed.
New ordinary text columns opt in through `IndexConfig::quantized_norms` (default
false). Chunked and explicitly reorderable fields retain their existing geometry
and normalization in this implementation. The BM25 owner prepares a 256-float
normalization table from the query's actual
average length and field parameters. Scalar and batch scoring share its formula.
Zero norms preserve the existing TF fallback.

Posting minimum lengths, ratios and impact envelopes must be constructed from
the same representative lengths used by scoring. Mixed-version merges preserve
source scoring semantics or reject incompatible conversion; they must not copy
exact-length bounds under quantized scoring. Ranking changes are measured against
the old fixture separately from exact pruned-versus-exhaustive checks on the new
fixture. All-byte-norm merges copy codes. Mixed u16/byte columns expand only the
byte representatives to u16 under bounded scratch; they do not requantize the
old column. Row compaction retains byte encoding when its norm columns all use
it. Scoring, token totals and copied pruning bounds remain consistent.

## Acceptance

Test old/new/mixed streams, every cursor through short copied blocks, unchanged
payload bytes, malformed descriptors/checkpoints, overflow, truncated input,
write failure, cancellation and compaction budgets. Run the search harness and
portable compilation and the WASM build/test harness. The September 17 review
adds a native-produced compact SIMD/byte-norm fixture for WASM.
Compare identical corpus and queries on ARM and x86, including latency, RSS,
file-component bytes and ranking overlap. Size savings alone do not establish a
speedup. Record retained and rejected experiments in the performance review.

## Trusted query reads

Normal segment reads trust Hermes writers for document order, skip bounds,
position checkpoints and payload semantics. Opening a term parses its footer and
constructs borrowed views; it does not scan every block, retain admission proofs,
or verify decoded document order. This applies equally to sync, async and WASM.
There is no full index scan at open and no posting validation cache setting.
Explicit `BlockPostingList::deserialize` and `PositionStream::open` remain strict,
as does merge admission. Altered index payloads are outside the ordinary search
contract: search is not an integrity audit. Format versions, I/O failures and
input/output extents needed by unsafe SIMD still have checked boundaries.
The persisted encodings and writer bytes are unchanged. Posting opens still
copy the small L1 arrays (one entry per eight blocks) into the existing search
representation; removing validation does not eliminate that allocation. Position
opens inspect only the footer and final checkpoint group (at most seven descriptors).
