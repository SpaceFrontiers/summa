# Summa Schema Definition Language (SDL)

SDL defines index fields, tokenizers, storage, and search options.

## Basic Syntax

An SDL file defines one or more indexes, each containing field definitions:

```
index <index_name> {
    field <field_name>: <field_type><tokenizer> [<attributes>]
    ...
}
```

## Example

```
# Article index schema
index articles {
    # Unique article URL (primary key, deduplicates on insert)
    field url: text [indexed, stored, primary]

    # Text fields with English stemming
    field title: text<en_stem> [indexed, stored]

    # Body content with default tokenizer
    field body: text<default> [indexed]

    # Author name - no stemming needed
    field author: text [indexed, stored]

    # Publication timestamp
    field published_at: i64 [indexed, stored]

    # View count
    field views: u64 [indexed, stored]

    # Rating score
    field rating: f64 [indexed, stored]

    # Caller-supplied content hash: skip unchanged upserts
    field content_hash: bytes [stored, content_hash]
}
```

## Field Types

| Type                  | Aliases            | Description                                |
| --------------------- | ------------------ | ------------------------------------------ |
| `text`                | `string`, `str`    | UTF-8 text, tokenized for full-text search |
| `u64`                 | `uint`, `unsigned` | Unsigned 64-bit integer                    |
| `i64`                 | `int`, `integer`   | Signed 64-bit integer                      |
| `f64`                 | `float`, `double`  | 64-bit floating point number               |
| `bytes`               | `binary`, `blob`   | Raw binary data                            |
| `json`                |                    | Arbitrary JSON values                      |
| `dense_vector`        | `vector`           | Dense float vectors                        |
| `binary_dense_vector` | `binary_vector`    | Packed bits for Hamming search             |
| `sparse_vector`       |                    | Sparse vector for learned sparse retrieval |

## Attributes

Attributes control how fields are processed and stored:

| Attribute      | Description                                                                                                                                                                                                            |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `indexed`      | Field is indexed for searching                                                                                                                                                                                         |
| `stored`       | Field value is stored and can be retrieved                                                                                                                                                                             |
| `content_hash` | Stored scalar fingerprint for unchanged upserts; requires a primary key                                                                                                                                                |
| `primary`      | Field is the primary key (enforces uniqueness, deduplicates)                                                                                                                                                           |
| `fast`         | Field is a fast field (column-oriented storage for range queries)                                                                                                                                                      |
| `reorder`      | Opt an indexed BMP sparse or text field (plain or chunked) into BP reordering. Sparse Seismic maintenance is scheduled from nomination debt. See [field-level reordering](lexical-vertical.md#field-level-reordering). |

Index-level options (inside the `index { ... }` block):

`max_l1_phrase_terms` sets the maximum retained tokens in each L1 phrase feature,
for both formula ranking and feature collection. Set it when creating the index:

```sdl
index documents {
    max_l1_phrase_terms: 256  # Default: 64
    field body: text<simple> [indexed<chunked, token_position>]
}
```

It accepts positive 32-bit integers, is persisted in `metadata.json` under
`schema.max_l1_phrase_terms`, and is included in the schema returned by index
info. Omission, including in existing metadata, means 64. The JSON creation
schema accepts the same top-level key alongside `fields`. Rust callers can use
`SchemaBuilder::set_max_l1_phrase_terms(NonZeroU32::new(256).unwrap())`.
The server's separate `--max-text-query-tokens` limit still applies (default 256).
See [candidate rescoring](candidate-rescoring.md) for scoring and memory bounds.

```sdl
index articles {
    reorder_on_merge: true   # Apply configured text reorder policy during merge.
                             # Absent = disabled: merges block-copy and the background
                             # optimizer reorders afterwards.
    field body: text<simple> [indexed, reorder]
}
```

### Attribute Syntax

Attributes are specified in square brackets after the field type:

```
field name: text [indexed, stored]    # Both indexed and stored
field name: text [indexed]            # Indexed only (not stored)
field name: text [stored]             # Stored only (not indexed)
field name: text                      # Default: indexed and stored
field id: text [indexed, stored, primary]  # Primary key field
```

### Primary Key

The `primary` attribute designates a field as the primary key. When a primary key is defined:

- Documents with duplicate primary key values are rejected during indexing
- The server automatically initializes deduplication tracking on index open
- At most one field per index may be marked as `primary`
- Requires a single-valued `text` field; implies `fast` and `indexed`

```
index articles {
    field url: text [indexed, stored, primary]
    field title: text<en_stem> [indexed, stored]
    field body: text [indexed]
}
```

### Content Hash

Mark one stored, single-valued `text`, `bytes`, or `u64` field with `content_hash`
to skip reindexing when its primary key and hash match the latest staged or
committed live document.
The hash does not need `indexed` or `fast`. Equality is exact and case-sensitive;
the caller must change the hash whenever any part of the full document changes.
Missing hashes cause normal replacement. Wrong types and multiple values error.
JSON documents encode byte hashes as standard padded Base64 strings; invalid
hashes (including explicit nulls or arrays) are rejected.

```sdl
index documents {
    field id: text<raw> [primary, stored]
    field digest: bytes [stored, content_hash]
    field body: text [indexed]
}
```

The JSON creation schema exposes the same settings:

```json
{
  "fields": [
    { "name": "id", "type": "text", "primary_key": true },
    { "name": "digest", "type": "bytes", "stored": true, "content_hash": true }
  ]
}
```

Native Rust uses `builder.set_content_hash(digest)` and
`writer.upsert_document(doc).await?`. A hash matching the latest staged or
committed version of the same key succeeds without new rows or tombstones.
Staged rows can be replaced or deleted without committing; a pending deletion
always permits its replacement insertion.
Ordinary inserts still reject duplicate primary keys. RPC accepted counts include
no-ops. See [content deduplication](content-deduplication.md) for cost and lifecycle
guarantees. Hash comparison remains opt-in; staged replacement and deletion also work without the marker.

### Multi-Value Fields

Fields can store multiple values per document using `stored<multi>`:

```
field tags: text [indexed, stored<multi>]
field embeddings: dense_vector<768> [indexed, stored<multi>]
```

Multi-value fields are useful for documents with multiple embeddings (e.g., chunked passages) or multiple values for the same attribute. Dense and sparse vector queries support configurable multi-value combiners (Sum, Max, Avg, LogSumExp, WeightedTopK).

### Default Behavior

If no attributes are specified, fields default to **both indexed and stored**.

## Tokenizers

Text fields can specify a tokenizer using angle brackets after the type:

```
field title: text<en_stem> [indexed, stored]    # English stemmer
field body: text<german> [indexed]              # German stemmer
field name: text<simple> [indexed, stored]      # Lowercase, punctuation stripped
field raw: text [indexed, stored]               # Default tokenizer (simple)
```

Unknown tokenizer names are rejected when the schema is parsed.

### Dynamic per-document stemming

A single field can hold documents in many languages and stem each one with
its own Snowball algorithm. The stemmer is selected per document from the
values of another text field, and per query from the `tokenizer_hint`
argument of `TermQuery`, `MatchQuery` and `PhraseQuery`:

```
field languages: text<raw_ci> [fast]
field content: text<lex(by: languages, default: en, stop_words: true)> [indexed<token_position>]
```

- `by` names a text field of the same index; all of its values in a document
  form the hint (`"ru,en"`).
- `default` is the language used when a document (or query) carries no
  recognised hint; `simple` means no stemming.
- `stop_words` (default `false`) drops the routed language's stop words
  before stemming. Surviving tokens keep their positions, so phrase queries
  keep the original word distances (`"quantum of the art"` is
  `quantum@0 art@3` on both sides and does not match `quantum art`).
- `segmenter` (default `simple`): `unicode` splits at UAX #29 word
  boundaries (`float-zero` → `float`, `zero`), bigrams runs of Han, Hiragana
  and Katakana, and folds diacritics of Latin, Cyrillic and Greek tokens
  after stemming.
- `reorder` on indexed text lets the reorder pass permute the field's virtual
  units with Recursive Graph Bisection over its own postings (smaller postings,
  tighter block bounds). Plain and chunked text are supported. Only that field's
  postings, positions and unit map are rewritten; logical document IDs never
  move. The attribute also supports indexed BMP sparse fields. Seismic and binary ANN
  maintenance follow their own debt automatically; other field types reject it.
- Each token is stemmed with the first hinted language whose script matches
  the token (Snowball stemmers are script-local), so Cyrillic and Latin text
  in one document both stem correctly; same-script languages use the first
  listed one.
- `indexed<token_position>` enables phrase queries on the field.
- `indexed<k1: 0.9, b: 0.4>` sets the field's BM25 parameters (defaults
  1.2 and 0.75; `b` must lie in 0..=1). They apply to every BM25 path
  (MaxScore bounds and scores, term and phrase scorers) at query time; the
  stored block bounds are parameter-free, so no rebuild is needed to change
  them.

See `docs/dynamic-tokenizer-and-phrase.md` for the design.

### Chunked text fields

A multi-valued text field declared `chunked` indexes **every value as its own
BM25 unit**, the way sparse and dense vector fields treat every value as one
vector:

```
field content: text<lex(by: languages, default: en)> [indexed<chunked, token_position>]
```

- Postings, IDF and the average length are computed over chunks; BM25 uses
  each chunk's real token count for length normalisation.
- Hits report the matching chunks as `ordinal_scores` (ordinal `n` = the
  `n`-th value sent for the field), so `FusionQuery` compounds a chunk found
  by a text query and by a vector query when both fields receive their values
  in the same chunk order, and clients can render the matching passage.
- The document score is the best chunk (`Max`).
- `token_position` keeps phrase queries per chunk; positions restart in every
  value, so a phrase never matches across two chunks. `positions` and
  `ordinal` are rejected on chunked fields (the chunk is the ordinal).
- `chunked` implies `multi`; prefix (`term*`) queries are not supported on
  chunked fields.

See `docs/chunked-text-fields.md` for the design.

### Available Tokenizers

| Name                          | Aliases      | Description                                                      |
| ----------------------------- | ------------ | ---------------------------------------------------------------- |
| `simple`                      | `default`    | Whitespace split, punctuation stripped, lowercased               |
| `unicode_word`                |              | UAX #29 words and lowercase; preserves internal word punctuation |
| `raw`                         |              | Whole value as one token, unchanged                              |
| `raw_ci`                      |              | Whole value as one token, lowercased                             |
| `lex(by: F, default: L, ...)` |              | Lexical tokenizer hinted by field `F` (see above)                |
| `en_stem`                     | `english`    | English Snowball stemmer                                         |
| `de_stem`                     | `german`     | German Snowball stemmer                                          |
| `fr_stem`                     | `french`     | French Snowball stemmer                                          |
| `es_stem`                     | `spanish`    | Spanish Snowball stemmer                                         |
| `it_stem`                     | `italian`    | Italian Snowball stemmer                                         |
| `pt_stem`                     | `portuguese` | Portuguese Snowball stemmer                                      |
| `ru_stem`                     | `russian`    | Russian Snowball stemmer                                         |
| `ar_stem`                     | `arabic`     | Arabic Snowball stemmer                                          |
| `da_stem`                     | `danish`     | Danish Snowball stemmer                                          |
| `nl_stem`                     | `dutch`      | Dutch Snowball stemmer                                           |
| `fi_stem`                     | `finnish`    | Finnish Snowball stemmer                                         |
| `el_stem`                     | `greek`      | Greek Snowball stemmer                                           |
| `hu_stem`                     | `hungarian`  | Hungarian Snowball stemmer                                       |
| `no_stem`                     | `norwegian`  | Norwegian Snowball stemmer                                       |
| `ro_stem`                     | `romanian`   | Romanian Snowball stemmer                                        |
| `sv_stem`                     | `swedish`    | Swedish Snowball stemmer                                         |
| `ta_stem`                     | `tamil`      | Tamil Snowball stemmer                                           |
| `tr_stem`                     | `turkish`    | Turkish Snowball stemmer                                         |

### Custom Tokenizers

`unicode_word` performs no stemming, normalization, elision, or CJK expansion.
Words longer than 255 Unicode scalar values are skipped while retaining the
position gap. It uses the Unicode version of the bundled `unicode-segmentation`
crate; it does not promise exact Lucene StandardAnalyzer equivalence. Changing
an existing field to this tokenizer requires reindexing its source documents.

You can register custom tokenizers programmatically:

```rust
use summa_core::TokenizerRegistry;

let registry = TokenizerRegistry::new();
registry.register("my_tokenizer", MyCustomTokenizer::new());
```

## Comments

Line comments start with `#`:

```
# This is a comment
index articles {
    # Title field for searching
    field title: text [indexed, stored]
}
```

## Multiple Indexes

A single SDL file can define multiple indexes:

```
index articles {
    field title: text [indexed, stored]
    field body: text [indexed]
}

index users {
    field name: text [indexed, stored]
    field email: text [indexed, stored]
    field created_at: i64 [indexed, stored]
}
```

## CLI Usage

### Create index from SDL file

```bash
summa-tool create -i ./myindex -s schema.sdl
```

### Create index from inline SDL

```bash
summa-tool init -i ./myindex -s 'index test { field title: text [indexed, stored] }'
```

## Grammar (PEG)

The authoritative grammar is [sdl.pest](../summa-core/src/dsl/sdl/sdl.pest).

## Dense Vectors

Dense vector fields store high-dimensional embeddings for semantic search. Vectors are quantized on write and scored using native-precision SIMD (no dequantization on the hot path).

### Syntax

```
field embedding: dense_vector<DIM> [indexed]              # f32 (default)
field embedding: dense_vector<DIM, f16> [indexed]         # half-precision, 2× less storage
field embedding: dense_vector<DIM, uint8> [indexed]       # scalar quantized, 4× less storage
```

### Quantization Types

| Type    | Alias | Bytes/dimension |
| ------- | ----- | --------------: |
| `f32`   |       |               4 |
| `f16`   |       |               2 |
| `uint8` | `u8`  |               1 |

This controls vector value storage; ANN codes have their own layout. Measure
recall on your corpus before reducing precision.

### Index Types and Routing

Float fields have three ANN formats. Two are built on the TurboQuant codec
(`docs/turboquant-quantization.md`). `ivf_tq` (the default) combines the
corpus-trained coarse centroid router with training-free TQ leaf codes of
each normalized vector's centroid residual — only centroids are trained,
never a codebook. Training samples, ANN copies, and queries are normalized
for cosine routing while original stored values remain available for exact
reranking. `tq` is fully training-free: each segment carries a 4-bit
compressed payload built at commit with no global artifacts, scanned
exhaustively with SIMD lookup tables and exact-reranked. Ordinary segment
merges copy both formats' immutable run columns byte-for-byte and rewrite
only their compact document-base directory. `flat` remains available for
exact brute-force search and as the accumulation format before an `ivf_tq`
build. `scann` uses an index-wide hierarchical partitioner and shared
asymmetric-hashing codebook. Every immutable segment references the same
trained generation, so commits only route/encode new vectors and merges copy
compatible leaf runs without retraining.

Only the cosine-normalized IVF-TQ generation is supported. Older unmarked
trained generations and ANN payloads are rejected while opening the index;
rebuild the index with a current Summa version.

`ivf_pq` (residual product quantization) was removed after IVF-TQ superseded
it on recall, latency, and training cost; indexes created with it must be
recreated with `ivf_tq` and reindexed.

```
field e: dense_vector<768, f16> [indexed]                                      # global IVF-TQ
field e: dense_vector<768, f16> [indexed<ivf_tq, routing: hnsw, nprobe: 64>]
field e: dense_vector<768, f16> [indexed<tq>]                                  # training-free TQ
field e: dense_vector<768, f16> [indexed<scann, num_clusters: 10000000, tree_levels: 2, nprobe: 1024>]
field e: dense_vector<768, f16> [indexed<flat>]                                # exact full scan
field e: dense_vector<768> [stored]                                            # stored, not indexed
```

`tq` ignores `num_clusters`, `nprobe`, `routing`, and `soar` (it scans every
code) and warns when they are set; `rerank_factor` applies unchanged.
`ivf_tq` accepts all IVF knobs.

For `scann`, `num_clusters` is the terminal leaf count and `tree_levels` is
the routing-tree depth. Both may be omitted for corpus-size autopilot. Explicit
depths must be in `1..=3`, explicit leaf counts must not exceed 30 million, and
`nprobe` must not exceed an explicit leaf count. Training readiness is derived
by the runtime from the requested geometry and available sample; it is not a
schema knob. `tree_levels` is an error on other index types rather than an
ignored setting.

At approximately one billion vectors, a high-selectivity explicit geometry is:

```
field e: dense_vector<1024, f16> [
  indexed<scann, num_clusters: 10000000, tree_levels: 3,
          nprobe: 1024>
]
```

An explicit ten-million-leaf tree requires at least 80 million sampled rows
(the internal eight-rows-per-leaf floor), so its operational training memory
and sample limits must be raised accordingly. The desired quality sample is
200 rows per leaf when the corpus and limits permit it.

When `num_clusters` is omitted, autopilot follows AlloyDB's recall-oriented
balanced geometry: `ceil(sqrt(rows))` below 100 million rows,
`ceil(rows^(2/3))` through one billion, and
`min(30,000,000, ceil(rows^(3/4)))` above one billion. A one-billion-row field
selects 1,000,000 leaves in two centroid levels. Builder sample/memory ceilings
never silently change that shared topology: the build fails or remains
untrained until it can retain the hardcoded eight-samples-per-leaf minimum.
Crossing 100 million rows requires 1,723,552 samples; crossing one billion
requires 44,987,312. `tree_levels` can still pin a depth from one through three.
The default `nprobe` is 64; intermediate routing keeps a 64-parent recall floor
and widens it when necessary to keep the full requested probe count reachable.
Exact reranking streams candidate values through fixed-size buffers.

### SOAR (higher recall for IVF indexes)

IVF-TQ supports SOAR — spilling each vector
into a secondary cluster with an orthogonality-amplified residual. This
improves recall at the same `nprobe` in exchange for larger cluster storage
(~1.3× assignments for the build-calibrated selective preset, 2× for full).
Omitting `soar` enables the selective preset by default: training calibrates
one secondary assignment for at most 30% of the calibration sample. Use
`soar: off` to disable secondary assignments explicitly:

```
field e: dense_vector<768, f16> [indexed<ivf_tq>]                   # default: selective, at most 30% spill
field e: dense_vector<768, f16> [indexed<ivf_tq, soar: selective>]  # calibrate to at most a 30% spill budget
field e: dense_vector<768, f16> [indexed<ivf_tq, soar: full>]       # spill every vector once
field e: dense_vector<768, f16> [indexed<ivf_tq, soar: aggressive>] # full one-secondary spill
field e: dense_vector<768, f16> [indexed<ivf_tq, soar: off>]        # explicitly disable spilling
```

### Example

```
index documents {
    field title: text<en_stem> [indexed, stored]
    field embedding: dense_vector<768, f16> [indexed<ivf_tq>, stored<multi>]
    field span: json [stored<multi>]
}
```

### Binary Vector IVF and ScaNN

Binary dense vector fields use the same global IVF router and HNSW topology as
float fields, with metric-specific k-majority centroids and exact packed-code
leaf scanning. The only approximation is which clusters get probed; there is
no lossy PQ stage or rerank for exact Hamming codes. Use `flat` explicitly for
brute-force SIMD Hamming scan. Binary `scann` uses hierarchical Hamming
partitioning and exact XOR-popcount leaf scoring; it does not expand packed
bits to floats or apply a float AH codec. It accepts the same `num_clusters`,
`tree_levels`, and `nprobe` parameters as float ScaNN. Float-only options such
as `soar` are rejected:

```
field hash: binary_dense_vector<512> [indexed<ivf, routing: hnsw, nprobe: 64>]
field hash: binary_dense_vector<1024> [indexed<scann, num_clusters: 10000000, tree_levels: 2, nprobe: 1024>]
field hash: binary_dense_vector<512> [indexed<flat>]
```

An existing IVF or ScaNN field can be changed atomically through the
`AlterVectorIndex` gRPC method. Pass the index name, field name, and the field's
replacement SDL type/options, for example
`dense_vector<768, f16> [indexed<scann, tree_levels: 2, nprobe: 1024>]`.
The response reports whether the ANN generation was built immediately, kept
flat until the geometry has enough training vectors, or only changed search
parameters.

## Sparse Vectors

Sparse vector fields store learned sparse representations (SPLADE, uniCOIL, etc.) using an inverted index with quantized weights. BMP is the default. Explicit `format: maxscore` selects the shared Block-Max MaxScore query pipeline; `format: seismic` selects geometric approximate nomination with exact forward scoring. BMP uses UInt8 impacts; the `quantization` setting selects stored precision for MaxScore and Seismic.

### Syntax

```
field sparse_emb: sparse_vector [indexed]
field sparse_emb: sparse_vector [indexed, stored]
```

Sparse vectors are indexed as posting lists with float weights. At query time, you can provide raw `(indices, values)` pairs or raw text (tokenized server-side if a HuggingFace tokenizer is configured).

### Document Mass Cropping

`doc_mass` crops the excessive low-weight tail of each document's sparse vector
at indexing time: entries are ranked by |weight| and only the head covering the
given fraction of the vector's total |weight| mass is kept. SPLADE-style vectors
often concentrate importance in a few head terms, but the quality impact is
corpus-dependent. Measure Recall@K or judged relevance before enabling it.
Vectors with at most `min_terms` entries are never cropped.

```
field emb: sparse_vector [indexed<quantization: uint8, doc_mass: 0.9>]
```

Use `summa-tool info --index <path>` to inspect the resulting average sparse vector
length (`avg terms/vector`).

### BMP Format Options

The BMP (block-max pruning) format takes additional `indexed<...>` options:

```
field emb: sparse_vector<u32> [indexed<format: bmp, dims: 105879, max_weight: 5.0,
    bmp_block_size: 32, bmp_grid_bits: 4,
    query<lsp_gamma: 1000>>, reorder]
```

- `bmp_block_size` (power of two, max 256; default 32) — vectors per BMP
  block, uniform across every segment of the field. Larger blocks reduce the
  number of locally bit-packed maximum cells but also coarsen pruning.
  Multi-valued fields count each ordinal as a vector. Do not increase the block
  size solely from document count; benchmark relevance and tail latency first.
- `bmp_grid_bits` (2 or 4; default 4) — bits per block-grid upper bound.
  Two-bit bounds are ceil-quantized and remain rank-safe, but are looser. They
  cap compressed D payload widths at two bits; the exact saving is
  workload-dependent.
- `bmp_forward_index` (boolean; default `true`) — store quantized vectors for
  L1 candidate backfill and BP reads. Set `false` to omit this additional storage
  from ingestion, merge and BP output. Ordinary BMP search uses inverted postings
  regardless of this setting. Without forward values, L1 backfill requires a
  logically ordered BMP document map; BP-reordered fields require enabling
  storage and explicit reorder/rebuild, or `l1.backfill: false`. Existing files
  are immutable. See [optional forward storage](bmp-forward-index.md#configuration-and-scoring).
- `query<pruning: 0.33>` — retain the highest-weight third of query
  dimensions for BMP candidate generation, matching the LSP/0 zero-shot beta.
  Visited documents are still scored with the bounded full query. This is
  disabled by default because candidate and block selection can still lose
  recall; enable it only after a representative Recall@K benchmark.
- `query<lsp_gamma: N>` — cap traversal to the global top-N superblocks across
  all physical segments. When omitted, Summa derives gamma from candidate
  depth (3000 through depth 100, 4000 through depth 1000, then the greater
  of 4000 and depth). Set zero for exhaustive traversal. Query-language
  `emb:sparse({...})` searches inherit this field setting, including zero.

See `docs/bmp-grid-compression.md` for current size formulas, LSP/0 behavior,
merge/reorder invariants, and the Flat-Inv versus Fwd design choice.

### Quantization and Pruning

Sparse posting lists support configurable weight quantization and pruning via `SparseVectorConfig`:

| Preset         | Quantization | Destructive pruning                   |
| -------------- | ------------ | ------------------------------------- |
| `conservative` | Float16      | disabled                              |
| `splade`       | UInt8        | disabled                              |
| `compact`      | UInt4        | enabled; benchmark quality before use |

These are configured programmatically, not in SDL.

### Seismic Format Options

```sdl
field embedding: sparse_vector<u32> [indexed<format: seismic, quantization: float32,
    seismic_postings: 4096, seismic_cluster_size: 64,
    seismic_summary_energy: 0.4,
    query<seismic_cut: 10, seismic_factor: 0.85, exhaustive: false>>]
```

For Seismic, the default is approximate nomination followed by exact candidate scoring.
`query<exhaustive: true>` scans the same forward values exhaustively. Query
configuration applies to both query-language and vector API searches. Sparse
queries have at most 64 effective dimensions; query pruning changes nominations,
while candidate scores use the full bounded query.

- `seismic_postings`: maximum retained postings per term in a new run, 1–65536.
- `seismic_cluster_size`: target cluster size, 1–`seismic_postings`.
- `seismic_summary_energy`: retained summary magnitude fraction, (0,1].
- `seismic_forward_compression`: lossless adaptive dimension compression, default
  `true` (`false` keeps raw U32 IDs). Uses U16/U24 or aligned gaps while preserving full U32 IDs and the
  configured weight precision; see [forward compression](seismic-forward-compression.md).
- `seismic_cut`: nomination query dimensions, 1–64.
- `seismic_factor`: summary pruning factor, [0,1]. Approximate recall must be
  measured on representative queries; exact scoring does not make nominations exact.

`quantization` selects Float32 (default), Float16, UInt8 or UInt4 storage using
the shared sparse weight codecs. Exact scoring refers to these stored values.
`dims` optionally bounds vocabulary IDs. Forward values are mandatory; retrieval,
all ordinal combiners, backfill and maintenance use this one encoded copy.

`doc_mass`, `weight_threshold`, and `pruning` change stored values; see
[document mass cropping](#document-mass-cropping) and measure recall before use.

Ordinary merge copies encoded runs without clustering. The existing background
optimizer services nomination fragmentation in bounded term passes; retained
reader generations remain valid across replacement. `summa-tool diagnose`
reports nominations, clusters, runs, encoded bytes and pending term debt.

## JSON Fields

JSON fields store arbitrary JSON values. They must be `stored` (not indexed):

```
field metadata: json [stored]
field spans: json [stored<multi>]    # Multi-value JSON
```
