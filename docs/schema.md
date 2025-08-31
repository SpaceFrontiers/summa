# Hermes Schema Definition Language (SDL)

Hermes uses a simple, readable schema definition language for defining index schemas. This document describes the SDL syntax and features.

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

    # Raw content hash (not indexed, just stored)
    field content_hash: bytes [stored]
}
```

## Field Types

| Type            | Aliases            | Description                                   |
| --------------- | ------------------ | --------------------------------------------- |
| `text`          | `string`, `str`    | UTF-8 text, tokenized for full-text search    |
| `u64`           | `uint`, `unsigned` | Unsigned 64-bit integer                       |
| `i64`           | `int`, `integer`   | Signed 64-bit integer                         |
| `f64`           | `float`, `double`  | 64-bit floating point number                  |
| `bytes`         | `binary`, `blob`   | Raw binary data                               |
| `json`          |                    | Arbitrary JSON values                         |
| `dense_vector`  |                    | Dense float vector for ANN search (see below) |
| `sparse_vector` |                    | Sparse vector for learned sparse retrieval    |

## Attributes

Attributes control how fields are processed and stored:

| Attribute | Description                                                                                                                                                                                                     |
| --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `indexed` | Field is indexed for searching                                                                                                                                                                                  |
| `stored`  | Field value is stored and can be retrieved                                                                                                                                                                      |
| `primary` | Field is the primary key (enforces uniqueness, deduplicates)                                                                                                                                                    |
| `fast`    | Field is a fast field (column-oriented storage for range queries)                                                                                                                                               |
| `reorder` | Opt this BMP sparse field into BP (graph bisection) reordering — used by the background optimizer, `hermes-tool reorder`, and reorder-on-merge. Fields without it keep insertion order (blob copied unchanged). |

Index-level options (inside the `index { ... }` block):

```sdl
index articles {
    reorder_on_merge: true   # BP-reorder `reorder`-attributed BMP fields inside merges.
                             # Absent = disabled: merges block-copy and the background
                             # optimizer reorders afterwards.
    field splade: sparse_vector<...> [indexed, reorder]
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
- Only one field per index should be marked as `primary`
- Works with `text`, `u64`, `i64`, and `bytes` field types

```
index articles {
    field url: text [indexed, stored, primary]
    field title: text<en_stem> [indexed, stored]
    field body: text [indexed]
}
```

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
field content: text<stem(by: languages, default: simple)> [indexed<token_position>]
```

- `by` names a text field of the same index; all of its values in a document
  form the hint (`"ru,en"`).
- `default` is the language used when a document (or query) carries no
  recognised hint; `simple` means no stemming.
- Each token is stemmed with the first hinted language whose script matches
  the token (Snowball stemmers are script-local), so Cyrillic and Latin text
  in one document both stem correctly; same-script languages use the first
  listed one.
- `indexed<token_position>` enables phrase queries on the field.

See `docs/dynamic-tokenizer-and-phrase.md` for the design.

### Available Tokenizers

| Name                      | Aliases      | Description                                        |
| ------------------------- | ------------ | -------------------------------------------------- |
| `simple`                  | `default`    | Whitespace split, punctuation stripped, lowercased |
| `raw`                     |              | Whole value as one token, unchanged                |
| `raw_ci`                  |              | Whole value as one token, lowercased               |
| `stem(by: F, default: L)` |              | Dynamic stemmer hinted by field `F` (see above)    |
| `en_stem`                 | `english`    | English Snowball stemmer                           |
| `de_stem`                 | `german`     | German Snowball stemmer                            |
| `fr_stem`                 | `french`     | French Snowball stemmer                            |
| `es_stem`                 | `spanish`    | Spanish Snowball stemmer                           |
| `it_stem`                 | `italian`    | Italian Snowball stemmer                           |
| `pt_stem`                 | `portuguese` | Portuguese Snowball stemmer                        |
| `ru_stem`                 | `russian`    | Russian Snowball stemmer                           |
| `ar_stem`                 | `arabic`     | Arabic Snowball stemmer                            |
| `da_stem`                 | `danish`     | Danish Snowball stemmer                            |
| `nl_stem`                 | `dutch`      | Dutch Snowball stemmer                             |
| `fi_stem`                 | `finnish`    | Finnish Snowball stemmer                           |
| `el_stem`                 | `greek`      | Greek Snowball stemmer                             |
| `hu_stem`                 | `hungarian`  | Hungarian Snowball stemmer                         |
| `no_stem`                 | `norwegian`  | Norwegian Snowball stemmer                         |
| `ro_stem`                 | `romanian`   | Romanian Snowball stemmer                          |
| `sv_stem`                 | `swedish`    | Swedish Snowball stemmer                           |
| `ta_stem`                 | `tamil`      | Tamil Snowball stemmer                             |
| `tr_stem`                 | `turkish`    | Turkish Snowball stemmer                           |

### Custom Tokenizers

You can register custom tokenizers programmatically:

```rust
use hermes_core::TokenizerRegistry;

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
hermes-tool create -i ./myindex -s schema.sdl
```

### Create index from inline SDL

```bash
hermes-tool init -i ./myindex -s 'index test { field title: text [indexed, stored] }'
```

## Grammar (PEG)

The SDL is parsed using [pest](https://pest.rs/). The complete grammar:

```pest
file = { SOI ~ index_def+ ~ EOI }

index_def = { "index" ~ identifier ~ "{" ~ field_def* ~ "}" }

field_def = { "field" ~ identifier ~ ":" ~ field_type ~ (sparse_vector_config | dense_vector_config | binary_dense_vector_config | tokenizer_spec)? ~ attributes? }

field_type = {
    "text" | "string" | "str" |
    "u64" | "uint" | "unsigned" |
    "i64" | "int" | "integer" |
    "f64" | "float" | "double" |
    "binary_dense_vector" | "binary_vector" |
    "bytes" | "binary" | "blob" |
    "json" |
    "sparse_vector" |
    "dense_vector" | "vector"
}

tokenizer_spec = { "<" ~ identifier ~ tokenizer_params? ~ ">" }
tokenizer_params = { "(" ~ tokenizer_param ~ ("," ~ tokenizer_param)* ~ ")" }
tokenizer_param = { identifier ~ ":" ~ identifier }

attributes = { "[" ~ attribute ~ ("," ~ attribute)* ~ "]" }
attribute = { indexed_with_config | "indexed" | stored_with_config | "stored" | "fast" | "primary" | "reorder" }

identifier = @{ (ASCII_ALPHA | "_") ~ (ASCII_ALPHANUMERIC | "_")* }

WHITESPACE = _{ " " | "\t" | "\r" | "\n" }
COMMENT = _{ "#" ~ (!"\n" ~ ANY)* }
```

## Dense Vectors

Dense vector fields store high-dimensional embeddings for semantic search. Vectors are quantized on write and scored using native-precision SIMD (no dequantization on the hot path).

### Syntax

```
field embedding: dense_vector<DIM> [indexed]              # f32 (default)
field embedding: dense_vector<DIM, f16> [indexed]         # half-precision, 2× less storage
field embedding: dense_vector<DIM, uint8> [indexed]       # scalar quantized, 4× less storage
```

### Quantization Types

| Type    | Aliases | Bytes/dim | Recall impact | Use case            |
| ------- | ------- | --------- | ------------- | ------------------- |
| `f32`   |         | 4         | Baseline      | Maximum precision   |
| `f16`   |         | 2         | <0.1% loss    | Recommended default |
| `uint8` | `u8`    | 1         | 1-3% loss     | Maximum compression |

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
rebuild the index with a current Hermes version.

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

Sparse vector fields store learned sparse representations (SPLADE, uniCOIL, etc.) using an inverted index with quantized weights. They support the same Block-Max MaxScore query pipeline as BM25 text fields.

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

Use `hermes-tool info <index>` to inspect the resulting average sparse vector
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
- `bmp_grid_bits` (2 or 4; default 4) — bits per block-grid upper bound.
  Two-bit bounds are ceil-quantized and remain rank-safe, but are looser. They
  cap compressed D payload widths at two bits; the exact saving is
  workload-dependent.
- `query<pruning: 0.33>` — retain the highest-weight third of query
  dimensions for BMP candidate generation, matching the LSP/0 zero-shot beta.
  Visited documents are still scored with the bounded full query. This is
  disabled by default because candidate and block selection can still lose
- `query<lsp_gamma: N>` — cap traversal to the global top-N superblocks across
  all physical segments. When omitted, Hermes derives gamma from candidate
  depth (250/500/1000 for depths 10/100/1000). Set zero for exhaustive
  traversal.

See `docs/bmp-grid-compression.md` for current size formulas, LSP/0 behavior,
merge/reorder invariants, and the Flat-Inv versus Fwd design choice.

### Quantization and Pruning

Sparse posting lists support configurable weight quantization and pruning via `SparseVectorConfig`:

| Preset         | Quantization | Destructive pruning                   |
| -------------- | ------------ | ------------------------------------- |
| `conservative` | Float16      | disabled                              |
| `splade`       | UInt8        | disabled                              |

These are configured programmatically, not in SDL.

## JSON Fields

JSON fields store arbitrary JSON values. They must be `stored` (not indexed):

```
field metadata: json [stored]
field spans: json [stored<multi>]    # Multi-value JSON
```

## Best Practices

1. **Use descriptive field names** - Field names should clearly indicate their purpose
2. **Only store what you need** - Use `[indexed]` without `stored` for large fields you only search but don't retrieve
3. **Use appropriate types** - Use numeric types for numbers to enable range queries
4. **Comment your schemas** - Add comments to explain field purposes
5. **Group related indexes** - Keep related indexes in the same SDL file
