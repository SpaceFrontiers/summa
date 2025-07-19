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
field name: text<default> [indexed, stored]     # Default (lowercase)
field raw: text [indexed, stored]               # Default tokenizer
```

### Available Tokenizers

| Name      | Aliases      | Description                                            |
| --------- | ------------ | ------------------------------------------------------ |
| `default` | `lowercase`  | Lowercase tokenizer (splits on whitespace, lowercases) |
| `simple`  | `raw`        | Simple whitespace tokenizer (no lowercasing)           |
| `en_stem` | `english`    | English Snowball stemmer                               |
| `de_stem` | `german`     | German Snowball stemmer                                |
| `fr_stem` | `french`     | French Snowball stemmer                                |
| `es_stem` | `spanish`    | Spanish Snowball stemmer                               |
| `it_stem` | `italian`    | Italian Snowball stemmer                               |
| `pt_stem` | `portuguese` | Portuguese Snowball stemmer                            |
| `ru_stem` | `russian`    | Russian Snowball stemmer                               |
| `ar_stem` | `arabic`     | Arabic Snowball stemmer                                |
| `da_stem` | `danish`     | Danish Snowball stemmer                                |
| `nl_stem` | `dutch`      | Dutch Snowball stemmer                                 |
| `fi_stem` | `finnish`    | Finnish Snowball stemmer                               |
| `el_stem` | `greek`      | Greek Snowball stemmer                                 |
| `hu_stem` | `hungarian`  | Hungarian Snowball stemmer                             |
| `no_stem` | `norwegian`  | Norwegian Snowball stemmer                             |
| `ro_stem` | `romanian`   | Romanian Snowball stemmer                              |
| `sv_stem` | `swedish`    | Swedish Snowball stemmer                               |
| `ta_stem` | `tamil`      | Tamil Snowball stemmer                                 |
| `tr_stem` | `turkish`    | Turkish Snowball stemmer                               |

### Custom Tokenizers

You can register custom tokenizers programmatically:

```rust
use hermes_core::{TokenizerRegistry, LowercaseTokenizer};

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

field_def = { "field" ~ identifier ~ ":" ~ field_type ~ tokenizer_spec? ~ attributes? }

field_type = {
    "text" | "string" | "str" |
    "u64" | "uint" | "unsigned" |
    "i64" | "int" | "integer" |
    "f64" | "float" | "double" |
    "bytes" | "binary" | "blob" |
    "json" |
    "dense_vector" | "sparse_vector"
}

tokenizer_spec = { "<" ~ identifier ~ ">" }

attributes = { "[" ~ attribute ~ ("," ~ attribute)* ~ "]" }
attribute = { indexed_with_config | "indexed" | stored_with_config | "stored" | "fast" | "primary" }

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

Float fields have one production ANN format: corpus-trained global IVF-PQ.
Every segment shares the index's centroid router and residual-PQ codebook;
segments store only compact assignments and PQ codes. `flat` remains available
for exact brute-force search and as the accumulation format before ANN build.

```
field e: dense_vector<768, f16> [indexed]                                      # global IVF-PQ
field e: dense_vector<768, f16> [indexed<ivf_pq, routing: hnsw, nprobe: 64>]
field e: dense_vector<768, f16> [indexed<flat>]                                # exact full scan
field e: dense_vector<768> [stored]                                            # stored, not indexed
```

When `num_clusters` is omitted, training chooses a corpus-sized leaf count
using an 8×sqrt(N) target bounded by sample quality and artifact memory. Routing
can be `auto`, `flat`, `two_level`, or `hnsw`; `auto` uses flat centroid scoring
below 4,096 leaves and HNSW above it. HNSW is a global coarse-quantizer index,
so its work is done once per query rather than once per segment. The default
`nprobe` is 64. Candidate collection keeps distinct documents, is bounded to
3×k, and exact reranking reads at most that bounded set across all segments.

### SOAR (higher recall for IVF indexes)

IVF-PQ supports SOAR — spilling each vector
into a secondary cluster with an orthogonality-amplified residual. This
improves recall at the same `nprobe` in exchange for larger cluster storage
(~1.2-2x assignments):

```
field e: dense_vector<768, f16> [indexed<ivf_pq, soar: selective>]  # spill boundary vectors
field e: dense_vector<768, f16> [indexed<ivf_pq, soar: full>]       # spill every vector once
field e: dense_vector<768, f16> [indexed<ivf_pq, soar: aggressive>] # spill every vector twice
field e: dense_vector<768, f16> [indexed<ivf_pq, soar: off>]        # no spilling (default)
```

### Example

```
index documents {
    field title: text<en_stem> [indexed, stored]
    field embedding: dense_vector<768, f16> [indexed<ivf_pq>, stored<multi>]
    field span: json [stored<multi>]
}
```

### Binary Vector IVF

Binary dense vector fields use the same global IVF router and HNSW topology as
float fields, with metric-specific k-majority centroids and exact packed-code
leaf scanning. The only approximation is which clusters get probed; there is
no lossy PQ stage or rerank for exact Hamming codes. Use `flat` explicitly for
brute-force SIMD Hamming scan:

```
field hash: binary_dense_vector<512> [indexed<ivf, routing: hnsw, nprobe: 64>]
field hash: binary_dense_vector<512> [indexed<flat>]
```

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
concentrate importance in a few head terms, so `doc_mass: 0.9` typically drops
20-40% of postings with <1% nDCG loss. Vectors with at most `min_terms` entries
are never cropped.

```
field emb: sparse_vector [indexed<quantization: uint8, doc_mass: 0.9>]
```

Use `hermes-tool info <index>` to inspect the resulting average sparse vector
length (`avg terms/vector`).

### BMP Format Options

The BMP (block-max pruning) format takes additional `indexed<...>` options:

```
field emb: sparse_vector<u32> [indexed<format: bmp, dims: 105879, max_weight: 5.0,
    bmp_block_size: 256>, reorder]
```

- `bmp_block_size` (power of two, max 256; default 64) — docs per BMP
  block, uniform across every segment of the field. The dense 4-bit grid
  is `dims × num_blocks / 2` bytes, so grid memory scales as
  1/block_size; smaller blocks give finer pruning granularity. Large
  corpora (10M+ docs at 100k-dim vocabularies) should set 256 — at 50M
  docs the grid is ~41 GB at block 64 vs ~10 GB at 256. See
  `docs/bmp-grid-compression.md`.

### Quantization and Pruning

Sparse posting lists support configurable weight quantization and pruning via `SparseVectorConfig`:

| Preset         | Quantization | Compression | Speed    | Quality loss |
| -------------- | ------------ | ----------- | -------- | ------------ |
| `conservative` | Float16      | 2x          | Baseline | <1%          |
| `splade`       | UInt8        | 5-7x        | 40-60%   | 2-4%         |
| `compact`      | UInt4        | 7-10x       | 50-70%   | 3-5%         |

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
