# Index compatibility

Summa 2.1 reads only indexes written in the current format. It performs no
migration on open: an index written by an older layout fails with an error
that says to rebuild it, instead of being upgraded or read through a fallback
path.

## What 2.1 opens

- `metadata.json` at format **11** (`INDEX_META_FORMAT_VERSION`). Indexes
  written by Summa 2.0.x are already at format 11. Fields added later inside
  format 11 keep their serde defaults, so metadata from any 2.0.x writer still
  loads.
- Chunk maps (`.chunks`) at versions 3–5. Versions 1 and 2 (unaddressed chunk
  maps and the 24-byte TOC) are rejected.
- Posting lists with the 44-byte `BPL2` footer and packed `(max_tf, min_len)`
  block bounds. The 24-byte footer with `f32` block maxima is rejected.
- Segments that persist norms or a chunk map for every scored text field,
  `.rowstats` for every indexed text and sparse field, and `.fast` whenever the
  schema has a columnar field. A segment missing any of these fails at open;
  queries no longer fall back to `tf` as the length.
- Plain `reorder` text fields with a document map. Merges no longer synthesize
  identity maps for segments that lack one.
- Sparse vector configs that name their `format`. A missing `format` used to
  mean MaxScore; it is now an error.
- IVF-TQ centroids and payloads stamped with the cosine generation marker. The
  marker is checked once when the index opens: metadata load checks centroids
  and ANN open checks payloads.

## Removed

- `ivf_pq` (SDL keyword, `VectorIndexType::IvfPq`, `codebook_file` metadata).
  ANN TOC type 2 and `AnnKind` 1 stay reserved and are never reused.
- The SDL alias `soar: aggressive`. Use `soar: full`.
- The JSON `SchemaConfig` schema format. `parse_schema` accepts SDL only.
- `summa-tool train-centroids` and the `retrain-centroids` placeholder. Use
  the server's `RetrainVectorIndex`.
- The `pnpm lab:*` forwarding scripts in `summa-web`. Run them from
  `summa-model-lab` directly.

Standalone posting codecs remain readable and writable; see
[posting codecs](posting-codecs.md).

## Upgrading

Rebuild and republish any index that 2.1 rejects. Production indexes written
by 2.0.x need no rebuild: they are at format 11, with chunk maps v3, the
current posting footer, row statistics and fast columns.
