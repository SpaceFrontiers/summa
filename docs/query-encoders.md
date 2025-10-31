# Query encoders

A vector field can name the model its queries are encoded with:

```
field sparse_vectors: sparse_vector<u32> [indexed<format: bmp, dims: 128256, query<encoder: "akin_embed_v1">>]
field binary_embeddings: binary_dense_vector<2560> [indexed<ivf, query<encoder: "pplx_embed_v1_4b">>]
```

The engine never calls the model. The declaration lets clients learn, per
physical index, which encodings a query needs instead of inferring it from
field names or keeping a side table that drifts from the index.

## Contract

- Allowed on `sparse_vector`, `dense_vector` and `binary_dense_vector` fields
  only. The name is 1..=128 characters of `[A-Za-z0-9._-:/]`. Anything else
  fails schema admission.
- A sparse field cannot declare both `query<encoder>` and
  `query<tokenizer>`: an encoded field takes explicit `indices`/`values` only.
  A `SparseVectorQuery` with `text` on such a field fails with
  `Field '<f>' takes queries encoded by '<encoder>': send explicit
  indices/values, not text` instead of falling through to the tokenizer path.
- `GetIndexInfo` reports the declarations in `query_encoders`
  (`QueryEncoderInfo { field, encoder }`, schema order); the rendered SDL in
  `schema` carries them too. The broker returns the first partition's list,
  like `text_fields`: partitions of one index share one schema.
- Query-side policy (pruning mass, term caps, timeouts, caching) stays with the
  client: the name identifies the model, not how a caller budgets it.

## Compatibility

`FieldEntry.query_encoder` is `#[serde(default, skip_serializing_if =
"Option::is_none")]`: existing indexes reopen unchanged, and metadata without
an encoder serializes byte-identically. Older servers ignore the persisted
field but reject the SDL option at creation, so create encoder-declaring
indexes only on 2.2.2+. Declaring an encoder on an existing index is a schema
metadata update; no segment data changes.
