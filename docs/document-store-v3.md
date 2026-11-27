# Document store v3

Document store v3 removes the `u16` ceiling on the number of stored
field-values in one document. The count at the start of every serialized
document is now a little-endian `u32`; the per-value field ID remains `u16`.
Repeated values count separately.

This is an intentionally incompatible format change. Readers accept only v3
stores, so indexes containing v2 segments must be rebuilt before deployment.
Raw block copying during merge remains safe because an old segment cannot be
opened and admitted to a v3 merge.

The wider count adds two uncompressed bytes per document. Vector ordinals
remain `u16`, so an indexed vector field can contain 65,536 values, represented
by ordinals `0..=65535`. Writers reject a document that exceeds that independent
limit before it enters an indexing generation; this keeps one invalid document
from poisoning the rest of a commit batch.

The maximum decompressed document-store block size is 256 MiB (formerly 64
MiB). A serialized document, including its four-byte block length prefix, must
fit within that ceiling.

## Windowed value decoding (2.2.7)

`SearchRequest.ordinal_window` returns a multi-valued stored field only at a
hit's best ordinals, but the store still validated, parsed and allocated every
value of the field before the server discarded most of them: for a book with
thousands of chunk spans, thousands of JSON parses per hit. `StoredValueWindow`
(field ids plus ascending positions) now travels with the read
(`Searcher::get_document_windowed`, `DocumentReadBatch::get_windowed`). Values of
those fields outside the window keep their position in the returned document as
empty placeholders (empty text or bytes, JSON null); other fields and value
types decode as before. The caller must read only the positions it asked for.
Block lookup, decompression and the shared block cache are unchanged.
Regression: `stored_value_window_decodes_only_requested_positions`.
