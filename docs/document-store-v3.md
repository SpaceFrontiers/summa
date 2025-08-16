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
