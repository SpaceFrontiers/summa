# Owned byte views

## Current behavior

`directories::OwnedBytes` retains an `Arc<Vec<u8>>` or `Arc<Mmap>` and a
`NonNull<[u8]>` view validated at construction and slicing. Posting/position
readers, norms, fast fields and dictionaries share this owner; filesystem,
memory and HTTP directories construct it. Native mmap support is feature-gated;
the Vec owner also serves portable/WASM code. No persisted or wire format
includes this Rust struct.

The direct view replaces the previous byte range at the same struct footprint
(32 bytes in the native build, 24 in the portable build). Subsequent `as_slice`
reads its pointer/length without resolving the backing enum, following the Arc
or checking the same range again. The backing owner and mmap classification
remain intact; no extra owner allocation or resident payload copy is added.
The [ownedbytes implementation](https://docs.rs/ownedbytes/latest/src/ownedbytes/lib.rs.html)
used in Tantivy illustrates the same direct-view ownership principle.

Matched screens supported retaining the change. The complete selected reader's
latency and residency are reported in the block-execution results;
individual stage gains must not be multiplied. The owner passes native heap,
mmap, cross-thread lifetime tests and three strict-provenance Miri tests on an
isolated extraction of the actual implementation.

## Invariants and safety argument

- Every stored pointer/length comes from a checked slice of the owned allocation.
  A child slice is bounded by its parent view, including empty and nested views.
- Both backing allocations are stable while their Arc exists. Moving or cloning
  `OwnedBytes` moves/clones the handle, without moving the Vec buffer or mapping.
- OwnedBytes exposes immutable slices only. Its private Arc<Vec> is never mutated
  or returned; an external Arc clone cannot obtain unique mutable access while
  the view retains another strong reference. Mmaps retain the existing immutable
  mapping contract.
- Every clone retains the backing Arc. Dropping a parent, external Arc, file
  handle or sibling cannot release storage still used by a surviving view.
- `as_slice` returns a lifetime tied to `&self`; it cannot expose an independent
  static reference. Dereferencing the stored view is the sole unsafe read.
- Send/Sync follow from the immutable, stable storage and the Send/Sync owners.
  Document these conditions beside explicit implementations for the raw view.
- Mmap-only advice and locking still inspect the backing owner. Heap data must
  never enter mmap advice, and a view does not introduce residency or pinning.

Construction and slicing reject invalid ranges immediately. The old range-only
implementation could let a nested slice escape its parent while still remaining
inside the original allocation; the regression first demonstrated that boundary failure.
Tests cover dropped owners, cross-thread clones, empty views, unaligned
subviews, invalid/reversed ranges and mmap ownership. The search harness,
native-without-sync and portable compilation pass, along with whole-fixture
score/count and immutable-byte comparisons. WASM rebuilding remains skipped by user request.
