# Intra-shard topic placement and redistribution

Status: proposed, September 20, 2026. No routing, merge, wire-format, or default
changes are implemented by this document. Scope is exclusively intra-shard:
primary-key hashing continues to select shards for inserts, deletes, and upserts.

Use one configured single-valued dense embedding field to place related documents
in segments within their existing shard. All other dense, sparse, text, and
multivalued fields travel with their document and use their existing pruning.

## Goal and current behavior

The objective is better segment and block locality, with bounded indexing memory
and rewrite costs. Search keeps its existing shard fanout, global text statistics,
and result combination. Similarity is a placement heuristic; each field's scorer
retains its own bounds. Gains in pruning and latency remain to be measured.

| Responsibility        | Today                                                      | Proposed change                                                                          |
| --------------------- | ---------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Distributed ingestion | Broker hashes the primary key to select a shard            | Keep that selection; optionally compute a topic hint using the destination shard's model |
| Native ingestion      | Shared bounded MPMC queue; workers collect mixed documents | Schedule bounded batches into topic-group builders                                       |
| ANN assignment        | Computed during segment construction                       | Compute early and reuse when artifact and assignment semantics match                     |
| ANN training          | Independent shard indexes                                  | Reuse each shard's existing trained model; shared cross-shard training is unnecessary    |
| Automatic merge       | Size-tier policy; segment ID and document count            | Add compact cell composition and locality-aware candidates                               |
| Rewrite               | Ordinary merge N→1, compaction 1→1, field-specific BP      | Explicit bounded whole-document N→M within one shard                                     |
| Delete/upsert         | Primary-key hash identifies the shard                      | Same path; an upsert's replacement may enter a different local topic group               |

Code owners: [broker partitioning](../summa-broker/src/partition.rs),
[broker writes](../summa-broker/src/index_service.rs),
[broker mutations](../summa-broker/src/index_service/mutations.rs),
[native writer](../summa-core/src/index/writer.rs),
[dense construction](../summa-core/src/segment/builder/dense.rs),
[merge policy](../summa-core/src/merge/mod.rs), and
[lossless compaction](../summa-core/src/segment/merger/compact.rs).
Direct server, CLI, and embedded writers also exist; they use the same core topic
assignment and grouping without requiring a broker.

## Stable cells, changing segments

Separate three identities:

1. **Model identity:** selected field, normalization, assignment parameters, and
   artifact hash, scoped to a physical shard index.
2. **Topic cell:** a leaf or an ancestor/group of leaves in that frozen model.
3. **Segment:** an immutable physical output holding one or more cells.

```text
primary key → existing hash → destination shard
                                  │
configured embedding → that shard's ANN router → topic cell
                                                   │
                                      local active builder group
                                                   │
                                       size-capped segments
```

The broker may execute the ANN-router step after selecting the destination shard;
it does not use the embedding to change that destination. Each shard owns the
cell-to-builder grouping. Broker hints name cells, never builder or segment IDs.

New documents select a topic group for new output. One cell can occupy many
segments; one segment can contain several nearby cells. Merging or splitting
segments does not rename cells.

Example: cell 17 has segments A and B. Merge A+B into C. New cell-17 rows continue
into active builder D; later C and D are natural merge candidates. C is never
reopened for appending. If A and B contain cells 17 and 18, C records both;
future builders may collect both under the shard's grouping plan.

A segment centroid is optional merge-selection metadata. A mean alone hides
multimodal segments and outliers. Prefer cell counts: merging adds counts and
preserves recognizable topics. A weighted centroid derived from those counts can
shortlist geometric neighbors; it is not an exact mean of document embeddings.

## Per-shard placement model

Reuse IVF-TQ's coarse centroids or ScaNN's routing tree for the configured field.
Independently trained shards may use different models and cell IDs. No shared
logical-index model, key-owner directory, shard reassignment, or distributed
cutover protocol is part of this design.

For each shard, reference an existing trained artifact as its placement model.
Choose a cell granularity with useful occupancy under that shard's builder and
segment budgets. Prefer an existing hierarchy level when possible; otherwise
combine nearby leaf centroids once. Numeric centroid IDs and modulo arithmetic
are not geometric grouping rules.

The placement model remains frozen across ordinary ANN retraining, retaining the
referenced artifact through the existing artifact lifecycle. Reuse an early ANN
assignment only while the placement and build models and parameters match. A
placement-model change is an explicit local rewrite/rebuild; normal merges never
train or move centroids. Do not compare cell IDs across different model identities.

If no trained model exists yet, index through ordinary local builders with an
unclassified placement label. Once the first model exists, new rows use it and a
bounded local N→M pass can classify earlier rows. Missing routing values use the
unclassified group and are counted separately. Bootstrap never changes shard
ownership or mutation semantics.

## Metadata and cost model

The existing shard index metadata owner publishes the selected field, placement
model reference, and grouping configuration. The broker caches immutable model
snapshots keyed by index, shard, and artifact identity. There is no new logical-index
control service. Local grouping changes do not invalidate the broker's cell hints.

Each segment records its placement model identity and exact compact cell counts.
Keep full counts in cold metadata, with bounded top-cell summaries plus other mass
for merge planning. A pure segment needs only a cell ID and count. Counts describe
physical rows; deletes can make them approximate for live composition until a
rewrite refreshes them. Redistribution uses captured visibility masks.

Persist one topic-cell label per document through existing integer column/block
codecs; uniform segments need only a constant label. There is no initial ownership
anchor or second routing label. Ordinals do not receive separate placement labels.
At 150M documents a two-byte column is 300 MB decimal before compression, versus
450 MB for three-byte IDs across the corpus. These are disk payload estimates,
not mandatory resident RAM. Width follows cell cardinality, not embedding vocabulary.

Placement runs once per document: about 150M assignments for the stated corpus,
not once for each of its 2–3B embedding values. Flat routing costs O(K × D) per
assignment; the existing hierarchical/graph routers avoid a full leaf scan at large
K. Compatible assignment reuse moves existing work earlier rather than repeating
it. Other fields still build their own indexes.

Caching a full model for every shard can make broker memory significant. Bound
that cache and share identical artifacts where available. A hierarchy-only cache
reduces broker RAM but leaves leaf assignment to segment construction. A cache
miss forwards documents without hints; the shard computes placement using the
same core router. Broker hinting is an optimization, not an availability dependency.
Report where assignments execute and why reuse was unavailable.

Redistribution costs at least input bytes read plus output bytes written, with
decode/repack CPU and bounded mapping scratch. Trigger it for measured dispersion
or fragmentation, with cooldowns and rewrite budgets. Do not assume that compact
placement metadata makes a multi-terabyte rewrite cheap.

## Ingestion and mutation flow

1. **Select the shard.** Broker admits the request and applies the existing primary-key
   hash before any embedding assignment. Preserve original batch positions.
2. **Compute optional hints.** For each destination, use its cached placement model
   to batch-normalize and assign vectors on a bounded CPU executor. Resolve the
   selected field through core schema handling, not another schema parser.
3. **Forward batches.** Include per-document cell/model identity and optionally the
   complete reusable ANN assignment in an internal routed-write envelope. Public
   documents do not gain hidden user fields. Preserve current timeout, acceptance,
   error indexing, and explicit commit behavior.
4. **Resolve placement at the shard.** Pin the local placement snapshot for the
   admitted batch. Use matching hints; absent or stale hints are recomputed locally
   before builder admission, with an observable counter. A stale hint cannot change
   ownership because the primary key already selected the shard. Trusted matching
   hints do not trigger a second vector assignment.
5. **Assign the builder.** Map the cell to a local topic group and enqueue bounded
   work under the existing writer's admission and pending-row ownership. Keep one
   placement identity per classified builder. Full ANN assignment reuse includes
   build parameters and SOAR secondary assignments where applicable; a coarse cell
   alone is not a full leaf assignment.
6. **Flush and publish.** Build immutable segments and publish every output of the
   prepared writer generation together through the existing lifecycle.

Deletes continue to hash the key and use the existing shard-local PK/deletion
path; no embedding is needed. Upserts hash the same key, compute a cell from the
replacement embedding, and atomically replace the old row through the current local
writer transaction. The replacement can change cells and segments freely within
its shard. Key uniqueness and pending replacement cancellation remain owned by the
existing writer. No key format change or separate key lookup is introduced.

Native direct-server/CLI writers compute placement before segment building using
the same core functions. The pure assignment/grouping representation stays portable;
WASM can execute it sequentially with its supported local builder/trained-artifact
capabilities. Native scheduling, RPC hints, and CPU pools do not become portable
core dependencies. Initial implementation and tests must explicitly report which
local/WASM placement capabilities are delivered rather than silently ignoring config.

## Shard-local builders and ordinary merges

The shard owns a local grouping plan: topic cell → builder group. Changing a
local grouping changes only future segment composition. Group cold neighboring cells to avoid tiny outputs.
Allow multiple builder lanes for a hot group so one popular topic does not bind
ingestion to one thread. Membership and ordering need not be perfect for locality.

Replace one unstructured builder per worker with a bounded scheduler over active
groups. Workers process bounded batches, with exclusive ownership of each active
builder while writing. Cap active builders and queued bytes under the existing
writer-wide memory budget, including finalization scratch. Flush large builders
under pressure; coalesce cold cells through the grouping plan rather than forcing
a flush on every cell switch. The worker pool size stays independent of cell count.
Preserve queued row cancellation/PK reservations and the existing all-or-nothing
prepared commit across every output builder.

Persist cell counts during construction. Extend merge metadata with compact
summaries and estimated encoded bytes/vector counts. Apply current eligibility,
fan-in, size, and resource budgets first. Generate bounded candidate sets from
same-cell and nearby-cell segments **within size tiers**, in addition to the
existing size-window candidates; a tie-breaker alone cannot select combinations
the current window generator never considers.

Prefer shared-cell mass, then geometric closeness of cell distributions, when
size/write-amplification costs are comparable. A simple first overlap score is
`sum_c min(p_A(c), p_B(c))`, where p is a normalized cell-mass histogram. Approximate
summaries may rank candidates; hard budgets use authoritative size metadata.
Keep policy thresholds explicit and benchmarked. Under serious segment backlog,
size reduction can override affinity. Never let locality prevent necessary merges.

Ordinary merge still copies compatible encoded runs/blocks and combines placement
columns/counts. Optional field BP preserves its existing independent ordinals.
Mixed segments are allowed; mark dispersion for later redistribution. A forced
single-segment merge explicitly sacrifices this layout and is not equivalent to
the new redistribution operation.

## Reuse of existing RGB/BP

Reuse the existing Recursive Graph Bisection kernel rather than implementing
another graph partitioner. The code already shares
[`ForwardIndex::from_csr` and `graph_bisection_with_progress`](../summa-core/src/segment/builder/graph_bisection.rs)
between text and BMP. It accepts entity-to-term memberships, returns
`order[new_position] = old_entity`, and supports depth/time budgets, bounded degree
scratch, cancellation, and warm starts through input ordering. Text and BMP already
plan across several source segments before writing a replacement.

The reusable pieces have different scopes:

| Piece                                          | Reuse for this proposal                     | Remaining work                                                                  |
| ---------------------------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------- |
| RGB partition kernel                           | Coarse offline grouping and finer ordering  | Adapt input entities/features to the chosen placement guide                     |
| Multi-source text/BMP forward construction     | Guide graph over selected sources           | Whole-document entities instead of independently movable chunks/vector ordinals |
| Depth/time/memory controls and background pool | Bound a redistribution planning pass        | Charge row maps, other fields, and all output writers too                       |
| Existing field RGB writers                     | Fine text/BMP order inside each destination | Compose destination row mapping with existing field-local maps                  |
| BMP block-level RGB                            | Reorder already coherent BMP blocks cheaply | Does not move entire documents or align blocks of unrelated fields              |
| Existing merge/compaction ownership            | N→M row selection and representation output | Generalize local publication from one replacement to the complete output set    |

Today's RGB is field-local: plain text has one scoring unit per document, chunked
text has one per value, and BMP can have several vectors per document. It leaves
the document store and unrelated fields in place. Its output cannot therefore be
treated as a whole-document destination map without adaptation. For document-level
RGB, aggregate/deduplicate the guiding field's memberships by document, retain
rows with no guide features, and let every field follow that one document mapping.
The shared graph stores membership, not floating-point embedding weights.

Dense centroid routing and RGB remain complementary. RGB is a batch optimizer;
it does not currently produce an out-of-sample classifier for arriving documents.
Keep centroid-derived cell IDs stable for ingestion. RGB partition positions and
branch paths are temporary planning results, not persistent routing identities.

For ordering, use existing per-field RGB inside the chosen output segments. For
coarse redistribution, a text/sparse guide can drive a shallow document-level RGB
pass within selected cell groups, after which the planner cuts its order into
size-capped outputs. This is an optional guide choice, not an assumption that the
configured dense field is directly consumable by the RGB kernel. Preserve source
order within each destination initially; use RGB's coarse order for destination
membership and existing per-field RGB for physical scoring order.

If clustering must use only the configured dense field, a possible adapter builds
a small graph over its existing centroids: each centroid entity receives shared
feature IDs representing nearby centroids or geometric neighborhoods. Run the same
RGB kernel on that graph to order/group cells, then assign documents via their
existing cell IDs. This is a new, unmeasured graph-construction adapter, not an
existing dense RGB implementation. A one-hot ID per centroid has no connection
between different cells and cannot establish geometric ordering; raw dense
coordinate presence also provides no useful sparse membership signal. Compare
this adapter with simply using the existing ANN hierarchy before adopting it.

The kernel bisects by entity count (`mid = n / 2`), not encoded bytes or vector
count. Cell occupancies can differ greatly. Keep hard capacity accounting in the
destination planner, split oversized groups, and allow cuts away from RGB subtree
boundaries for arbitrary M. A size/depth cap alone does not guarantee exactly M
balanced outputs. Exposing completed partition ranges can help planning, but do
not introduce a second bisection implementation or claim weighted balancing from
the current kernel.

The initial recommendation is centroid-based ingestion plus existing field RGB,
then bounded coarse RGB as an alternative redistribution planner. Benchmark guide
quality, graph construction, and rewrite cost separately. The graph's deadline
does not eliminate the source-read/output-write cost of the maintenance operation.
See [budgeted BP](budgeted-reorder.md), [block-level reorder](block-level-reorder.md),
and [merge-time reordering](merge-time-reorder.md) for implemented behavior.

## Local redistribution: N segments → M segments

This is an explicit maintenance operation with source-byte, output-count, scratch,
temporary-disk, concurrency, and time budgets. It can consolidate N→fewer, repair
N→N, or split N→more. It does not merge into one intermediate segment first.

1. **Select and claim.** Pick a bounded connected group of compatible segments
   using cell overlap/dispersion and small-segment backlog. Claim sources and all
   output IDs through the existing segment lifecycle owner. Capture visibility
   masks and the placement/model identities.
2. **Plan destinations.** Stream row placement labels and cost estimates. Pack
   neighboring cells into M size-capped outputs, splitting hot cells by stable
   row order where needed. Respect bytes and vector counts as well as docs.
   Existing unclassified/mixed data can compute a cell from the configured dense
   field once; this assignment pass must be counted in I/O and CPU estimates.
3. **Build mappings.** Assign each visible source row to exactly one destination.
   Within each output preserve `(source order, source doc ID)` order initially.
   This makes each source-to-output mapping monotone, reducing changes to posting
   codecs. Bound output fanout. Use packed destination labels and compact ranks,
   with disk-backed scratch where necessary; do not allocate M full-sized document
   arrays. Field chunk/ordinal maps derive from the same row ownership mapping.
4. **Rewrite through existing owners.** Extend lossless compaction's row selection
   and normal merge's multi-source machinery. Stream postings, positions, stores,
   fast fields, sparse forwards, and ANN runs directly into destination writers.
   Indexed-only fields come from their index structures, not document hydration.
   Copy independently reusable blocks/runs when every row has one destination;
   decode/repack mixed blocks and remap labels. Keep fixed per-vector ANN codes
   and trained artifacts when their representation permits it. Do not promise
   byte-copying for compressed blocks that straddle destinations.
5. **Bound passes.** Route each decoded source block to its outputs in one bounded
   fanout pass where the codec permits. Do not implement N→M by running complete
   compaction M times over the same inputs. If writer or codec budgets require
   multiple passes, bound and report the read amplification explicitly. Use cold
   bulk writers and bounded prefetch.
6. **Publish together.** Reconcile captured deletion generations under the existing
   publication owner. Initially abort/replan if any source visibility changed;
   this follows compaction's conservative rule and may defer very hot sources.
   Replace all N inputs with all M outputs in one local metadata generation,
   retaining unrelated new segments. Refresh the PK view before resuming affected
   writer publication. Old readers retain their old sources/masks.
7. **Recover.** Before publication, failure leaves the old set authoritative and
   outputs owned until cleanup. After the metadata commit, finish ownership and
   durability handling through the existing lifecycle. Cancellation never exposes
   half the output set. Optional bounded BP can run within destinations later.

The destination plan already narrows each output's topic coverage; a further global
document sort is not required for the first version. Existing per-field BP can improve
block order independently. This keeps redistribution distinct from retraining.

## Implementation stages and validation

1. Core per-shard model references, topic hints/columns, and bounded local builders;
   direct and broker-fed indexing use the same shard-local implementation.
2. Similarity-aware merge candidates and compact composition metadata.
3. Bounded local N→M redistribution through existing storage and lifecycle owners.
4. Broker computation/reuse of topic hints, measured against shard-only assignment.

These are parts of the same intra-shard scope. Broker hinting can be developed
alongside local routing but is not required to establish document placement.
Configuration names and format versions are finalized with implementation; this
proposal does not declare accepted schema attributes or change defaults. Old
indexes can be rebuilt; no legacy placement-format branch is required.

Core vector structures own normalized assignment and hierarchy traversal. Core
index owns placement state, builder scheduling, and writer epochs. Segment owners
write labels and N→M output; the existing segment manager owns claims, publication,
and retirement. Broker owns hash routing and optional hint computation/cache;
server owns wire conversion, admission, and existing lifecycle leases.

Behavior tests cover identical primary-key shard routing with and without hints,
new data after a merge, hot/cold builder groups, embedding-changing upserts, deletes
without embeddings, missing routing fields, and mixed/multivalued/indexed-only
fields. Compare hinted and local assignment, model-cache misses and stale hints,
and N→M query equivalence across scorers. Test queued replacement cancellation,
source deletions during redistribution, failures/panics, and readers held across
publication. All local outputs must publish together. Ordinary merge still copies
compatible encoded representations; optional redistribution owns the extra rewrite.

Measure on the same corpus/machine/compiler: current placement, local topic
placement, affinity merges, and N→M repair. Compare shard-only assignment with
broker hints. Report per-field blocks/candidates visited, bytes/faults, warm/cold
p50/p95/p99, QPS, ingest throughput, builder/scratch/metadata memory, broker model
cache residency, segment sizes, and rewritten/read bytes. Use the actual
150M-document / 2–3B-vector shape where feasible. Current pruning is the baseline;
no performance improvement has yet been measured.
