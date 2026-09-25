# Summa 2 migration

Summa 2 uses the `SpaceFrontiers/summa` repository and the `summa` namespace
throughout its packages, clients, commands, configuration, metrics and UI.
This is a breaking namespace release, not a migration of the original Summa
0.x engine's indexes or API. Rebuild 0.x indexes from their source documents.

## Compatibility and ownership

The core remains the sole owner of search, scoring and index encodings. Native,
async-only and WASM adapters use the same renamed core. Search-index format
versions, magic bytes, field ordinals and encoded segment layouts are unchanged
from the immediately preceding 1.9.1 implementation. Renaming does not add a
query-time translation layer, allocation, cache, scorer or writer. Query and
merge cost models are unchanged; no performance improvement is claimed.

Upgrade servers, brokers and clients together. Protobuf message field numbers
and types are unchanged, but fully qualified RPC service names now begin with
`summa` (and `summa.broker` for broker control). Regenerate bindings from
`summa-proto/summa.proto`; mixed namespace deployments are unsupported.

- Rust crates and binaries use `summa-*`; Rust imports use `summa_*`.
- Python uses `summa-client-python` / `summa_client_python` and
  `summa-mal` / `summa_mal`. The TypeScript client is
  `summa-client-typescript`; the browser package is `summa-wasm`.
- Rename environment variables to `SUMMA_*`, monitoring series to `summa_*`,
  Kubernetes labels to `summa.spacefrontiers.org/*`, and deployment names and
  volume references to their `summa` counterparts before restarting services.
- Browser databases, local storage keys and model artifact cache directories
  use the new namespace. Export needed local browser data before upgrading;
  the new UI does not discover databases under the previous namespace.
- Model trace identifiers and training hash-domain identifiers use the new
  namespace. Start fresh training runs and regenerate receipts, trace files and
  content-addressed training artifacts; search-index compatibility does not
  imply training checkpoint compatibility.

The original Summa blog and its assets are retained in the GitHub Pages source.
Historical articles describe their original implementation; current package and
API documentation is maintained alongside the code. The website build reads
its API, schema and operations guides directly from these maintained sources
through `scripts/build_website.py`, keeping the published guides in sync.

## Release and validation

The first namespace release is 2.0.0. Internal crate requirements must match
the released version, and crates must be published in dependency order.
Packaged server and broker sources must contain their protobuf inputs so
registry consumers can build them outside the monorepo.

The `summa-proto` source crate owns those inputs. Server and broker build
scripts copy its embedded schemas and shared mutation validation into private
Cargo output directories before invoking their existing generators. This adds
build-time file copies only, with no runtime parser or search changes.

Validate the renamed search stack with `python3 scripts/check_search.py check`
and `full`, regenerate both clients, build and test WASM, and check package
contents and documentation links. Compare unchanged binary index fixtures and
the protobuf schema's field definitions; record results and environmental
limitations in [the performance review](search-performance-review.md).
