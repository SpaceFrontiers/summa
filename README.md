![Maintenance](https://img.shields.io/badge/maintenance-activly--developed-brightgreen.svg)

# summa

Fast full-text search server

### Features
- Simple GRPC API for managing multiple indices and search
- Indexing documents through Kafka or directly
- IPFS API for replication (oncoming)
- Tracing with [OpenTelemetry](https://github.com/open-telemetry/opentelemetry-rust) and exposing metrics in Prometheus format
- Various configurable tokenizers (including CJK)
- Fine CLI and client library written in Python

### Getting started
- Look at [examples](src/examples/README.md)
- Explore [Docs.rs](https://docs.rs/crate/summa/latest)
