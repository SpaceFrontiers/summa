![Maintenance](https://img.shields.io/badge/maintenance-activly--developed-brightgreen.svg)
[![Crates.io](https://img.shields.io/crates/v/summa.svg)](https://crates.io/crates/summa)
[![Docs](https://docs.rs/summa/badge.svg)](https://docs.rs/crate/summa/)

# Summa

Fast full-text search server written in Rust with CLI and client library in Python.
Index documents and do search queries through Python client, CLI and/or GRPC API

Documentation: [Github.io](https://izihawa.github.io/summa/) and [docs.rs](https://docs.rs/crate/summa/latest)

### Features
- Indexing documents directly through GRPC API or Kafka
- Fine CLI and asynchronous client library [aiosumma](aiosumma/README.md) written in Python
- Simple GRPC API for managing multiple indices and for search
- Extendable query parsing. Add synonyms and build query context with Python library
- Fast document ranking with custom and/or user-defined scoring functions
- Tracing with [OpenTelemetry](https://github.com/open-telemetry/opentelemetry-rust) and exposing metrics in Prometheus format
- Reflection API for deep insights about text data

### Getting started
- Quick start [guide](https://izihawa.github.io/summa/quick-start)
- Explore [Docs.rs](https://docs.rs/crate/summa/latest)
