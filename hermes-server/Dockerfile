# Build stage
# Fully pinned builder: exact Rust version AND the same Debian release as
# the runtime stage below. The previous `rustlang/rust:nightly-slim` was a
# moving tag on BOTH axes — when upstream rebased it from bookworm to
# trixie (glibc 2.38), the binary stopped loading on the bookworm runtime
# ("GLIBC_2.38 not found", prod CrashLoopBackOff 2026-07-15). The server
# needs no nightly features; it builds on stable.
FROM rust:1.97.0-slim-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    protobuf-compiler \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy workspace files
COPY Cargo.toml Cargo.lock ./
COPY hermes-core ./hermes-core
COPY hermes-llm ./hermes-llm
COPY hermes-mal ./hermes-mal
COPY hermes-mal-python ./hermes-mal-python
COPY hermes-tokenizer ./hermes-tokenizer
COPY hermes-train ./hermes-train
COPY hermes-server ./hermes-server
COPY hermes-tool ./hermes-tool
COPY hermes-wasm ./hermes-wasm
COPY hermes-proto ./hermes-proto

# Build release binary
RUN cargo build --release -p hermes-server --bin hermes-server

# Runtime stage
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/hermes-server /usr/local/bin/hermes-server

# Default gRPC port
EXPOSE 50051

ENTRYPOINT ["hermes-server"]
