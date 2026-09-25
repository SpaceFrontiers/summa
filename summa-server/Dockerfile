# Build stage
# Fully pinned builder: exact Rust version AND the same Debian release as
# the runtime stage below. The previous `rustlang/rust:nightly-slim` was a
# moving tag on BOTH axes — when upstream rebased it from bookworm to
# trixie (glibc 2.38), the binary stopped loading on the bookworm runtime
# ("GLIBC_2.38 not found", prod CrashLoopBackOff 2026-07-15). The server
# needs no nightly features; it builds on stable.
FROM rust:1.98.0-slim-bookworm AS builder

# Bootstrap from the published base until official 1.98.1 image tags exist.
RUN rustup toolchain install 1.98.1 --profile minimal \
    && rustup default 1.98.1

# Install build dependencies
RUN apt-get update && apt-get install -y \
    protobuf-compiler \
    pkg-config \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy workspace files
COPY Cargo.toml Cargo.lock ./
COPY summa-broker ./summa-broker
COPY summa-core ./summa-core
COPY summa-llm ./summa-llm
COPY summa-mal ./summa-mal
COPY summa-mal-python ./summa-mal-python
COPY summa-tokenizer ./summa-tokenizer
COPY summa-train ./summa-train
COPY summa-server ./summa-server
COPY summa-tool ./summa-tool
COPY summa-wasm ./summa-wasm
COPY summa-proto ./summa-proto

# Build release binary
RUN cargo build --release -p summa-server --bin summa-server

# Runtime stage
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/summa-server /usr/local/bin/summa-server

# Default gRPC port
EXPOSE 50051

ENTRYPOINT ["summa-server"]
