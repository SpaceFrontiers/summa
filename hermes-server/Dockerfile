# Build stage
FROM rustlang/rust:nightly-slim AS builder

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
