# syntax=docker/dockerfile:1

# summa-k8s-manager: SearchCluster operator and console. Build from the
# repository root: docker build -f summa-k8s-manager/Dockerfile .
# The builder matches the runtime's Debian release so the binary loads
# against the runtime glibc.
FROM rust:1.98.0-slim-bookworm AS builder

# Bootstrap from the published base until official 1.98.1 image tags exist.
RUN rustup toolchain install 1.98.1 --profile minimal \
    && rustup default 1.98.1

RUN apt-get update && apt-get install -y \
    protobuf-compiler \
    pkg-config \
    make \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cargo resolves the whole workspace, so every member manifest must exist.
COPY Cargo.toml Cargo.lock ./
COPY summa-broker ./summa-broker
COPY summa-core ./summa-core
COPY summa-k8s-manager ./summa-k8s-manager
COPY summa-llm ./summa-llm
COPY summa-mal ./summa-mal
COPY summa-mal-python ./summa-mal-python
COPY summa-tokenizer ./summa-tokenizer
COPY summa-train ./summa-train
COPY summa-server ./summa-server
COPY summa-tool ./summa-tool
COPY summa-wasm ./summa-wasm
COPY summa-proto ./summa-proto

RUN --mount=type=cache,id=summa-k8s-manager-cargo-registry,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,id=summa-k8s-manager-cargo-git,target=/usr/local/cargo/git,sharing=locked \
    --mount=type=cache,id=summa-k8s-manager-cargo-target,target=/app/target,sharing=locked \
    cargo build --locked --release -p summa-k8s-manager --bin summa-k8s-manager \
    && install -Dm755 target/release/summa-k8s-manager /out/summa-k8s-manager

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /out/summa-k8s-manager /usr/local/bin/summa-k8s-manager

# Console (localhost only by default) and Prometheus metrics.
EXPOSE 8080 9184
USER 65532:65532
ENTRYPOINT ["summa-k8s-manager"]
CMD ["run"]
