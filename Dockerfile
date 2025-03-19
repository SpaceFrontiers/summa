FROM rust:1.94

# System deps
RUN apt-get update && apt-get install -y \
    clang \
    llvm \
    protobuf-compiler \
    curl \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1000 rustuser && \
    useradd --uid 1000 --gid 1000 --create-home rustuser

ENV RUSTUP_HOME=/home/rustuser/.rustup
ENV CARGO_HOME=/home/rustuser/.cargo
ENV PATH="/home/rustuser/.cargo/bin:$PATH"

USER rustuser
WORKDIR /app

# Setup rustup
RUN rustup default stable

# Install wasm-pack
RUN cargo install wasm-pack