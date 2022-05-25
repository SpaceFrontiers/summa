FROM rust as builder
WORKDIR app
COPY .cargo .cargo
COPY examples examples
COPY ipfs-hamt-directory ipfs-hamt-directory
COPY ipfs-hamt-directory-py ipfs-hamt-directory-py
COPY summa-core summa-core
COPY summa-embed-py summa-embed-py
COPY summa-proto summa-proto
COPY summa-server summa-server
COPY summa-wasm summa-wasm
COPY Cargo.toml Cargo.toml
COPY rustfmt.toml rustfmt.toml
RUN cargo build --profile release -p summa-server

FROM gcr.io/distroless/cc
COPY --from=builder /app/target/release/summa-server-bin /bin/summa-server
ENTRYPOINT ["/bin/summa-server"]
