FROM rust as builder
WORKDIR app
COPY .cargo .cargo
COPY examples examples
COPY summa-core summa-core
COPY summa-embed-py summa-embed-py
COPY summa-proto summa-proto
COPY summa-server summa-server
COPY summa-wasm summa-wasm
COPY Cargo.toml Cargo.toml
COPY Cargo.lock Cargo.lock
COPY rustfmt.toml rustfmt.toml
RUN cargo build --profile release -p summa-server

FROM gcr.io/distroless/cc
COPY --from=builder /app/target/release/summa-server-bin /bin/summa-server
ENTRYPOINT ["/bin/summa-server"]
