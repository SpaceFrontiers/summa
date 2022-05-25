FROM rust as builder
WORKDIR app
COPY . .
RUN cargo build --profile release -p summa-server

FROM gcr.io/distroless/cc
COPY --from=builder /app/target/release/summa-server-bin /bin/summa-server
ENTRYPOINT ["/bin/summa-server"]
