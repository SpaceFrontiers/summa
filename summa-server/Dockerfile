FROM ubuntu:22.04 as build-env
RUN apt install zlib1g

FROM gcr.io/distroless/cc
COPY --from=build-env /lib/x86_64-linux-gnu/libz.so.1 /lib/x86_64-linux-gnu/libz.so.1
COPY summa-server-bin /bin/summa-server
ENTRYPOINT ["/bin/summa-server"]