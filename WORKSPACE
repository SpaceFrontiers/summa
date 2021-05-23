load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "io_bazel_rules_go",
    urls = ["https://github.com/bazelbuild/rules_go/releases/download/v0.32.0/rules_go-v0.32.0.zip"],
)
http_archive(
    name = "bazel_gazelle",
    urls = ["https://github.com/bazelbuild/bazel-gazelle/releases/download/v0.25.0/bazel-gazelle-v0.25.0.tar.gz"],
)
http_archive(
    name = "com_github_grpc_grpc",
    sha256 = "291db3c4e030164421b89833ee761a2e6ca06b1d1f8e67953df762665d89439d",
    strip_prefix = "grpc-1.46.1",
    urls = ["https://github.com/grpc/grpc/archive/v1.46.1.tar.gz"],
)
http_archive(
    name = "io_bazel_rules_docker",
    sha256 = "59536e6ae64359b716ba9c46c39183403b01eabfbd57578e84398b4829ca499a",
    strip_prefix = "rules_docker-0.22.0",
    urls = ["https://github.com/bazelbuild/rules_docker/releases/download/v0.22.0/rules_docker-v0.22.0.tar.gz"],
)
http_archive(
    name = "rules_rust",
    sha256 = "d0f505b8801e05900b126e239259152062a6ee523b4f2013c0d7ca332b915a26",
    strip_prefix = "rules_rust-0.5.0",
    urls = ["https://github.com/bazelbuild/rules_rust/archive/0.5.0.tar.gz"],
)
http_archive(
    name = "rules_proto_grpc",
    sha256 = "507e38c8d95c7efa4f3b1c0595a8e8f139c885cb41a76cab7e20e4e67ae87731",
    strip_prefix = "rules_proto_grpc-4.1.1",
    urls = ["https://github.com/rules-proto-grpc/rules_proto_grpc/archive/4.1.1.tar.gz"],
)
http_archive(
    name = "rules_python",
    sha256 = "cdf6b84084aad8f10bf20b46b77cb48d83c319ebe6458a18e9d2cebf57807cdd",
    strip_prefix = "rules_python-0.8.1",
    urls = ["https://github.com/bazelbuild/rules_python/archive/0.8.1.tar.gz"],
)

# Go
load("@io_bazel_rules_go//go:deps.bzl", "go_register_toolchains", "go_rules_dependencies")
load("@bazel_gazelle//:deps.bzl", "gazelle_dependencies")
go_rules_dependencies()
go_register_toolchains(version = "1.18.2")
gazelle_dependencies()

# GRPC
load("@rules_proto_grpc//:repositories.bzl", "rules_proto_grpc_repos", "rules_proto_grpc_toolchains")
rules_proto_grpc_toolchains()
rules_proto_grpc_repos()
load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")
grpc_deps()

# Rust
load("@rules_rust//rust:repositories.bzl", "rust_register_toolchains", "rust_repository_set", "rules_rust_dependencies")
rust_repository_set(
    name = "rust_darwin_x86_64_cross",
    exec_triple = "x86_64-apple-darwin",
    extra_target_triples = ["x86_64-unknown-linux-gnu"],
    version = "1.61.0"
)
rules_rust_dependencies()
rust_register_toolchains()
load("@rules_rust//crate_universe:repositories.bzl", "crate_universe_dependencies")
crate_universe_dependencies(bootstrap = True)
load("@rules_rust//crate_universe:defs.bzl", "crate", "crates_repository")
crates_repository(
    name = "crate_index",
    lockfile = "//:Cargo.lock",
    manifests = ["//:Cargo.toml"],
    generator = "@cargo_bazel_bootstrap//:cargo-bazel",
    supported_platform_triples = [
        "aarch64-apple-darwin",
        "i686-apple-darwin",
        "i686-unknown-linux-gnu",
        "x86_64-apple-darwin",
        "x86_64-unknown-linux-gnu",
        "wasm32-unknown-unknown",
        "wasm32-wasi",
    ],
    annotations = {
        "rdkafka-sys": [crate.annotation(
            build_script_env = {"CARGO_MAKEFLAGS": ""},
        )],
        "prost-build": [crate.annotation(
            build_script_data = [
                "@com_google_protobuf//:protoc",
            ],
            build_script_env = {
                "PROTOC": "$(execpath @com_google_protobuf//:protoc)",
                "PROTOC_NO_VENDOR": "1",
            },
        )],
        "tonic-reflection": [crate.annotation(
            build_script_data = [
                "@com_google_protobuf//:protoc",
            ],
            build_script_env = {
                "PROTOC": "$(execpath @com_google_protobuf//:protoc)",
            },
        )],
    },
)
load("@crate_index//:defs.bzl", "crate_repositories")
crate_repositories()

# Python
load("@rules_python//python:repositories.bzl", "python_register_toolchains")
python_register_toolchains(
    name = "python3_10",
    python_version = "3.10",
)

load("@python3_10//:defs.bzl", "interpreter")
load("@rules_python//python:pip.bzl", "pip_parse")
pip_parse(
    name = "pip_modules",
    python_interpreter_target = interpreter,
    requirements_lock = "//aiosumma:requirements-lock.txt",
)
load("@pip_modules//:requirements.bzl", "install_deps")
install_deps()

# Proto / gRPC
load("@rules_proto_grpc//:repositories.bzl", "rules_proto_grpc_repos", "rules_proto_grpc_toolchains")
rules_proto_grpc_toolchains()
rules_proto_grpc_repos()

# Docker
load("@io_bazel_rules_docker//toolchains/docker:toolchain.bzl", docker_toolchain_configure="toolchain_configure")
docker_toolchain_configure(name="docker_config")
load("@io_bazel_rules_docker//repositories:repositories.bzl", container_repositories = "repositories")
container_repositories()
load("@io_bazel_rules_docker//repositories:deps.bzl", container_deps = "deps")
container_deps()
load("@io_bazel_rules_docker//rust:image.bzl", rust_image_repos = "repositories")
rust_image_repos()
