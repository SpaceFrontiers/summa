load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "io_bazel_rules_go",
    urls = ["https://github.com/bazelbuild/rules_go/releases/download/v0.34.0/rules_go-v0.34.0.zip"],
)
http_archive(
    name = "bazel_gazelle",
    sha256 = "501deb3d5695ab658e82f6f6f549ba681ea3ca2a5fb7911154b5aa45596183fa",
    urls = ["https://github.com/bazelbuild/bazel-gazelle/releases/download/v0.26.0/bazel-gazelle-v0.26.0.tar.gz"],
)
http_archive(
    name = "com_github_grpc_grpc",
    sha256 = "9b1f348b15a7637f5191e4e673194549384f2eccf01fcef7cc1515864d71b424",
    strip_prefix = "grpc-1.48.0",
    urls = ["https://github.com/grpc/grpc/archive/v1.48.0.tar.gz"],
)
http_archive(
    name = "io_bazel_rules_docker",
    sha256 = "b1e80761a8a8243d03ebca8845e9cc1ba6c82ce7c5179ce2b295cd36f7e394bf",
    urls = ["https://github.com/bazelbuild/rules_docker/releases/download/v0.25.0/rules_docker-v0.25.0.tar.gz"],
)
http_archive(
    name = "rules_rust",
    sha256 = "f3d443e9ad1eca99fbcade1c649adbd8200753cf22e47846b3105a43a550273b",
    strip_prefix = "rules_rust-0.8.1",
    urls = ["https://github.com/bazelbuild/rules_rust/archive/0.8.1.tar.gz"],
)
http_archive(
    name = "com_google_protobuf",
    sha256 = "85d42d4485f36f8cec3e475a3b9e841d7d78523cd775de3a86dba77081f4ca25",
    strip_prefix = "protobuf-3.21.4",
    urls = [
        "https://mirror.bazel.build/github.com/protocolbuffers/protobuf/archive/v3.21.4.tar.gz",
        "https://github.com/protocolbuffers/protobuf/archive/v3.21.4.tar.gz",
    ],
)
http_archive(
    name = "rules_proto_grpc",
    sha256 = "507e38c8d95c7efa4f3b1c0595a8e8f139c885cb41a76cab7e20e4e67ae87731",
    strip_prefix = "rules_proto_grpc-4.1.1",
    urls = ["https://github.com/rules-proto-grpc/rules_proto_grpc/archive/4.1.1.tar.gz"],
)
http_archive(
    name = "rules_python",
    sha256 = "f5f26431f86ab8dc991644d1577d6129e2707e6ae580ee9b500d1eb27547a265",
    strip_prefix = "rules_python-ffab6b87dd8b85822c7d1f60a3fef42bc464485e",
    url = "https://github.com/ppodolsky/rules_python/archive/ffab6b87dd8b85822c7d1f60a3fef42bc464485e.tar.gz",
)

# Proto
load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")
protobuf_deps()

# GRPC
load("@rules_proto_grpc//:repositories.bzl", "rules_proto_grpc_repos", "rules_proto_grpc_toolchains")
rules_proto_grpc_toolchains()
rules_proto_grpc_repos()
load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")
grpc_deps()
load("@com_github_grpc_grpc//bazel:grpc_extra_deps.bzl", "grpc_extra_deps")
grpc_extra_deps()

# Rust
load("@rules_rust//rust:repositories.bzl", "rust_register_toolchains", "rules_rust_dependencies")
rules_rust_dependencies()
rust_register_toolchains(
    version="1.62.0",
)
load("@rules_rust//crate_universe:repositories.bzl", "crate_universe_dependencies")
crate_universe_dependencies(bootstrap = True)
load("@rules_rust//crate_universe:defs.bzl", "crate", "crates_repository", "render_config")
crates_repository(
    name = "crate_index",
    cargo_lockfile = "//:Cargo.lock",
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
    }
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
load("@io_bazel_rules_docker//container:container.bzl", "container_pull")
container_pull(
    name = "izihawa-base-image",
    digest = "sha256:878db474e52ad144ecf5faafb64c795989a83e3bbcccfaa005b743e51942a653",
    registry = "index.docker.io",
    repository = "izihawa/base-image",
)